"""Lens validation gate (pre-registration section 7, first bullet; metric refined per
docs/DEVIATIONS.md entry 1 — the original pass@10 min-over-35-layers saturated at an
exact tie for both lenses).

Official companion-repo protocol (data/evaluations/README.md): readout at the single
position immediately preceding `target`, min-over-all-layers rank, pass@k = mean over
items of the fraction of intermediates ranked <= k. Order-ops intermediates expand to
synonym sets (numbers -> digit and word forms; operations -> symbol and word forms);
rank is the min over single-token synonyms. We report k in {1, 5, 10}; the gate compares
J-lens vs vanilla logit lens (use_jacobian=False) on pass@1 and pass@5, identical
treatment for both.

Usage: .venv/bin/python -m hw0.validate_lens
"""

import json
import sys

import torch

from hw0 import core

from hw0.core import jlens_data_dir

ORDER_OPS = f"{jlens_data_dir()}/data/evaluations/lens-eval-order-ops.json"
MULTIHOP = f"{jlens_data_dir()}/data/evaluations/lens-eval-multihop.json"
OUT = ("results/lens_validation_pen.json" if __import__("os").environ.get("HW0_LENS_PATH")
       else "results/lens_validation.json")
KS = (1, 5, 10)

WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
         13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
         17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
OPS = {
    "multiplication": ["multiplication", "multiply", "multiplied", "times", "product", "*", "×"],
    "addition": ["addition", "add", "added", "plus", "sum", "+"],
    "subtraction": ["subtraction", "subtract", "subtracted", "minus", "difference", "-", "−"],
    "division": ["division", "divide", "divided", "quotient", "/", "÷"],
}


def synonyms(inter: str) -> list[str]:
    if inter in OPS:
        return OPS[inter]
    try:
        n = int(inter)
        return [inter] + ([WORDS[n]] if n in WORDS else [])
    except ValueError:
        return [inter]


def single_token_ids(tok, words: list[str]) -> list[int]:
    """Ids of surface forms that are GENUINELY single tokens (with/without leading
    space, capitalized). Multi-token forms are excluded per the single-token protocol —
    crediting a first fragment would reward e.g. 'mult' for 'multiplication'."""
    ids = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
    return sorted(ids)


def best_rank(logits: torch.Tensor, target_ids: list[int]) -> int:
    order = logits.argsort(descending=True)
    lookup = torch.empty_like(order)
    lookup[order] = torch.arange(len(order))
    return int(min(lookup[t].item() for t in target_ids))


def evaluate(setup, items, use_jacobian: bool, expand: bool) -> dict[int, float]:
    layers = sorted(setup.lens.jacobians.keys())
    fractions = {k: [] for k in KS}
    for it in items:
        lens_logits, _, _ = setup.lens.apply(
            setup.model, it["prompt"], positions=[-1], layers=layers,
            use_jacobian=use_jacobian)
        best = []
        for inter in it["intermediates"]:
            forms = synonyms(str(inter)) if expand else [str(inter)]
            tids = single_token_ids(setup.tok, forms)
            if not tids:
                continue
            best.append(min(best_rank(lens_logits[l][0], tids) for l in layers))
        for k in KS:
            fractions[k].append(sum(b < k for b in best) / max(len(best), 1))
    return {k: sum(v) / len(v) for k, v in fractions.items()}


def main():
    setup = core.load()
    report = {}
    for name, path, expand in [("order-ops", ORDER_OPS, True),
                               ("multihop", MULTIHOP, False)]:
        items = json.load(open(path))["items"]
        report[name] = {
            "n": len(items),
            "jlens": evaluate(setup, items, True, expand),
            "logit_lens": evaluate(setup, items, False, expand),
        }
    # Gate: J-lens >= logit lens on pass@1 AND pass@5, strictly better on at least one,
    # on at least one of the two evals.
    def wins(r):
        j, g = r["jlens"], r["logit_lens"]
        return j[1] >= g[1] and j[5] >= g[5] and (j[1] > g[1] or j[5] > g[5])
    report["gate"] = "PASS" if any(wins(report[n]) for n in ("order-ops", "multihop")) else "FAIL"
    json.dump(report, open(OUT, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
