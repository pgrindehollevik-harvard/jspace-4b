"""Lens validation gate (pre-registration section 7, first bullet).

The pre-fitted Qwen3-4B lens must beat the vanilla logit lens (use_jacobian=False) at
ranking annotated intermediates on the companion repo's order-ops eval before anything
downstream is allowed to run.

Metric (companion repo data/evaluations/README.md): pass@k = mean fraction of an item's
intermediates whose min-over-layers full-vocab rank is <= k at the readout position (the
final prompt position). Both lenses read out at the same positions/layers; intermediates
are matched over leading-space and capitalization token variants, identically for both.

Usage: .venv/bin/python -m hw0.validate_lens [path/to/lens-eval-order-ops.json]
"""

import json
import sys

import torch

from hw0 import core

DEFAULT_EVAL = ".context/jacobian-lens/data/evaluations/lens-eval-order-ops.json"
PASS_K = 10


def token_variants(tok, text: str) -> list[int]:
    ids = set()
    for v in (text, " " + text, text.capitalize(), " " + text.capitalize()):
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def ranks_at(logits: torch.Tensor, target_ids: list[int]) -> int:
    """Best (min) full-vocab rank among the target's token variants; 0 = top."""
    order = logits.argsort(descending=True)
    pos = {tid: i for i, tid in enumerate(order.tolist())}
    return min(pos[t] for t in target_ids)


def evaluate(setup, items, use_jacobian: bool) -> float:
    per_item = []
    layers = sorted(setup.lens.jacobians.keys())
    for it in items:
        lens_logits, _, _ = setup.lens.apply(
            setup.model, it["prompt"], positions=[-1], layers=layers,
            use_jacobian=use_jacobian)
        hits = 0
        for inter in it["intermediates"]:
            tids = token_variants(setup.tok, str(inter))
            if not tids:
                continue
            best = min(ranks_at(lens_logits[l][0], tids) for l in layers)
            hits += best < PASS_K
        per_item.append(hits / max(len(it["intermediates"]), 1))
    return sum(per_item) / len(per_item)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL
    items = json.load(open(path))["items"]
    setup = core.load()
    jlens_score = evaluate(setup, items, use_jacobian=True)
    logit_score = evaluate(setup, items, use_jacobian=False)
    verdict = "PASS" if jlens_score > logit_score else "FAIL"
    print(json.dumps({"eval": path, "n_items": len(items), "pass_k": PASS_K,
                      "jlens_pass@k": round(jlens_score, 4),
                      "logit_lens_pass@k": round(logit_score, 4),
                      "gate": verdict}))


if __name__ == "__main__":
    main()
