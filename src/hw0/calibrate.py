"""Calibration pilot + positive-control stop-gate (pre-registration section 7).

Hypothesis-blind by construction: this stage sees only
  (a) the multihop positive control (clean / J-ablated / random-ablated),
  (b) wikitext teacher-forced selectivity (top-1 agreement with clean),
  (c) degenerate-output rate on 30 GSM8K TRAIN CoT generations,
and never any CoT-vs-direct accuracy comparison.

Band choice rule: from the pre-registered candidates, pick the band maximizing the
multihop J-ablation drop SUBJECT TO degenerate rate < 10% and wikitext top-1 > 80%.
STOP-GATE: the chosen band's multihop drop must be significant AND >= 2x the
random-control drop (Fisher exact), else the ladder (k=10 -> k=5 -> light band) applies;
if every rung fails, the main grid must not run.

Usage: .venv/bin/python -m hw0.calibrate          (writes results/calibration.json)
"""

import json
import time

import torch

from hw0 import core
from hw0.ablation import JSpaceAblator
from hw0.data import load_gsm8k, load_wikitext_heldout
from hw0.generate import generate
from hw0.grading import max_ngram_repetition

MULTIHOP = ".context/jacobian-lens/data/evaluations/lens-eval-multihop.json"
OUT = "results/calibration.json"

BANDS = core.BAND_PRIMARY_CANDIDATES  # [(14, 24), (14, 31)]
LADDER = [  # (band, k) tried in order until the stop-gate passes
    *[(b, core.K_ABLATE) for b in BANDS],
    (BANDS[0], 5),
    (core.BAND_LIGHT, core.K_ABLATE),
]


def multihop_accuracy(setup, items, ablator=None) -> list[bool]:
    """Greedy 8-token completion; hit = target substring appears (case-insensitive)."""
    hits = []
    for it in items:
        r = generate(setup, it["prompt"], max_new_tokens=8, seed=0, ablator=ablator,
                     greedy=True)
        hits.append(it["target"].strip().lower() in r.text.lower())
    return hits


def wikitext_top1_match(setup, seqs, ablator) -> float:
    """Teacher-forced: fraction of positions where ablated argmax == clean argmax."""
    matches, total = 0, 0
    for text in seqs:
        ids = setup.tok(text, return_tensors="pt", truncation=True,
                        max_length=128).input_ids.to(setup.device)
        with torch.no_grad():
            ablator.enabled = False
            clean_logits = setup.hf(ids).logits[0]
            ablator.exempt_ids = clean_logits.topk(core.K_CLEAN_EXEMPT, dim=-1).indices
            ablator.enabled = True
            abl_logits = setup.hf(ids).logits[0]
            ablator.enabled = False
        m = (clean_logits.argmax(-1) == abl_logits.argmax(-1))
        matches += int(m.sum().item())
        total += m.numel()
    return matches / total


def degenerate_rate(setup, problems, ablator) -> tuple[float, float]:
    """(degenerate fraction, ablated tok/s) over pilot CoT generations."""
    degen, toks, secs = 0, 0, 0.0
    for p in problems:
        prompt = core.chat_prompt(setup, p["problem"], "cot")
        t0 = time.time()
        r = generate(setup, prompt, max_new_tokens=640, seed=1, ablator=ablator)
        secs += time.time() - t0
        toks += r.n_new
        degen += (max_ngram_repetition(r.token_ids) > 0.5) or (r.hit_cap and r.n_new >= 640)
    return degen / len(problems), toks / secs


def mcnemar_exact_p(hits_a: list[bool], hits_b: list[bool]) -> float:
    """Exact McNemar (paired, two-sided) on per-item hit lists (deviation 3:
    replaces the pre-registered unpaired Fisher, which ignores item pairing)."""
    from math import comb
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if not x and y)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, x) for x in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def evaluate_rung(setup, items, pilot, wiki, clean_hits, band, k) -> dict:
    clean_acc = sum(clean_hits) / len(clean_hits)
    rung = {"band": band, "k": k}
    hits = {}
    for mode in ("jspace", "random"):
        abl = JSpaceAblator(setup, band, k=k, mode=mode).install()
        try:
            hits[mode] = multihop_accuracy(setup, items, ablator=abl)
            rung[f"multihop_{mode}"] = sum(hits[mode]) / len(hits[mode])
            rung[f"wikitext_top1_{mode}"] = wikitext_top1_match(setup, wiki, abl)
        finally:
            abl.remove()
    abl = JSpaceAblator(setup, band, k=k).install()
    try:
        rung["degenerate_rate"], rung["ablated_tok_s"] = degenerate_rate(
            setup, pilot, abl)
    finally:
        abl.remove()

    rung["j_drop"] = clean_acc - rung["multihop_jspace"]
    rung["r_drop"] = clean_acc - rung["multihop_random"]
    rung["mcnemar_p_drop"] = mcnemar_exact_p(clean_hits, hits["jspace"])
    rung["mcnemar_p_j_vs_random"] = mcnemar_exact_p(hits["random"], hits["jspace"])
    rung["passes_constraints"] = (
        rung["degenerate_rate"] < 0.10 and rung["wikitext_top1_jspace"] > 0.80)
    rung["passes_gate"] = (
        rung["passes_constraints"] and rung["mcnemar_p_drop"] < 0.05
        and rung["j_drop"] >= 2 * max(rung["r_drop"], 0.0) and rung["j_drop"] > 0)
    return rung


def main():
    setup = core.load()
    items = json.load(open(MULTIHOP))["items"]
    pilot = load_gsm8k(n=30, seed=100, split="train")
    wiki = load_wikitext_heldout(n=50)

    clean_hits = multihop_accuracy(setup, items)
    clean_acc = sum(clean_hits) / len(clean_hits)
    report = {"clean_multihop": clean_acc, "rungs": []}
    print(f"clean multihop accuracy: {clean_acc:.3f} ({sum(clean_hits)}/{len(clean_hits)})")

    # Prereg section 7: evaluate BOTH primary candidates, choose the one maximizing
    # the positive-control drop subject to constraints+gate; ladder only if none pass.
    chosen = None
    for band, k in [(b, core.K_ABLATE) for b in BANDS]:
        rung = evaluate_rung(setup, items, pilot, wiki, clean_hits, band, k)
        report["rungs"].append(rung)
        print(json.dumps(rung), flush=True)
    passing = [r for r in report["rungs"] if r["passes_gate"]]
    if passing:
        chosen = max(passing, key=lambda r: r["j_drop"])
    else:
        for band, k in [(BANDS[0], 5), (core.BAND_LIGHT, core.K_ABLATE)]:
            rung = evaluate_rung(setup, items, pilot, wiki, clean_hits, band, k)
            report["rungs"].append(rung)
            print(json.dumps(rung), flush=True)
            if rung["passes_gate"]:
                chosen = rung
                break

    report["chosen"] = chosen
    report["stop_gate"] = "PASS" if chosen else "FAIL"
    json.dump(report, open(OUT, "w"), indent=1)
    print(json.dumps({"stop_gate": report["stop_gate"], "chosen": chosen}))


if __name__ == "__main__":
    main()
