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


def fisher_exact_p(a_hit, a_n, b_hit, b_n) -> float:
    """One-sided Fisher exact via hypergeometric tail (no scipy dependency)."""
    from math import comb
    total, hits = a_n + b_n, a_hit + b_hit
    p = 0.0
    for x in range(a_hit, min(hits, a_n) + 1):
        p += comb(a_n, x) * comb(b_n, hits - x) / comb(total, hits)
    return min(p, 1.0)


def main():
    setup = core.load()
    items = json.load(open(MULTIHOP))["items"]
    pilot = load_gsm8k(n=30, seed=100, split="train")
    wiki = load_wikitext_heldout(n=50)

    report = {"clean_multihop": None, "rungs": []}
    clean_hits = multihop_accuracy(setup, items)
    clean_acc = sum(clean_hits) / len(clean_hits)
    report["clean_multihop"] = clean_acc
    print(f"clean multihop accuracy: {clean_acc:.3f} ({sum(clean_hits)}/{len(clean_hits)})")

    chosen = None
    for band, k in LADDER:
        rung = {"band": band, "k": k}
        for mode in ("jspace", "random"):
            abl = JSpaceAblator(setup, band, k=k, mode=mode).install()
            try:
                hits = multihop_accuracy(setup, items, ablator=abl)
                rung[f"multihop_{mode}"] = sum(hits) / len(hits)
            finally:
                abl.remove()
        abl = JSpaceAblator(setup, band, k=k).install()
        try:
            rung["wikitext_top1"] = wikitext_top1_match(setup, wiki, abl)
            rung["degenerate_rate"], rung["ablated_tok_s"] = degenerate_rate(
                setup, pilot, abl)
        finally:
            abl.remove()

        j_drop = clean_acc - rung["multihop_jspace"]
        r_drop = clean_acc - rung["multihop_random"]
        n = len(items)
        rung["j_drop"], rung["r_drop"] = j_drop, r_drop
        rung["fisher_p"] = fisher_exact_p(
            round(rung["multihop_random"] * n), n, round(rung["multihop_jspace"] * n), n)
        rung["passes_constraints"] = (
            rung["degenerate_rate"] < 0.10 and rung["wikitext_top1"] > 0.80)
        rung["passes_gate"] = (
            rung["passes_constraints"] and rung["fisher_p"] < 0.05
            and j_drop >= 2 * max(r_drop, 0.0) and j_drop > 0)
        report["rungs"].append(rung)
        print(json.dumps(rung))
        if rung["passes_gate"]:
            chosen = rung
            break

    report["chosen"] = chosen
    report["stop_gate"] = "PASS" if chosen else "FAIL"
    json.dump(report, open(OUT, "w"), indent=1)
    print(json.dumps({"stop_gate": report["stop_gate"], "chosen": chosen}))


if __name__ == "__main__":
    main()
