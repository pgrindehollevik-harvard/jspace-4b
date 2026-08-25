"""Calibration pilot + positive-control stop-gate (pre-registration section 7).

Hypothesis-blind by construction: this stage sees only
  (a) the positive control (multihop primary; order-ops fallback if multihop is not
      near-ceiling for the clean model — checked FIRST, as pre-registered),
  (b) wikitext teacher-forced selectivity (top-1 agreement with clean),
  (c) degenerate-output rate on 30 GSM8K TRAIN CoT generations,
and never any CoT-vs-direct accuracy comparison.

Band choice rule: evaluate BOTH primary candidates; choose the one maximizing the
positive-control drop SUBJECT TO degenerate rate < 10% and wikitext top-1 > 80%.
STOP-GATE: the drop must be significant (exact McNemar, deviation 3) AND >= 2x the
random-control drop; the ladder (k=5, light band) applies only if no candidate passes.
If every rung fails, the main grid must not run.

Every expensive stage checkpoints to results/calibration_state.json, so a crash (or an
OOM kill — observed twice on the original 48GB MPS machine) restarts from a clean
process with the finished stages loaded. The appropriate accelerator cache is emptied
between stages; memory telemetry is printed with each progress line.

Usage: .venv/bin/python -u -m jspace.calibrate     (writes results/calibration.json)
"""

import json
import os
import time

import torch

from jspace import core
from jspace.ablation import JSpaceAblator
from jspace.data import load_gsm8k, load_wikitext_heldout
from jspace.generate import generate, generate_resumable
from jspace.grading import max_ngram_repetition

from jspace.core import jlens_data_dir

MULTIHOP = f"{jlens_data_dir()}/data/evaluations/lens-eval-multihop.json"
ORDER_OPS = f"{jlens_data_dir()}/data/evaluations/lens-eval-order-ops.json"
# A locally fitted lens gets its own namespaced ladder run, so the original verdict's
# provenance is preserved. core also accepts the historical HW0_LENS_PATH alias.
_SUFFIX = "_pen" if core.using_local_lens() else ""
OUT = f"results/calibration{_SUFFIX}.json"
STATE = f"results/calibration_state{_SUFFIX}.json"

BANDS = core.BAND_PRIMARY_CANDIDATES  # [(14, 24), (14, 31)]
NEAR_CEILING = 0.90


def mem() -> str:
    backend, allocated = core.accelerator_memory_gb()
    return f"rss={core.process_rss_gb():.1f}G {backend}={allocated:.1f}G"


def load_state() -> dict:
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {}


def stage(state: dict, key: str, fn):
    """Run fn() once ever: checkpointed across process restarts."""
    if key not in state:
        state[key] = fn()
        json.dump(state, open(STATE, "w"))
        core.empty_accelerator_cache()
        print(f"stage[{key}] done ({mem()})", flush=True)
    else:
        print(f"stage[{key}] loaded from checkpoint", flush=True)
    return state[key]


def control_accuracy(setup, items, ablator=None, tag="") -> list[bool]:
    """Greedy 8-token completion; hit = target substring appears (case-insensitive)."""
    hits = []
    for i, it in enumerate(items):
        r = generate(setup, it["prompt"], max_new_tokens=8, seed=0, ablator=ablator,
                     greedy=True)
        hits.append(it["target"].strip().lower() in r.text.lower())
        if i % 20 == 0:
            print(f"  control[{tag}] {i}/{len(items)} ({mem()})", flush=True)
    return hits


def wikitext_top1_match(setup, seqs, ablator) -> float:
    """Teacher-forced: fraction of positions where ablated argmax == clean argmax."""
    matches, total = 0, 0
    for text in seqs:
        ids = setup.tok(text, return_tensors="pt", truncation=True,
                        max_length=128).input_ids.to(setup.device)
        with torch.no_grad():
            ablator.enabled = False
            logits = setup.hf(ids).logits[0]
            clean_am = logits.argmax(-1)
            ablator.exempt_ids = logits.topk(core.K_CLEAN_EXEMPT, dim=-1).indices
            del logits
            ablator.enabled = True
            abl_am = setup.hf(ids).logits[0].argmax(-1)
            ablator.enabled = False
        m = (clean_am == abl_am)
        matches += int(m.sum().item())
        total += m.numel()
    return matches / total


def degenerate_rate(setup, problems, ablator, state, key) -> tuple[float, float]:
    """(degenerate fraction, ablated tok/s) over pilot CoT generations.

    Sub-checkpointed per generation: this is the longest calibration stage (~90s per
    generation) and the one the silent SIGKILLs keep landing in — a crash may only
    cost the in-flight generation."""
    partial = state.setdefault(key, {"flags": [], "toks": 0, "secs": 0.0})
    for i in range(len(partial["flags"]), len(problems)):
        p = problems[i]
        prompt = core.chat_prompt(setup, p["problem"], "cot")
        cur = partial.setdefault("cur", {})
        pre_existing = len(cur.get("tokens", []))
        t0 = time.time()
        r = generate_resumable(setup, prompt, 640, seed=1, ablator=ablator,
                               state=cur, save=lambda: json.dump(state, open(STATE, "w")))
        # tok/s counts only tokens produced within this process's timing window,
        # so resumed prefixes don't inflate the throughput estimate.
        partial["secs"] += time.time() - t0
        partial["toks"] += len(r.token_ids) - pre_existing
        partial["flags"].append(bool(
            (max_ngram_repetition(r.token_ids) > 0.5)
            or (r.hit_cap and len(r.token_ids) >= 640)))
        partial["cur"] = {}
        json.dump(state, open(STATE, "w"))
        if i % 5 == 0:
            print(f"  degen {i}/{len(problems)} ({mem()})", flush=True)
    return sum(partial["flags"]) / len(partial["flags"]), (
        partial["toks"] / max(partial["secs"], 1e-9))


def mcnemar_exact_p(hits_a: list[bool], hits_b: list[bool]) -> float:
    """Exact McNemar (paired, two-sided) on per-item hit lists (deviation 3)."""
    from math import comb
    b = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    c = sum(1 for x, y in zip(hits_a, hits_b) if not x and y)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, x) for x in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def evaluate_rung(setup, state, items, pilot, wiki, clean_hits, band, k) -> dict:
    tag = f"{band[0]}-{band[1]}:{k}"
    clean_acc = sum(clean_hits) / len(clean_hits)
    rung = {"band": band, "k": k}
    hits = {}
    for mode in ("jspace", "random"):
        def run(mode=mode):
            abl = JSpaceAblator(setup, band, k=k, mode=mode).install()
            try:
                h = control_accuracy(setup, items, ablator=abl, tag=f"{tag}-{mode}")
                w = wikitext_top1_match(setup, wiki, abl)
            finally:
                abl.remove()
            return {"hits": h, "wikitext": w}
        r = stage(state, f"rung:{tag}:{mode}", run)
        hits[mode] = r["hits"]
        rung[f"control_{mode}"] = sum(r["hits"]) / len(r["hits"])
        rung[f"wikitext_top1_{mode}"] = r["wikitext"]

    def run_degen():
        abl = JSpaceAblator(setup, band, k=k).install()
        try:
            d, t = degenerate_rate(setup, pilot, abl, state, f"rung:{tag}:degen_partial")
        finally:
            abl.remove()
        return {"degenerate_rate": d, "tok_s": t}
    d = stage(state, f"rung:{tag}:degen", run_degen)
    rung["degenerate_rate"], rung["ablated_tok_s"] = d["degenerate_rate"], d["tok_s"]

    rung["j_drop"] = clean_acc - rung["control_jspace"]
    rung["r_drop"] = clean_acc - rung["control_random"]
    rung["mcnemar_p_drop"] = mcnemar_exact_p(clean_hits, hits["jspace"])
    rung["mcnemar_p_j_vs_random"] = mcnemar_exact_p(hits["random"], hits["jspace"])
    rung["passes_constraints"] = (
        rung["degenerate_rate"] < 0.10 and rung["wikitext_top1_jspace"] > 0.80)
    rung["passes_gate"] = (
        rung["passes_constraints"] and rung["mcnemar_p_drop"] < 0.05
        and rung["j_drop"] >= 2 * max(rung["r_drop"], 0.0) and rung["j_drop"] > 0)
    return rung


def main():
    import faulthandler
    faulthandler.enable()
    state = load_state()
    setup = core.load()
    mh_items = json.load(open(MULTIHOP))["items"]
    oo_items = json.load(open(ORDER_OPS))["items"]
    pilot = load_gsm8k(n=30, seed=100, split="train")
    wiki = load_wikitext_heldout(n=50)

    # Pre-registered control choice, checked FIRST: multihop primary; order-ops
    # fallback if the clean model is not near-ceiling on multihop.
    mh_clean = stage(state, "clean_multihop",
                     lambda: control_accuracy(setup, mh_items, tag="clean-mh"))
    oo_clean = stage(state, "clean_orderops",
                     lambda: control_accuracy(setup, oo_items, tag="clean-oo"))
    mh_acc = sum(mh_clean) / len(mh_clean)
    oo_acc = sum(oo_clean) / len(oo_clean)
    if mh_acc >= NEAR_CEILING or mh_acc >= oo_acc:
        control, items, clean_hits = "multihop", mh_items, mh_clean
    else:
        control, items, clean_hits = "order-ops", oo_items, oo_clean
    report = {"clean_multihop": mh_acc, "clean_orderops": oo_acc,
              "control_used": control,
              "control_note": ("multihop not near-ceiling on clean Qwen3-4B; "
                               "order-ops fallback per prereg section 7"
                               if control == "order-ops" else "multihop primary"),
              "rungs": []}
    print(f"clean multihop={mh_acc:.3f} order-ops={oo_acc:.3f} -> control={control}",
          flush=True)

    chosen = None
    for band in BANDS:
        rung = evaluate_rung(setup, state, items, pilot, wiki, clean_hits,
                             band, core.K_ABLATE)
        report["rungs"].append(rung)
        print(json.dumps(rung), flush=True)
    passing = [r for r in report["rungs"] if r["passes_gate"]]
    if passing:
        chosen = max(passing, key=lambda r: r["j_drop"])
    else:
        for band, k in [(BANDS[0], 5), (core.BAND_LIGHT, core.K_ABLATE)]:
            rung = evaluate_rung(setup, state, items, pilot, wiki, clean_hits, band, k)
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
