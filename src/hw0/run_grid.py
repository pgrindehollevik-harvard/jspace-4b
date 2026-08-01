"""Main experiment grid: interleaved, resumable, priority-ordered (prereg sections 3-4, 9).

Execution is problem-wise INTERLEAVED across all cells of a dataset so an interruption
leaves balanced partial data in every arm, and datasets run in priority order
GSM8K -> MATH-500 -> AIME so the replication (H1) survives any budget overrun.

Every generation appends one JSON line to results/grid.jsonl:
  {cell, dataset, mode, condition, problem_id, seed_idx, level, gold, text, n_new,
   hit_cap, norm_log, seconds, band, k}
Already-present (cell, problem_id, seed_idx) triples are skipped on restart.

Usage: .venv/bin/python -m hw0.run_grid [--band a,b] [--dataset gsm8k|math500|aime]
The band/k come from results/calibration.json unless overridden.
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

from hw0 import core
from hw0.ablation import JSpaceAblator
from hw0.data import CAPS, load_aime, load_gsm8k, load_math500
from hw0.generate import generate_resumable
from hw0.grading import max_ngram_repetition

PARTIALS = "results/partials"

OUT = "results/grid.jsonl"

# (dataset, mode, condition, n_problems or None=all, seeds_per_problem)
CELLS = [
    ("gsm8k", "cot", "clean", 150, 1), ("gsm8k", "cot", "jspace", 150, 1),
    ("gsm8k", "cot", "random", 100, 1),
    ("gsm8k", "direct", "clean", 150, 1), ("gsm8k", "direct", "jspace", 150, 1),
    ("gsm8k", "direct", "random", 100, 1),
    ("math500", "cot", "clean", 150, 1), ("math500", "cot", "jspace", 150, 1),
    ("math500", "cot", "random", 100, 1),
    ("math500", "direct", "clean", 150, 1), ("math500", "direct", "jspace", 150, 1),
    ("math500", "direct", "random", 100, 1),
    ("aime", "cot", "clean", None, 2), ("aime", "cot", "jspace", None, 2),
    ("aime", "cot", "random", None, 1),
    ("aime", "direct", "clean", None, 4), ("aime", "direct", "jspace", None, 4),
    ("aime", "direct", "random", None, 4),
    # Dose-response anchor (prereg 6.8): light band on GSM8K.
    ("gsm8k", "cot", "jspace-light", 75, 1), ("gsm8k", "direct", "jspace-light", 75, 1),
]

DATASET_ORDER = ["gsm8k", "math500", "aime"]


def load_problems():
    return {"gsm8k": load_gsm8k(n=150, seed=0),
            "math500": load_math500(per_level=30, seed=0),
            "aime": load_aime()}


def seed_for(problem_id: str, seed_idx: int) -> int:
    # Deterministic across processes (Python's hash() is salted per run); NOT shared
    # across conditions in any meaningful sense (streams diverge at the first ablated
    # position; pairing is problem-level).
    import zlib
    return zlib.crc32(f"{problem_id}#{seed_idx}".encode()) % (2**31)


def done_keys(path: str, band, k) -> set:
    """Collect completed keys; repair a torn final line; refuse mixed ablation settings."""
    keys = set()
    if not os.path.exists(path):
        return keys
    good = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write from an interrupted run; dropped and redone
            if r["condition"] in ("jspace", "random") and (tuple(r["band"]) != band
                                                           or r["k"] != k):
                raise SystemExit(
                    f"grid.jsonl contains records with band={r['band']} k={r['k']} but "
                    f"the current calibration chose band={band} k={k}; refusing to mix. "
                    f"Move results/grid.jsonl aside before rerunning.")
            good.append(line)
            keys.add((r["cell"], r["problem_id"], r["seed_idx"]))
    with open(path, "w") as f:  # rewrite without any torn line so appends stay clean
        f.writelines(good)
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", default=None, help="a,b override; default from calibration")
    ap.add_argument("--calibration", default="results/calibration.json")
    ap.add_argument("--dataset", default=None, choices=DATASET_ORDER)
    args = ap.parse_args()

    if args.band:
        a, b = map(int, args.band.split(","))
        band, k = (a, b), core.K_ABLATE
    else:
        cal = json.load(open(args.calibration))
        assert cal["stop_gate"] == "PASS", "stop-gate failed; the main grid must not run"
        band, k = tuple(cal["chosen"]["band"]), cal["chosen"]["k"]
    print(f"band={band} k={k}")

    setup = core.load()
    problems = load_problems()
    os.makedirs(PARTIALS, exist_ok=True)
    done = done_keys(OUT, band, k)
    print(f"{len(done)} generations already recorded")

    ablators = {
        "jspace": JSpaceAblator(setup, band, k=k, mode="jspace"),
        "random": JSpaceAblator(setup, band, k=k, mode="random"),
        "jspace-light": JSpaceAblator(setup, core.BAND_LIGHT, k=k, mode="jspace"),
        "clean": None,
    }

    out = open(OUT, "a")
    for dataset in ([args.dataset] if args.dataset else DATASET_ORDER):
        cells = [c for c in CELLS if c[0] == dataset]
        max_n = max((n or len(problems[dataset])) for _, _, _, n, _ in cells)
        # Interleave: for each problem index, visit every cell that includes it.
        for i in range(max_n):
            for ds, mode, condition, n, seeds in cells:
                pool = problems[ds][: (n or len(problems[ds]))]
                if i >= len(pool):
                    continue
                p = pool[i]
                for s in range(seeds):
                    key = (f"{ds}-{mode}-{condition}", p["id"], s)
                    if key in done:
                        continue
                    abl = ablators[condition]
                    prompt = core.chat_prompt(setup, p["problem"], mode)
                    t0 = time.time()
                    # Chunk-level persistence: a SIGKILL mid-generation (an observed
                    # failure mode on this machine — see docs/DEVIATIONS.md) costs at
                    # most one chunk. Sidecar removed once the full record is written.
                    pfile = os.path.join(
                        PARTIALS, f"{key[0]}__{p['id'].replace('/', '_')}__{s}.json")
                    pstate = json.load(open(pfile)) if os.path.exists(pfile) else {}

                    def save_partial(pstate=pstate, pfile=pfile):
                        json.dump(pstate, open(pfile + ".tmp", "w"))
                        os.replace(pfile + ".tmp", pfile)

                    # One transient MPS error must not kill a 24h unattended run:
                    # retry once with full teardown, then skip (the absent key makes
                    # a later restart redo it; partial chunks persist either way).
                    r = None
                    for attempt in (0, 1):
                        try:
                            if abl is not None:
                                abl.install()
                                if abl.mode == "random":
                                    abl.reseed(seed_for(p["id"], s * 7919 + 1))
                            r = generate_resumable(
                                setup, prompt, CAPS[(ds, mode)],
                                seed_for(p["id"], s), abl, pstate, save_partial)
                            break
                        except Exception:
                            print(f"FAIL {key} attempt {attempt}", file=sys.stderr)
                            traceback.print_exc()
                            if abl is not None:
                                abl.enabled = False
                                abl.pop_norm_summary()
                            torch.mps.empty_cache()
                            time.sleep(10)
                        finally:
                            if abl is not None:
                                abl.remove()
                    if r is None:
                        continue
                    rec = {"cell": key[0], "dataset": ds, "mode": mode,
                           "condition": condition, "problem_id": p["id"],
                           "seed_idx": s, "level": p["level"], "gold": p["answer"],
                           "text": r.text, "n_new": r.n_new, "hit_cap": r.hit_cap,
                           "repetition": round(max_ngram_repetition(r.token_ids), 4),
                           "norm_log": r.norm_log, "seconds": round(time.time() - t0, 2),
                           "band": list(band if condition != "jspace-light"
                                        else core.BAND_LIGHT), "k": k}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    if os.path.exists(pfile):
                        os.remove(pfile)
            if i % 10 == 0:
                print(f"[{dataset}] problem {i}/{max_n} done at {time.strftime('%H:%M:%S')}",
                      flush=True)
    out.close()
    print("GRID_COMPLETE")


if __name__ == "__main__":
    main()
