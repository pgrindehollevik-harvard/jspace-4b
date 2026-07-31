"""Analysis: grading, estimands, bootstrap CIs, and the headline figure (prereg section 8).

Reads results/grid.jsonl, writes results/graded.jsonl, results/summary.json, and
figures under results/figures/. Rerunnable end-to-end; the report's principal numbers
and plots all regenerate from here.
"""

import json
import os
from collections import defaultdict

import numpy as np

from hw0.grading import grade

GRID = "results/grid.jsonl"
GRADED = "results/graded.jsonl"
SUMMARY = "results/summary.json"
FIGDIR = "results/figures"
FLOOR = 0.10          # prereg 8: clean acc below this -> ratio endpoints excluded
N_BOOT = 10_000
RNG = np.random.default_rng(0)

DIFFICULTY = ["gsm8k", "math500", "aime"]


def grade_all():
    graded = []
    with open(GRID) as f:
        for line in f:
            r = json.loads(line)
            g = grade(r["dataset"], r["gold"], r["text"], r.get("repetition", 0.0),
                      r["hit_cap"])
            r.update(g)
            graded.append(r)
    with open(GRADED, "w") as f:
        for r in graded:
            f.write(json.dumps(r) + "\n")
    return graded


def cell_table(graded):
    """problem-level accuracy per cell: {cell: {problem_id: mean correct over seeds}}."""
    by_cell = defaultdict(lambda: defaultdict(list))
    for r in graded:
        by_cell[r["cell"]][r["problem_id"]].append(bool(r["correct"]))
    return {c: {p: float(np.mean(v)) for p, v in probs.items()}
            for c, probs in by_cell.items()}


def acc(cell):
    return float(np.mean(list(cell.values()))) if cell else float("nan")


def boot_ci(fn, tables, n=N_BOOT):
    """Problem-level cluster bootstrap over the union of problem ids."""
    ids = sorted(set().union(*[set(t) for t in tables]))
    stats = []
    for _ in range(n):
        take = RNG.choice(len(ids), len(ids), replace=True)
        sampled = [ids[i] for i in take]
        stats.append(fn([{p: t[p] for p in sampled if p in t} for t in tables]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def retention(tables):
    clean, abl = tables
    a_clean, a_abl = acc(clean), acc(abl)
    return a_abl / a_clean if a_clean > 0 else float("nan")


def survival(tables):
    """P(correct ablated | correct clean), problem-level."""
    clean, abl = tables
    ok = [abl[p] for p in clean if clean[p] >= 0.5 and p in abl]
    return float(np.mean(ok)) if ok else float("nan")


def p_d(tables):
    """[R(cot,J) - R(direct,J)] - [R(cot,rand) - R(direct,rand)] from 8 tables."""
    (cc, cj, dc, dj, ccr, cr, dcr, dr) = tables
    return (retention([cc, cj]) - retention([dc, dj])) - (
        retention([ccr, cr]) - retention([dcr, dr]))


def main():
    graded = grade_all()
    cells = cell_table(graded)
    summary = {"cells": {}, "retention": {}, "survival": {}, "p_d": {}, "artifacts": {}}

    art = defaultdict(lambda: defaultdict(int))
    for r in graded:
        art[r["cell"]][r["class"]] += 1
    summary["artifacts"] = {c: dict(v) for c, v in art.items()}

    for ds in DIFFICULTY:
        for mode in ("cot", "direct"):
            clean = cells.get(f"{ds}-{mode}-clean", {})
            summary["cells"][f"{ds}-{mode}-clean"] = acc(clean)
            for cond in ("jspace", "random", "jspace-light"):
                cell = cells.get(f"{ds}-{mode}-{cond}", {})
                if not cell:
                    continue
                summary["cells"][f"{ds}-{mode}-{cond}"] = acc(cell)
                key = f"{ds}-{mode}-{cond}"
                floored = acc(clean) < FLOOR
                summary["retention"][key] = {
                    "value": retention([clean, cell]) if not floored else None,
                    "ci": boot_ci(retention, [clean, cell]) if not floored else None,
                    "floor_excluded": floored,
                }
                summary["survival"][key] = {
                    "value": survival([clean, cell]),
                    "ci": boot_ci(survival, [clean, cell]),
                }
        needed = [f"{ds}-cot-clean", f"{ds}-cot-jspace", f"{ds}-direct-clean",
                  f"{ds}-direct-jspace", f"{ds}-cot-clean", f"{ds}-cot-random",
                  f"{ds}-direct-clean", f"{ds}-direct-random"]
        if all(cells.get(c) for c in needed):
            tabs = [cells[c] for c in needed]
            floored = acc(cells[f"{ds}-direct-clean"]) < FLOOR
            summary["p_d"][ds] = {
                "value": p_d(tabs) if not floored else None,
                "ci": boot_ci(p_d, tabs) if not floored else None,
                "floor_excluded": floored,
            }

    # H2 primary contrast (prereg 8): P_D(gsm8k) - P_D(math500).
    if all(ds in summary["p_d"] and summary["p_d"][ds]["value"] is not None
           for ds in ("gsm8k", "math500")):
        needed = lambda ds: [f"{ds}-cot-clean", f"{ds}-cot-jspace", f"{ds}-direct-clean",
                             f"{ds}-direct-jspace", f"{ds}-cot-clean", f"{ds}-cot-random",
                             f"{ds}-direct-clean", f"{ds}-direct-random"]
        g, m = [cells[c] for c in needed("gsm8k")], [cells[c] for c in needed("math500")]

        def contrast(tables):
            return p_d(tables[:8]) - p_d(tables[8:])
        summary["h2_primary"] = {
            "value": contrast(g + m),
            "ci": boot_ci(contrast, g + m),
        }

    os.makedirs(FIGDIR, exist_ok=True)
    _headline_figure(summary)
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k in ("p_d", "h2_primary")},
                     indent=1))


def _headline_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(DIFFICULTY))
    series = [("cot", "jspace", "CoT / J-space", "#1a80bb", "-"),
              ("direct", "jspace", "direct / J-space", "#b8291d", "-"),
              ("cot", "random", "CoT / random", "#1a80bb", "--"),
              ("direct", "random", "direct / random", "#b8291d", "--")]
    for mode, cond, label, color, ls in series:
        ys, los, his = [], [], []
        for ds in DIFFICULTY:
            r = summary["retention"].get(f"{ds}-{mode}-{cond}")
            if r and r["value"] is not None:
                ys.append(r["value"]); los.append(r["ci"][0]); his.append(r["ci"][1])
            else:
                ys.append(np.nan); los.append(np.nan); his.append(np.nan)
        ys, los, his = map(np.array, (ys, los, his))
        ax.plot(x, ys, ls, color=color, label=label, marker="o")
        ax.fill_between(x, los, his, color=color, alpha=0.12, linewidth=0)
    ax.set_xticks(x, ["GSM8K", "MATH-500", "AIME"])
    ax.set_ylabel("retention  (ablated acc / clean acc)")
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Does written reasoning still protect when problems get hard?")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/headline_retention.png", dpi=200)


if __name__ == "__main__":
    main()
