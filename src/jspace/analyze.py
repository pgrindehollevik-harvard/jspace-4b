"""Analysis: grading, estimands, bootstrap CIs, and the headline figure (prereg section 8).

Reads results/grid.jsonl, writes results/graded.jsonl, results/summary.json, and
figures under results/figures/. Rerunnable end-to-end; the report's principal numbers
and plots all regenerate from here.
"""

import json
import os
from collections import defaultdict

import numpy as np

from jspace.grading import grade

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
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write from an interrupted run; the runner redoes it
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
    """Problem-level cluster bootstrap over the union of problem ids.

    Resampled instances are keyed by draw index so multiplicity is honored (a problem
    drawn m times appears under m distinct keys) while the same draw still maps to the
    same problem in every table — pairing survives, duplicates count.
    """
    ids = sorted(set().union(*[set(t) for t in tables]))
    stats = []
    for _ in range(n):
        take = RNG.choice(len(ids), len(ids), replace=True)
        sampled = [ids[i] for i in take]
        stats.append(fn([{i: t[p] for i, p in enumerate(sampled) if p in t}
                         for t in tables]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def retention(tables):
    """Paired retention: both accuracies computed over the problems present in BOTH
    tables, so a smaller ablated arm is compared against its own clean subset."""
    clean, abl = tables
    shared = [p for p in clean if p in abl]
    if not shared:
        return float("nan")
    a_clean = float(np.mean([clean[p] for p in shared]))
    a_abl = float(np.mean([abl[p] for p in shared]))
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


def p_d_abs(tables):
    """Co-primary absolute-point DiD (prereg 8): same structure as p_d but with
    accuracy differences instead of ratios (floor-insensitive)."""
    (cc, cj, dc, dj, ccr, cr, dcr, dr) = tables

    def diff(clean, abl):
        shared = [p for p in clean if p in abl]
        if not shared:
            return float("nan")
        return float(np.mean([abl[p] for p in shared])) - float(
            np.mean([clean[p] for p in shared]))

    return (diff(cc, cj) - diff(dc, dj)) - (diff(ccr, cr) - diff(dcr, dr))


def mcnemar_exact(clean, abl):
    """Exact McNemar p for clean-vs-ablated at problem level (prereg 8, the AIME
    endpoint of record). Problem counts as correct if >= half its seeds are."""
    from math import comb
    shared = [p for p in clean if p in abl]
    b = sum(1 for p in shared if clean[p] >= 0.5 and abl[p] < 0.5)
    c = sum(1 for p in shared if clean[p] < 0.5 and abl[p] >= 0.5)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, x) for x in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def trend_model(graded):
    """Pre-registered trend statistic (prereg 8): logistic regression
    correct ~ mode x condition x ordinal difficulty, problem-clustered SEs; the
    registered statistic is the J-vs-random contrast of the three-way interaction."""
    import pandas as pd
    import statsmodels.formula.api as smf

    DIFF = {"gsm8k": 1, "aime": 7}  # math500 uses 1 + level (2..6)
    rows = [{"correct": int(r["correct"]), "mode": r["mode"],
             "condition": r["condition"], "problem": r["problem_id"],
             "difficulty": DIFF.get(r["dataset"], 1 + r["level"])}
            for r in graded if r["condition"] in ("clean", "jspace", "random")]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    try:
        m = smf.logit("correct ~ C(mode)*C(condition, Treatment('clean'))*difficulty",
                      data=df).fit(disp=0, cov_type="cluster",
                                   cov_kwds={"groups": df["problem"]})
        out = {"n": len(df), "terms": {}}
        for t in m.params.index:
            if "difficulty" in t and ("jspace" in t or "random" in t):
                out["terms"][t] = {"coef": float(m.params[t]),
                                   "se": float(m.bse[t]), "p": float(m.pvalues[t])}
        return out
    except Exception as e:  # separation/singularity on partial data must not kill analysis
        return {"error": str(e)}


def length_covariate(graded):
    """Retention-vs-CoT-length within dataset (prereg 8): does damage scale with the
    number of ablated generation steps? Correlation of per-problem clean CoT length
    with survival under J-ablation."""
    from collections import defaultdict as dd
    by = dd(dict)
    for r in graded:
        if r["mode"] == "cot" and r["condition"] in ("clean", "jspace"):
            by[(r["dataset"], r["problem_id"])][r["condition"]] = (
                r["n_new"], bool(r["correct"]))
    out = {}
    for ds in DIFFICULTY:
        pairs = [(v["clean"][0], int(v["jspace"][1]))
                 for (d, _), v in by.items()
                 if d == ds and "clean" in v and "jspace" in v and v["clean"][1]]
        if len(pairs) >= 20:
            x = np.array([p[0] for p in pairs], float)
            y = np.array([p[1] for p in pairs], float)
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
            out[ds] = {"n": len(pairs), "pointbiserial_r": r,
                       "mean_len_survived": float(x[y == 1].mean()) if y.any() else None,
                       "mean_len_died": float(x[y == 0].mean()) if (y == 0).any() else None}
    return out


def boot_ci_2strata(fn, tables_a, tables_b, n=N_BOOT):
    """Stratified cluster bootstrap for cross-dataset contrasts: resample each
    dataset's problems independently, preserving per-dataset sample sizes."""
    ids_a = sorted(set().union(*[set(t) for t in tables_a]))
    ids_b = sorted(set().union(*[set(t) for t in tables_b]))
    stats = []
    for _ in range(n):
        sa = [ids_a[i] for i in RNG.choice(len(ids_a), len(ids_a), replace=True)]
        sb = [ids_b[i] for i in RNG.choice(len(ids_b), len(ids_b), replace=True)]
        ta = [{i: t[p] for i, p in enumerate(sa) if p in t} for t in tables_a]
        tb = [{i: t[p] for i, p in enumerate(sb) if p in t} for t in tables_b]
        stats.append(fn(ta + tb))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


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
                    "mcnemar_p": mcnemar_exact(clean, cell),
                }
        needed = [f"{ds}-cot-clean", f"{ds}-cot-jspace", f"{ds}-direct-clean",
                  f"{ds}-direct-jspace", f"{ds}-cot-clean", f"{ds}-cot-random",
                  f"{ds}-direct-clean", f"{ds}-direct-random"]
        if all(cells.get(c) for c in needed):
            tabs = [cells[c] for c in needed]
            # Floor rule (prereg 8) on BOTH clean denominators of the ratio estimand.
            floored = (acc(cells[f"{ds}-direct-clean"]) < FLOOR
                       or acc(cells[f"{ds}-cot-clean"]) < FLOOR)
            summary["p_d"][ds] = {
                "value": p_d(tabs) if not floored else None,
                "ci": boot_ci(p_d, tabs) if not floored else None,
                "abs_did": p_d_abs(tabs),
                "abs_did_ci": boot_ci(p_d_abs, tabs),
                "floor_excluded": floored,
            }

    # H2 primary contrast (prereg 8): P_D(gsm8k) - P_D(math500), stratified bootstrap.
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
            "ci": boot_ci_2strata(contrast, g, m),
        }

    # Within-MATH-500 level 1-5 gradient (prereg 6.5): format-constant difficulty axis.
    levels = {r["problem_id"]: r["level"] for r in graded if r["dataset"] == "math500"}
    grad = {}
    for lv in sorted(set(levels.values())):
        sub = lambda cell: {p: v for p, v in cells.get(cell, {}).items()
                            if levels.get(p) == lv}
        clean, abl = sub("math500-cot-clean"), sub("math500-cot-jspace")
        if clean and abl and acc(clean) >= FLOOR:
            grad[lv] = {"value": retention([clean, abl]),
                        "ci": boot_ci(retention, [clean, abl])}
    summary["math_gradient_cot_jspace"] = grad

    # Direct-compliance audit + CoT length distribution (prereg 6.6, 8).
    lengths, compliance = defaultdict(list), defaultdict(list)
    for r in graded:
        lengths[r["cell"]].append(r["n_new"])
        if r["mode"] == "direct":
            compliance[r["cell"]].append(r["n_new"] <= 40)
    summary["cot_length_mean"] = {c: float(np.mean(v)) for c, v in lengths.items()
                                  if "-cot-" in c}
    summary["direct_compliance"] = {c: float(np.mean(v)) for c, v in compliance.items()}
    summary["trend_model"] = trend_model(graded)
    summary["cot_length_covariate"] = length_covariate(graded)

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
