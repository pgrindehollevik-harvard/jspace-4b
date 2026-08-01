"""Report figure: the calibration ladder as a two-panel dose-response dumbbell chart.

Panel A: positive-control accuracy drop, J-space vs per-slot norm-matched random, per
rung (paired dumbbells — the estimand is the gap between the paired dots).
Panel B: wikitext top-1 selectivity of the J arm against the frozen 0.80 gate line.
Two measures of different scales -> two panels sharing the rung axis (never dual-axis).
Colors: validated categorical slots (blue #2a78d6 = J-space, orange #eb6834 = random);
identity is double-encoded by legend + direct labels on the top rung.

Usage: .venv/bin/python -m hw0.fig_ladder   (writes results/figures/ladder.pdf/.png)
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#1a1a19", "#6b6a63"


def load_rungs():
    rungs = []
    for path, label in [("results/calibration.json", "stock lens"),
                        ("results/calibration_pen.json", "penult. lens (n=32)")]:
        c = json.load(open(path))
        for r in c["rungs"]:
            band = f"{r['band'][0]}–{r['band'][1]}"
            rungs.append({
                "name": f"{label}, L{band}, k={r['k']}",
                "j": 100 * r["j_drop"], "rand": 100 * r["r_drop"],
                "sel": r["wikitext_top1_jspace"],
            })
    return rungs


def main():
    rungs = load_rungs()
    y = list(range(len(rungs)))[::-1]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(7.0, 2.9), sharey=True,
        gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.06})

    for yi, r in zip(y, rungs):
        lo, hi = sorted([r["rand"], r["j"]])
        ax.plot([lo, hi], [yi, yi], color=MUTED, lw=1.2, zorder=1)
        ax.plot(r["j"], yi, "o", color=BLUE, ms=7, zorder=3)
        ax.plot(r["rand"], yi, "o", color=ORANGE, ms=7, zorder=3,
                markerfacecolor="none", markeredgewidth=1.8)
    ax.plot([], [], "o", color=BLUE, ms=7, label="J-space ablation")
    ax.plot([], [], "o", color=ORANGE, ms=7, markerfacecolor="none",
            markeredgewidth=1.8, label="norm-matched random")
    ax.set_yticks(y, [r["name"] for r in rungs], fontsize=8)
    ax.set_xlabel("positive-control accuracy drop (points)", fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_xlim(-2, 22)

    for yi, r in zip(y, rungs):
        bx.plot(r["sel"], yi, "o", color=BLUE, ms=7)
    bx.axvline(0.80, color=INK, lw=1.0, ls="--")
    bx.text(0.803, y[0] + 0.45, "gate: >0.80", fontsize=7.5, color=INK)
    bx.set_xlabel("wikitext top-1 agreement\nwith clean model (J arm)", fontsize=8)
    bx.set_xlim(0.4, 1.0)

    for a in (ax, bx):
        a.tick_params(labelsize=8, length=0)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.grid(axis="x", color="#e8e7e0", lw=0.7)
        a.set_axisbelow(True)
    fig.suptitle("No rung achieves both a J-specific drop and selectivity",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"results/figures/ladder.{ext}", dpi=220, bbox_inches="tight")
    print("wrote results/figures/ladder.{pdf,png}")


if __name__ == "__main__":
    main()
