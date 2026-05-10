"""A/B bench: constrained_layout vs tight_layout vs no-layout on
a 6-panel multiclass figure (CONFUSION + PR_F1 + ROC + CALIB_GRID +
PROB_DIST + TOP_K_ACC), the panel template c0134 reports use.

Drove the 2026-05-11 ``FigureSpec.constrained_layout`` default flip
from True -> False after the 1M-row fuzz profile (c0134:
lgb-multiclass + 2 weight schemas + ensembles) attributed 75 s of
wall-time across 112 panel-figure constrained_layout calls.

Usage:
    python -m mlframe.profiling.bench_layout

Reports wall mean / std over 5 repeats per layout option AND saves
four PNGs to ``D:/Temp/layout_ab_*.png`` so the operator can eyeball
the visual diff between layout engines on a representative figure.
"""
import time
import statistics
import os
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg", force=False)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def make_fig(layout):
    fig = Figure(figsize=(18, 8), layout=layout)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 3)
    rng = np.random.default_rng(0)

    # Panel 1: heatmap (confusion) — has colorbar
    ax = fig.add_subplot(gs[0, 0])
    cm = rng.uniform(0, 1, (3, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix (row-normalised)")
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    fig.colorbar(im, ax=ax)

    # Panel 2: PR_F1 bar
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(3); w = 0.25
    ax.bar(x - w, rng.uniform(0.6, 1, 3), width=w, label="P")
    ax.bar(x, rng.uniform(0.6, 1, 3), width=w, label="R")
    ax.bar(x + w, rng.uniform(0.6, 1, 3), width=w, label="F1")
    ax.set_title("Per-class P/R/F1")
    ax.set_xticks(x); ax.set_xticklabels(["c0", "c1", "c2"])
    ax.legend(fontsize=8)

    # Panel 3: ROC
    ax = fig.add_subplot(gs[0, 2])
    g = np.linspace(0, 1, 200)
    for k in range(3):
        ax.plot(g, np.power(g, 0.5 + k * 0.2), label=f"c{k} (AUC=0.85)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("Per-class ROC (one-vs-rest)")
    ax.legend(fontsize=8)

    # Panel 4: CALIB_GRID
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 11)
    ax.plot(bins, bins, "g:", label="perfect")
    for k in range(3):
        ax.plot(bins, bins + rng.normal(0, 0.03, 11), label=f"c{k}")
    ax.set_title("Per-class reliability curves")
    ax.legend(fontsize=8)

    # Panel 5: PROB_DIST violin (small after sampling)
    ax = fig.add_subplot(gs[1, 1])
    groups = [rng.uniform(0, 1, 5000) ** (1 + i) for i in range(3)]
    ax.violinplot(groups, showmeans=False, showextrema=False, showmedians=True)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["c0 (n=333_333)", "c1 (n=333_333)", "c2 (n=333_334)"], rotation=30, ha="right", fontsize=8)
    ax.set_title("P(y=true_class) per true class")

    # Panel 6: TOP_K_ACC
    ax = fig.add_subplot(gs[1, 2])
    ax.plot([1, 2, 3], [0.85, 0.95, 1.0], "o-")
    ax.set_xticks([1, 2, 3])
    ax.set_title("Top-k accuracy")
    ax.set_xlabel("k"); ax.set_ylabel("Top-k accuracy")

    fig.suptitle("VAL lgb on multiclass [PROB_DIST sampled cap=5000]", fontsize=11)
    return fig


def bench(layout, label, n_repeat=5):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bench.png")
        # warm
        f = make_fig(layout); f.savefig(out)
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            f = make_fig(layout)
            f.savefig(out)
            times.append(time.perf_counter() - t0)
    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"  {label:<45} {m*1000:>8.1f} ms +/- {s*1000:>5.1f} ms")


def make_fig_with_adjust():
    """layout=None + explicit subplots_adjust tuned for our typical
    grid (2 rows x 3 cols, 18x8 figsize, 1 colorbar)."""
    fig = make_fig(None)
    # Margins tuned for an 18x8 figure with 1-2 line suptitle + 2-line
    # subplot titles, leaving room for legends + xtick rotation.
    fig.subplots_adjust(
        left=0.05, right=0.97, top=0.92, bottom=0.10,
        wspace=0.25, hspace=0.40,
    )
    return fig


def bench_adjust(label, n_repeat=5):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bench.png")
        f = make_fig_with_adjust(); f.savefig(out)
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            f = make_fig_with_adjust()
            f.savefig(out)
            times.append(time.perf_counter() - t0)
    import statistics
    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"  {label:<45} {m*1000:>8.1f} ms +/- {s*1000:>5.1f} ms")


print("# 6-panel multiclass figure with heatmap+colorbar")
bench("constrained", "constrained (current)")
bench("tight", "tight")
bench(None, "None (no auto layout)")
bench_adjust("None + subplots_adjust (tuned margins)")

# Visual A/B: save one PNG of each layout so we can eyeball the diff
print()
print("# Visual A/B PNGs:")
for lay, label in [("constrained", "constrained"), ("tight", "tight"), (None, "no_layout")]:
    out = f"D:/Temp/layout_ab_{label}.png"
    f = make_fig(lay)
    f.savefig(out)
    print(f"  {label:<15} -> {out}")
out = "D:/Temp/layout_ab_subplots_adjust.png"
f = make_fig_with_adjust(); f.savefig(out)
print(f"  {'subplots_adjust':<15} -> {out}")
