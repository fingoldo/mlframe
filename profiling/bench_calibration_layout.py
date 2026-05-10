"""A/B bench: constrained vs None vs None+adjust on the canonical
2-panel calibration plot (scatter + histogram + colorbar spanning
both axes).

Wave 4 1M-row fuzz aggregate attributed 146 s of wall across 108
calls to ``show_calibration_plot`` (1.35 s / call). All 108 calls
build their own ``Figure(layout="constrained")`` outside the
FigureSpec pipeline, so Wave 3's flip doesn't reach them.

This bench mimics the calibration plot's structure precisely
(scatter + multi-axis colorbar + histogram with sharex) and checks
whether layout=None produces visually-equivalent output AND a real
speedup.

Usage:
    python -m mlframe.profiling.bench_calibration_layout
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def make_calibration_fig(layout):
    """Reproduce the calibration plot's structure with layout knob."""
    fig = Figure(figsize=(12, 6), layout=layout)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_main)

    # Calibration scatter (top)
    n_bins = 10
    rng = np.random.default_rng(0)
    freqs_pred = np.linspace(0.05, 0.95, n_bins)
    freqs_true = freqs_pred + rng.normal(0, 0.02, n_bins)
    hits = rng.integers(500, 50000, n_bins)
    cm = matplotlib.colormaps["RdYlBu"]
    sc = ax_main.scatter(
        x=freqs_pred, y=freqs_true, marker="o",
        s=5000 * hits / hits.sum(), c=hits, label="Observed", cmap=cm,
    )
    ax_main.plot(freqs_pred, freqs_pred, "g--", label="Perfect")
    cbar = fig.colorbar(sc, ax=[ax_main, ax_hist])  # spans BOTH axes
    cbar.set_label("Bin population")
    ax_main.set_ylabel("Observed frequency")
    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_main.set_title(
        "ICE=0.0123, BR_DECOMP REL=0.001 RES=0.034 UNC=0.245, CMAEW=0.027\n"
        "LL=0.450, ROC_AUC=0.812, PR_AUC=0.451"
    )

    # Histogram (bottom)
    ax_hist.bar(
        freqs_pred, hits,
        width=0.08, align="center",
        color=cm(hits / hits.max()),
        edgecolor="white", linewidth=0.5,
    )
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Bin population")
    return fig


def make_calibration_fig_with_adjust():
    """layout=None + tuned subplots_adjust margins for the calibration
    figsize=(12, 6) layout with 2-line title + multi-axis colorbar.

    Margins were tuned by trial-and-error to leave room for the
    2-line title at top, colorbar at right, x-labels at bottom, and
    y-labels at left. If matplotlib's default geometry already fits
    on this figsize the visual diff vs constrained is negligible.
    """
    fig = make_calibration_fig(None)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.10, hspace=0.05)
    return fig


def bench(fn, label, n_repeat=5):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bench.png")
        f = fn(); f.savefig(out); plt.close(f)
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            f = fn()
            f.savefig(out)
            plt.close(f)
            times.append(time.perf_counter() - t0)
    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"  {label:<55} {m*1000:>9.1f} ms +/- {s*1000:>6.1f} ms")


def main() -> None:
    print("# Calibration plot (2-subplot + multi-axis colorbar + 2-line title)\n")
    bench(lambda: make_calibration_fig("constrained"), "constrained (current)")
    bench(lambda: make_calibration_fig(None), "None (no auto layout)")
    bench(make_calibration_fig_with_adjust, "None + subplots_adjust (tuned)")

    print()
    print("# Visual A/B PNGs:")
    for lay, label in [("constrained", "constrained"), (None, "no_layout")]:
        out = f"D:/Temp/calib_ab_{label}.png"
        f = make_calibration_fig(lay); f.savefig(out); plt.close(f)
        print(f"  {label:<15} -> {out}")
    out = "D:/Temp/calib_ab_subplots_adjust.png"
    f = make_calibration_fig_with_adjust(); f.savefig(out); plt.close(f)
    print(f"  {'subplots_adjust':<15} -> {out}")


if __name__ == "__main__":
    main()
