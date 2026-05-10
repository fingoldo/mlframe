"""Micro-bench: violin / KDE rendering with vs without sampling.

Validates the assumption that capping per-group violin data at 5000
samples is a measurable speedup vs the un-sampled N/K-per-group form
used by ``_prob_dist_panel`` (multiclass) and ``_score_by_rel_panel``
(LTR) on production-scale 1M-row inputs.

Also runs the multilabel Jaccard panel (vectorised vs prior Python
row-loop form) for the same input.

Usage:
    python -m mlframe.profiling.bench_violin_sampling

Reports wall-time mean / std over 5 repeats per scenario. Each
scenario builds the panel spec AND renders the figure to PNG to
capture the full chart pipeline cost (KDE + draw + savefig).
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

# Import the panel builders + matplotlib renderer.
from mlframe.reporting.charts.multiclass import _prob_dist_panel
from mlframe.reporting.charts.ltr import _score_by_rel_panel
from mlframe.reporting.charts.multilabel import _jaccard_dist_panel
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer


def _bench(label: str, fn, n_repeat: int = 5) -> Tuple[float, float]:
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"  {label:<70} {mean*1000:>10.1f} ms ± {std*1000:>6.1f} ms")
    return mean, std


def _render_violin(panel, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    MatplotlibRenderer()._violin(ax, panel)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def _render_hist(panel, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(panel.values, bins=panel.bins, density=getattr(panel, "density", False))
    ax.set_title(panel.title)
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def _build_prob_dist_unsampled(y_true, y_proba, classes):
    """OLD form: pre-fix _prob_dist_panel without subsample_for_density.

    Reproduced verbatim so the bench reports a fair NEW-vs-OLD delta
    against the same render path. Kept INSIDE the bench file (not
    shipped) so the production builder is the only one in the
    package.
    """
    from mlframe.reporting.spec import ViolinPanelSpec
    K = len(classes)
    groups, labels = [], []
    for k in range(K):
        mask = np.asarray(y_true) == k
        if not mask.any():
            groups.append(np.array([0.0]))
            labels.append(f"{classes[k]} (n=0)")
        else:
            groups.append(y_proba[mask, k])
            labels.append(f"{classes[k]} (n={int(mask.sum())})")
    return ViolinPanelSpec(
        groups=tuple(groups), group_labels=tuple(labels),
        title="P(y=true_class) per true class",
        xlabel="True class", ylabel="Predicted P(y = true_class)",
    )


def _build_jaccard_pyrowloop(y_true, y_proba, labels):
    """OLD form: pre-fix _jaccard_dist_panel with Python row-loop."""
    from mlframe.reporting.spec import HistogramPanelSpec
    y_pred = (y_proba >= 0.5).astype(np.int8)
    n = y_true.shape[0]
    jaccards = np.zeros(n)
    for i in range(n):
        intersection = int(((y_true[i] == 1) & (y_pred[i] == 1)).sum())
        union = int(((y_true[i] == 1) | (y_pred[i] == 1)).sum())
        jaccards[i] = (intersection / union) if union > 0 else 1.0
    return HistogramPanelSpec(
        values=jaccards, bins=20,
        title=f"Per-row Jaccard (mean={jaccards.mean():.3f})",
        xlabel="Jaccard score", ylabel="Density", density=True,
    )


def main() -> None:
    rng = np.random.default_rng(42)
    n = 1_000_000

    print(f"# Bench: violin / hist panels on N={n:_}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bench.png")

        # 1) Multiclass PROB_DIST (3 classes)
        print("## multiclass _prob_dist_panel (K=3 classes)")
        K = 3
        y_true = rng.integers(0, K, n)
        y_proba = rng.uniform(0, 1, (n, K))
        # Normalise rows to make it look like probs
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        def _multiclass_bench_new():
            panel = _prob_dist_panel(y_true, y_proba, classes=list(range(K)))
            _render_violin(panel, out)

        def _multiclass_bench_old():
            panel = _build_prob_dist_unsampled(y_true, y_proba, classes=list(range(K)))
            _render_violin(panel, out)

        # Warm pass first to amortise gaussian_kde / matplotlib JIT.
        _multiclass_bench_new()
        _bench("OLD (un-sampled, N/K=333k per violin)        ", _multiclass_bench_old, n_repeat=3)
        _bench("NEW (sampled, cap=5000 per violin)           ", _multiclass_bench_new)
        print()

        # 2) LTR SCORE_BY_REL (4 quartile bins, continuous relevance)
        print("## ltr _score_by_rel_panel (4 quartile bins, continuous relevance)")
        y_rel = rng.uniform(0, 5, n).astype(np.float64)
        y_score = rng.normal(0, 1, n)
        gids = rng.integers(0, 5000, n)

        def _ltr_bench():
            panel = _score_by_rel_panel(y_rel, y_score, gids)
            _render_violin(panel, out)

        _bench("NEW (sampled, cap=5000 per quartile)         ", _ltr_bench)
        print()

        # 3) Multilabel JACCARD_DIST
        print("## multilabel _jaccard_dist_panel (K=10 labels, density hist)")
        Kml = 10
        y_true_ml = (rng.uniform(0, 1, (n, Kml)) < 0.3).astype(np.int8)
        y_proba_ml = rng.uniform(0, 1, (n, Kml)).astype(np.float32)

        def _jaccard_bench_new():
            panel = _jaccard_dist_panel(y_true_ml, y_proba_ml, labels=[f"l{i}" for i in range(Kml)])
            _render_hist(panel, out)

        def _jaccard_bench_old():
            panel = _build_jaccard_pyrowloop(y_true_ml, y_proba_ml, labels=[f"l{i}" for i in range(Kml)])
            _render_hist(panel, out)

        # Warm numba JIT cache (first call lowers the parallel kernel,
        # ~150 ms cold; we want to measure steady-state per-panel cost).
        _jaccard_bench_new()
        _bench("OLD (Python row-loop, N=1M)                  ", _jaccard_bench_old, n_repeat=1)
        _bench("NEW (numba parallel, cached)                 ", _jaccard_bench_new)
        print()


if __name__ == "__main__":
    main()
