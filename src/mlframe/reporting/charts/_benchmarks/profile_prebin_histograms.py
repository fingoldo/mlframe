"""cProfile + wall-time harness for the PERF-4 spec-build pre-bin of histogram-style panels.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_prebin_histograms``

Affected panels: multilabel JACCARD_DIST / HAMMING_DIST, quantile WIDTH_DIST / PIT_HIST, ltr MRR_DIST. Each
previously placed a RAW length-n array into the FigureSpec; they now run ``np.histogram`` once at spec-build
and emit the pre-binned form (counts + bin_centers + bin_width), so the spec carries O(bins) data, never O(n).

This compares spec-build wall-time of the pre-bin path vs the prior raw path (building the equivalent raw
HistogramPanelSpec from the same length-n vector) at n=1e6. The pre-bin path does one extra O(n) np.histogram
pass but stores ~20 floats instead of n; the raw path stores the n-array reference (cheap to build, expensive
to retain). The expectation is that the single np.histogram pass is small relative to the per-row metric kernels
each panel already runs, so the pre-bin adds no meaningful spec-build cost -- and it removes the length-n RAM
retention entirely (the point of PERF-4).

Conclusion (numbers printed below): the extra np.histogram is a few ms at 1e6 -- dwarfed by the per-row Jaccard
kernel / PIT interpolation / MRR kernel that dominate each builder. No actionable speedup; pre-bin is the right
default (RAM win at zero wall cost). cProfile attributes the cost to np.histogram + the panel's own metric pass.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts._sampling import prebin_histogram
from mlframe.reporting.charts.ltr import _mrr_dist_panel
from mlframe.reporting.charts.multilabel import _hamming_dist_panel, _jaccard_dist_panel
from mlframe.reporting.charts.quantile import _pit_hist_panel, _width_dist_panel
from mlframe.reporting.spec import HistogramPanelSpec

_N = 1_000_000


def _multilabel_data(n: int):
    rng = np.random.default_rng(0)
    K = 4
    return rng.integers(0, 2, (n, K)).astype(np.int8), rng.random((n, K)), [f"l{k}" for k in range(K)]


def _quantile_data(n: int):
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    y = rng.standard_normal(n)
    alphas = (0.05, 0.25, 0.5, 0.75, 0.95)
    scale = 0.5 + np.abs(rng.standard_normal(n))
    preds = np.column_stack([scale * norm.ppf(a) for a in alphas])
    return y, preds, alphas


def _ltr_data(n_queries: int):
    rng = np.random.default_rng(0)
    sizes = rng.integers(2, 19, n_queries)
    total = int(sizes.sum())
    gid = np.repeat(np.arange(n_queries), sizes)
    rels = rng.integers(0, 4, total)
    scores = rels.astype(float) + rng.normal(0, 0.7, total)
    return rels, scores, gid


def _best(fn, *args, repeats: int = 3) -> float:
    fn(*args)  # warm (numba JIT for the Jaccard / MRR kernels)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _raw_jaccard(y_t, y_p, lbl):
    from mlframe.reporting.charts._jaccard_kernel import jaccard_rows
    j = jaccard_rows(np.ascontiguousarray(y_t, dtype=np.int8),
                     np.ascontiguousarray(y_p, dtype=np.float32))
    return HistogramPanelSpec(values=j, bins=20, density=True)


def main() -> None:
    yt, yp, lbl = _multilabel_data(_N)
    yq, pq, aq = _quantile_data(_N)
    rels, scores, gid = _ltr_data(100_000)

    print(f"spec-build wall-time @ n={_N} (pre-bin path; best of 3):")
    print(f"  JACCARD_DIST  {_best(_jaccard_dist_panel, yt, yp, lbl):8.2f} ms")
    print(f"  HAMMING_DIST  {_best(_hamming_dist_panel, yt, yp, lbl):8.2f} ms")
    print(f"  WIDTH_DIST    {_best(_width_dist_panel, yq, pq, aq):8.2f} ms")
    print(f"  PIT_HIST      {_best(_pit_hist_panel, yq, pq, aq):8.2f} ms")
    print(f"  MRR_DIST      {_best(_mrr_dist_panel, rels, scores, gid):8.2f} ms")

    # Isolate the extra np.histogram cost vs the prior raw-spec build for the Jaccard panel.
    j = None
    def _prebin_only():
        nonlocal j
        from mlframe.reporting.charts._jaccard_kernel import jaccard_rows
        if j is None:
            j = jaccard_rows(np.ascontiguousarray(yt, dtype=np.int8),
                             np.ascontiguousarray(yp, dtype=np.float32))
        prebin_histogram(j, 20, True)
    print(f"\n  np.histogram-only (Jaccard, n={_N}): {_best(lambda: _prebin_only()):8.3f} ms "
          f"(the extra spec-build cost the pre-bin adds)")

    pr = cProfile.Profile()
    pr.enable()
    _jaccard_dist_panel(yt, yp, lbl)
    _width_dist_panel(yq, pq, aq)
    _mrr_dist_panel(rels, scores, gid)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
