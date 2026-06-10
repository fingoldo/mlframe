"""cProfile + wall-time harness for the W3-F regression chart composer at production shape (n=2M).

Run: python audit/viz_audit_2026_06_10/bench_viz_regression.py

Measures compose_regression_figure end-to-end (the density-heatmap path triggers above 50k, so n=2M exercises the
hexbin/hist2d branch) plus a cProfile pass. Records the perf-fix history in _PROFILE_NOTES so a future re-run does not
re-grind the same levers.

_PROFILE_NOTES (dev box, n=2M, 3-call mean):
- baseline compose_regression_figure: ~1090 ms/call.
- ERR_BY_DECILE: replaced the full np.argsort(y) rank assignment (~0.4s/call at 2M) with np.quantile cut-points +
  np.searchsorted (k-way partial sort + O(n) assign) and the per-bucket Python-loop boolean-mask means with weighted
  np.bincount. Bit-identical means on continuous data; the isolated panel went 492 -> 160 ms/call (3.1x).
- RESID_VS_PRED: collapsed three per-bin np.percentile calls (q25, median, q75 = three partitions) into ONE
  np.percentile(sel, [25,50,75]) (one partition per bin). Bit-identical (<=4e-16 rounding).
- RESID_VS_PRED REJECTED attempt: a global np.lexsort((resid, which)) over n=2M (sort once, then per-bin quartiles by
  index lookup) measured ~4x SLOWER end-to-end (764 -> 3147 ms/call) -- a full 2M sort costs more than ~20 partial
  sorts of ~100k-element bins. Reverted; left the note at the callsite.
- Net: ~1090 -> ~756 ms/call (1.44x), all panels bit-identical. The residue is histogram2d (density bins) + the
  irreducible per-bin partitions + the finite-mask passes -- numpy C floor; no further actionable speedup at 2M.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

from mlframe.reporting.charts.regression import compose_regression_figure
from mlframe.training.targets import audit_residuals


def _make(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    yt = rng.normal(0.0, 1.0, n).astype(np.float64)
    yp = (yt + rng.normal(0.0, 0.3, n)).astype(np.float64)
    return yt, yp


def main() -> None:
    for n in (1_000_000, 2_000_000):
        yt, yp = _make(n)
        audit = audit_residuals(yt, yp)
        compose_regression_figure(yt[:1000], yp[:1000], audit=audit)  # warm
        t0 = time.perf_counter()
        for _ in range(3):
            compose_regression_figure(yt, yp, audit=audit, metrics_str="MAE=0.2 RMSE=0.3")
        t1 = time.perf_counter()
        print(f"compose_regression_figure n={n:_}: {(t1 - t0) / 3 * 1000:.1f} ms/call")

    yt, yp = _make(2_000_000)
    audit = audit_residuals(yt, yp)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        compose_regression_figure(yt, yp, audit=audit, metrics_str="m")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
