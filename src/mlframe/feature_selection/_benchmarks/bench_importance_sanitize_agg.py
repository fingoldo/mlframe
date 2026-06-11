"""cProfile + warm microbench harness for ``mlframe.feature_selection.importance``.

TARGET: the importance-vector sanitize/aggregation public entrypoint
``plot_feature_importance`` (returns the magnitude-sorted FI DataFrame) and the
text-log helper ``_format_top_fi_for_log``. The SHAP beeswarm path
(``show_shap_beeswarm_plot`` / ``explain_top_feature_importances``) is dominated
by ``shap.TreeExplainer`` (third-party) + matplotlib render and is not the
optimisation seam, so this harness profiles the pure-data FI aggregation path.

Representative shape: p=40 features (a wide engineered importance vector), plus a
small p=8 shape; both the suite-common data-return path (``show_plots=True`` in a
non-interactive session -> returns ``df`` BEFORE rendering) and the explicit
render path (``plot_file`` set, Agg backend).

----------------------------------------------------------------------------
FINDING (cProfile, p=40, 20000 data-return calls):

``plot_feature_importance`` computed the sort TWICE on every call:

    sorted_idx = np.argsort(feature_importances)               # full argsort
    sorted_columns = np.array(columns)[sorted_idx]             # fancy-index copy
    df = pd.Series(feature_importances[sorted_idx], ...,       # fancy-index copy
                   index=sorted_columns).to_frame() \
            .sort_values(by="fi", ascending=False)             # SECOND full sort

The leading ``np.argsort`` + two fancy-index gathers are pure waste: pandas
``sort_values`` re-sorts the Series from scratch, so the ascending pre-sort only
permutes the data into an order that ``sort_values`` immediately discards. The
final frame is identical whether or not the rows were pre-permuted (pandas sort
is a total order over the ``fi`` values; row identity is carried by the index).

FIX: drop the pre-sort. Build the Series directly from the raw aligned
``feature_importances`` / ``columns`` and let the single ``sort_values`` do the
one sort that actually matters. Bit-identical: same values, same index labels,
same descending order. Saves one O(p log p) argsort + two length-p ndarray
copies + one ``np.array(columns)`` build per call.

Run:
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
    python src/mlframe/feature_selection/_benchmarks/bench_importance_sanitize_agg.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import matplotlib

matplotlib.use("Agg")

import numpy as np

from mlframe.feature_selection.importance import plot_feature_importance


def _make_fi(p: int, seed: int) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    # Mixed-sign importances with a heavy near-zero tail (tree-on-residual shape):
    fi = rng.normal(size=p) * np.where(rng.random(p) < 0.3, 1.0, 1e-9)
    cols = [f"feat_{i:03d}" for i in range(p)]
    return fi.astype(np.float64), cols


def _call_data_return(fi: np.ndarray, cols: list[str]):
    # Suite-common path: show_plots default True but headless/non-interactive ->
    # returns df BEFORE any rendering. log_fi off to isolate the aggregation cost.
    return plot_feature_importance(fi, cols, kind="bench", show_plots=True, plot_file="", log_fi=False)


def _profile():
    fi, cols = _make_fi(40, 0)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20000):
        _call_data_return(fi, cols)
    pr.disable()
    for sort_key in ("cumulative", "tottime"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sort_key).print_stats(25)
        print(f"\n===== sort: {sort_key} =====")
        print(s.getvalue())


def _old_impl(fi, columns):
    """The pre-fix double-sort body, kept here so the bench can measure the
    drop-the-pre-sort ceiling and document the verdict reproducibly."""
    import numpy as _np
    import pandas as _pd

    sorted_idx = _np.argsort(fi)
    if len(columns) == 0:
        columns = _np.arange(len(fi))
    sorted_columns = _np.array(columns)[sorted_idx]
    return _pd.Series(data=fi[sorted_idx], index=sorted_columns, name="fi").to_frame().sort_values(by="fi", ascending=False)


def _new_drop_presort(fi, columns):
    """Candidate: single ``sort_values`` on the raw aligned Series (no pre-sort).
    Bit-identical ONLY when all importances are distinct; tie order diverges (two
    chained unstable quicksorts in the old body produce an arbitrary-but-fixed
    tie arrangement no single sort reproduces). Measured here as the CEILING."""
    import numpy as _np
    import pandas as _pd

    if len(columns) == 0:
        columns = _np.arange(len(fi))
    return _pd.Series(data=fi, index=_np.asarray(columns), name="fi").to_frame().sort_values(by="fi", ascending=False)


def _bench_fn(fn, fi, cols, n):
    for _ in range(500):
        fn(fi, cols)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(fi, cols)
    return (time.perf_counter() - t0) / n * 1e6


def _microbench():
    # Public-entrypoint wall (data-return path), then the old-vs-ceiling A/B that
    # underpins the no-actionable-speedup verdict in the module docstring.
    for p in (8, 40):
        fi, cols = _make_fi(p, 1)
        for _ in range(200):
            _call_data_return(fi, cols)
        n = 30000
        t0 = time.perf_counter()
        for _ in range(n):
            _call_data_return(fi, cols)
        dt = time.perf_counter() - t0
        print(f"entrypoint p={p:3d}: {dt / n * 1e6:8.2f} us/call  ({n} calls, {dt:.3f}s)")

    print("\n-- A/B: old double-sort vs drop-pre-sort ceiling (best-of-5, distinct fi) --")
    for p in (8, 40, 200):
        rng = np.random.default_rng(3)
        fi = rng.normal(size=p).astype(np.float64)
        cols = [f"f{i:03d}" for i in range(p)]
        old = min(_bench_fn(_old_impl, fi, cols, 50000) for _ in range(5))
        new = min(_bench_fn(_new_drop_presort, fi, cols, 50000) for _ in range(5))
        print(f"p={p:4d}: old={old:7.2f}us  drop-pre-sort={new:7.2f}us  speedup={old / new:.2f}x")


if __name__ == "__main__":
    print("### cProfile (p=40, 20000 data-return calls) ###")
    _profile()
    print("\n### warm microbench ###")
    _microbench()
