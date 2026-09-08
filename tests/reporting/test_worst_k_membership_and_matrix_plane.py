"""Two hot paths that answered a cheap question expensively, and one that contradicted its own docstring.

``_append_missing_worst_k`` built a Python set over the WHOLE plotted subsample to ask whether each worst-K
point was already in it, then re-scanned the full ``(pred, true)`` arrays once per point that was, to find
where. Both are the same lexicographic lookup.

``_resolve_feature_matrix`` promised a matrix "without a full frame copy" and then called
``np.column_stack``, which materialises a second dense copy of everything gathered while the gathered list
is still alive -- roughly double the peak of the result it returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.reporting.charts._error_analysis_shared import ordinal_codes
from mlframe.reporting.charts.error_analysis import _resolve_feature_matrix
from mlframe.reporting.charts.regression import _append_missing_worst_k


def test_a_worst_k_point_already_plotted_is_not_plotted_twice():
    """It must be highlighted where it already sits, not appended beside itself."""
    yp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yt = np.array([0.0, 1.5, 2.5, 3.5, 9.0])
    s_pred, s_true = yp[:4].copy(), yt[:4].copy()
    out_pred, _, highlight = _append_missing_worst_k(s_pred, s_true, yp, yt, np.array([1, 2]))
    assert len(out_pred) == 4, f"the panel grew to {len(out_pred)} points for two it already had"
    assert sorted(highlight.tolist()) == [1, 2], f"highlighted {highlight.tolist()} instead of the points' own positions"


def test_a_worst_k_point_the_subsample_dropped_is_appended_once():
    """The other half: a point the subsample never kept has to be added, exactly once."""
    yp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yt = np.array([0.0, 1.5, 2.5, 3.5, 9.0])
    out_pred, out_true, highlight = _append_missing_worst_k(yp[:4].copy(), yt[:4].copy(), yp, yt, np.array([4]))
    assert len(out_pred) == 5 and out_pred[-1] == 4.0 and out_true[-1] == 9.0
    assert highlight.tolist() == [4], f"the appended point is highlighted at {highlight.tolist()}"


def test_the_highlighted_points_are_the_worst_k_points():
    """The contract that matters: whatever the indices are, they must land on the right coordinates."""
    rng = np.random.default_rng(0)
    yp, yt = rng.normal(size=400), rng.normal(size=400)
    keep = rng.choice(400, 250, replace=False)
    wk = rng.choice(400, 30, replace=False)
    out_pred, out_true, highlight = _append_missing_worst_k(yp[keep].copy(), yt[keep].copy(), yp, yt, wk)
    assert highlight.size == wk.size, f"{highlight.size} highlights for {wk.size} worst-K rows"
    assert np.allclose(np.sort(out_pred[highlight]), np.sort(yp[wk]))
    assert np.allclose(np.sort(out_true[highlight]), np.sort(yt[wk]))


def test_duplicate_coordinates_do_not_multiply_the_highlights():
    """Guard on the lookup: several plotted points can share one (pred, true), and only one may be claimed."""
    s_pred = np.array([1.0, 1.0, 1.0, 2.0])
    s_true = np.array([5.0, 5.0, 5.0, 6.0])
    yp, yt = s_pred.copy(), s_true.copy()
    out_pred, _, highlight = _append_missing_worst_k(s_pred.copy(), s_true.copy(), yp, yt, np.array([0]))
    assert len(out_pred) == 4, "a duplicated coordinate was appended as though it were missing"
    assert highlight.size == 1, f"one worst-K row produced {highlight.size} highlights"


def _mixed_frame(n: int = 20_000, p: int = 40) -> pd.DataFrame:
    """A frame with the numeric, categorical, boolean and integer columns the resolver has to coerce."""
    rng = np.random.default_rng(0)
    data = {f"f{i}": rng.normal(size=n) for i in range(p - 3)}
    data["cat"] = rng.choice(list("abcde"), n)
    data["flag"] = rng.random(n) < 0.5
    data["count"] = rng.integers(0, 9, n)
    return pd.DataFrame(data)


def test_the_plane_matches_what_column_stack_produced():
    """Filling a preallocated plane must not change a single value."""
    df = _mixed_frame()
    plane, names = _resolve_feature_matrix(df, None)
    stacked = np.column_stack(
        [(ordinal_codes(df[c].to_numpy()) if df[c].to_numpy().dtype.kind in "OUSb" else df[c].to_numpy().astype(np.float64)) for c in df.columns]
    )
    assert names == list(df.columns)
    assert np.array_equal(plane, stacked, equal_nan=True), "the preallocated plane differs from the stacked one"


def test_the_matrix_is_column_major():
    """Every consumer of this matrix reads it by column; ``_bin_matrix`` documents the same choice."""
    plane, _ = _resolve_feature_matrix(_mixed_frame(n=1_000, p=6), None)
    assert plane.flags["F_CONTIGUOUS"], "the feature plane is row-major, so each per-column read is a strided gather"


def test_the_gathered_columns_are_not_held_alongside_the_result():
    """The docstring promises no full-frame copy; column_stack made one and kept the inputs alive beside it."""
    import tracemalloc

    df = _mixed_frame(n=60_000, p=150)
    tracemalloc.start()
    plane, _ = _resolve_feature_matrix(df, None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result_bytes = plane.nbytes
    assert peak < result_bytes * 1.6, f"peak was {peak / 1e6:.0f}MB for a {result_bytes / 1e6:.0f}MB result; the gathered columns are still alive"
