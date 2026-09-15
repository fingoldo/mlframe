"""The missingness families' copy-free path allocates materially less than the augmented-frame path, on the same frame.

The indicator, count and pattern families returned the whole frame with their columns appended (``pd.concat`` / ``X.copy()``) although the
cascade keeps only the appended columns, and the indicator's raw MI floor was built from a ``frame[[...]]`` sub-frame. Scoring the floor still
needs a float64 matrix of the raw numeric columns and its binned codes, so the absolute peak cannot drop below that; what must disappear is
the per-family frame copies. Both paths run here on the same frame, in the same process, so the comparison is not sensitive to warm-up.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._missingness_fe import (
    missing_indicator_with_recipes,
    missingness_count_with_recipes,
    missingness_pattern_with_recipes,
)

# Measured on this fixture (60k x 60 and 120k x 40 float64, warm, three repeats each): the augmented path peaked at 3.03-4.34 frame copies and
# the copy-free path at 1.03-1.04, the float64 matrix the MI floor genuinely needs. The saving was 2.00 frames on every warm repeat.
_MIN_COPIES_SAVED = 1.5
_MAX_COPY_FREE_FRAMES = 1.25


def _wide_frame(n: int = 60_000, p: int = 60, seed: int = 0):
    """A float frame with four MNAR columns (NaN at 5%) and a binary target."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, p))
    miss = rng.random((n, 4)) < 0.05
    for j in range(4):
        data[miss[:, j], j] = np.nan
    X = pd.DataFrame(data, columns=[f"c{i}" for i in range(p)])
    y = ((np.nan_to_num(data[:, 5]) + 0.8 * miss[:, 0]) > 0.3).astype(np.int64)
    return X, y


def _legacy_path(X, y, cols):
    """The shape of the old cascade calls: augmented frames and a sub-frame floor reference."""
    raw_cols = list(X.columns)
    ind, app, _ = missing_indicator_with_recipes(X, cols=cols, mi_gate=True, y=y, raw_X=X[raw_cols])
    cnt, _, _ = missingness_count_with_recipes(X, cols=cols)
    pat, _, _ = missingness_pattern_with_recipes(X, cols=cols, top_k=3)
    return ind[app], cnt, pat


def _copy_free_path(X, y, cols):
    """The new cascade calls: encoding-only frames and the floor reading columns by name."""
    raw_cols = list(X.columns)
    ind, _app, _ = missing_indicator_with_recipes(X, cols=cols, mi_gate=True, y=y, raw_X=X, raw_columns=raw_cols, return_augmented=False)
    cnt, _, _ = missingness_count_with_recipes(X, cols=cols, return_augmented=False)
    pat, _, _ = missingness_pattern_with_recipes(X, cols=cols, top_k=3, return_augmented=False)
    return ind, cnt, pat


def _peak(fn, *args) -> int:
    """Peak traced bytes while ``fn(*args)`` runs."""
    tracemalloc.start()
    try:
        fn(*args)
        _, pk = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(pk)


def test_copy_free_path_saves_frame_copies():
    """The copy-free path's peak is lower than the augmented path's by at least the per-family frame copies."""
    X, y = _wide_frame()
    cols = ["c0", "c1", "c2", "c3"]
    frame_bytes = int(X.memory_usage(index=False, deep=False).sum())
    _copy_free_path(X.iloc[:2000], y[:2000], cols)  # warm imports and compiled kernels
    _legacy_path(X.iloc[:2000], y[:2000], cols)
    legacy = _peak(_legacy_path, X, y, cols)
    copy_free = _peak(_copy_free_path, X, y, cols)
    saved = (legacy - copy_free) / frame_bytes
    assert saved >= _MIN_COPIES_SAVED, f"copy-free peak {copy_free / 1e6:.1f} MB vs augmented {legacy / 1e6:.1f} MB saves {saved:.2f} frame copies"
    assert copy_free <= _MAX_COPY_FREE_FRAMES * frame_bytes, f"copy-free peak {copy_free / frame_bytes:.2f} frames exceeds the floor matrix's ~1 frame"


def test_copy_free_encodings_equal_augmented_columns():
    """Encoding-only frames carry exactly the appended columns of the augmented path, on the frame's index."""
    X, _y = _wide_frame(n=5000, p=12, seed=3)
    cols = ["c0", "c1", "c2", "c3"]
    for fn, kw in (
        (missing_indicator_with_recipes, dict(cols=cols)),
        (missingness_count_with_recipes, dict(cols=cols)),
        (missingness_pattern_with_recipes, dict(cols=cols, top_k=3)),
    ):
        full, app_full, _ = fn(X, **kw)
        enc, app_enc, _ = fn(X, return_augmented=False, **kw)
        assert app_full == app_enc and app_enc
        assert list(enc.columns) == app_enc and enc.index.equals(X.index)
        pd.testing.assert_frame_equal(enc, full[app_full])
        pd.testing.assert_frame_equal(full[list(X.columns)], X)
