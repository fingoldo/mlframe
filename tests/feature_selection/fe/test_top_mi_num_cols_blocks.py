"""_top_mi_num_cols scores the numeric block in float32 column blocks, never as one full float64 copy.

A 1M x 500 numeric frame made the old full float64 copy 3.7 GB and the call failed with MemoryError on a 16 GB host.
The ranking must stay the same as scoring the whole matrix at once.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _extra_fe_families as fam
from mlframe.feature_selection.filters import _orthogonal_univariate_fe as orth
from mlframe.feature_selection.filters._unified_fe_gate import _coerce_y_classes


def _frame(n: int, p: int, seed: int = 0):
    """A float32 frame of p standard-normal columns and a binary target driven by columns c3 and c[p//2]."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)).astype(np.float32), columns=[f"c{i}" for i in range(p)])
    y = (X["c3"].to_numpy() * 0.7 + X[f"c{p // 2}"].to_numpy() * 0.4 + rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def test_blocks_are_bounded_and_float32(monkeypatch):
    """Every MI call receives at most one chunk of float32 columns, and together the chunks cover every numeric column once."""
    X, y = _frame(n=2000, p=3 * fam._TOP_MI_CHUNK_COLS + 7)
    seen_shapes, seen_dtypes = [], []
    real = orth._mi_classif_batch

    def spy(block, y_bin, **kw):
        """Record the block's shape and dtype, then score it with the real MI function."""
        seen_shapes.append(block.shape)
        seen_dtypes.append(block.dtype)
        return real(block, y_bin, **kw)

    monkeypatch.setattr(orth, "_mi_classif_batch", spy)
    fam._top_mi_num_cols(X, list(X.columns), y, max_cols=10)

    assert all(shape[1] <= fam._TOP_MI_CHUNK_COLS for shape in seen_shapes)
    assert all(dtype == np.float32 for dtype in seen_dtypes)
    assert sum(shape[1] for shape in seen_shapes) == X.shape[1]


@pytest.mark.parametrize("seed,n,p,k", [(0, 20_000, 150, 10), (1, 8_000, 90, 5)])
def test_topk_matches_whole_matrix_scoring(seed, n, p, k):
    """The block-scored top-k equals the top-k from scoring the whole float32 matrix in one call."""
    X, y = _frame(n=n, p=p, seed=seed)
    cols = list(X.columns)
    whole = np.asarray(orth._mi_classif_batch(X[cols].to_numpy(dtype=np.float32), _coerce_y_classes(y), nbins=10))
    expected = [cols[i] for i in np.argsort(-whole, kind="stable")[:k]]
    assert fam._top_mi_num_cols(X, cols, y, max_cols=k) == expected
