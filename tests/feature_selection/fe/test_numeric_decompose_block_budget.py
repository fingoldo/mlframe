"""The numeric decompose family scores its candidates in blocks of source columns under a byte cap (PLAN-10).

At 100000 x 400 the old budget (a quarter of the available RAM) gave two blocks of 285 columns and a 2.8 GB peak; the cap of 512 MB gives blocks of 60 columns and 0.76 GB, with the same result (the candidates are
independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _numeric_decompose_fe as D


def _frame(n: int, p: int, seed: int = 0):
    """Gaussian columns and a target that depends on the first two."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)).astype(np.float32), columns=[f"x{i}" for i in range(p)])
    y = ((X["x0"] + 0.5 * X["x1"] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    return X, y


def test_the_block_size_is_set_by_the_byte_cap(monkeypatch):
    """With a small cap the frame is split so that one block stays under it; with a big cap there is one block."""
    monkeypatch.delenv("MLFRAME_NUMERIC_DECOMPOSE_BLOCK_COLS", raising=False)
    X, _ = _frame(2000, 40)
    per_col = 2000 * 7 * 4 * 3
    monkeypatch.setattr(D, "_DECOMPOSE_BLOCK_BYTES", 5 * per_col)
    block = D._decompose_block_columns(X, None, 7)
    assert block == 5
    monkeypatch.setattr(D, "_DECOMPOSE_BLOCK_BYTES", 1000 * per_col)
    assert D._decompose_block_columns(X, None, 7) is None


def test_the_cap_is_never_above_the_default_512_mb():
    """The module constant is the 512 MB of the measurement."""
    assert D._DECOMPOSE_BLOCK_BYTES == 512 * 1024 * 1024


def test_blocks_give_the_same_result_as_one_frame(monkeypatch):
    """Forcing blocks of 7 columns gives the same selected columns and scores as one block."""
    X, y = _frame(3000, 30)
    monkeypatch.delenv("MLFRAME_NUMERIC_DECOMPOSE_BLOCK_COLS", raising=False)
    one = D.hybrid_numeric_decompose_fe_with_recipes(X, y)
    monkeypatch.setenv("MLFRAME_NUMERIC_DECOMPOSE_BLOCK_COLS", "7")
    blocked = D.hybrid_numeric_decompose_fe_with_recipes(X, y)
    assert list(one[0].columns) == list(blocked[0].columns)
    assert list(one[1]) == list(blocked[1])
