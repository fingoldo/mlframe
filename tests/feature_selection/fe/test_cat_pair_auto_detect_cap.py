"""Cat-pair auto-detection covers wide categorical frames, not just the first eight columns."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_pair_fe import auto_detect_cat_pair_cols, hybrid_cat_pair_fe


def _frame(n=6000, m=14):
    """Fourteen 4-level categorical columns; the target is a noisy modular sum of c11 and c12."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"c{i}": rng.integers(0, 4, n).astype(str) for i in range(m)})
    y = (((X["c11"].astype(int) + X["c12"].astype(int)) % 4 == 0) ^ (rng.random(n) < 0.1)).astype(int).to_numpy()
    return X, y


def test_auto_detect_returns_all_columns_up_to_the_cap():
    """All fourteen columns are detected by default."""
    X, _ = _frame()
    assert len(auto_detect_cat_pair_cols(X)) == 14


def test_planted_cross_beyond_eighth_column_is_found():
    """The (c11, c12) cross is emitted with default auto-detection."""
    X, y = _frame()
    assert "cross_c11_c12" in hybrid_cat_pair_fe(X, y)[1]
