"""``_top_mi_num_cols`` scores a raw column once per fit scope and returns the ranking of the unmemoised call."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _extra_fe_families as fam
import mlframe.feature_selection.filters._orthogonal_univariate_fe as orth
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_scoring_memo import orth_scoring_memo_scope


def _frame() -> "tuple[pd.DataFrame, np.ndarray]":
    """Twelve float32 columns, two of them informative, and a binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((4000, 12)).astype(np.float32), columns=[f"c{i}" for i in range(12)])
    y = (X["c3"] + 0.5 * X["c7"] + 0.5 * rng.standard_normal(4000) > 0).astype(np.int64).to_numpy()
    return X, y


def test_second_family_ranks_from_the_table(monkeypatch):
    """Inside a fit scope a second call on the same frame object, column list and target launches no MI batch and returns the same top columns; another frame or target is scored."""
    X, y = _frame()
    cols = list(X.columns)
    want = fam._top_mi_num_cols(X, cols, y, 4)  # outside a scope: the plain route
    assert set(want[:2]) == {"c3", "c7"}
    widths = []
    real = orth._mi_classif_batch

    def spy(mat, *args, **kwargs):
        """Record the width of each batch MI call."""
        widths.append(int(mat.shape[1]))
        return real(mat, *args, **kwargs)

    monkeypatch.setattr(orth, "_mi_classif_batch", spy)
    with orth_scoring_memo_scope():
        assert fam._top_mi_num_cols(X, cols, y, 4) == want
        assert fam._top_mi_num_cols(X, cols, y.copy(), 4) == want  # the same frame object and target values: from the table
        assert widths == [12]
        other = X.copy()
        assert fam._top_mi_num_cols(other, cols, y, 4) == want  # another frame object is scored on its own
        assert widths == [12, 12]
        assert fam._top_mi_num_cols(X, cols, 1 - y, 4) == want  # another target is scored on its own (the mirrored labels give the same MI)
        assert widths == [12, 12, 12]
    fam._top_mi_num_cols(X, cols, y, 4)  # outside the scope nothing is remembered
    assert widths == [12, 12, 12, 12]
