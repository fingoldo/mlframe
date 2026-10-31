"""The raw-MI noise floor equals the one computed from the column-stacked matrix, on a miss and on a memo hit."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _unified_fe_gate as G
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch


def test_floor_equals_the_stacked_matrix_floor_and_memo_hits_agree():
    """A fresh call, a repeated call (memo hit) and the reference built with ``np.column_stack`` return the same float."""
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({f"c{i}": rng.normal(size=n) for i in range(12)})
    X["c3"] = np.round(X["c3"], 1)
    y = (X["c0"].to_numpy() + rng.normal(size=n) > 0).astype(int)
    G._RAW_MI_FLOOR_MEMO.clear()
    first = G.raw_mi_noise_floor(X, y, nbins=10)
    again = G.raw_mi_noise_floor(X, y, nbins=10)
    stacked = np.column_stack([X[c].to_numpy(dtype=G._crit_np_dtype()) for c in X.columns])
    reference = G._floor_from_raw_mi(_mi_classif_batch(stacked, G._coerce_y_classes(y), nbins=10), n, 10, G._coerce_y_classes(y), 3.5)
    assert first == again == reference
    assert len(G._RAW_MI_FLOOR_MEMO) == 1
