"""FS_WRAPPERS-7 (2026-08-05 audit): ``split_into_train_test``'s X_estimator fast-path must raise a clear
error naming ``col_pos`` when ``X_estimator`` is supplied, ``col_pos=None``, and ``features_indices`` are
column-name strings -- not an opaque numpy string-to-int cast ``ValueError`` from the fallback branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers._helpers import split_into_train_test


def test_string_features_indices_without_col_pos_raises_clear_error():
    """X_estimator + string features_indices + col_pos=None must raise ValueError naming col_pos."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
    y = pd.Series([0, 1, 0, 1])
    X_estimator = X.to_numpy()

    with pytest.raises(ValueError, match="col_pos"):
        split_into_train_test(
            X,
            y,
            train_index=np.array([0, 1]),
            test_index=np.array([2, 3]),
            features_indices=np.array(["a"]),
            X_estimator=X_estimator,
            col_pos=None,
        )
