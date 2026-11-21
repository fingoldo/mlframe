"""A frame without NaN skips the NaN mask and the fill; a frame with NaN still routes them to the dedicated bin; the tile check agrees with a full scan."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import categorize_dataset
from mlframe.feature_selection.filters.discretization._discretization_dataset import _array_has_nan


@pytest.mark.parametrize("shape", [(10, 3), (70000, 2), (65536, 1)])
def test_array_has_nan_agrees_with_a_full_scan(shape):
    """NaN in the first, a middle and the last tile, and none at all."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(shape).astype(np.float32)
    assert _array_has_nan(a) is False
    for r in (0, shape[0] // 2, shape[0] - 1):
        b = a.copy()
        b[r, -1] = np.nan
        assert _array_has_nan(b) is True and np.isnan(b).any()
    assert _array_has_nan(a.astype(np.int32)) is False and _array_has_nan(np.empty((0, 3), dtype=np.float32)) is False


@pytest.mark.parametrize("strategy", ["separate_bin", "propagate", "fillna_zero"])
def test_codes_are_identical_with_and_without_the_skip(strategy, monkeypatch):
    """Without NaN the codes equal those of the path that always built the mask; with NaN the NaN rows keep the dedicated bin."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((3000, 4)), columns=list("abcd"))
    clean, _c, _nb = categorize_dataset(X, method="quantile", n_bins=8, dtype=np.int32, missing_strategy=strategy)
    monkeypatch.setattr("mlframe.feature_selection.filters.discretization._discretization_dataset._array_has_nan", lambda arr, tile_rows=65536: True)
    forced, _c2, _nb2 = categorize_dataset(X, method="quantile", n_bins=8, dtype=np.int32, missing_strategy=strategy)
    np.testing.assert_array_equal(clean, forced)
    monkeypatch.undo()
    Xn = X.copy()
    Xn.loc[::7, "b"] = np.nan
    withnan, _c3, nb3 = categorize_dataset(Xn, method="quantile", n_bins=8, dtype=np.int32, missing_strategy="separate_bin")
    assert (withnan[::7, 1] == withnan[::7, 1].max()).all() and int(nb3[1]) >= 9
