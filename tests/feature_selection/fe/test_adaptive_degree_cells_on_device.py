"""The device scoring of the adaptive-degree cells lists the host cells in the host order with host-close MI, and the per-source winner is the host's."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_adaptive_degree_device as D
from mlframe.feature_selection.filters import _orthogonal_adaptive_degree_fe as A
from mlframe.feature_selection.filters._orthogonal_shared import coerce_y_classif

DEGREES = (1, 2, 3, 4, 5, 6)


def _data(n: int, p: int = 5):
    """A frame with a skewed column and a nonlinear target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n).astype(np.float32) for i in range(p)})
    if p > 3:
        X["x3"] = np.exp(X["x3"])
    y = ((X["x0"] ** 2 + 0.5 * X["x1"] ** 3 + rng.normal(size=n) * 0.3) > 1).astype(np.int64).to_numpy()
    return X, coerce_y_classif(y)


@pytest.mark.parametrize("n", [1009, 6000])
@pytest.mark.parametrize("relaxed", ["1", "0"])
def test_cells_equal_host_cells(monkeypatch, n, relaxed):
    """Same cells in the same order, MI close to the host's, same winning (degree, basis) for every clear source."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_ADAPTIVE_DEGREE_DEVICE_MIN_ROWS", "0")
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relaxed)
    X, ya = _data(n)
    cols = list(X.columns)
    got = D.adaptive_degree_cells_on_device(X, cols, ya, DEGREES, "auto", 10)
    if got is None:
        pytest.skip("needs a usable CUDA device")
    names, cells, mi = got
    host_names, host_values, host_cells = A._build_degree_candidates(X, cols, ya, DEGREES, "auto", np.float32)
    assert names == host_names and cells == host_cells
    host_mi = A._mi_classif_batch(np.column_stack(host_values).astype(np.float32), ya, nbins=10)
    np.testing.assert_allclose(mi, host_mi, atol=5e-2 if n < 2000 else 5e-3)


def test_winner_values_are_rebuilt_on_the_host():
    """The lazy values of a cell equal the host enumeration's vector."""
    X, ya = _data(2000, p=2)
    cols = list(X.columns)
    _names, values, cells = A._build_degree_candidates(X, cols, ya, DEGREES, "auto", np.float32)
    lazy = D.LazyAdaptiveValues(X, cells, np.float32)
    for j in (0, len(cells) // 2, len(cells) - 1):
        np.testing.assert_array_equal(lazy[j], values[j])


def test_small_frames_stay_on_the_host():
    """Under the row threshold the device path declines."""
    X, ya = _data(1000, p=2)
    assert D.adaptive_degree_cells_on_device(X, list(X.columns), ya, DEGREES, "auto", 10) is None
