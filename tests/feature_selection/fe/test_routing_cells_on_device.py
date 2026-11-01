"""The device scoring of the conditional-routing cells lists the host cells in the host order with the host scores (up to the device rounding), and the winners are the host ones."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _orthogonal_routing_device as D
from mlframe.feature_selection.filters import _orthogonal_routing_fe as R
from mlframe.feature_selection.filters._orthogonal_shared import coerce_y_classif

BASES = ("hermite", "legendre", "chebyshev", "laguerre")


def _data(n=20000, p=6):
    """A frame where two columns carry a nonlinear signal."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n).astype(np.float32) for i in range(p)})
    y = ((X["x0"] ** 2 + 0.5 * X["x1"] ** 3 + rng.normal(size=n) * 0.3) > 1).astype(np.int64).to_numpy()
    return X, y


def test_device_cells_equal_host_cells():
    """Same cell names in the same order; MI within 1e-4 and the routing score within 1e-5 of the host values."""
    pytest.importorskip("cupy")
    X, y = _data()
    ya = coerce_y_classif(y)
    cols = list(X.columns)
    got = D.routing_scores_on_device(X, cols, R.PRE_TRANSFORM_NAMES, BASES, (2, 3), ya, 10, "corr")
    if got is None:
        pytest.skip("needs a usable CUDA device")
    names, meta, mi, score = got
    host_names, host_values, host_meta = R._enumerate_routing_cells(X, cols, R.PRE_TRANSFORM_NAMES, BASES, (2, 3))
    assert names == host_names and meta == host_meta
    mat = R._stack_candidates(host_values)
    host_mi = R._mi_classif_batch(mat, ya, nbins=10)
    host_score = R._routing_scores("corr", ya, mat, host_mi)
    np.testing.assert_allclose(mi, host_mi, atol=1e-4)
    np.testing.assert_allclose(score, host_score, atol=1e-5)


def test_winner_values_are_rebuilt_on_the_host():
    """The lazy values of a cell equal the host enumeration's vector."""
    X, _y = _data(n=3000, p=2)
    cols = list(X.columns)
    _names, values, meta = R._enumerate_routing_cells(X, cols, R.PRE_TRANSFORM_NAMES, BASES, (2, 3))
    lazy = D.LazyRoutingValues(X, meta)
    for j in (0, len(meta) // 2, len(meta) - 1):
        np.testing.assert_array_equal(lazy[j], values[j])


def test_small_frames_stay_on_the_host():
    """Below the row threshold the device path declines."""
    X, y = _data(n=1000, p=2)
    assert D.routing_scores_on_device(X, list(X.columns), R.PRE_TRANSFORM_NAMES, BASES, (2, 3), coerce_y_classif(y), 10, "corr") is None


@pytest.mark.parametrize("n", [129, 1009, 4097])
@pytest.mark.parametrize("relaxed", ["1", "0"])
def test_cells_equal_host_across_sizes_and_criterion_dtypes(monkeypatch, n, relaxed):
    """Odd row counts around block boundaries, float32 and float64 criterion dtypes: same cell list, MI and score close to the host's."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_ROUTING_DEVICE_MIN_ROWS", "0")
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relaxed)
    X, y = _data(n=n, p=3)
    ya = coerce_y_classif(y)
    cols = list(X.columns)
    got = D.routing_scores_on_device(X, cols, R.PRE_TRANSFORM_NAMES, BASES, (2, 3), ya, 10, "corr")
    if got is None:
        pytest.skip("needs a usable CUDA device")
    names, meta, mi, score = got
    host_names, host_values, host_meta = R._enumerate_routing_cells(X, cols, R.PRE_TRANSFORM_NAMES, BASES, (2, 3))
    assert names == host_names and meta == host_meta
    mat = R._stack_candidates(host_values)
    host_mi = R._mi_classif_batch(mat, ya, nbins=10)
    host_score = R._routing_scores("corr", ya, mat, host_mi)
    np.testing.assert_allclose(score, host_score, atol=1e-4)
    # the device bins by edge selection, so a small frame can move a boundary row: the MI only has to stay close
    np.testing.assert_allclose(mi, host_mi, atol=5e-2 if n < 200 else 5e-3)
