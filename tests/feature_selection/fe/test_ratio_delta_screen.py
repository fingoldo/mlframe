"""The ratio / log-ratio pair screen: the device kernel and the numba form decide every pair alike, and the grouped-delta gather equals the dict lookup."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _ratio_delta_fe as rd
from mlframe.feature_selection.filters import _ratio_delta_screen as sc


def _columns(n=4000, p=10):
    """Float32 sources with NaNs, a near-collinear pair, a near-constant denominator and a constant column."""
    rng = np.random.default_rng(0)
    m = rng.normal(size=(p, n)).astype(np.float32)
    m[1, ::97] = np.nan
    m[3] = m[2] * 2.0 + 0.001 * m[3]
    m[5] = 1.0 + 0.001 * m[5]
    m[7] = 0.0
    return m


@pytest.mark.parametrize("kind", [sc.RATIO, sc.LOG_RATIO])
def test_host_screen_rejects_the_redundant_pairs(kind):
    """a / (near-constant) is redundant with a, a constant source makes no ratio, independent normals pass."""
    m = _columns()
    ok = sc.ratio_pair_screen_host(list(m), kind, 1e-9, 0.99)
    assert not ok.diagonal().any()
    if kind == sc.RATIO:
        assert not ok[0, 5] and ok[0, 4]
    assert ok.sum() > 0


@pytest.mark.parametrize("kind", [sc.RATIO, sc.LOG_RATIO])
def test_device_screen_matches_host(kind):
    """Identical accept matrices on the host and the device."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _ratio_delta_screen_gpu as gpu

    m = _columns()
    host = sc.ratio_pair_screen_host(list(m), kind, 1e-9, 0.99)
    dev = gpu.ratio_pair_screen_device(cp.asarray(m), kind, 1e-9, 0.99)
    assert np.array_equal(host, dev)


def test_grouped_delta_gather_matches_the_lookup():
    """The recipe's lookup applied through ``apply_grouped_delta`` reproduces the fitted columns exactly."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=3000).astype(np.float32), "g": rng.integers(0, 17, 3000)})
    enc, recipes = rd.grouped_delta_features(X, "g", ["x"])
    for name, recipe in recipes.items():
        assert np.array_equal(np.asarray(enc[name]), np.asarray(rd.apply_grouped_delta(X, recipe)))
