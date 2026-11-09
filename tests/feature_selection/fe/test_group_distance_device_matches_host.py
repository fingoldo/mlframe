"""The device group statistics give the host loop's z, KL and Wasserstein-1 lookups per group, with ties, NaN rows and a small group."""

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _group_distance_fe as host
from mlframe.feature_selection.filters import _group_distance_gpu as gpu


def _frame(n, seed):
    """Float32 numeric columns (one rounded to few levels, one with NaN rows), a 9-group key with one tiny group."""
    r = np.random.default_rng(seed)
    g = r.integers(0, 8, n).astype(object)
    g[:5] = "tiny"
    X = pd.DataFrame(
        {
            "g": g,
            "a": (r.standard_normal(n) + (r.integers(0, 8, n) == 3)).astype(np.float32),
            "b": np.round(r.standard_normal(n) * 2).astype(np.float32),
            "c": r.standard_normal(n).astype(np.float32),
        }
    )
    X.loc[X.index[::13], "c"] = np.nan
    return X


@pytest.mark.parametrize("n", [3000, 20000])
def test_device_lookups_equal_the_host_loop(n, monkeypatch):
    """Every lookup of the recipes (z, KL, W1 per group key) agrees with the host loop; the host loop is forced by choosing the host path."""
    X = _frame(n, 0)
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: False)
    enc_h, rec_h = host.generate_group_distance_features(X, ["g"], ["a", "b", "c"])
    monkeypatch.setattr(gpu, "use_device", lambda n_rows: True)
    enc_d, rec_d = host.generate_group_distance_features(X, ["g"], ["a", "b", "c"])
    assert list(enc_d.columns) == list(enc_h.columns)
    for name in rec_h:
        for key in ("z_lookup", "kl_lookup", "w_lookup"):
            h, d = rec_h[name][key], rec_d[name][key]
            assert set(h) == set(d)
            for k in h:
                assert d[k] == pytest.approx(h[k], rel=1e-5, abs=1e-9), (name, key, k)
    np.testing.assert_allclose(enc_d.to_numpy(), enc_h.to_numpy(), rtol=1e-5, atol=1e-9)


def test_too_many_groups_go_back_to_the_host():
    """More groups than the shared tables hold: the device entry hands the block back."""
    n = 2000
    xs = np.random.default_rng(1).standard_normal((2, n)).astype(np.float32)
    gid = (np.arange(n) % (gpu.MAX_GROUPS + 1)).astype(np.int32)
    bins = np.zeros(xs.shape, dtype=np.int16)
    assert gpu.group_stats_device(gid, gpu.MAX_GROUPS + 1, xs, bins, [np.sort(r.astype(np.float64)) for r in xs], 20) is None
