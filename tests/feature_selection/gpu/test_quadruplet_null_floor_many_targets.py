"""The quadruplet permutation-null floor builds the product matrix once and scores every shuffled target against it; the floor equals the per-permutation loop of the scorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a CUDA device answers."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:  # no driver: skipped
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _fixture(n: int = 30_000):
    """A raw frame, its quadruplet product columns and a target that depends on one four-way product."""
    from mlframe.feature_selection.filters._orthogonal_quadruplet_fe import generate_quadruplet_cross_basis_features

    rng = np.random.default_rng(3)
    raw = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(5)})
    y = ((raw["x0"] * raw["x1"] * raw["x2"] * raw["x3"] + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64).to_numpy()
    quads = [("x0", "x1", "x2", "x3"), ("x1", "x2", "x3", "x4")]
    eng = generate_quadruplet_cross_basis_features(raw, quads, max_degree=1, basis="auto")
    return raw, eng, y


def test_the_floor_with_one_product_build_equals_the_per_permutation_loop(monkeypatch):
    """Same permutations, same product columns, same resident MI: the floor is the one of the loop, and the device path is the one taken."""
    from mlframe.feature_selection.filters import _orthogonal_quadruplet_fe as Q

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")
    raw, eng, y = _fixture()
    assert not eng.empty
    kw = dict(nbins=10, basis="auto", n_perm=4, quantile=0.95)

    calls = {"many": 0}
    real = Q._null_maxima_device

    def counting(*a, **k):
        """Count the device path and pass through."""
        out = real(*a, **k)
        calls["many"] += int(out is not None)
        return out

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _gpu_resident_cross_basis as cb

    monkeypatch.setattr(cb, "crossbasis_device_chosen", lambda n_rows: True)
    monkeypatch.setattr(Q, "_null_maxima_device", counting)
    fast = Q._quadruplet_null_engineered_mi_floor(raw, eng, y, **kw)
    assert calls["many"] > 0, "the device-born path declined although its per-host choice was forced to device"
    monkeypatch.setattr(Q, "_null_maxima_device", lambda *a, **k: None)
    slow = Q._quadruplet_null_engineered_mi_floor(raw, eng, y, **kw)
    assert fast > 0.0
    np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-9)


def test_the_host_default_path_is_untouched(monkeypatch):
    """Without STRICT residency the many-target path declines and the per-permutation loop gives the floor."""
    from mlframe.feature_selection.filters import _orthogonal_quadruplet_fe as Q

    for k in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_FE_GPU_STRICT_RESIDENT", "MLFRAME_CMI_GPU"):
        monkeypatch.delenv(k, raising=False)
    raw, eng, y = _fixture(8_000)
    perms = [np.random.default_rng(i).permutation(y) for i in range(2)]
    assert Q._null_maxima_device(raw, eng, perms, 10, "auto") is None
