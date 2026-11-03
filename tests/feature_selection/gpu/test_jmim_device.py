"""JMIM scores on the device equal the host njit loop (plug-in I((x, z); y), clipped at 0, minimum over the references)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orthogonal_jmim_fe as M


def test_device_equals_host(monkeypatch):
    """Signal, interaction, constant and noise candidates against a raw reference set."""
    rng = np.random.default_rng(0)
    n = 20000
    raw = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(6)})
    eng = pd.DataFrame({
        "x0__sq": raw["x0"] ** 2,
        "x1__mix": raw["x0"] * raw["x1"],
        "x2__const": np.full(n, 1.5),
        "x3__noise": rng.standard_normal(n),
        "x4__abs": np.abs(raw["x4"]),
    })
    y = ((raw["x0"] ** 2 + raw["x0"] * raw["x1"] + 0.5 * rng.standard_normal(n)) > 1.0).astype(int).to_numpy()
    monkeypatch.setenv("MLFRAME_JMIM_DEVICE", "0")
    want = M.score_features_by_jmim(raw, eng, y)
    monkeypatch.setenv("MLFRAME_JMIM_DEVICE", "1")
    monkeypatch.setattr("mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch.lookup_mi_classif_backend", lambda *a, **k: "cuda")
    called = []
    real = M._jmim_scores_device
    monkeypatch.setattr(M, "_jmim_scores_device", lambda *a, **k: (lambda r: called.append(r is not None) or r)(real(*a, **k)))
    got = M.score_features_by_jmim(raw, eng, y)
    assert called == [True]
    assert list(got["engineered_col"]) == list(want["engineered_col"])
    np.testing.assert_allclose(got["engineered_mi"].to_numpy(), want["engineered_mi"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert (want["engineered_mi"] > 0).sum() >= 2
