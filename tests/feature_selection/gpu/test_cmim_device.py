"""CMIM scores on the device equal the host loop (same Miller-Madow CMI, minimum over references of another source, clipped at 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orthogonal_cmim_fe as M


def test_device_equals_host(monkeypatch):
    """Signal candidates, a candidate redundant with another source, noise; self-source references excluded."""
    rng = np.random.default_rng(0)
    n = 20000
    raw = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(8)})
    eng = pd.DataFrame({
        "x0__sq": raw["x0"] ** 2,
        "x1__cube": raw["x1"] ** 3,
        "x2__copy3": raw["x3"] + 0.01 * rng.standard_normal(n),
        "x4__noise": rng.standard_normal(n),
        "x5__mix": raw["x0"] * raw["x5"],
    })
    y = ((raw["x0"] ** 2 + raw["x1"] + 0.5 * raw["x3"] + 0.5 * rng.standard_normal(n)) > 1.0).astype(int).to_numpy()
    monkeypatch.setenv("MLFRAME_CMIM_DEVICE", "0")
    want = M.score_features_by_cmim(raw, eng, y)
    monkeypatch.setenv("MLFRAME_CMIM_DEVICE", "1")
    from mlframe.feature_selection.filters._scorer_device_dispatch import forced_backend

    called = []
    real = M._cmim_scores_device
    monkeypatch.setattr(M, "_cmim_scores_device", lambda *a, **k: (lambda r: called.append(r is not None) or r)(real(*a, **k)))
    with forced_backend("device"):
        got = M.score_features_by_cmim(raw, eng, y)
    assert called == [True]
    assert list(got["engineered_col"]) == list(want["engineered_col"])
    np.testing.assert_allclose(got["engineered_mi"].to_numpy(), want["engineered_mi"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert (want["engineered_mi"] > 0).sum() >= 2
