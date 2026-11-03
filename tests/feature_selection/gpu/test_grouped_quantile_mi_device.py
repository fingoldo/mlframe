"""Grouped-quantile candidate MI on the device equals the host binner's (same partitions), with low-cardinality integer columns kept on the host."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._grouped_quantile_fe import generate_grouped_quantile_features, score_grouped_quantile_by_mi_uplift


def test_device_scores_equal_host(monkeypatch):
    """Percentile-rank, IQR and p90-p10 broadcasts of a grouped frame plus a few-valued integer column: same scores table up to round-off."""
    rng = np.random.default_rng(4)
    n = 40000
    g = rng.integers(0, 6, n)
    a = rng.standard_normal(n) + g
    X = pd.DataFrame({"g": g, "a": a, "b": rng.exponential(2, n), "c": rng.standard_normal(n)})
    y = ((a - g) > 0.3).astype(int) ^ (rng.random(n) < 0.1)
    enc, rec = generate_grouped_quantile_features(X, ["g"], ["a", "b", "c"])
    enc["few"] = rng.integers(0, 4, n).astype(float)
    src = {nm: rec[nm]["num_col"] for nm in rec}
    src["few"] = "a"
    monkeypatch.setenv("MLFRAME_GROUPED_QUANTILE_MI_DEVICE", "0")
    want = score_grouped_quantile_by_mi_uplift(X, enc, y, eng_to_source=src)
    monkeypatch.setenv("MLFRAME_GROUPED_QUANTILE_MI_DEVICE", "1")
    got = score_grouped_quantile_by_mi_uplift(X, enc, y, eng_to_source=src)
    assert list(got["engineered_col"]) == list(want["engineered_col"])
    np.testing.assert_allclose(got["mi"].to_numpy(), want["mi"].to_numpy(), rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(got.drop(columns=["mi", "uplift"]), want.drop(columns=["mi", "uplift"]))
