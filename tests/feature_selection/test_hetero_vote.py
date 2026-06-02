"""Cross-model shadow-voting all-relevant selection (BorutaShap-B7): keeps signal, drops the model-specific
noise leak that a single-model shadow gate admits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _data(seed=0, n=1500, p_sig=4, p_noise=20):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_sig))
    logit = z @ np.array([1.5, -1.2, 1.0, 0.9])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {f"sig_{i}": z[:, i] for i in range(p_sig)}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), pd.Series(y), [f"sig_{i}" for i in range(p_sig)], [f"noise_{j}" for j in range(p_noise)]


def test_hetero_vote_keeps_signal_drops_noise():
    pytest.importorskip("sklearn")
    from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote

    X, y, signal, noise = _data(seed=0)
    accepted, info = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=4, vote_threshold=0.5, random_state=0)
    acc = set(accepted)
    # All strong (marginal) signals survive the majority vote.
    for s in signal:
        assert s in acc, f"signal {s} dropped (vote_fraction={info['vote_fraction'][s]})"
    # Cross-model voting admits very few noise columns (single-model gates leak more).
    n_noise = len(acc & set(noise))
    assert n_noise <= 1, f"cross-model vote admitted {n_noise} noise columns: {sorted(acc & set(noise))}"
    # Reported diagnostics are well-formed.
    assert info["n_models"] == 3
    assert set(info["vote_fraction"]) == set(X.columns)
    assert all(0.0 <= v <= 1.0 for v in info["vote_fraction"].values())
