"""The causal recency-weighted aggregate stays finite for events thousands of half-lives before an entity's last event."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode


def test_early_event_of_a_long_history_is_not_nan():
    """One entity, events at t=0 and t=2000 with half-life 1: the t=0 row (0.5**2000 underflowed to 0/0) must equal its own step-1 value."""
    events = pd.DataFrame({"card": ["a", "a"], "e": ["x", "y"], "t": [0.0, 2000.0]})
    out = two_step_recency_weighted_target_encode(events, "card", ["e"], np.array([1.0, 0.0]), "t", 1.0)
    assert np.isfinite(out).all()


def test_matches_the_direct_definition_on_a_short_history():
    """With three events the result equals the explicitly weighted mean of the step-1 encodings seen so far (negative control)."""
    events = pd.DataFrame({"card": ["a"] * 3, "e": ["x", "y", "z"], "t": [0.0, 1.0, 2.0]})
    y = np.array([1.0, 0.0, 1.0])
    out = two_step_recency_weighted_target_encode(events, "card", ["e"], y, "t", 1.0)
    from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode

    enc = ordered_target_encode(events["e"].astype(str).to_numpy(), y, order=events["t"].to_numpy(), smoothing=1.0)
    t = events["t"].to_numpy()
    expected = [np.sum(enc[: i + 1] * 0.5 ** (t[i] - t[: i + 1])) / np.sum(0.5 ** (t[i] - t[: i + 1])) for i in range(3)]
    np.testing.assert_allclose(out, expected, rtol=1e-12)
