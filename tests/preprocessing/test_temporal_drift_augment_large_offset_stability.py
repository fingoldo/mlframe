"""Regression test: augment_temporal_drift's expanding variance must stay numerically stable on
large-offset/small-spread columns (e.g. a revenue-scale feature), not collapse to a fake zero.

Pre-fix, the running sum(x^2) - (sum(x))^2/n formula suffered catastrophic cancellation on such
data, sometimes going NEGATIVE; clipped to 0, the "safe_std" guard then silently treated a
genuinely-varying column as zero-variance and z-scored every synthetic row to a flat 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift


def _make_large_offset_panel(seed: int) -> pd.DataFrame:
    """Build a per-entity panel with a large-offset/small-spread numeric column."""
    rng = np.random.default_rng(seed)
    rows = []
    for entity_id in range(30):
        n_periods = rng.integers(4, 8)
        offset = 1e9 * rng.uniform(0.5, 2.0)
        series = offset + rng.standard_normal(n_periods) * 0.5
        for t, val in enumerate(series):
            rows.append({"entity_id": entity_id, "t": t, "x": val})
    return pd.DataFrame(rows)


def test_augmented_rows_are_not_flattened_to_zero_on_large_offset_data():
    """Synthetic rows on large-offset data must not collapse to a flat 0.0 z-score."""
    df = _make_large_offset_panel(seed=0)
    out = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,), min_history=2)
    synth = out.loc[out["_temporal_drift_augmented"]]
    assert len(synth) > 0
    # pre-fix: every synthetic row's "x" was silently forced to 0.0 (fake zero-variance guard).
    assert not np.allclose(synth["x"].to_numpy(), 0.0), "synthetic rows collapsed to a flat 0.0 z-score"
    assert synth["x"].std() > 0.1


def test_expanding_variance_matches_stable_reference_on_large_offset_data():
    """Augmented z-scores must match an independent stable expanding-variance reference."""
    # Directly cross-check the augmented z-scores against an independent, brute-force stable
    # per-entity expanding-variance reference (two-pass, no cancellation risk).
    df = _make_large_offset_panel(seed=1)
    out = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,), min_history=2)
    synth = out.loc[out["_temporal_drift_augmented"]].sort_values(["entity_id", "t"]).reset_index(drop=True)

    for _, row in synth.iterrows():
        entity_id, drop_last_t = row["entity_id"], row["t"]
        history = df.loc[(df["entity_id"] == entity_id) & (df["t"] <= drop_last_t), "x"].to_numpy()
        ref_mean = history.mean()
        ref_std = history.std(ddof=1)
        ref_z = (history[-1] - ref_mean) / ref_std
        assert abs(row["x"] - ref_z) < 1e-5, f"entity {entity_id}: got {row['x']}, expected {ref_z}"
