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


def test_synthetic_row_keeps_its_truncated_vintage_timestamp():
    """A synthetic row's `time_col` must stay at the period it was truncated to, not the entity's true last.

    Regression: carrying non-feature columns back from the entity's TRUE last period (so a per-period label is
    not taken from the truncated vintage) initially excluded only `entity_col`, which swept up `time_col` too.
    Every synthetic row then claimed the entity's final timestamp while carrying deliberately truncated
    features, so a caller slicing history by time reconstructed the untruncated window and the augmentation
    silently did nothing. Nothing raised; the rows simply described a period they did not come from.
    """
    df = _make_large_offset_panel(seed=3)
    out = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,), min_history=2)
    synth = out.loc[out["_temporal_drift_augmented"]]
    assert not synth.empty, "the fixture produced no synthetic rows, so this test would pass vacuously"

    true_last_t = df.groupby("entity_id")["t"].max()
    for entity_id, group in synth.groupby("entity_id"):
        # n_drop_options=(1,), so each synthetic row sits exactly one period before that entity's true last.
        expected = sorted(df.loc[df["entity_id"] == entity_id, "t"].unique())[-2]
        assert set(group["t"]) == {expected}, f"entity {entity_id}: synthetic t is {sorted(set(group['t']))}, expected {expected}"
        assert expected != true_last_t[entity_id], "fixture too short to distinguish the truncated vintage from the true last period"
