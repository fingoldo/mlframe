"""biz_value + unit tests for ``feature_engineering.state_duration.time_since_state_change``.

The win: on synthetic churn-style panel data where the target depends on RECENCY of cancellation (just
cancelled -> high reactivation propensity, cancelled long ago -> low), ``cancellation_duration`` carries
strong correlation with the target while a single raw lag-1 state flag cannot distinguish "cancelled 1
period ago" from "cancelled 20 periods ago" (both show False at lag-1) and so carries far less signal —
quantifying exactly why condensing many raw lag flags into a duration-since-transition feature is a genuine
information gain, not just a stylistic preference.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from mlframe.feature_engineering.state_duration import time_since_state_change


def _make_state_panel(n_entities: int, n_periods: int, seed: int, transition_prob: float = 0.08):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    state = np.empty(entity_ids.shape[0], dtype=bool)
    pos = 0
    for _e in range(n_entities):
        cur = bool(rng.random() < 0.5)
        for _t in range(n_periods):
            if rng.random() < transition_prob:
                cur = not cur
            state[pos] = cur
            pos += 1
    return entity_ids, state


def test_time_since_state_change_basic_semantics_small_case():
    entity_ids = np.array([1, 1, 1, 1, 1])
    state = np.array([True, True, False, False, True])
    out = time_since_state_change(state, entity_ids)
    assert out["possession_duration"][0] == 1.0
    assert out["possession_duration"][1] == 2.0
    assert np.isnan(out["possession_duration"][2])
    assert np.isnan(out["cancellation_duration"][0])
    assert out["cancellation_duration"][2] == 1.0
    assert out["cancellation_duration"][3] == 2.0
    assert out["possession_duration"][4] == 1.0  # re-acquired, resets
    assert np.isnan(out["cancellation_duration"][4])


def test_time_since_state_change_never_true_stays_nan_for_both():
    entity_ids = np.array([1, 1, 1])
    state = np.array([False, False, False])
    out = time_since_state_change(state, entity_ids)
    assert np.all(np.isnan(out["possession_duration"]))
    assert np.all(np.isnan(out["cancellation_duration"]))  # never possessed -> "cancelled" is meaningless


def test_time_since_state_change_boundaries_never_bleed_across_entities():
    entity_ids = np.array([1, 1, 2, 2])
    state = np.array([True, True, False, False])
    out = time_since_state_change(state, entity_ids)
    # entity 2 has no prior True at all -- its own history, not entity 1's, determines its features.
    assert np.isnan(out["cancellation_duration"][2])
    assert np.isnan(out["cancellation_duration"][3])


def test_time_since_state_change_returns_expected_shapes():
    entity_ids, state = _make_state_panel(20, 15, seed=0)
    out = time_since_state_change(state, entity_ids)
    assert out["possession_duration"].shape == entity_ids.shape
    assert out["cancellation_duration"].shape == entity_ids.shape


def test_biz_val_cancellation_duration_beats_raw_lag1_flag_on_recency_dependent_target():
    entity_ids, state = _make_state_panel(n_entities=400, n_periods=25, seed=42)
    out = time_since_state_change(state, entity_ids)
    cancellation_duration = out["cancellation_duration"]

    # target: high reactivation propensity shortly after cancellation, decaying with recency; undefined
    # (never triggered) rows get a low base rate. Only rows with a defined cancellation_duration matter.
    rng = np.random.default_rng(7)
    n = state.shape[0]
    base_prob = 0.05
    recency_boost = np.where(np.isnan(cancellation_duration), 0.0, np.exp(-0.3 * np.nan_to_num(cancellation_duration, nan=999.0)))
    y_prob = np.clip(base_prob + 0.8 * recency_boost, 0.0, 1.0)
    y = (rng.random(n) < y_prob).astype(np.float64)

    # raw lag-1 state flag: True/False at t-1, computed the naive way (shift within entity, no library dep
    # needed for this comparison baseline -- pure numpy per-entity shift).
    lag1 = np.empty(n, dtype=np.float64)
    lag1[:] = np.nan
    for start in range(0, n, 25):
        seg = state[start : start + 25].astype(np.float64)
        lag1[start + 1 : start + 25] = seg[:-1]

    valid_mask = ~np.isnan(cancellation_duration)
    corr_duration, _ = spearmanr(cancellation_duration[valid_mask], y[valid_mask])
    corr_lag1, _ = spearmanr(lag1[valid_mask], y[valid_mask])

    # cancellation_duration should show strong recency-driven correlation; lag1 (constant False for every
    # "currently cancelled" row regardless of recency) should show far weaker correlation on the same rows.
    assert abs(corr_duration) > 0.15, f"cancellation_duration should predict the recency-dependent target, got corr={corr_duration:.3f}"
    assert abs(corr_duration) > abs(corr_lag1) + 0.10, (
        f"cancellation_duration should carry materially more signal than a raw lag-1 flag: "
        f"duration_corr={corr_duration:.3f} lag1_corr={corr_lag1:.3f}"
    )
