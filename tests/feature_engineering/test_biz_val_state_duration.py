"""biz_value + unit tests for ``feature_engineering.state_duration.time_since_state_change``.

The win: on synthetic churn-style panel data where the target depends on RECENCY of cancellation (just
cancelled -> high reactivation propensity, cancelled long ago -> low), ``cancellation_duration`` carries
strong correlation with the target while a single raw lag-1 state flag cannot distinguish "cancelled 1
period ago" from "cancelled 20 periods ago" (both show False at lag-1) and so carries far less signal --
quantifying exactly why condensing many raw lag flags into a duration-since-transition feature is a genuine
information gain, not just a stylistic preference.

A second, independent gap: two entities can share the exact same ``cancellation_duration`` (e.g. both
"cancelled 4 periods ago") yet have wildly different churn HISTORIES -- one a first-time cancellation, the
other the entity's 6th acquire/cancel cycle. Pure duration-since-change cannot separate them; the opt-in
``activation_count`` output (number of prior acquire/reacquire events) can, and predicts churn-recidivism
driven outcomes (permanent, non-reactivating cancellation) that duration alone is blind to by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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


def test_time_since_state_change_default_omits_activation_count_bit_identical():
    """Opt-in param must be provably opt-in: default call output is unaffected by the new code path."""
    entity_ids, state = _make_state_panel(30, 20, seed=1)
    out_default = time_since_state_change(state, entity_ids)
    out_explicit_false = time_since_state_change(state, entity_ids, include_activation_count=False)
    assert set(out_default.keys()) == {"possession_duration", "cancellation_duration"}
    assert set(out_explicit_false.keys()) == {"possession_duration", "cancellation_duration"}
    for key in out_default:
        np.testing.assert_array_equal(out_default[key], out_explicit_false[key])


def test_time_since_state_change_activation_count_semantics_small_case():
    entity_ids = np.array([1, 1, 1, 1, 1, 1, 1])
    #                       T     T     F     F     T     T     F     -- wait, keep 7 values below
    state = np.array([True, True, False, False, True, True, False])
    out = time_since_state_change(state, entity_ids, include_activation_count=True)
    # first True run (rows 0-1) is activation #1; second True run (rows 4-5) is activation #2.
    assert list(out["activation_count"]) == [1, 1, 1, 1, 2, 2, 2]


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


def _make_churn_recidivism_panel(n_entities: int, seed: int, cancel_run_length: int = 4, acquire_run_length: int = 3):
    """Every entity's LAST row has an IDENTICAL ``cancellation_duration`` (= ``cancel_run_length``), so
    duration-since-change carries zero information about which entities differ -- only the number of prior
    acquire/cancel cycles (``activation_count``) varies, from 1 to 6 cycles per entity."""
    rng = np.random.default_rng(seed)
    n_cycles_per_entity = rng.integers(1, 7, size=n_entities)  # 1..6 inclusive
    entity_ids_list = []
    state_list = []
    last_row_idx = np.empty(n_entities, dtype=np.int64)
    pos = 0
    for e in range(n_entities):
        k = int(n_cycles_per_entity[e])
        seq = []
        for _cycle in range(k):
            seq += [True] * acquire_run_length + [False] * cancel_run_length
        entity_ids_list.append(np.full(len(seq), e))
        state_list.append(np.array(seq, dtype=bool))
        pos += len(seq)
        last_row_idx[e] = pos - 1
    entity_ids = np.concatenate(entity_ids_list)
    state = np.concatenate(state_list)
    return entity_ids, state, n_cycles_per_entity, last_row_idx


def test_biz_val_activation_count_beats_duration_alone_on_churn_recidivism_target():
    n_entities = 800
    entity_ids, state, n_cycles_per_entity, last_idx = _make_churn_recidivism_panel(n_entities=n_entities, seed=11)
    out = time_since_state_change(state, entity_ids, include_activation_count=True)

    cancellation_duration = out["cancellation_duration"][last_idx]
    activation_count = out["activation_count"][last_idx]

    # sanity: duration-since-change is constant across ALL entities by construction (proves it is
    # structurally blind to recidivism), while activation_count reproduces the true cycle count.
    assert np.all(cancellation_duration == cancellation_duration[0])
    np.testing.assert_array_equal(activation_count, n_cycles_per_entity)

    # target: permanent (non-reactivating) churn probability rises with prior cycle count -- a real
    # recidivism pattern (repeat cancel/reacquire customers are more likely to eventually leave for good).
    rng = np.random.default_rng(23)
    logit = 0.7 * (activation_count - 3.0)
    y_prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_entities) < y_prob).astype(np.float64)

    idx = rng.permutation(n_entities)
    split = int(0.6 * n_entities)
    train_idx, test_idx = idx[:split], idx[split:]

    X_duration_only = cancellation_duration.reshape(-1, 1)
    X_duration_plus_count = np.column_stack([cancellation_duration, activation_count])

    model_duration_only = LogisticRegression().fit(X_duration_only[train_idx], y[train_idx])
    # a constant single feature has zero variance -> predict_proba degenerates to the class prior; that IS
    # the point (duration alone cannot separate these entities at all).
    proba_duration_only = model_duration_only.predict_proba(X_duration_only[test_idx])[:, 1]
    auc_duration_only = roc_auc_score(y[test_idx], proba_duration_only)

    model_combined = LogisticRegression().fit(X_duration_plus_count[train_idx], y[train_idx])
    proba_combined = model_combined.predict_proba(X_duration_plus_count[test_idx])[:, 1]
    auc_combined = roc_auc_score(y[test_idx], proba_combined)

    assert auc_duration_only < 0.55, f"duration-only AUC should sit near chance (constant feature), got {auc_duration_only:.3f}"
    assert auc_combined > 0.75, f"adding activation_count should give strong separation, got AUC={auc_combined:.3f}"
    assert auc_combined > auc_duration_only + 0.20, (
        f"activation_count should add material predictive signal beyond duration alone: "
        f"duration_only_auc={auc_duration_only:.3f} combined_auc={auc_combined:.3f}"
    )
