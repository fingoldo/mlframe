"""biz_value test for ``feature_engineering.multi_window_aggregate``.

The win: a genuine behavioral shift concentrated in the RECENT window (e.g. a worsening trend) is diluted
away by a single all-history aggregate but clearly visible in a short-lookback window -- so a classifier
using the short-window aggregate as a feature should beat one using only the all-history aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.multi_window_aggregate import multi_window_aggregate


def _make_data(seed: int):
    """Helper: Make data."""
    rng = np.random.default_rng(seed)
    n_entities = 400
    # a long history (~4 years) so a small recent-90-day shift is heavily diluted in the all-history mean,
    # but still clearly visible in a 90-day window -- the realistic scenario the source technique targets.
    cutoff = 1500.0
    rows = []
    labels = {}
    for entity_id in range(n_entities):
        worsening = rng.random() < 0.5
        labels[entity_id] = int(worsening)
        n_events = rng.integers(80, 120)
        times = np.sort(rng.uniform(0, cutoff - 1, n_events))
        if worsening:
            # a modest elevated level ONLY in the last 90 days; earlier history is identical to the stable class.
            values = np.where(times > cutoff - 90, rng.normal(7, 1.5, n_events), rng.normal(5, 1.5, n_events))
        else:
            values = rng.normal(5, 1.5, n_events)
        for t, v in zip(times, values):
            rows.append({"entity": entity_id, "t": float(t), "amount": float(v)})

    history_df = pd.DataFrame(rows)
    query_df = pd.DataFrame({"entity": list(labels.keys()), "as_of": cutoff})
    y = np.array([labels[e] for e in query_df["entity"]])
    return history_df, query_df, y


def test_biz_val_multi_window_aggregate_recent_window_beats_all_history_aggregate():
    """Biz val multi window aggregate recent window beats all history aggregate."""
    history_df, query_df, y = _make_data(seed=0)

    result = multi_window_aggregate(
        history_df,
        entity_col="entity",
        time_col="t",
        as_of=query_df,
        agg_funcs={"amount": ["sum", "count", "mean"]},
        lookback_horizons=[90, 10_000],
    )

    recent_mean = result[["amount_mean_last_90"]].fillna(0.0)
    all_history_mean = result[["amount_mean_last_10000"]].fillna(0.0)

    auc_recent = cross_val_score(LogisticRegression(), recent_mean, y, cv=5, scoring="roc_auc").mean()
    auc_all_history = cross_val_score(LogisticRegression(), all_history_mean, y, cv=5, scoring="roc_auc").mean()

    assert auc_recent > auc_all_history + 0.2, (
        f"the recent-window aggregate should detect the concentrated shift far better than the all-history aggregate: "
        f"recent={auc_recent:.4f} all_history={auc_all_history:.4f}"
    )
    assert auc_recent > 0.9


def test_multi_window_aggregate_matches_manual_windowed_sum():
    """Multi window aggregate matches manual windowed sum."""
    history_df = pd.DataFrame({"entity": ["a"] * 5, "t": [1, 5, 10, 15, 20], "amount": [10.0, 20.0, 30.0, 40.0, 50.0]})
    query_df = pd.DataFrame({"entity": ["a"], "as_of": [21]})

    result = multi_window_aggregate(
        history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["sum", "count", "mean"]}, lookback_horizons=[10, 100]
    )
    row = result.iloc[0]
    assert row["amount_sum_last_10"] == 90.0  # rows at t=15,20 (window [11,21))
    assert row["amount_count_last_10"] == 2.0
    assert row["amount_mean_last_10"] == 45.0
    assert row["amount_sum_last_100"] == 150.0  # all 5 rows


def test_multi_window_aggregate_guards_against_catastrophic_cancellation():
    """FE_ROOT_B-6 (2026-08-05 audit): windowed sum/mean is derived as the difference of two independently-
    computed CUMULATIVE aggregates (upper - lower); when an entity's history span greatly exceeds the
    horizon, both cumulative snapshots can be large and nearly equal, so plain float64 subtraction can lose
    most (or all) of the result's significant digits.

    Concretely: one huge early-history row (1e18) dominates BOTH the upper (cumulative-to-cutoff) and lower
    (cumulative-to-window-start) snapshots, so a naive ``upper - lower`` silently returns 0.0 instead of the
    true windowed sum 8.3 (0.9 + 4.1 + 3.3, the three rows actually inside the 10-unit window) -- verified
    directly below via plain float64 arithmetic on the exact same numbers. The fix must recover the exact
    windowed sum regardless of the dominating early-history magnitude.
    """
    # Sanity: confirm the naive difference really does lose the signal at this magnitude (not testing our
    # own fix's math twice -- this is the ground-truth failure mode being guarded against).
    upper_naive = 1e18 + 1.3 + 2.7 + 0.9 + 4.1 + 3.3
    lower_naive = 1e18 + 1.3 + 2.7
    assert upper_naive - lower_naive == 0.0, "the naive-subtraction repro no longer demonstrates cancellation at this magnitude"

    history_df = pd.DataFrame(
        {
            "entity": ["a"] * 6,
            "t": [0.0, 850.0, 900.0, 993.0, 996.0, 999.0],
            "amount": [1e18, 1.3, 2.7, 0.9, 4.1, 3.3],
        }
    )
    query_df = pd.DataFrame({"entity": ["a"], "as_of": [1000.0]})

    result = multi_window_aggregate(
        history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["sum", "count", "mean"]}, lookback_horizons=[10]
    )
    row = result.iloc[0]

    assert row["amount_sum_last_10"] == pytest.approx(8.3, abs=1e-9), f"expected the exact windowed sum 8.3, got {row['amount_sum_last_10']}"
    assert row["amount_count_last_10"] == 3.0
    assert row["amount_mean_last_10"] == pytest.approx(8.3 / 3.0, abs=1e-9)


def test_multi_window_aggregate_empty_horizons_raises():
    """Multi window aggregate empty horizons raises."""
    import pytest

    history_df = pd.DataFrame({"entity": ["a"], "t": [1.0], "amount": [1.0]})
    query_df = pd.DataFrame({"entity": ["a"], "as_of": [2.0]})
    with pytest.raises(ValueError):
        multi_window_aggregate(history_df, "entity", "t", query_df, {"amount": ["sum"]}, lookback_horizons=[])


def test_multi_window_aggregate_auto_select_default_off_is_bit_identical():
    """auto_select is opt-in: omitting the new params must reproduce the exact pre-extension output."""
    history_df, query_df, _ = _make_data(seed=1)
    kwargs = dict(
        history_df=history_df,
        entity_col="entity",
        time_col="t",
        as_of=query_df,
        agg_funcs={"amount": ["sum", "count", "mean"]},
        lookback_horizons=[90, 365, 10_000],
    )
    baseline = multi_window_aggregate(**kwargs)
    with_defaults = multi_window_aggregate(**kwargs, auto_select=False)
    pd.testing.assert_frame_equal(baseline, with_defaults)


def test_biz_val_multi_window_aggregate_auto_select_keeps_useful_drops_redundant():
    """auto_select mode should keep the horizons that carry real incremental signal and drop the rest.

    Dataset: the label-driving shift is concentrated in the last 90 days (see ``_make_data``). A 90-day
    horizon is genuinely predictive; a 10_000-day (all-history) horizon is diluted to ~no signal, and a
    5-day horizon is a near-strict subset of the 90-day window so it carries no *incremental* signal once
    the 90-day horizon is already in the feature set (greedy forward-selection, so 90 must be evaluated
    before its subset horizon 5 -- confirmed stable across 5 seeds). This proves the selection correctly
    separates "useful" from "redundant/noise" horizons using CV lift, rather than the caller having to
    hand-pick which of a candidate grid of lookback windows to keep.
    """
    history_df, query_df, y = _make_data(seed=0)
    true_useful = {90}
    candidate_horizons = [90, 5, 10_000]

    selected_out, info = multi_window_aggregate(
        history_df,
        entity_col="entity",
        time_col="t",
        as_of=query_df,
        agg_funcs={"amount": ["sum", "count", "mean"]},
        lookback_horizons=candidate_horizons,
        auto_select=True,
        target=y,
        cv=5,
        scoring="roc_auc",
        min_lift=0.02,
        estimator=LogisticRegression(max_iter=1000, C=0.1),
        return_selection_info=True,
    )

    kept = set(info["kept_horizons"])
    precision = len(kept & true_useful) / len(kept) if kept else 0.0
    recall = len(kept & true_useful) / len(true_useful)

    assert precision == 1.0, f"kept horizons should contain no redundant/noise horizon: kept={kept}"
    assert recall == 1.0, f"kept horizons should contain every genuinely useful horizon: kept={kept}"
    assert any(c.endswith("_last_90") for c in selected_out.columns)
    assert not any(c.endswith("_last_10000") for c in selected_out.columns)
    assert not any(c.endswith("_last_5") for c in selected_out.columns)
