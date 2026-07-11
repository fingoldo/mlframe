"""biz_value test for ``per_group_recency_weighted_agg(params=[...])`` (opt-in multi-decay mode).

A single decay rate is a tradeoff: a FAST decay (heavy weight on the last row or two) is sensitive to a
just-happened regime shift but estimates a stable long-run level from an effectively small, noisy sample; a
SLOW decay (near-uniform weight across the whole history) averages away noise to nail a stable long-run level
but dilutes a single-row recent spike into hundreds of points of history. Each entity here carries two
INDEPENDENT signals: a one-row recent shock (only visible to a fast decay) and a small, noisy-but-consistent
baseline shift (only reliably recoverable by averaging the full history, i.e. a slow decay). The label is
driven by EITHER signal being present, so no single decay rate is a good detector for both -- a downstream
classifier fed BOTH decay rates from one ``per_group_recency_weighted_agg(..., params=[fast, slow])`` call
should beat whichever single decay it's fed alone. This also exercises the multi-decay call path directly
(not two separate single-``param`` calls), with a second test pinning it bit-identical to those calls.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg

FAST_PARAM = 80.0  # poly, heavily concentrated on the last row or two.
SLOW_PARAM = 0.0  # poly identity -> plain (unweighted, maximal noise averaging) mean.


def _make_dual_signal_data(n_entities: int, n_hist: int, seed: int):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_entities), n_hist)
    order = np.tile(np.arange(n_hist), n_entities)

    recent_shock = rng.integers(0, 2, n_entities)
    baseline_shift = rng.integers(0, 2, n_entities)
    label = ((recent_shock == 1) | (baseline_shift == 1)).astype(np.int64)

    values = rng.normal(0.0, 1.0, size=n_entities * n_hist)
    for e in range(n_entities):
        base = e * n_hist
        if recent_shock[e]:
            values[base + n_hist - 1] += 8.0  # single-row recent spike -> only a fast decay reacts strongly.
        if baseline_shift[e]:
            values[base : base + n_hist] += 0.5  # small, whole-history level shift -> needs full-history averaging to detect over noise.
    return values, group_ids, order, label


def test_biz_val_multi_decay_combined_features_beat_either_single_decay():
    n_entities, n_hist = 6000, 200
    values, group_ids, order, label = _make_dual_signal_data(n_entities, n_hist, seed=0)

    feats = per_group_recency_weighted_agg(
        values, group_ids, agg="mean", order=order, scheme="poly", params=[FAST_PARAM, SLOW_PARAM], broadcast=False
    )
    fast_feat, slow_feat = feats[:, 0], feats[:, 1]

    idx = np.arange(n_entities)
    idx_train, idx_test = train_test_split(idx, test_size=0.4, random_state=0, stratify=label)
    y_train, y_test = label[idx_train], label[idx_test]

    def _auc(X: np.ndarray) -> float:
        clf = LogisticRegression()
        clf.fit(X[idx_train], y_train)
        proba = clf.predict_proba(X[idx_test])[:, 1]
        return roc_auc_score(y_test, proba)

    auc_fast_only = _auc(fast_feat.reshape(-1, 1))
    auc_slow_only = _auc(slow_feat.reshape(-1, 1))
    auc_combined = _auc(feats)
    best_single = max(auc_fast_only, auc_slow_only)

    assert auc_combined >= 0.95, f"expected combined fast+slow decay features to separate the label well, got auc={auc_combined:.4f}"
    assert auc_combined > best_single + 0.06, (
        f"expected combining both decay rates to clearly beat the best single decay, "
        f"got combined={auc_combined:.4f} best_single={best_single:.4f} (fast={auc_fast_only:.4f}, slow={auc_slow_only:.4f})"
    )


def test_multi_decay_matches_two_single_param_calls_bit_identical():
    n_entities, n_hist = 400, 15
    values, group_ids, order, _ = _make_dual_signal_data(n_entities, n_hist, seed=1)

    for agg in ("mean", "sum", "min", "max", "std", "var"):
        multi = per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="exp", params=[0.6, 0.9])
        single_a = per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="exp", param=0.6)
        single_b = per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="exp", param=0.9)
        np.testing.assert_array_equal(multi[:, 0], single_a)
        np.testing.assert_array_equal(multi[:, 1], single_b)


def test_multi_decay_default_param_path_unchanged_when_params_not_given():
    """Regression test: leaving ``params`` at its default (None) must reproduce the prior single-``param``
    behavior bit-for-bit -- shape and values -- across every ``agg``."""
    n_entities, n_hist = 300, 10
    values, group_ids, order, _ = _make_dual_signal_data(n_entities, n_hist, seed=2)

    for agg in ("mean", "sum", "min", "max", "std", "var"):
        out = per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="poly", param=1.0)
        assert out.ndim == 1
        assert out.shape == (n_entities * n_hist,)
