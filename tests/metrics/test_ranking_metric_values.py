"""Value-level edge coverage for ``mlframe.metrics.ranking`` public API.

Complements the existing dispatch/bit-identity sensors (``test_ranking_public_batch_dispatch_cpx24``)
and the k<=0 guard tests with HAND-COMPUTED expected values for NDCG / MAP / MRR: exponential-gain
formula (distinct from sklearn's linear gain), k>n clamping, no-relevant -> NaN, group_ids=None,
non-contiguous / shuffled group ids, tied-score determinism, and the batched summary-kernel identity.
Expected numbers are computed by hand (and cross-checked against sklearn where the linear/exponential
gains coincide, i.e. binary relevance).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.metrics.ranking import (
    compute_ranking_summary,
    map_at_k,
    mrr,
    ndcg_at_k,
)


# ----------------------------------------------------------------------------
# NDCG value semantics
# ----------------------------------------------------------------------------


def test_ndcg_binary_known_value_matches_sklearn():
    # Single query, binary relevance. Predicted order = score-descending = [0,1,2,3].
    # DCG@4 = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG@4 = 1/log2(2) + 1/log2(3).
    y = np.array([1.0, 0.0, 1.0, 0.0])
    s = np.array([0.4, 0.3, 0.2, 0.1])
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    got = ndcg_at_k(y, s, group_ids=None, k=10)
    assert got == pytest.approx(expected, abs=1e-9)
    # Binary relevance: exponential (2^rel-1) and sklearn's linear gain coincide.
    from sklearn.metrics import ndcg_score

    assert got == pytest.approx(ndcg_score([y], [s]), abs=1e-9)


def test_ndcg_uses_exponential_gain_not_linear():
    # Graded relevance where exponential (2^rel-1) and linear (rel) gains DIVERGE.
    # y=[2,1], scores=[0.1,0.9] -> predicted order puts rel=1 first, rel=2 second.
    y = np.array([2.0, 1.0])
    s = np.array([0.1, 0.9])
    # Exponential gain: DCG = (2^1-1)/log2(2) + (2^2-1)/log2(3); IDCG = 3/log2(2) + 1/log2(3).
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    expected_exp = dcg / idcg
    got = ndcg_at_k(y, s, group_ids=None, k=10)
    assert got == pytest.approx(expected_exp, abs=1e-6)  # ~0.79671
    # sklearn uses LINEAR gain -> ~0.85972; ours must be measurably different.
    from sklearn.metrics import ndcg_score

    sklearn_linear = ndcg_score([y], [s])
    assert abs(got - sklearn_linear) > 0.05


def test_ndcg_k_greater_than_n_clamps_to_n():
    y = np.array([1.0, 0.0, 1.0])
    s = np.array([0.9, 0.5, 0.1])
    # k far beyond the 3-item group must clamp to n -> identical to k=n.
    assert ndcg_at_k(y, s, None, k=1000) == pytest.approx(ndcg_at_k(y, s, None, k=3), abs=1e-12)


def test_ndcg_tied_scores_deterministic_stable_order():
    # All scores tied: stable mergesort keeps original row order, so the rel=1 item (row 0)
    # ranks first -> perfect NDCG=1.0, reproducibly.
    y = np.array([1.0, 0.0])
    s = np.array([0.5, 0.5])
    first = ndcg_at_k(y, s, None, k=10)
    second = ndcg_at_k(y, s, None, k=10)
    assert first == 1.0
    assert first == second  # deterministic across repeated calls


# ----------------------------------------------------------------------------
# MAP / MRR value semantics
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,expected",
    [
        # y=[1,0,1,1], scores desc -> rels order [1,0,1,1]; n_relevant=3.
        # k=10: precisions at hits = 1/1, 2/3, 3/4 ; denom=min(10,3)=3.
        (10, (1.0 + 2.0 / 3.0 + 3.0 / 4.0) / 3.0),
        # k=2: only first two positions -> one hit prec 1/1 ; denom=min(2,3)=2.
        (2, 1.0 / 2.0),
    ],
)
def test_map_at_k_known_value(k, expected):
    y = np.array([1.0, 0.0, 1.0, 1.0])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert map_at_k(y, s, None, k=k) == pytest.approx(expected, abs=1e-12)


def test_mrr_reciprocal_of_first_relevant_rank():
    # First relevant item is at rank 3 (0-indexed position 2) -> RR = 1/3.
    y = np.array([0.0, 0.0, 1.0, 0.0])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert mrr(y, s, None) == pytest.approx(1.0 / 3.0, abs=1e-12)


# ----------------------------------------------------------------------------
# Degenerate / no-relevance
# ----------------------------------------------------------------------------


def test_all_metrics_nan_when_no_relevant_items():
    y = np.zeros(5)  # no positives anywhere -> every query degenerate
    s = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    assert np.isnan(ndcg_at_k(y, s, None, k=10))
    assert np.isnan(map_at_k(y, s, None, k=10))
    assert np.isnan(mrr(y, s, None))


# ----------------------------------------------------------------------------
# Grouping: None vs explicit, non-contiguous ids, order-invariance
# ----------------------------------------------------------------------------


def test_group_ids_none_equals_single_explicit_group():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    s = np.array([0.4, 0.3, 0.2, 0.1])
    assert ndcg_at_k(y, s, None, k=5) == pytest.approx(
        ndcg_at_k(y, s, np.zeros(4), k=5), abs=1e-12
    )


def test_noncontiguous_and_shuffled_group_ids_group_by_value():
    # Group A rows (id 50): y=[1,0], s=[0.9,0.1] -> NDCG 1.0.
    # Group B rows (id 20): y=[1,0,1], s=[0.1,0.9,0.5] -> NDCG ~0.693419.
    yt = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    ys = np.array([0.9, 0.1, 0.1, 0.9, 0.5])
    gids = np.array([50, 50, 20, 20, 20])
    ndcg_A = ndcg_at_k(np.array([1.0, 0.0]), np.array([0.9, 0.1]), None, k=10)
    ndcg_B = ndcg_at_k(np.array([1.0, 0.0, 1.0]), np.array([0.1, 0.9, 0.5]), None, k=10)
    expected = (ndcg_A + ndcg_B) / 2.0
    got = ndcg_at_k(yt, ys, gids, k=10)
    assert got == pytest.approx(expected, abs=1e-12)
    # Row order must not matter: a permutation of the (row, group) tuples yields the same mean.
    perm = np.array([4, 0, 2, 3, 1])
    got_shuffled = ndcg_at_k(yt[perm], ys[perm], gids[perm], k=10)
    assert got_shuffled == pytest.approx(got, abs=1e-12)


# ----------------------------------------------------------------------------
# Batched summary kernel: hand-computed values + cross-path identity
# ----------------------------------------------------------------------------


def test_compute_ranking_summary_values_and_cross_path_identity():
    # Two groups (A id0: y=[1,0] s=[0.9,0.1]; B id1: y=[1,0,1] s=[0.1,0.9,0.5]).
    yt = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    ys = np.array([0.9, 0.1, 0.1, 0.9, 0.5])
    gids = np.array([0, 0, 1, 1, 1])
    summary = compute_ranking_summary(yt, ys, gids, eval_at=(1, 3))

    ndcg_B3 = (1.0 / math.log2(3) + 1.0 / math.log2(4)) / (1.0 / math.log2(2) + 1.0 / math.log2(3))
    # ndcg@1: A=1.0 (top item relevant), B=0.0 (top item irrelevant) -> mean 0.5.
    assert summary["ndcg@1"] == pytest.approx(0.5, abs=1e-9)
    # ndcg@3: A=1.0, B=ndcg_B3 (~0.693419) -> mean.
    assert summary["ndcg@3"] == pytest.approx((1.0 + ndcg_B3) / 2.0, abs=1e-9)
    # map@3: A=1.0 (denom min(3,1)=1), B=(1/2+2/3)/2=0.583333 -> mean 0.791667.
    assert summary["map@3"] == pytest.approx((1.0 + (0.5 + 2.0 / 3.0) / 2.0) / 2.0, abs=1e-9)
    # mrr: A first-rel rank1=1.0, B first-rel rank2=0.5 -> mean 0.75.
    assert summary["mrr"] == pytest.approx(0.75, abs=1e-9)

    # The batched summary kernel must agree with the per-metric public functions (independent path).
    assert summary["ndcg@3"] == pytest.approx(ndcg_at_k(yt, ys, gids, k=3), abs=1e-12)
    assert summary["map@3"] == pytest.approx(map_at_k(yt, ys, gids, k=3), abs=1e-12)
    assert summary["mrr"] == pytest.approx(mrr(yt, ys, gids), abs=1e-12)
