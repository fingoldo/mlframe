"""Regression test for the MDL-binning pairwise combo-count vectorization.

``compute_mdl_binning_pairwise_features._process`` counts, per query row, how many train rows
share the same (feature-0, feature-1) bin combo. The OLD code built a ``collections.Counter`` over
the train combo codes and looked each query row up with a per-row Python ``counts.get(int(c), 0)``
list comprehension (plus ``len(set(query_combo))`` for the unique-combo feature). The optimization
replaces both with ``np.unique(return_counts=True)`` + ``np.searchsorted``.

Counts are integers, so the vectorised result is BIT-IDENTICAL to the dict path. This test pins:

1. The vectorised lookup equals the ``Counter.get`` reference for random combo arrays, including
   query combos absent from train (count 0) and the unique-combo scalar.
2. The full feature function produces finite, correctly-shaped, deterministic output.

bench: src/mlframe/feature_engineering/_benchmarks/bench_mdl_binning_combo_count_vectorized.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest


def _old_combo(train_combo: np.ndarray, query_combo: np.ndarray):
    """Helper: Old combo."""
    combo_counts = Counter(train_combo)
    out = np.array([combo_counts.get(int(c), 0) for c in query_combo], dtype=np.float32)
    return out, float(len(set(query_combo)))


def _new_combo(train_combo: np.ndarray, query_combo: np.ndarray):
    """Helper: New combo."""
    uniq_combo, uniq_counts = np.unique(train_combo, return_counts=True)
    pos = np.searchsorted(uniq_combo, query_combo)
    pos_clipped = np.clip(pos, 0, uniq_combo.shape[0] - 1)
    matched = uniq_combo[pos_clipped] == query_combo
    out = np.where(matched, uniq_counts[pos_clipped], 0).astype(np.float32)
    return out, float(np.unique(query_combo).shape[0])


def test_vectorized_combo_count_bit_identical_to_counter():
    """Vectorized combo count bit identical to counter."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        n_train = int(rng.integers(100, 3000))
        n_query = int(rng.integers(100, 3000))
        hi = int(rng.integers(2, 12))
        # Intentionally let some query combos be absent from train (count 0).
        train_combo = rng.integers(0, hi, n_train) * 100 + rng.integers(0, hi, n_train)
        query_combo = rng.integers(0, hi + 2, n_query) * 100 + rng.integers(0, hi + 2, n_query)
        out_old, uc_old = _old_combo(train_combo, query_combo)
        out_new, uc_new = _new_combo(train_combo, query_combo)
        assert np.array_equal(out_old, out_new)
        assert uc_old == uc_new


def test_combo_code_no_collision_when_bin1_reaches_100():
    """FE_TRANSFORMER_B-4: the pairwise combo encoding (bin0 * base + bin1) must not collide distinct
    (bin0, bin1) pairs once bin1 can reach 100 -- a hardcoded base=100 makes (bin0=0, bin1=100) and
    (bin0=1, bin1=0) both encode to 100."""
    # A hardcoded base=100 collides these two genuinely distinct pairs.
    combo_a = 0 * 100 + 100  # (bin0=0, bin1=100)
    combo_b = 1 * 100 + 0  # (bin0=1, bin1=0)
    assert combo_a == combo_b, "sanity: this is the exact collision the hardcoded base=100 produces"

    # The fix's dynamic base (len(edges_for_feature_1) + 1, always > any real bin1 index) must not.
    n_edges_feat1 = 150  # enough that bin1 indices can legitimately reach up to 150
    combo_base = n_edges_feat1 + 1
    combo_a_fixed = 0 * combo_base + 100
    combo_b_fixed = 1 * combo_base + 0
    assert combo_a_fixed != combo_b_fixed


def test_combo_count_matches_ground_truth_when_bin1_edges_exceed_100():
    """FE_TRANSFORMER_B-4 end-to-end: with feature-1's edge count long enough that real bin1 indices
    exceed 100, the fixed dynamic-base combo encoding must match a ground-truth Python Counter over
    (bin0, bin1) pairs, while the old hardcoded base=100 encoding must NOT (proving the fixture
    actually exercises the collision)."""
    rng = np.random.default_rng(3)
    n_train, n_query = 400, 100
    # bin0 in a small range; bin1 spans [0, 150) so e.g. (bin0=0, bin1=105) and (bin0=1, bin1=5) --
    # genuinely distinct pairs -- both encode to 105 under the old hardcoded base=100.
    train_bin0 = rng.integers(0, 3, n_train)
    train_bin1 = rng.integers(0, 150, n_train)
    query_bin0 = rng.integers(0, 3, n_query)
    query_bin1 = rng.integers(0, 150, n_query)

    n_edges_feat1 = 150  # len(edges) == 150 -> combo_base == 151 under the fix

    # Ground truth: count (bin0, bin1) pairs directly, no combo-code collision possible.
    from collections import Counter

    truth_counts = Counter(zip(train_bin0.tolist(), train_bin1.tolist()))
    ground_truth = np.array([truth_counts.get((int(b0), int(b1)), 0) for b0, b1 in zip(query_bin0, query_bin1)], dtype=np.float32)

    # Reproduce the module's own fixed combo encoding directly (mirrors the code under test).
    combo_base = n_edges_feat1 + 1
    train_combo = train_bin0 * combo_base + train_bin1
    query_combo = query_bin0 * combo_base + query_bin1
    uniq_combo, uniq_counts = np.unique(train_combo, return_counts=True)
    pos = np.searchsorted(uniq_combo, query_combo)
    pos_clipped = np.clip(pos, 0, uniq_combo.shape[0] - 1)
    matched = uniq_combo[pos_clipped] == query_combo
    fixed_result = np.where(matched, uniq_counts[pos_clipped], 0).astype(np.float32)

    np.testing.assert_array_equal(fixed_result, ground_truth)

    # And confirm the OLD hardcoded base=100 would have gotten this wrong on the same data.
    old_train_combo = train_bin0 * 100 + train_bin1
    old_query_combo = query_bin0 * 100 + query_bin1
    old_uniq_combo, old_uniq_counts = np.unique(old_train_combo, return_counts=True)
    old_pos = np.clip(np.searchsorted(old_uniq_combo, old_query_combo), 0, old_uniq_combo.shape[0] - 1)
    old_matched = old_uniq_combo[old_pos] == old_query_combo
    old_result = np.where(old_matched, old_uniq_counts[old_pos], 0).astype(np.float32)
    assert not np.array_equal(old_result, ground_truth), "the old base=100 encoding must diverge from ground truth on this straddling data (else the fixture doesn't exercise the bug)"


def test_combo_base_is_derived_from_the_edge_count_at_the_production_site(monkeypatch):
    """FE_TRANSFORMER_B-4 asserted through `compute_mdl_binning_pairwise_features`, not a local copy.

    Every other test in this file compares local re-implementations: `_old_combo` vs `_new_combo`, and an
    inline duplicate of the module's combo encoding vs a Counter. None reads `combo_base` from the module,
    so reverting `mdl_binning_pairwise.py` to a hardcoded `combo_base = 100` leaves all of them green while
    production silently sums the counts of colliding (bin0, bin1) pairs.

    The collision cannot currently be reached through the real binner: `_mdl_bin_edges` performs a SINGLE
    top-level Fayyad-Irani split (its own docstring says `_split` "is only ever invoked once"), so it
    returns at most one edge whatever `max_bins` says, bin1 indices stay in [0, 1], and a base of 100 is
    accidentally safe. That makes the fix a guard against a future or caller-supplied binner rather than a
    live bug -- and it is exactly why the guard needs a test that does not depend on the current binner's
    conservatism. The edge list is stubbed here to the >100 edges the fix is written for, and the assertion
    runs against production's own combo output.
    """
    pytest.importorskip("sklearn")

    from collections import Counter

    import mlframe.feature_engineering.transformer.mdl_binning_pairwise as mdl

    rng = np.random.default_rng(11)
    n_train, n_query, d = 3000, 400, 3
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] + 0.1 * rng.standard_normal(n_train)).astype(np.float32)
    X_query = rng.standard_normal((n_query, d)).astype(np.float32)

    # 150 edges on every feature: bin1 indices then span [0, 150], straddling the old hardcoded base.
    fine_edges = np.quantile(X_train[:, 1], np.linspace(0.005, 0.995, 150)).tolist()
    monkeypatch.setattr(mdl, "_mdl_bin_edges", lambda x, y_class, n_classes, max_bins=8, min_size=20: list(fine_edges))

    df = mdl.compute_mdl_binning_pairwise_features(X_train, y_train, X_query, None, seed=0, task="regression", standardize=False)
    combo_col = [c for c in df.columns if "combo_count" in c]
    assert combo_col, f"no combo-count column in {list(df.columns)}"
    emitted = df[combo_col[0]].to_numpy()

    # Ground truth over (bin0, bin1) PAIRS -- no combo code, so no encoding collision is possible.
    train_bins = np.column_stack([np.digitize(X_train[:, i], fine_edges) for i in (0, 1)])
    query_bins = np.column_stack([np.digitize(X_query[:, i], fine_edges) for i in (0, 1)])
    truth = Counter(map(tuple, train_bins.tolist()))
    ground_truth = np.array([truth.get((int(b0), int(b1)), 0) for b0, b1 in query_bins], dtype=np.float32)

    np.testing.assert_array_equal(emitted, ground_truth)

    # ...and a hardcoded base of 100 must genuinely disagree on this data, or the fixture does not straddle
    # it and this test would pass against the reverted production code too.
    old_train = train_bins[:, 0] * 100 + train_bins[:, 1]
    old_query = query_bins[:, 0] * 100 + query_bins[:, 1]
    old_uniq, old_counts = np.unique(old_train, return_counts=True)
    old_pos = np.clip(np.searchsorted(old_uniq, old_query), 0, old_uniq.shape[0] - 1)
    old_result = np.where(old_uniq[old_pos] == old_query, old_counts[old_pos], 0).astype(np.float32)
    assert not np.array_equal(old_result, ground_truth), "base=100 agrees with ground truth here; the fixture does not exercise the collision"


def test_the_mdl_binner_emits_a_single_split_today(monkeypatch):
    """Records why the collision above has to be reached with a stubbed edge list.

    `_mdl_bin_edges` is a single top-level split, so `max_bins`/`max_bins_per_feat` cannot raise the edge
    count past one and bin indices stay in [0, 1]. If that ever becomes recursive, the hardcoded-base
    collision becomes reachable for real and this test should start failing -- which is the signal to
    exercise the case above through the genuine binner rather than a stub.
    """
    from mlframe.feature_engineering.transformer.mdl_binning_pairwise import _mdl_bin_edges

    rng = np.random.default_rng(5)
    n = 20_000
    x = rng.standard_normal(n)
    y = 3.0 * x + 0.01 * rng.standard_normal(n)
    qs = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
    y_class = np.digitize(y, qs).astype(np.int32)

    edges = _mdl_bin_edges(x, y_class, 5, max_bins=160)
    assert len(edges) <= 1, (
        f"_mdl_bin_edges now returns {len(edges)} edges on a strongly monotone target; it has become "
        "recursive, so the combo-base collision is reachable through the real binner and the test above "
        "should drive it without stubbing"
    )


def test_full_feature_function_runs_and_deterministic():
    """Full feature function runs and deterministic."""
    pytest.importorskip("sklearn")
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer.mdl_binning_pairwise import (
        compute_mdl_binning_pairwise_features,
    )

    rng = np.random.default_rng(7)
    n, d = 600, 6
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)).astype(np.float32)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)

    df1 = compute_mdl_binning_pairwise_features(X, y, None, splitter, seed=0, task="regression")
    df2 = compute_mdl_binning_pairwise_features(X, y, None, splitter, seed=0, task="regression")
    assert df1.shape == (n, 5)
    arr = df1.to_numpy()
    assert np.isfinite(arr).all()
    assert np.allclose(arr, df2.to_numpy())
