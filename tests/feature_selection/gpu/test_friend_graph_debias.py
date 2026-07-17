"""Regression tests for the friend-graph F3 (bias-mismatched garbage threshold) and F4 (CMI clamp) fixes.

F3: ``build_friend_graph`` previously compared a multi-term, 2-variable-inflated ``total_unique`` sum against a single-term, 1-variable ``rel[i]`` -- a finite-sample bias
mismatch that, on small n with a high-cardinality node, could flip a node's red flag (and hence which features ``prune_by_friend_graph`` removes). The fix Miller-Madow
debiases both sides before the compare.

F4: ``neighbor_unique_target`` previously clamped the per-neighbor CMI to >=0, silently zeroing noisy-but-real (negative-fluctuating) chain-rule estimates; the detail now
carries the RAW CMI so the negative finite-sample noise is visible and a weakly-positive justifier is not discarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.friend_graph import (
    build_friend_graph,
    neighbor_unique_target,
)
from mlframe.feature_selection.filters.info_theory import mi


def _small_n_highcard_dataset(n=300, seed=1):
    """Small n with high-cardinality nodes that share entropy (strong feature-feature edges) and a weakly-relevant target.

    High cardinality + small n maximises the plug-in MI inflation, which is exactly where the old bias-mismatched threshold (inflated multi-term ``total_unique`` sum vs
    less-inflated single-term ``rel[i]``) flips a red flag. Returns (data, nbins, target_indices, names, selected_vars). seed=1/n=300 is chosen so node 0 is red under the
    raw-bias rule but green under the debiased rule for garbage_unique_ratio in (0, 0.31).
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 7, n).astype(np.int32)
    y = ((x % 2) ^ (rng.random(n) < 0.2)).astype(np.int32)
    nbs = [((x + rng.integers(0, 2, n)) % 7).astype(np.int32) for _ in range(5)]
    data = np.column_stack([x] + nbs + [y]).astype(np.int32)
    nbins = np.array([7, 7, 7, 7, 7, 7, 2], dtype=np.int64)
    names = ["x", "c1", "c2", "c3", "c4", "c5", "y"]
    return data, nbins, np.array([6], dtype=np.int64), names, [0, 1, 2, 3, 4, 5]


def test_regression_neighbor_unique_target_cached_mis_equivalence():
    """Wave 13 finding 4a: ``neighbor_unique_target`` now accepts ``cached_MIs`` so a reciprocal
    ``I((X_i,X_j);Y)`` is computed once and reused when both ends are visited. Equivalence: the
    cached-path detail must equal the uncached-path detail bit-for-bit (same underlying mi() call,
    just memoized); call-count: with the cache shared across two calls covering the same pair from
    both sides, the joint MI for that pair is computed at most once."""
    rng = np.random.default_rng(7)
    n = 80
    x = rng.integers(0, 2, n).astype(np.int32)
    y = x.copy()
    noise = [rng.integers(0, 4, n).astype(np.int32) for _ in range(6)]
    data = np.column_stack([x, *noise, y]).astype(np.int32)
    nbins = np.array([2, 4, 4, 4, 4, 4, 4, 2], dtype=np.int64)
    target = np.array([7], dtype=np.int64)
    rel_i = float(mi(data, np.array([0], dtype=np.int64), target, nbins))
    rel_j = float(mi(data, np.array([1], dtype=np.int64), target, nbins))

    # Uncached reference: two independent calls, one per side.
    total_i_uncached, detail_i_uncached = neighbor_unique_target(data, 0, [1], target, rel_i=rel_i, factors_nbins=nbins)
    total_j_uncached, detail_j_uncached = neighbor_unique_target(data, 1, [0], target, rel_i=rel_j, factors_nbins=nbins)

    # Cached path: shared dict across both sides, plus a counting wrapper on the underlying mi() call
    # to prove the second side is a cache hit, not a recompute.
    from mlframe.feature_selection.filters import friend_graph as fg

    orig_mi = fg.mi
    calls = {"n": 0}

    def counted_mi(*a, **kw):
        calls["n"] += 1
        return orig_mi(*a, **kw)

    fg.mi = counted_mi
    try:
        cache: dict = {}
        total_i_cached, detail_i_cached = neighbor_unique_target(data, 0, [1], target, rel_i=rel_i, factors_nbins=nbins, cached_MIs=cache)
        n_after_first = calls["n"]
        total_j_cached, detail_j_cached = neighbor_unique_target(data, 1, [0], target, rel_i=rel_j, factors_nbins=nbins, cached_MIs=cache)
        n_after_second = calls["n"]
    finally:
        fg.mi = orig_mi

    assert n_after_first == 1, f"expected exactly 1 mi() call for the first side, got {n_after_first}"
    assert n_after_second == n_after_first, (
        f"second side re-called mi() for the same (i,j) pair despite the shared cache (n_after_first={n_after_first}, n_after_second={n_after_second})"
    )
    assert total_i_cached == total_i_uncached
    assert total_j_cached == total_j_uncached
    assert detail_i_cached == detail_i_uncached
    assert detail_j_cached == detail_j_uncached


def test_regression_build_friend_graph_klass_unchanged_with_mi_cache():
    """Wave 13 findings 4a/4b: wiring the shared joint-MI cache + the detail-by-j index into
    ``build_friend_graph``'s suspect loop must not change any node's classification, weighted degree,
    or neighbors_unique_target versus the pre-fix per-side-recompute / linear-scan behaviour --
    selection-equivalence pinned against the existing F3 fixture."""
    data, nbins, target, names, sel = _small_n_highcard_dataset()
    g = build_friend_graph(
        sel,
        data,
        nbins,
        target,
        feature_names=names,
        garbage_min_degree=2,
        garbage_unique_ratio=0.1,
        mi_eps=1e-6,
        compute_layout=False,
    )
    # Reference: recompute each suspect's neighbor_unique_target independently (no shared cache),
    # exactly as the pre-fix code path did, and check the aggregate + per-node detail match exactly.
    by_idx = {n.idx: n for n in g.nodes}
    neighbors_ref: dict = {i: [] for i in sel}
    for e in g.edges:
        neighbors_ref[e.a].append(e.b)
        neighbors_ref[e.b].append(e.a)
    for i in sel:
        node = by_idx[i]
        degree = len(neighbors_ref[i])
        if degree < 2:
            continue
        rel_i = node.relevance
        total_ref, detail_ref = neighbor_unique_target(data, i, neighbors_ref[i], target, rel_i=rel_i, factors_nbins=nbins)
        assert total_ref == pytest.approx(node.neighbors_unique_target, abs=1e-12)
        detail_shipped = dict(g._neighbor_unique_detail.get(i, []))
        detail_reference = dict(detail_ref)
        assert detail_shipped == pytest.approx(detail_reference, abs=1e-12)


def test_f4_neighbor_unique_detail_carries_raw_unclamped_cmi():
    """The per-neighbor CMI in ``detail`` is the raw chain-rule value; on independent (noisy) pairs at small n at least one term is negative (was clamped to exactly 0)."""
    # Construct a node X strongly relevant to y, with neighbours that are PURE NOISE (target-independent). The chain-rule CMI I(Y; noise | X) is ~0 in truth, so the
    # finite-sample plug-in estimate (joint MI of (X, noise) with y minus I(X;y)) fluctuates around 0 -- negative on some neighbours at small n. Pre-fix this was clamped to 0.
    rng = np.random.default_rng(7)
    n = 80
    x = rng.integers(0, 2, n).astype(np.int32)
    y = x.copy()  # X fully determines y -> rel_i is maximal
    noise = [rng.integers(0, 4, n).astype(np.int32) for _ in range(6)]
    data = np.column_stack([x] + noise + [y]).astype(np.int32)
    nbins = np.array([2, 4, 4, 4, 4, 4, 4, 2], dtype=np.int64)
    target = np.array([7], dtype=np.int64)
    rel_i = float(mi(data, np.array([0], dtype=np.int64), target, nbins))
    total_unique, detail = neighbor_unique_target(
        data,
        0,
        [1, 2, 3, 4, 5, 6],
        target,
        rel_i=rel_i,
        factors_nbins=nbins,
    )
    vals = [c for _j, c in detail]
    assert any(c < 0.0 for c in vals), f"expected at least one raw negative CMI (pre-fix clamp would force all >=0); got {vals}"
    # total_unique is still the clamped non-negative aggregate.
    assert total_unique >= 0.0


def test_f3_debiased_threshold_changes_red_flag_on_small_n_highcard():
    """At small n with a high-cardinality hub the debiased garbage threshold must differ from the raw-bias threshold for at least one node.

    Reproduces F3 by reconstructing the pre-fix (bias-mismatched) decision from the same graph quantities and asserting the shipped (debiased) classification differs.
    """
    data, nbins, target, names, sel = _small_n_highcard_dataset()
    gur = 0.1
    g = build_friend_graph(
        sel,
        data,
        nbins,
        target,
        feature_names=names,
        garbage_min_degree=2,
        garbage_unique_ratio=gur,
        mi_eps=1e-6,
        compute_layout=False,
    )
    by_idx = {nd.idx: nd for nd in g.nodes}
    n = float(data.shape[0])
    n_y = int(nbins[int(target[0])])

    raw_flags = {}
    debiased_flags = {}
    for nd in g.nodes:
        detail = g._neighbor_unique_detail.get(nd.idx, [])
        if not detail:
            continue
        n_i = int(nbins[nd.idx])
        raw_total = sum(max(0.0, c) for _j, c in detail)
        raw_red = raw_total > max(1e-6, gur * nd.relevance)
        db_total = 0.0
        for j, c in detail:
            n_j = int(nbins[j])
            bias = (n_i * n_j - 1 - (n_i - 1)) * (n_y - 1) / (2.0 * n)
            db_total += max(0.0, c - bias)
        rel_db = max(0.0, nd.relevance - (n_i - 1) * (n_y - 1) / (2.0 * n))
        db_red = db_total > max(1e-6, gur * rel_db)
        raw_flags[nd.idx] = raw_red
        debiased_flags[nd.idx] = db_red
        # The shipped classification must match the debiased decision (this is what build_friend_graph now computes).
        assert (nd.klass == "red") == db_red, f"node {nd.name}: shipped klass={nd.klass} disagrees with debiased red={db_red}"

    assert raw_flags != debiased_flags, (
        f"expected the bias mismatch to flip at least one red flag on small-n high-cardinality data; raw={raw_flags} debiased={debiased_flags}"
    )
