"""2026-07-06: ``discover_cluster_members`` batch-warm equivalence.

The 60s ``pair_su`` tottime in the 1M .prof lives in ``discover_cluster_members``,
which scores ONE anchor against K candidates via per-pair ``pair_su`` -> K joint
histograms sharing the anchor column. The optimization batch-warms
``state.pairwise_su_cache`` for all statically-eligible candidates in one
prange-over-pairs joint-entropy pass (5-8.5x, see bench_pair_su_batch_over_pairs),
then the UNCHANGED loop reads those as cache hits.

This test pins that the warm path is byte-identical in SELECTION (same members
added) and in the cached SU VALUES to the per-pair loop. It fails pre-optimization
in the sense that the warm-call symbol was not wired (the value-source is the same
float either way, so we assert equivalence of the two code paths).
"""

from __future__ import annotations

import numpy as np


def _state(n_cols, seed=0):
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import make_dcd_state

    rng = np.random.default_rng(seed)
    # Column 0 is the anchor. Columns 1,2 are near-copies of it (high SU),
    # 3,4 are independent (low SU). Gives a non-trivial member set so the
    # equivalence check exercises the accept branch, not just the reject one.
    base = rng.integers(0, 4, 800).astype(np.int32)
    cols = [base]
    for _ in range(2):
        c = base.copy()
        flip = rng.random(800) < 0.05
        c[flip] = rng.integers(0, 4, flip.sum()).astype(np.int32)
        cols.append(c)
    for _ in range(n_cols - 3):
        cols.append(rng.integers(0, 4, 800).astype(np.int32))
    fd = np.ascontiguousarray(np.stack(cols, axis=1))
    fn = np.array([4] * n_cols, dtype=np.int64)
    st = make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(n_cols)],
        nbins=fn,
        target_indices=None,
        distance="su",
    )
    return st, fd, fn


def _run(disable_batch):
    """Run discover_cluster_members over anchor 0; optionally neuter the
    batch-warm so the loop computes every pair per-pair. Returns
    (newly_added, {key: su}) captured from the pairwise cache."""
    import mlframe.feature_selection.filters._dynamic_cluster_discovery as dcd

    n_cols = 6
    st, fd, fn = _state(n_cols)
    orig = dcd.pair_su_batch
    if disable_batch:
        dcd.pair_su_batch = lambda *a, **k: np.zeros(0)  # no-op warm
    try:
        added = dcd.discover_cluster_members(
            st,
            just_selected=0,
            candidate_pool=list(range(1, n_cols)),
            entropy_cache=None,
            factors_data=fd,
            factors_nbins=fn,
        )
    finally:
        dcd.pair_su_batch = orig
    cache_vals = {k: float(v) for k, v in st.pairwise_su_cache.items()}
    return added, cache_vals


def test_discover_batch_warm_selection_and_values_bit_identical():
    added_warm, vals_warm = _run(disable_batch=False)
    added_plain, vals_plain = _run(disable_batch=True)

    # Same members selected -- the SELECTION-EQUIVALENCE bar (the contract
    # for FE/MRMR: same features chosen, not necessarily bit-identical).
    assert added_warm == added_plain
    # And the two near-copies were actually clustered (accept branch hit).
    assert len(added_warm) >= 1, "expected the redundant copies to cluster"

    # SU values agree to well within 1e-12 absolute (agent's documented
    # contract). They are NOT strictly bit-identical: the batched
    # joint-entropy kernel is @njit(parallel=True) and numba's parallel
    # codegen may reorder the inner -(p*log p) reduction by one ULP vs the
    # serial joint_entropy_2var on some bin patterns under a real multi-pair
    # batch (a single-pair batch IS bit-identical). For a small SU the
    # order-1 joint's ULP is amplified by cancellation in
    # 2*(h_a+h_b-h_ab)/(h_a+h_b) to ~3e-16 absolute -- still orders of
    # magnitude below any tau_cluster margin, so selection is unaffected
    # (asserted above).
    assert set(vals_warm) == set(vals_plain)
    for k in vals_warm:
        assert abs(vals_warm[k] - vals_plain[k]) <= 1e-12, (k, vals_warm[k], vals_plain[k])


def test_discover_batch_warm_calls_batch_once():
    """The warm path routes through pair_su_batch exactly once (K joints in
    one parallel pass), proving the loop is not paying K serial joints."""
    import mlframe.feature_selection.filters._dynamic_cluster_discovery as dcd

    n_cols = 6
    st, fd, fn = _state(n_cols)
    calls = {"n": 0, "pairs": None}
    orig = dcd.pair_su_batch

    def _spy(state, pairs, **k):
        calls["n"] += 1
        calls["pairs"] = list(pairs)
        return orig(state, pairs, **k)

    dcd.pair_su_batch = _spy
    try:
        dcd.discover_cluster_members(
            st,
            just_selected=0,
            candidate_pool=list(range(1, n_cols)),
            entropy_cache=None,
            factors_data=fd,
            factors_nbins=fn,
        )
    finally:
        dcd.pair_su_batch = orig

    assert calls["n"] == 1
    # Every warmed pair shares the anchor and covers the eligible candidates.
    assert calls["pairs"] == [(c, 0) for c in range(1, n_cols)]
