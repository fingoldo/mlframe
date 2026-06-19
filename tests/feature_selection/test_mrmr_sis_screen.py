"""Tests for the SIS front gate (Gate A of the p=100k MRMR cascade).

Covers, per MRMR_100K_SCALING_DESIGN.md section 5:
  * chunk-vs-full screen rank-equivalence + determinism across runs;
  * biz_value: recall of planted operands (main-effect AND pure-pair) >= random baseline at L>=0.1
    on a small fast in-RAM frame;
  * one real-scale liveness test (p=20000-50000, memmap, modest n) -- screen completes within a wall
    budget and survivor set is << p and contains the planted signal;
  * a cProfile run confirming the hotspot.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_sis_screen import (
    sis_screen,
    fuse_scores,
    survivor_count,
)


# ----------------------------------------------------------------------------------------------------------
# synthetic-frame generator: planted main-effects + planted pure-pair interactions (realistic leakage)
# ----------------------------------------------------------------------------------------------------------
def _make_frame(n, p, n_main, n_pairs, L, seed, out=None):
    """Build an (n, p) frame.

    - first ``n_main`` columns are main effects driving a binary y;
    - next ``2*n_pairs`` columns are pure-pair operands: their interaction ``|a|*|b|`` (leakage L) drives
      y. Each operand alone has ~0 MARGINAL (first-moment) MI, but the interaction leaks into the operand's
      HIGHER moments (|a| relates to a^2), which is exactly the realistic non-XOR case the 2nd-moment
      propensity score targets (a perfectly-balanced XOR with zero higher-moment leakage is the irreducible
      floor invisible to ANY O(p) score -- explicitly out of scope per the design).
    - the rest are standard-normal noise.
    Returns (X, y, main_idx, operand_idx). When ``out`` (a memmap) is given the matrix is written chunked.
    """
    rng = np.random.default_rng(seed)
    main_idx = np.arange(n_main, dtype=int)
    operand_idx = np.arange(n_main, n_main + 2 * n_pairs, dtype=int)

    # latent signal -> binary y
    main_vals = rng.standard_normal((n, n_main)).astype(np.float32)
    op_vals = rng.standard_normal((n, 2 * n_pairs)).astype(np.float32)
    z = main_vals.sum(axis=1) * 0.5
    for k in range(n_pairs):
        a = op_vals[:, 2 * k]
        b = op_vals[:, 2 * k + 1]
        # |a|*|b| interaction: leaks into operand higher moments (the realistic, detectable case);
        # centre by E[|N|]^2 = (2/pi) so the interaction term has ~zero mean.
        z = z + L * 4.0 * (np.abs(a) * np.abs(b) - (2.0 / np.pi))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < prob).astype(np.int64)

    if out is None:
        X = np.empty((n, p), dtype=np.float32)
    else:
        X = out
    X[:, main_idx] = main_vals
    X[:, operand_idx] = op_vals
    # noise columns, chunked to keep peak RAM modest for the memmap path
    noise_cols = np.setdiff1d(np.arange(p), np.concatenate([main_idx, operand_idx]))
    cw = 2000
    for c0 in range(0, noise_cols.size, cw):
        cols = noise_cols[c0:c0 + cw]
        X[:, cols] = rng.standard_normal((n, cols.size)).astype(np.float32)
    if out is not None:
        out.flush()
    return X, y, main_idx, operand_idx


def _recall(survivors, planted):
    s = set(int(i) for i in survivors)
    if len(planted) == 0:
        return 1.0
    return sum(1 for i in planted if int(i) in s) / len(planted)


# ----------------------------------------------------------------------------------------------------------
# chunk-vs-full rank equivalence + determinism
# ----------------------------------------------------------------------------------------------------------
def test_chunk_vs_full_rank_equivalence_and_determinism():
    n, p = 1500, 800
    X, y, _, _ = _make_frame(n, p, n_main=6, n_pairs=4, L=0.2, seed=7)

    surv_full, sc_full = sis_screen(X, y, target_survivors=200, chunk_width=p, return_scores=True)
    surv_chunk, sc_chunk = sis_screen(X, y, target_survivors=200, chunk_width=137, return_scores=True)

    # block width must not change the per-column scores (single pass, no cross-block state)
    np.testing.assert_allclose(sc_full["fused"], sc_chunk["fused"], rtol=1e-5, atol=1e-5)
    assert np.array_equal(surv_full, surv_chunk), "chunked screen must select the same survivors as full"

    # determinism across repeated runs (no global RNG)
    surv_again = sis_screen(X, y, target_survivors=200, chunk_width=137)
    assert np.array_equal(surv_chunk, surv_again)

    # survivors are sorted ascending unique indices within range
    assert surv_chunk.tolist() == sorted(set(surv_chunk.tolist()))
    assert surv_chunk.min() >= 0 and surv_chunk.max() < p


def test_survivor_count_rule_data_derived():
    # concentrated signal: a few large scores above the MAD knee, rest at noise floor
    rng = np.random.default_rng(0)
    fused = rng.standard_normal(5000)
    fused[:15] += 12.0  # strong outliers
    m = survivor_count(fused, k_target=5, mad_c=3.0)
    # floor max(20*5, 1000) = 1000 dominates the 15-feature knee here
    assert m == 1000
    # with a huge k_target the floor scales
    m2 = survivor_count(fused, k_target=100, mad_c=3.0)
    assert m2 == 2000
    # ram_cap clamps
    m3 = survivor_count(fused, k_target=100, mad_c=3.0, ram_cap=1200)
    assert m3 == 1200
    # never exceeds p
    assert survivor_count(fused[:50], k_target=100) == 50


def test_fuse_keeps_both_signal_classes():
    p = 1000
    mi = np.zeros(p)
    prop = np.zeros(p)
    mi[10] = 5.0          # pure main effect (high MI, zero propensity)
    prop[20] = 5.0        # pure interaction operand (zero MI, high propensity)
    fused = fuse_scores(mi, prop)
    top2 = set(np.argsort(-fused)[:2].tolist())
    assert {10, 20} <= top2, "fusion must keep both a main effect and an interaction operand"


# ----------------------------------------------------------------------------------------------------------
# biz_value: recall of planted signal >= random baseline at L>=0.1
# ----------------------------------------------------------------------------------------------------------
def test_bizvalue_recall_beats_random_baseline():
    n, p = 3000, 3000
    n_main, n_pairs, L = 8, 6, 0.15
    X, y, main_idx, op_idx = _make_frame(n, p, n_main, n_pairs, L, seed=11)

    m = 600
    survivors, sc = sis_screen(X, y, target_survivors=m, return_scores=True)

    planted = np.concatenate([main_idx, op_idx])
    rec_all = _recall(survivors, planted)
    rec_main = _recall(survivors, main_idx)
    rec_op = _recall(survivors, op_idx)
    random_baseline = m / p

    print(
        f"[bizvalue] m={m}/{p} baseline={random_baseline:.3f} "
        f"recall_all={rec_all:.3f} recall_main={rec_main:.3f} recall_op={rec_op:.3f}"
    )
    # main effects must be essentially all recovered
    assert rec_main >= 0.9, f"main-effect recall {rec_main} too low"
    # interaction operands must beat the random baseline by a clear margin (the whole point of fusing 2nd-moment)
    assert rec_op > random_baseline + 0.1, f"operand recall {rec_op} <= baseline {random_baseline}"
    assert rec_all > random_baseline


# ----------------------------------------------------------------------------------------------------------
# real-scale liveness: memmap, wall budget, survivors << p, planted signal present
# ----------------------------------------------------------------------------------------------------------
def test_real_scale_memmap_liveness(tmp_path):
    n, p = 4000, 20000
    n_main, n_pairs, L = 10, 8, 0.15
    path = os.path.join(str(tmp_path), "X.dat")
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, p))
    X, y, main_idx, op_idx = _make_frame(n, p, n_main, n_pairs, L, seed=23, out=mm)
    del X
    mm.flush()
    # reopen read-only so the screen reads from disk in column blocks
    Xr = np.memmap(path, dtype=np.float32, mode="r", shape=(n, p))

    t0 = time.perf_counter()
    survivors, sc = sis_screen(Xr, y, k_target=30, return_scores=True)
    wall = time.perf_counter() - t0
    print(f"[liveness] p={p} n={n} wall={wall:.2f}s survivors={survivors.size} chunk_w={sc['chunk_width']}")

    # survivor set is much smaller than p
    assert survivors.size < p // 4
    # planted main effects all present; operands beat random
    rec_main = _recall(survivors, main_idx)
    rec_op = _recall(survivors, op_idx)
    baseline = survivors.size / p
    print(f"[liveness] recall_main={rec_main:.3f} recall_op={rec_op:.3f} baseline={baseline:.3f}")
    assert rec_main >= 0.9
    assert rec_op > baseline
    # wall budget (generous to tolerate concurrent disk load on this box)
    assert wall < 180.0, f"screen took {wall:.1f}s, over budget"


@pytest.mark.skipif(os.environ.get("SIS_PROFILE", "") != "1", reason="set SIS_PROFILE=1 to run cProfile")
def test_cprofile_hotspot(tmp_path):
    import cProfile
    import pstats
    import io

    n, p = 4000, 20000
    path = os.path.join(str(tmp_path), "Xp.dat")
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, p))
    _, y, _, _ = _make_frame(n, p, n_main=10, n_pairs=8, L=0.15, seed=5, out=mm)
    Xr = np.memmap(path, dtype=np.float32, mode="r", shape=(n, p))

    pr = cProfile.Profile()
    pr.enable()
    sis_screen(Xr, y, k_target=30)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())
