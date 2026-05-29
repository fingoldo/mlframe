"""DCD pre-implementation gate: SU pair scaling bench (Wave 9 §8).

Measures the cost of pairwise SU computation on already-binned MRMR data
matrices at production scales. The DCD plan claims a 0.7x wall-time win
vs the legacy post-hoc cluster_aggregate; this bench is the GATE that
decides whether to proceed with the ~570 LOC DCD impl or pivot to the
~80 LOC Tier-1 Pearson-replacement only patch.

Pass criterion (per plan v2 §8):
    extrapolated DCD wall-time at p=10000 <= 0.7x legacy post-hoc

If FAILS: pivot to Tier-1 (replace Pearson in _discover_clusters with SU,
keep post-hoc architecture intact).

Measurement protocol:
    1. Synthesise binned integer matrices at (N, p) = (5000, 100), (5000, 1000),
       (5000, 10000).
    2. Time the FULL O(p^2) pairwise SU matrix via symmetric_uncertainty (this
       is what legacy post-hoc cluster_aggregate effectively costs).
    3. Time DCD's expected access pattern: K greedy picks * (p - already-pruned)
       SU calls, with an LRU-bounded cache.
    4. Project both to wall-time at p=10000; compute ratio.

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_dcd_pair_su_scaling

(or via direct path: python benchmarks/bench_dcd_pair_su_scaling.py)
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Allow running via direct path OR via -m
try:
    from mlframe.feature_selection.filters.info_theory import symmetric_uncertainty
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mlframe.feature_selection.filters.info_theory import symmetric_uncertainty


# =============================================================================
# Synth data
# =============================================================================


def make_binned_matrix(n: int, p: int, n_bins: int = 10, n_clusters: int = 5,
                       cluster_size: int = 6, seed: int = 0) -> tuple:
    """Synthesise a binned integer matrix mimicking MRMR's post-categorize state.

    Produces:
      - factors_data: (n, p) int32 with values in [0, n_bins).
      - factors_nbins: (p,) int64 == n_bins for all columns.
      - cluster_groups: dict[anchor_col -> list[member_cols]] for ground-truth.
    """
    rng = np.random.default_rng(int(seed))
    data = rng.integers(0, n_bins, size=(n, p)).astype(np.int32)

    # Inject collinear clusters: copy anchor with mild bin-noise to first
    # n_clusters anchors.
    cluster_groups = {}
    for c_i, anchor in enumerate(range(n_clusters)):
        members = []
        for k in range(cluster_size - 1):
            target = n_clusters + c_i * (cluster_size - 1) + k
            if target >= p:
                break
            noise_mask = rng.random(n) < 0.10
            data[:, target] = np.where(
                noise_mask,
                rng.integers(0, n_bins, n).astype(np.int32),
                data[:, anchor],
            )
            members.append(target)
        cluster_groups[anchor] = members

    factors_nbins = np.full(p, n_bins, dtype=np.int64)
    return data, factors_nbins, cluster_groups


# =============================================================================
# Cost models
# =============================================================================


def _su_pair(factors_data, a: int, b: int, factors_nbins) -> float:
    """SU(X_a, X_b) wrapper using the vetted signature
    ``symmetric_uncertainty(factors_data, x: ndarray, y: ndarray, factors_nbins)``.
    """
    return float(symmetric_uncertainty(
        factors_data,
        np.array([a], dtype=np.int64),
        np.array([b], dtype=np.int64),
        factors_nbins,
    ))


def measure_full_pairwise(factors_data, factors_nbins, max_pairs: int = None
                           ) -> tuple[float, int]:
    """Cost A: full O(p^2) pairwise SU matrix (the legacy post-hoc upper bound).

    On p=10000 the full p*(p-1)/2 ~ 50M pairs are infeasible to measure.
    Caps at ``max_pairs`` and returns extrapolation.
    """
    p = factors_data.shape[1]
    total_pairs = p * (p - 1) // 2
    cap = int(max_pairs) if max_pairs else total_pairs
    sampled = min(cap, total_pairs)
    # Random pair sampling for fair extrapolation.
    rng = np.random.default_rng(0)
    # Reservoir of (i, j) pairs.
    pairs = []
    while len(pairs) < sampled:
        a = int(rng.integers(0, p))
        b = int(rng.integers(0, p))
        if a != b:
            pairs.append((min(a, b), max(a, b)))
    t0 = time.perf_counter()
    for a, b in pairs:
        _ = _su_pair(factors_data, a, b, factors_nbins)
    elapsed = time.perf_counter() - t0
    # Project to full pairwise.
    per_pair = elapsed / max(len(pairs), 1)
    projected_full = per_pair * total_pairs
    return projected_full, total_pairs


def measure_dcd_access(factors_data, factors_nbins,
                        n_selected: int = 20,
                        avg_cluster_size: int = 6,
                        cache_max: int = 50_000) -> tuple[float, dict]:
    """Cost B: DCD's expected access pattern.

    For each of ``n_selected`` greedy picks, compute SU(pool_member, anchor)
    over a shrinking pool. The pool shrinks by avg_cluster_size members per
    pick (DCD's pruning effect). LRU-bounded cache amortises repeated lookups.

    Returns (elapsed_seconds, stats_dict).
    """
    p = factors_data.shape[1]
    pool = set(range(p))
    cache: "OrderedDict[tuple[int,int], float]" = OrderedDict()
    selected = []
    n_su_calls = 0
    n_cache_hits = 0
    n_cache_misses = 0

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for k in range(int(n_selected)):
        if not pool:
            break
        # Pick a random pool member as the "winner" of this round.
        anchor = int(rng.choice(list(pool)))
        selected.append(anchor)
        pool.discard(anchor)
        # For each remaining pool member, compute SU(c, anchor).
        prune_set = set()
        for c in list(pool):
            key = (min(c, anchor), max(c, anchor))
            if key in cache:
                su = cache[key]
                cache.move_to_end(key)
                n_cache_hits += 1
            else:
                su = _su_pair(factors_data, c, anchor, factors_nbins)
                cache[key] = su
                n_cache_misses += 1
                while len(cache) > cache_max:
                    cache.popitem(last=False)
            n_su_calls += 1
            if su > 0.7:  # DCD's tau_cluster default
                prune_set.add(c)
        # Shrink pool by avg_cluster_size or actual prune_set, whichever is smaller.
        # (DCD's actual behaviour is to prune by SU threshold; we approximate.)
        n_prune = min(len(prune_set), int(avg_cluster_size))
        for c in list(prune_set)[:n_prune]:
            pool.discard(c)
    elapsed = time.perf_counter() - t0
    stats = dict(
        n_su_calls=n_su_calls,
        n_cache_hits=n_cache_hits,
        n_cache_misses=n_cache_misses,
        cache_hit_rate=n_cache_hits / max(n_su_calls, 1),
        final_pool_size=len(pool),
        final_cache_size=len(cache),
        n_selected=len(selected),
    )
    return elapsed, stats


# =============================================================================
# Bench driver
# =============================================================================


def main():
    out_path = Path(__file__).resolve().parent / "bench_dcd_pair_su_scaling_result.json"
    results = {}

    print("=" * 78)
    print("DCD PRE-IMPL MICRO-BENCH (Wave 9 §8 gate)")
    print("=" * 78)
    print()

    SCALES = [(5000, 100), (5000, 1000), (5000, 10000)]
    for n, p in SCALES:
        print(f"Scale n={n}, p={p}")
        print("-" * 78)
        data, nbins, _ = make_binned_matrix(n, p, n_bins=10, n_clusters=10,
                                             cluster_size=6, seed=42)
        # Cost A: full pairwise SU (legacy upper bound; sample if p too large).
        sample_cap = min(5000, p * (p - 1) // 2)
        t_a, total_pairs = measure_full_pairwise(data, nbins, max_pairs=sample_cap)
        print(f"  Cost A (full pairwise extrapolated to {total_pairs:,} pairs): "
              f"{t_a:.2f}s")
        # Cost B: DCD access pattern (n_selected=20 greedy picks).
        t_b, stats = measure_dcd_access(data, nbins, n_selected=20,
                                         avg_cluster_size=6, cache_max=50_000)
        print(f"  Cost B (DCD: 20 greedy picks + prune): {t_b:.2f}s")
        print(f"  Cost B stats: {stats}")
        ratio = t_b / t_a if t_a > 0 else float("inf")
        print(f"  Ratio B/A: {ratio:.3f}  (target <= 0.7)")
        # Pass criterion
        verdict = "PASS" if ratio <= 0.7 else "FAIL — pivot to Tier-1"
        print(f"  Verdict: {verdict}")
        print()
        results[f"n{n}_p{p}"] = dict(
            cost_a_seconds=float(t_a),
            cost_b_seconds=float(t_b),
            ratio_b_over_a=float(ratio),
            verdict=verdict,
            stats=stats,
        )

    # Overall verdict
    p10k = results.get("n5000_p10000", {})
    overall_pass = (p10k.get("ratio_b_over_a", 1.0) <= 0.7)
    print("=" * 78)
    print(f"OVERALL VERDICT: {'PASS — proceed with full DCD' if overall_pass else 'FAIL — pivot to Tier-1 Pearson-replacement patch'}")
    print("=" * 78)
    results["overall_verdict"] = "PASS" if overall_pass else "FAIL"

    try:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {out_path}")
    except Exception as exc:
        print(f"Could not save results: {exc!r}")


if __name__ == "__main__":
    main()
