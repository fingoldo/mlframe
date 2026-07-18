"""Benchmark: FCBF-style ordered-relevance pruning in ``_discover_clusters`` (06_docs_backlog_drift.md
"good, still-unimplemented idea" #1) -- does rejecting chain-transitivity artifacts (A-B-C merged via a
bridge member B even though A/C don't directly correlate) improve cluster-aggregate selection quality
on a fixture designed to trigger that failure mode?

Run: PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_cluster_aggregate_fcbf_pruning
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cluster_aggregate import _discover_clusters


def _chain_fixture(seed: int, n: int = 2000):
    """Two DISTINCT latent factors z1/z2 (uncorrelated), each with 3 noisy reflections. A bridge column
    'bridge' correlates moderately with BOTH the z1-cluster and z2-cluster reps (simulating a spurious
    connector), which a naive single-linkage connected-components pass could chain z1's and z2's
    reflections into one bogus cluster through."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)  # independent of z1
    cols = {
        "z1_a": z1 + 0.15 * rng.standard_normal(n),
        "z1_b": z1 + 0.15 * rng.standard_normal(n),
        "z1_c": z1 + 0.15 * rng.standard_normal(n),
        "z2_a": z2 + 0.15 * rng.standard_normal(n),
        "z2_b": z2 + 0.15 * rng.standard_normal(n),
        "z2_c": z2 + 0.15 * rng.standard_normal(n),
        # Bridge: correlates ~0.65 with z1 and ~0.65 with z2 (independently) via a shared but noisy
        # combination -- moderately above the default corr_threshold=0.6 to BOTH sides.
        "bridge": 0.72 * z1 + 0.72 * z2 + 0.3 * rng.standard_normal(n),
    }
    for i in range(4):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    y = (0.8 * z1 - 0.6 * z2 > 0).astype(int)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _run(seed: int):
    """Call ``_discover_clusters`` directly rather than the full ``MRMR.fit`` pipeline: the
    downstream aggregate-acceptance MI gate in ``run_cluster_aggregate_step`` independently rejects
    aggregates that don't beat their best member's own relevance, which would hide the pruning's
    effect on cluster COMPOSITION behind an unrelated accept/reject decision."""
    Xdf, y = _chain_fixture(seed)
    cols = list(Xdf.columns) + ["y"]
    frame = Xdf.copy()
    frame["y"] = y.values
    data = frame.values.astype(np.float64)
    nbins = np.full(data.shape[1], 20, dtype=np.int64)
    target_indices = [len(cols) - 1]
    clusters = _discover_clusters(
        data=data, cols=cols, nbins=nbins, X=frame, target_indices=target_indices,
        feature_names_in_=list(Xdf.columns), categorical_idx=[], cached_MIs=None,
        min_member_relevance=0.0, corr_threshold=0.6, min_cluster_size=3, max_cluster_size=12,
        homogeneity_tau=0.6, max_candidates=200, mi_eps=1e-6, edge_significance=3.0, dtype=np.int32,
    )
    n_clusters = len(clusters)
    max_cluster_size = max((len(c["members"]) for c in clusters), default=0)
    bridge_idx = cols.index("bridge")
    bridge_chained = any(bridge_idx in c["members"] for c in clusters)
    return n_clusters, max_cluster_size, bridge_chained


def main():
    """Run the chain-transitivity fixture across seeds and report cluster count/size (post-FCBF-pruning
    behavior; this bench documents the CURRENT (post-fix) behavior since the fix is unconditional, not
    a toggle -- a future re-run against a reverted copy of _cluster_aggregate.py would show the pre-fix
    numbers for comparison)."""
    results = [_run(s) for s in range(8)]
    n_clusters = [r[0] for r in results]
    max_sizes = [r[1] for r in results]
    bridge_chained = [r[2] for r in results]
    print(f"clusters found per seed: {n_clusters}")
    print(f"max cluster size per seed: {max_sizes}")
    print(f"bridge chained into a cluster per seed: {bridge_chained}")
    n_bridge_chained = sum(bridge_chained)
    print(f"seeds where the FCBF pruning FAILED to reject the bridge column: {n_bridge_chained}/{len(results)}")


if __name__ == "__main__":
    main()
