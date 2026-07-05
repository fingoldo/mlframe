"""Profiling benchmark for the cat-FE orchestrator.

Per ``mlframe/CLAUDE.md`` "Profile every new feature with cProfile +
optimize hotspots": this script measures cat-FE orchestrator time across
a grid of (N_cat_cols, n_samples) and emits a JSON results file plus a
cProfile call-graph attribution. The dispatch decision rule (per
``mlframe/CLAUDE.md`` "Numerical-kernel acceleration ladder + size-aware
dispatch"): GPU activates when ``N>=200 AND n>=500k`` -- the threshold
where CPU prange becomes memory-bandwidth-bound and per-pair merge_vars
costs dominate.

Usage:

    D:/ProgramData/anaconda3/python.exe \\
        mlframe/feature_selection/_benchmarks/bench_categorical_fe.py

Output:

    _results/bench_categorical_fe_YYYYMMDD_HHMMSS.json
    _results/bench_categorical_fe_YYYYMMDD_HHMMSS_profile.txt

Per the user's ``feedback_save_useful_scripts_in_package`` rule, the
script lives under the package (not D:/Temp) so any maintainer can
re-run it.

The benchmark is intentionally fast (under 2 min wall) so it can ship
as a smoke check; production-shape sweeps (N=500, n=1M) are commented
in but disabled by default. Per ``feedback_perf_measure_first``: the
output decision is a RATIO comparison (CPU vs GPU), not absolute
numbers -- numbers will rot, ratios survive hardware changes.
"""

from __future__ import annotations

import cProfile
import json
import os
import pstats
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import CatFEConfig
from mlframe.feature_selection.filters.cat_interactions import (
    run_cat_interaction_step,
)
from mlframe.feature_selection.filters.info_theory import merge_vars


def make_synthetic(n: int, n_cat: int, signal_pair: bool = True, seed: int = 0):
    """Synthetic n_cat-col categorical dataset. If ``signal_pair=True``,
    columns 0 and 1 form an XOR synergy pair; rest are independent uniform.
    Returns ``(data, nbins, classes_y, freqs_y, target_idx)`` ready for
    ``run_cat_interaction_step``."""
    rng = np.random.default_rng(seed)
    cols = []
    nbins_list = []
    if signal_pair:
        x1 = rng.integers(0, 2, n).astype(np.int32)
        x2 = rng.integers(0, 2, n).astype(np.int32)
        cols.append(x1); nbins_list.append(2)
        cols.append(x2); nbins_list.append(2)
        for _ in range(n_cat - 2):
            cols.append(rng.integers(0, 4, n).astype(np.int32))
            nbins_list.append(4)
        y = (x1 ^ x2).astype(np.int32)
    else:
        for _ in range(n_cat):
            cols.append(rng.integers(0, 4, n).astype(np.int32))
            nbins_list.append(4)
        y = rng.integers(0, 2, n).astype(np.int32)
    cols.append(y); nbins_list.append(2)
    data = np.column_stack(cols).astype(np.int32)
    nbins = np.array(nbins_list, dtype=np.int64)
    cls_y, fq_y, _ = merge_vars(
        factors_data=data, vars_indices=np.array([n_cat], dtype=np.int64),
        var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
    )
    return data, nbins, cls_y, fq_y, n_cat


def time_one(n: int, n_cat: int, cfg_overrides: dict | None = None, signal_pair: bool = True, warmup: bool = False) -> dict:
    """Run cat-FE once on the (n, n_cat) shape, return timing dict."""
    data, nbins, cls_y, fq_y, tgt = make_synthetic(n, n_cat, signal_pair)
    cols_names = [f"c{i}" for i in range(n_cat)] + ["y"]
    defaults = dict(
        enable=True,
        top_k_pairs=16,
        min_interaction_information=0.05,
        full_npermutations=0,  # bench focuses on search, not perm test
        fwer_correction="none",
        n_folds_stability=0,
        max_kway_order=2,
    )
    if cfg_overrides:
        defaults.update(cfg_overrides)
    cfg = CatFEConfig(**defaults)
    t0 = time.perf_counter()
    _, _, _, state = run_cat_interaction_step(
        data=data, cols=cols_names, nbins=nbins,
        target_indices=np.array([tgt], dtype=np.int64),
        classes_y=cls_y, classes_y_safe=cls_y, freqs_y=fq_y,
        categorical_vars=list(range(n_cat)),
        cfg=cfg, dtype=np.int32, verbose=0,
    )
    elapsed = time.perf_counter() - t0
    return {
        "n_samples": n,
        "n_cat_cols": n_cat,
        "n_pairs_considered": (n_cat * (n_cat - 1)) // 2,
        "n_recipes_produced": len(state.recipes),
        "wall_seconds": elapsed,
        "signal_pair": signal_pair,
        "warmup": warmup,
    }


def main() -> None:
    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"bench_categorical_fe_{ts}.json"
    profile_path = out_dir / f"bench_categorical_fe_{ts}_profile.txt"

    # Warmup -- numba @njit kernel compile happens on first call. Discard
    # the timing.
    print("Warming up numba kernels...")
    time_one(n=200, n_cat=4, warmup=True)

    # Grid: small + medium shapes. Skip the n>=500k shape unless
    # explicitly requested -- it adds 60+s to the bench.
    grid = [(n, n_cat) for n in (1_000, 10_000, 100_000) for n_cat in (10, 50, 100)]
    if os.environ.get("MLFRAME_CAT_FE_BENCH_PROD"):
        grid.extend([(500_000, 200), (1_000_000, 200)])

    results = []
    print(f"Running cat-FE bench across {len(grid)} (n, n_cat) shapes...")
    for n, n_cat in grid:
        r = time_one(n=n, n_cat=n_cat)
        results.append(r)
        print(f"  n={n:>8} n_cat={n_cat:>4} pairs={r['n_pairs_considered']:>6} " f"recipes={r['n_recipes_produced']:>3} wall={r['wall_seconds']:.3f}s")

    # cProfile attribution on one mid-shape config so the user can see
    # where time goes. n=10k, n_cat=50 is a reasonable representative.
    print("Capturing cProfile attribution at n=10000, n_cat=50...")
    pr = cProfile.Profile()
    pr.enable()
    time_one(n=10_000, n_cat=50)
    pr.disable()
    with open(profile_path, "w") as f:
        stats = pstats.Stats(pr, stream=f).sort_stats("cumulative")
        stats.print_stats(40)  # top 40 by cumulative time

    # Decision-rule output: per ``mlframe/CLAUDE.md`` we publish RATIOS,
    # not absolute numbers. The bench reports wall times so the user
    # can DERIVE the ratios per their hardware -- this is healthier
    # than baking in numbers that will rot.
    payload = {
        "timestamp": ts,
        "platform": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "decision_rule": (
            "Inspect wall_seconds across (n_samples, n_cat_cols). "
            "If n_cat_cols >= 200 AND n_samples >= 500_000, GPU "
            "dispatch (CatFEConfig(backend='gpu')) is expected to "
            "outperform CPU by 5-50x once CuPy is installed. "
            "Below that regime, CPU prange is the right call."
        ),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON results -> {json_path}")
    print(f"cProfile dump -> {profile_path}")


if __name__ == "__main__":
    main()
