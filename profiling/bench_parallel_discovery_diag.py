"""HIGH#6 diagnostic: WHY does discovery_n_jobs > 1 yield only 1.05x?

Instruments the per-transform closure to log thread name + start/end
times. If the closure ENTRIES interleave across threads, joblib is
dispatching in parallel and the issue is GIL contention inside the
closure body. If they DO NOT interleave, joblib isn't actually
running in parallel (threading-backend coordination issue).

Also runs cProfile on the parallel path so we can see WHERE the wall
time goes - if per-base setup (``_extract_column_array``,
``_build_feature_matrix``, ``_prebin_feature_columns``) dominates,
the closure body is irrelevant to the speedup ceiling.

Run: python profiling/bench_parallel_discovery_diag.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import threading
import time

import numpy as np
import pandas as pd


def _build_problem(n: int = 200_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "base1": rng.normal(50.0, 10.0, n),
        "base2": rng.normal(20.0, 5.0, n),
        "base3": rng.exponential(3.0, n),
        "f0": rng.normal(size=n), "f1": rng.normal(size=n),
        "f2": rng.normal(size=n), "f3": rng.normal(size=n),
        "y": rng.normal(size=n),
    })


def diag_thread_overlap(n_jobs: int = 4) -> None:
    """Instrument the per-transform closure to log thread + timing.

    Strategy: patch ``get_transform`` so each call records the thread name
    and timestamps. After fit, inspect the log: if start-end ranges of
    different transforms overlap, joblib is dispatching in parallel.
    """
    from mlframe.training import composite_discovery
    from mlframe.training import composite_transforms
    from mlframe.training.composite_discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    # Use a lock-protected list to record observation events.
    events: list[tuple[str, str, float, float | None]] = []
    events_lock = threading.Lock()

    # Patch ``_mi_to_target_prebinned`` instead of ``get_transform``: the
    # former runs inside the parallel-dispatched closure body; the latter
    # only runs in the serial pre-filter.
    real_mi_fn = composite_discovery._mi_to_target_prebinned

    def _wrapped_mi(*a, **kw):
        ev = ("_mi_to_target_prebinned",
              threading.current_thread().name,
              time.perf_counter(), None)
        with events_lock:
            events.append(ev)
        return real_mi_fn(*a, **kw)

    composite_discovery._mi_to_target_prebinned = _wrapped_mi

    try:
        df = _build_problem(n=200_000)
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=50_000,
            composite_skip_when_raw_dominates_ratio=0.0,
            transforms=[
                "linear_residual", "diff", "ratio", "logratio",
                "monotonic_residual", "quantile_residual",
                "ewma_residual", "cbrt_y", "log_y", "yeo_johnson_y",
            ],
            mi_nbins=8, mi_estimator="bin",
            top_k_after_mi=4, eps_mi_gain=-1.0,
            random_state=42, discovery_n_jobs=n_jobs,
            mi_gain_bootstrap_n=0,
            detect_linear_residual_alpha_drift=False,
            base_candidates=["base1", "base2", "base3"],
        )
        disc = CompositeTargetDiscovery(config=cfg)
        n = len(df)
        feature_cols = [c for c in df.columns if c != "y"]
        train_idx = np.arange(int(0.8 * n))
        disc.fit(df=df, target_col="y", feature_cols=feature_cols,
                 train_idx=train_idx)
    finally:
        composite_discovery._mi_to_target_prebinned = real_mi_fn

    threads_seen = sorted({e[1] for e in events})
    print(f"\n[diag thread overlap] n_jobs={n_jobs}")
    print(f"  total _mi_to_target_prebinned calls: {len(events)}")
    print(f"  unique threads used:       {len(threads_seen)}")
    print(f"  thread names:              {threads_seen}")
    if len(threads_seen) == 1:
        print(f"  >>> joblib did NOT dispatch in parallel! single thread "
              f"executed all transforms <<<")
    else:
        # Are there overlapping intervals across threads?
        # Simplification: count how many transforms each thread handled.
        from collections import Counter
        per_thread = Counter(e[1] for e in events)
        print(f"  transforms per thread: {dict(per_thread)}")


def diag_cprofile(n_jobs: int) -> None:
    from mlframe.training.composite_discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    df = _build_problem(n=200_000)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, mi_sample_n=50_000,
        composite_skip_when_raw_dominates_ratio=0.0,
        transforms=[
            "linear_residual", "diff", "ratio", "logratio",
            "monotonic_residual", "quantile_residual",
            "ewma_residual", "cbrt_y", "log_y", "yeo_johnson_y",
        ],
        mi_nbins=8, mi_estimator="bin",
        top_k_after_mi=4, eps_mi_gain=-1.0,
        random_state=42, discovery_n_jobs=n_jobs,
        mi_gain_bootstrap_n=0,
        detect_linear_residual_alpha_drift=False,
        base_candidates=["base1", "base2", "base3"],
    )
    disc = CompositeTargetDiscovery(config=cfg)
    n = len(df)
    feature_cols = [c for c in df.columns if c != "y"]
    train_idx = np.arange(int(0.8 * n))
    prof = cProfile.Profile()
    prof.enable()
    disc.fit(df=df, target_col="y", feature_cols=feature_cols,
             train_idx=train_idx)
    prof.disable()

    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(30)
    print(f"\n[cProfile top-30 cumulative] n_jobs={n_jobs}")
    print(stream.getvalue())


def main() -> int:
    print("=" * 70)
    print("HIGH#6 diagnostic: WHY discovery_n_jobs > 1 yields only 1.05x")
    print("=" * 70)
    # Warm up.
    _ = _build_problem(n=10_000)
    diag_thread_overlap(n_jobs=4)
    diag_cprofile(n_jobs=4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
