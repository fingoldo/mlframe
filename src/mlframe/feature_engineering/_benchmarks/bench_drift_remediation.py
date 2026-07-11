"""cProfile harness for ``feature_engineering.drift_remediation.remediate_drifting_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_drift_remediation``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.drift_remediation import remediate_drifting_features


def _make_data(n_time_ids: int, n_entities: int, n_features: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    time_ids = np.repeat(np.arange(n_time_ids), n_entities)
    n = time_ids.shape[0]
    data = {"time_id": time_ids}
    for i in range(n_features):
        if i % 3 == 0:
            data[f"feat_{i}"] = time_ids.astype(np.float64) * 5.0 + rng.standard_normal(n)
        else:
            data[f"feat_{i}"] = rng.standard_normal(n)
    df = pd.DataFrame(data)
    split = n_time_ids // 2
    return df[df["time_id"] < split].reset_index(drop=True), df[df["time_id"] >= split].reset_index(drop=True)


def _run(n_time_ids: int, n_entities: int, n_features: int, n_calls: int) -> None:
    train_df, test_df = _make_data(n_time_ids, n_entities, n_features, seed=0)
    for _ in range(n_calls):
        remediate_drifting_features(train_df, test_df, group_col="time_id", n_splits=2)


def _run_tiered(n_time_ids: int, n_entities: int, n_features: int, n_calls: int) -> None:
    """Same shape as ``_run`` but exercises the opt-in severity-tiered drop path (n_std/drop_n_std)."""
    train_df, test_df = _make_data(n_time_ids, n_entities, n_features, seed=0)
    for _ in range(n_calls):
        remediate_drifting_features(train_df, test_df, group_col="time_id", n_splits=2, n_std=0.5, drop_n_std=2.0)


def _run_auto_tune(n_time_ids: int, n_entities: int, n_features: int, n_calls: int) -> None:
    """Auto-tune path: one extra ``adversarial_auc`` re-check per candidate threshold, the expensive branch."""
    train_df, test_df = _make_data(n_time_ids, n_entities, n_features, seed=0)
    for _ in range(n_calls):
        remediate_drifting_features(train_df, test_df, group_col="time_id", n_splits=2, n_std=0.5, auto_tune_drop_threshold=True)


if __name__ == "__main__":
    for label, fn in [("default (uniform rank-transform)", _run), ("tiered (drop_n_std fixed)", _run_tiered), ("tiered (auto_tune_drop_threshold)", _run_auto_tune)]:
        print(f"--- {label} ---")
        for n_time_ids, n_entities, n_features, n_calls in [(60, 40, 10, 3), (200, 100, 30, 1)]:
            t0 = time.perf_counter()
            fn(n_time_ids, n_entities, n_features, n_calls)
            wall = time.perf_counter() - t0
            n_rows = n_time_ids * n_entities
            print(f"rows~{n_rows:>7,} feats={n_features:>3} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.2f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 60, 20, 2)
    _run_tiered(100, 60, 20, 2)
    _run_auto_tune(100, 60, 20, 2)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
