"""cProfile harness for ``feature_engineering.panel_sequence_tensor.build_panel_sequence_tensor``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_panel_sequence_tensor``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.panel_sequence_tensor import build_panel_sequence_tensor


def _make_panel(n_entities: int, max_hist: int, n_channels: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows_id, rows_t = [], []
    for e in range(n_entities):
        hist_len = rng.integers(3, max_hist)
        rows_id.append(np.full(hist_len, e))
        rows_t.append(np.arange(hist_len))
    ids = np.concatenate(rows_id)
    ts = np.concatenate(rows_t)
    data = {"id": ids, "t": ts}
    for c in range(n_channels):
        data[f"ch{c}"] = rng.normal(size=len(ids))
    return pd.DataFrame(data)


def _run(n_entities: int, max_hist: int, n_channels: int, n_calls: int, per_channel_normalize: bool = False) -> None:
    df = _make_panel(n_entities, max_hist, n_channels)
    channel_cols = [c for c in df.columns if c.startswith("ch")]
    for _ in range(n_calls):
        build_panel_sequence_tensor(df, "id", "t", channel_cols, max_lags=max_hist, normalize=True, per_channel_normalize=per_channel_normalize)


if __name__ == "__main__":
    for n_entities, max_hist, n_channels, n_calls in [(2000, 13, 5, 20), (50000, 13, 5, 20), (50000, 13, 20, 5)]:
        for per_channel_normalize in (False, True):
            t0 = time.perf_counter()
            _run(n_entities, max_hist, n_channels, n_calls, per_channel_normalize=per_channel_normalize)
            wall = time.perf_counter() - t0
            mode = "per_channel" if per_channel_normalize else "joint"
            print(f"n_entities={n_entities:>7} max_hist={max_hist:>3} n_channels={n_channels:>3} n_calls={n_calls:>4} mode={mode:>11} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 13, 20, 5, per_channel_normalize=False)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- joint normalize ---")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 13, 20, 5, per_channel_normalize=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- per_channel normalize ---")
    print(buf.getvalue())
