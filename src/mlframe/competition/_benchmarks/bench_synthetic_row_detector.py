"""cProfile benchmark for mlframe.competition.synthetic_row_detector.

COMPETITION/EXPLORATORY ONLY - see module docstring under src/mlframe/competition/.
Run: python -m mlframe.competition._benchmarks.bench_synthetic_row_detector
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.competition.synthetic_row_detector import count_encoding_shift_report, detect_synthetic_rows


def _make_bench_dataset(n_real: int, n_synthetic: int, n_cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = pd.DataFrame({f"col_{j}": np.round(rng.uniform(0, 1000, n_real), 4) for j in range(n_cols)})
    synthetic = pd.DataFrame(
        {f"col_{j}": rng.choice(real[f"col_{j}"].to_numpy(), size=n_synthetic, replace=True) for j in range(n_cols)}
    )
    return pd.concat([real, synthetic], ignore_index=True)


def run_once(n_real: int, n_synthetic: int, n_cols: int) -> float:
    test_df = _make_bench_dataset(n_real, n_synthetic, n_cols)

    start = time.perf_counter()
    mask = detect_synthetic_rows(test_df)
    count_encoding_shift_report(test_df, mask, warn=False)
    return time.perf_counter() - start


def main() -> None:
    for n_real, n_synthetic, n_cols in [(5_000, 5_000, 5), (50_000, 50_000, 5), (100_000, 100_000, 10)]:
        elapsed = run_once(n_real, n_synthetic, n_cols)
        print(f"n_real={n_real} n_synthetic={n_synthetic} n_cols={n_cols}: {elapsed:.3f}s wall")

    test_df = _make_bench_dataset(50_000, 50_000, 5)

    profiler = cProfile.Profile()
    profiler.enable()
    mask = detect_synthetic_rows(test_df)
    count_encoding_shift_report(test_df, mask, warn=False)
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
