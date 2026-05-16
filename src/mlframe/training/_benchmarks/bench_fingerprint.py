"""Microbenchmark for the three ``fingerprint_df`` sample-hash strategies.

Variants:
    (a) baseline: ``sub.to_arrow().to_pandas().to_csv()``
    (b) polars-native ``hash_rows`` + xxhash on the resulting Series bytes
    (c) ``write_ipc(BytesIO())`` + xxhash on the IPC payload

Run directly: ``python -m mlframe.training._benchmarks.bench_fingerprint``.

The script prints a markdown table with median wall times across N trials per (n_rows, n_cols)
combination. The fastest variant is what ``fingerprint_df`` should pick by default.

User memory rule ``feedback_save_useful_scripts_in_package`` -- this file lives in the package, not
``D:/Temp``, so the benchmark survives across sessions and is reachable from the test harness.
"""
from __future__ import annotations

import io
import statistics
import time
from typing import List, Tuple

import numpy as np
import polars as pl

try:
    import xxhash  # type: ignore[import-untyped]
    HAVE_XX = True
except ImportError:
    HAVE_XX = False


def _make_frame(n_rows: int, n_cols: int) -> pl.DataFrame:
    rng = np.random.default_rng(20260516)
    data = {
        f"f{i}": rng.standard_normal(n_rows).astype(np.float32) for i in range(n_cols)
    }
    return pl.DataFrame(data)


def _variant_a(sub: pl.DataFrame) -> str:
    arrow_table = sub.to_arrow()
    return arrow_table.to_pandas().to_csv(index=False)


def _variant_b(sub: pl.DataFrame) -> str:
    """Polars-native hash_rows + xxhash."""
    if not HAVE_XX:
        return ""
    h = xxhash.xxh3_64()
    # hash_rows returns a UInt64 Series; bytes form is portable across polars versions.
    h.update(sub.hash_rows().to_numpy().tobytes())
    return h.hexdigest()


def _variant_c(sub: pl.DataFrame) -> str:
    """write_ipc(BytesIO) + xxhash."""
    if not HAVE_XX:
        return ""
    buf = io.BytesIO()
    sub.write_ipc(buf)
    return xxhash.xxh3_64(buf.getvalue()).hexdigest()


def _bench(fn, sub: pl.DataFrame, trials: int = 7) -> float:
    """Return median wall time in ms across ``trials`` runs."""
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(sub)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def run(combinations: List[Tuple[int, int]] | None = None, trials: int = 7) -> None:
    combinations = combinations or [
        (256, 8),
        (4096, 16),
        (4096, 64),
        (4096, 256),
    ]
    print(f"# fingerprint_df sample-hash variants (median of {trials} trials, ms)")
    print(f"# xxhash available: {HAVE_XX}")
    print("| n_rows | n_cols | (a) arrow.to_pandas.to_csv | (b) hash_rows+xxhash | (c) write_ipc+xxhash | best |")
    print("|--------|--------|----------------------------|----------------------|----------------------|------|")
    for n, k in combinations:
        df = _make_frame(n, k)
        sub = df  # in real usage we sample-index first; benchmark on the worst case.
        ta = _bench(_variant_a, sub, trials=trials)
        tb = _bench(_variant_b, sub, trials=trials) if HAVE_XX else float("nan")
        tc = _bench(_variant_c, sub, trials=trials) if HAVE_XX else float("nan")
        candidates = [("a", ta)]
        if HAVE_XX:
            candidates.extend([("b", tb), ("c", tc)])
        best = min(candidates, key=lambda kv: kv[1])[0]
        print(f"| {n} | {k} | {ta:.3f} | {tb:.3f} | {tc:.3f} | {best} |")


if __name__ == "__main__":
    run()
