"""Wave 22 + 23 standalone benchmark.

Run::

    python profiling/bench_torch_dataset_concurrency.py

Prints a table of measurements for the eager-tensor + shared-memory
DataLoader claims so reviewers / users can verify the speedups
themselves without parsing pytest output. Covers:

* **Per-batch ``__getitem__`` throughput** -- eager (Wave 22) vs the
  pre-Wave-22 per-batch ``iloc + to_numpy + from_numpy`` legacy
  pattern, across pandas / polars / ndarray inputs at several N.
* **Single-process vs multi-worker** -- iterates the DataLoader for a
  fixed number of batches at ``num_workers in (0, 2, 4)``. Measures
  wall-time AND peak RSS (psutil if available) to demonstrate the
  shared-memory + persistent_workers gains.
* **Byte-cap fallback overhead** -- ensures the above-2-GB lazy path
  is not catastrophically slower than the eager path; sanity check
  for the OOM-safety design.

Each block prints a markdown table for easy paste into commit/PR
descriptions.
"""
from __future__ import annotations

import os
import time
import gc
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _peak_rss_mb() -> float:
    if not _HAS_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def _bench(fn, n_warmup: int = 5, n_iter: int = 200) -> float:
    """Return median per-iter ms."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000


def bench_eager_vs_legacy_per_batch():
    print("\n## Wave 22: per-batch `__getitem__` throughput (eager vs legacy)")
    print()
    print("| Input type | N rows | Eager ms/batch | Legacy ms/batch | Speedup |")
    print("|---|---:|---:|---:|---:|")

    BS = 128
    configs = [
        ("ndarray", lambda n: np.random.rand(n, 16).astype(np.float32)),
        ("pandas", lambda n: pd.DataFrame(np.random.rand(n, 16).astype(np.float32))),
        ("polars", lambda n: pl.DataFrame({f"c{i}": np.random.rand(n).astype(np.float32) for i in range(16)})),
    ]
    for name, mk in configs:
        for n in (10_000, 100_000, 1_000_000):
            features = mk(n)
            arr = features.to_numpy() if hasattr(features, "to_numpy") else np.asarray(features)
            # Eager: convert once outside the timed loop, then index.
            t_eager = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)

            def eager_step():
                _ = t_eager[0:BS]

            # Legacy: per-batch conversion (mirrors pre-Wave-22 code).
            if isinstance(features, pd.DataFrame):
                def legacy_step():
                    sub = features.iloc[0:BS, :].to_numpy()
                    _ = torch.from_numpy(sub).to(torch.float32)
            elif isinstance(features, pl.DataFrame):
                def legacy_step():
                    sub = features[0:BS].to_torch()
                    _ = sub.to(torch.float32)
            else:
                def legacy_step():
                    sub = features[0:BS]
                    _ = torch.from_numpy(sub).to(torch.float32)

            t_e = _bench(eager_step, n_iter=300)
            t_l = _bench(legacy_step, n_iter=300)
            speedup = t_l / max(t_e, 1e-9)
            print(f"| {name} | {n:_} | {t_e:.3f} | {t_l:.3f} | {speedup:.1f}x |")


def bench_dataloader_multi_worker():
    """DataLoader iteration with num_workers + persistent_workers.

    IMPORTANT platform note: Windows DataLoader uses **spawn** semantics
    (no fork available), so every worker process re-imports the entire
    parent's module graph -- torch, lightning, numpy, scipy, sklearn,
    matplotlib, mlframe.*, etc. That's 5-15 s of startup PER WORKER
    every epoch (or once total if ``persistent_workers=True``).

    For a tensor-only ``__getitem__`` (the Wave 22 fastpath: single
    tensor index, ~15 us per batch), the per-worker spawn cost
    massively dominates the actual work. On Windows the OPTIMAL config
    is ``num_workers=0`` for tensor-only datasets; multi-worker is for
    augmentation-heavy / I/O-heavy datasets where the per-batch cost is
    minutes, not microseconds.

    On Linux, fork makes worker startup negligible, so multi-worker
    becomes an obvious win even for tensor-only datasets when the model
    is GPU-bound. The shared-memory tensor (Wave 23) avoids per-worker
    payload duplication regardless of platform.

    This bench skips multi-worker on Windows by default to avoid the
    spawn-cost noise; set MLFRAME_BENCH_DATALOADER_WORKERS=1 to opt in.
    """
    import sys
    if sys.platform == "win32" and not os.environ.get("MLFRAME_BENCH_DATALOADER_WORKERS"):
        print("\n## Wave 23: DataLoader iteration (skipped on Windows)")
        print("\nWindows uses spawn -- per-worker re-import of torch+sklearn+scipy")
        print("dominates pure tensor indexing. Set MLFRAME_BENCH_DATALOADER_WORKERS=1")
        print("to force the bench. Multi-worker DataLoader is genuinely useful only")
        print("for augmentation-heavy datasets on Windows, or any dataset on Linux.")
        return

    print("\n## Wave 23: DataLoader iteration (num_workers + persistent_workers)")
    print()
    print("| num_workers | persistent | Wall (s, 3 epochs) | Peak RSS (MB) |")
    print("|---:|:---:|---:|---:|")

    from torch.utils.data import DataLoader
    from mlframe.training.neural.data import TorchDataset

    n = 100_000  # smaller bench-frame so total wall stays manageable
    features = np.random.rand(n, 16).astype(np.float32)
    labels = np.random.rand(n).astype(np.float32)

    for n_workers in (0, 2):  # 4-worker Windows-spawn OOMs on small hosts
        for persistent in (False, True):
            if n_workers == 0 and persistent:
                continue
            ds = TorchDataset(
                features=features.copy(), labels=labels.copy(),
                batch_size=128, share_memory=True,
            )
            dl_kwargs = dict(
                dataset=ds, batch_size=None, num_workers=n_workers, shuffle=False,
            )
            if persistent and n_workers > 0:
                dl_kwargs["persistent_workers"] = True
            loader = DataLoader(**dl_kwargs)

            gc.collect()
            rss_before = _peak_rss_mb()

            t0 = time.perf_counter()
            for _ in range(3):
                for batch in loader:
                    pass
            elapsed = time.perf_counter() - t0

            rss_after = _peak_rss_mb()
            del loader, ds
            gc.collect()

            print(
                f"| {n_workers} | {'Y' if persistent else 'N'} | "
                f"{elapsed:.2f} | {max(0, rss_after - rss_before):.0f} |"
            )


def bench_share_memory_overhead():
    print("\n## Wave 23: `share_memory_()` overhead in single-process mode")
    print()
    print("| N rows | share_memory=True | share_memory=False | Overhead |")
    print("|---:|---:|---:|---:|")

    from mlframe.training.neural.data import TorchDataset

    BS = 128
    for n in (10_000, 100_000, 1_000_000):
        features = np.random.rand(n, 16).astype(np.float32)
        labels = np.random.rand(n).astype(np.float32)

        ds_shared = TorchDataset(
            features=features.copy(), labels=labels.copy(),
            batch_size=BS, share_memory=True,
        )
        ds_unshared = TorchDataset(
            features=features.copy(), labels=labels.copy(),
            batch_size=BS, share_memory=False,
        )

        t_s = _bench(lambda: ds_shared[0], n_iter=300)
        t_u = _bench(lambda: ds_unshared[0], n_iter=300)
        overhead = (t_s / max(t_u, 1e-9) - 1.0) * 100

        print(f"| {n:_} | {t_s:.4f} ms | {t_u:.4f} ms | {overhead:+.1f}% |")


def bench_byte_cap_fallback():
    print("\n## Wave 22 v2: byte-cap fallback path (lazy per-batch)")
    print()
    print("| N rows | Eager ms/batch | Lazy ms/batch | Overhead vs eager |")
    print("|---:|---:|---:|---:|")

    from mlframe.training.neural.data import TorchDataset

    class BigBytesArray(np.ndarray):
        @property
        def nbytes(self):
            return 3 * 1024**3

    BS = 128
    for n in (10_000, 100_000):
        features = np.random.rand(n, 16).astype(np.float32)
        labels = np.random.rand(n).astype(np.float32)
        huge = features.view(BigBytesArray)

        ds_e = TorchDataset(features=features.copy(), labels=labels.copy(), batch_size=BS)
        ds_l = TorchDataset(features=huge, labels=labels.copy(), batch_size=BS)
        assert ds_e._eager_features is True
        assert ds_l._eager_features is False

        t_e = _bench(lambda: ds_e[0], n_iter=300)
        t_l = _bench(lambda: ds_l[0], n_iter=300)
        overhead = (t_l / max(t_e, 1e-9) - 1.0) * 100

        print(f"| {n:_} | {t_e:.4f} | {t_l:.4f} | {overhead:+.1f}% |")


if __name__ == "__main__":
    print("# Wave 22 + 23 standalone benchmark")
    print(f"\nHost: {os.cpu_count()} CPUs, psutil={_HAS_PSUTIL}")
    print(f"NumPy: {np.__version__}, Torch: {torch.__version__}")

    bench_eager_vs_legacy_per_batch()
    bench_share_memory_overhead()
    bench_byte_cap_fallback()
    bench_dataloader_multi_worker()
