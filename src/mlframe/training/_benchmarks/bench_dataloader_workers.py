"""Benchmark: does size-aware num_workers > 0 in MLP DataLoader give a
real speedup vs the current num_workers=0 default?

Context: the current default of num_workers=0 was set as a safety measure
because TorchDataset holds the entire frame in its closure (pickled per worker
on Windows / CoW-broken on Linux). On HUGE frames (10+ GB) any value > 0 is a
memory landmine. But on SMALL frames (< 2 GB) the data loading single-thread
becomes the bottleneck.

This bench varies (n_rows, n_features, num_workers) and reports
batches-per-second + wall time for one epoch.

Usage:
    python -m mlframe.training._benchmarks.bench_dataloader_workers

Only runs when torch is importable. Skip silently otherwise.
"""
from __future__ import annotations

import os
import time
from typing import Tuple

try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    _TORCH_OK = True
except Exception as e:
    print(f"torch import failed: {e}; skipping bench")
    _TORCH_OK = False


def _make_dataset(n_rows: int, n_features: int) -> "TensorDataset":
    """Build a tensor-backed dataset of given shape."""
    X = torch.randn(n_rows, n_features, dtype=torch.float32)
    y = torch.randn(n_rows, 1, dtype=torch.float32)
    return TensorDataset(X, y)


def _bench_one(n_rows: int, n_features: int, batch_size: int, num_workers: int) -> Tuple[float, float]:
    """Return (wall_seconds, batches_per_sec) for one full pass through the
    dataset at the given num_workers setting."""
    ds = _make_dataset(n_rows, n_features)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
        shuffle=False,
    )
    t0 = time.perf_counter()
    n_batches = 0
    for _x, _y in dl:
        n_batches += 1
    elapsed = time.perf_counter() - t0
    return elapsed, n_batches / elapsed if elapsed > 0 else float("inf")


def main() -> None:
    if not _TORCH_OK:
        return

    print("=" * 80)
    print(f"DataLoader num_workers bench (cpu_count={os.cpu_count()})")
    print("=" * 80)

    shapes = [
        # (n_rows, n_features, batch_size, label)
        (10_000, 25, 1024, "tiny narrow"),
        (100_000, 25, 4096, "small narrow"),
        (1_000_000, 25, 16384, "medium narrow"),
        (1_000_000, 1000, 1024, "medium wide"),
    ]
    worker_choices = [0, 2, 4]

    print(f"{'shape':<20} {'workers':>8} {'wall_s':>10} {'batches/s':>12} {'speedup':>10}")
    print("-" * 80)
    for n_rows, n_features, batch_size, label in shapes:
        baseline = None
        for nw in worker_choices:
            wall, bps = _bench_one(n_rows, n_features, batch_size, nw)
            if baseline is None:
                baseline = wall
                speedup = 1.0
            else:
                speedup = baseline / wall if wall > 0 else float("inf")
            print(
                f"{label:<20} {nw:>8} {wall:>10.3f} {bps:>12.1f} {speedup:>9.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
