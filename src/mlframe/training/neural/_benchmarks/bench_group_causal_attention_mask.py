"""cProfile harness for ``training.neural.group_causal_attention_mask.group_causal_attention_mask``.

Run: ``python -m mlframe.training.neural._benchmarks.bench_group_causal_attention_mask``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import torch

from mlframe.training.neural.group_causal_attention_mask import group_causal_attention_mask


def _make_group_ids(batch_size: int, seq_len: int, n_groups: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    raw = torch.randint(0, n_groups, (batch_size, seq_len))
    return torch.sort(raw, dim=1).values  # non-decreasing per row, matching the documented contract


def _run(batch_size: int, seq_len: int, n_groups: int, n_calls: int) -> None:
    group_ids = _make_group_ids(batch_size, seq_len, n_groups, seed=0)
    for _ in range(n_calls):
        group_causal_attention_mask(group_ids)


if __name__ == "__main__":
    for batch_size, seq_len, n_groups, n_calls in [(32, 64, 20, 200), (256, 128, 50, 200), (256, 512, 100, 50)]:
        t0 = time.perf_counter()
        _run(batch_size, seq_len, n_groups, n_calls)
        wall = time.perf_counter() - t0
        print(f"batch={batch_size:>4} seq_len={seq_len:>4} n_groups={n_groups:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(256, 512, 100, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
