"""Peak-memory diagnostic for MRMR.fit at large n -- pinpoints the OOM driver.

Background: a 1M-row fit OOMs on a 16GB box (bench_scaling: peak 9.87GB at 400k -> ~25GB projected
at 1M). The FE path is suspected to hold the discretised screening matrix of (n, raw + many
candidates) at once. This harness measures, for a single MRMR.fit:
  * peak process RSS (a daemon thread polls psutil every 50ms across the fit), and
  * the top Python allocations by traceback at peak (tracemalloc snapshot) -- which names the
    file:line where the largest blocks live, so a streaming fix can be aimed precisely.

tracemalloc only sees Python/numpy allocations made through the CPython allocator (it MISSES some
C-level / cupy buffers), so cross-check the tracemalloc total against the RSS peak: a large gap means
the big buffer is native (numba/cupy/arrow) and the traceback list points at the Python site that
triggered it.

Run on a machine with enough RAM for the target n (the dev box OOMs at 1M -- this is meant for the
big machine). CPU-only + sweep disabled for a clean, deterministic profile:

  CUDA_VISIBLE_DEVICES='' NUMBA_DISABLE_CUDA=1 PYUTILZ_KERNEL_DISABLE_SWEEP=1 MLFRAME_DISABLE_HNSW=1 \
  MLFRAME_DISABLE_GPU=1 python -m mlframe.feature_selection.filters._benchmarks.bench_fe_peak_memory --n 1000000

Not a pytest target (no asserts) -- a manual diagnostic.
"""
from __future__ import annotations

import argparse
import threading
import time
import tracemalloc

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _make_case(n: int, n_noise: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.uniform(1.0, 5.0, n) for _ in range(6))
    y = a**2 / b + f / 5.0 + 3.0 * np.log(c) * np.sin(d)
    cols = {"a": a, "b": b, "c": c, "d": d, "e": e}
    for j in range(n_noise):
        cols[f"n{j}"] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


class _RSSPoller(threading.Thread):
    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        import psutil  # local import so the module loads without psutil
        self._proc = psutil.Process()

    def run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / 1e6
            if rss > self.peak_mb:
                self.peak_mb = rss
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--n-noise", type=int, default=10)
    ap.add_argument("--fe-max-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20, help="top-N tracemalloc allocation sites to print")
    args = ap.parse_args()

    df, y = _make_case(args.n, args.n_noise, args.seed)
    base_mb = None
    try:
        import psutil
        base_mb = psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass

    poller = _RSSPoller()
    poller.start()
    tracemalloc.start(25)
    t0 = time.perf_counter()
    fs = MRMR(
        verbose=0, random_seed=args.seed, fe_max_steps=args.fe_max_steps,
        dcd_enable=False, build_friend_graph=False, cluster_aggregate_enable=False,
    ).fit(df, y)
    wall = time.perf_counter() - t0
    snap = tracemalloc.take_snapshot()
    poller.stop()
    tracemalloc.stop()

    print(f"n={args.n} fe_max_steps={args.fe_max_steps} noise={args.n_noise}")
    print(f"WALL = {wall:.1f}s")
    if base_mb is not None:
        print(f"baseline RSS (data loaded) = {base_mb:,.0f} MB")
    print(f"PEAK process RSS during fit  = {poller.peak_mb:,.0f} MB")
    tm_total = sum(s.size for s in snap.statistics("filename")) / 1e6
    print(f"tracemalloc end-of-fit total = {tm_total:,.0f} MB (Python/numpy only; gap vs RSS = native buffers)")
    print(f"selected: {list(fs.get_feature_names_out())[:8]}")
    print(f"\n=== top {args.top} allocation sites by size (tracemalloc, traceback) ===")
    for st in snap.statistics("traceback")[: args.top]:
        frame = st.traceback[0]
        print(f"  {st.size/1e6:8.1f} MB  x{st.count:<7} {frame.filename.split('mlframe')[-1]}:{frame.lineno}")


if __name__ == "__main__":
    main()
