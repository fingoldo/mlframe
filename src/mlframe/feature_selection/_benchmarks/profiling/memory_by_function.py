"""Which functions hold the host memory at the peak of an MRMR fit (PLAN-10): ``tracemalloc`` snapshots taken when the process RSS reaches a new peak, grouped by the allocating line.

``rss_scale_fit.py`` names the innermost frame of the MAIN thread at each new peak, but the memory of a fit with concurrent stages is held by worker threads, so that name says little. numpy registers its array
allocations with ``tracemalloc``, so a snapshot at the peak lists the code lines that own the live bytes, whichever thread allocated them.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.memory_by_function ROWS COLS [all|default|nofe]`` (env ``MIN_STEP_MB`` the growth that triggers a snapshot, default 400; ``TOP`` default 18; ``FRAMES`` the stack depth kept per allocation, default 1: with more, each row lists the calling lines). Tracing slows the
fit by a factor of two to three: use a smaller frame than the one under study (the allocation sites do not change with the row count) and nothing else heavy beside it.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tracemalloc
import warnings

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1"}.items():
    os.environ.setdefault(_k, _v)


def main() -> None:
    """Fit once under ``tracemalloc``; print the live bytes by allocating line at the highest RSS the sampler saw."""
    import numpy as np
    import pandas as pd
    import psutil

    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    n, p = int(sys.argv[1]), int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    min_step = int(os.environ.get("MIN_STEP_MB", "400")) * 1024 * 1024
    top = int(os.environ.get("TOP", "18"))
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, p), dtype=np.float32), columns=[f"x{i}" for i in range(p)])
    y = pd.Series((X["x0"].to_numpy() * X["x1"].to_numpy() + 0.7 * X["x2"].to_numpy() + 0.5 * rng.standard_normal(n, dtype=np.float32) > 0).astype(np.int8))
    kw = F.all_families_kwargs() if mode == "all" else ({"fe_max_steps": 0} if mode == "nofe" else {})  # nofe: no feature engineering, the categorisation and the screen only (the early phase of a fit)
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = {"rss": base, "snap": None, "t": 0.0}
    stop = threading.Event()
    t0 = time.perf_counter()

    def watch() -> None:
        """Poll the RSS; at every new peak above the previous snapshot by ``min_step`` take a ``tracemalloc`` snapshot (the last one is the peak's)."""
        last_snap_rss = base
        while not stop.wait(0.25):
            rss = proc.memory_info().rss
            peak["rss"] = max(peak["rss"], rss)
            if rss > last_snap_rss + min_step:
                peak["snap"] = tracemalloc.take_snapshot()
                peak["t"] = time.perf_counter() - t0
                last_snap_rss = rss

    frames = int(os.environ.get("FRAMES", "1"))
    tracemalloc.start(frames)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    MRMR(verbose=0, random_seed=0, n_jobs=1, skip_retraining_on_same_content=False, **kw).fit(X, y)
    stop.set()
    th.join()
    wall = time.perf_counter() - t0
    print(f"MEM rows={n} cols={p} mode={mode} wall {wall:.0f}s (traced); RSS growth at the highest peak {(peak['rss'] - base) / 2**20:.0f} MB; snapshot of the last new peak taken at t={peak['t']:.0f}s", flush=True)
    snap = peak["snap"]
    if snap is None:
        print("MEM no snapshot: the RSS never grew by MIN_STEP_MB")
        return
    stats = snap.statistics("lineno" if frames == 1 else "traceback")
    total = sum(s.size for s in stats)
    print(f"MEM traced live bytes in that snapshot {total / 2**20:.0f} MB (the RSS growth also holds untraced native and device-driver memory)")
    for s in stats[:top]:
        fr = s.traceback[-1] if frames > 1 else s.traceback[0]
        print(f"MEM {s.size / 2**20:8.1f} MB  {s.count:7d} blocks  {os.path.relpath(fr.filename).replace(chr(92), '/')[-70:]}:{fr.lineno}", flush=True)
        if frames > 1:
            for caller in list(s.traceback)[:-1][-int(os.environ.get("CALLERS", "4")):][::-1]:
                print(f"MEM      called from {os.path.relpath(caller.filename).replace(chr(92), '/')[-60:]}:{caller.lineno}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))
    main()
