"""All-thread sampler of the all-families fit: where EVERY thread of the stage workers spends its time, not only the main thread.

The main-thread sampler (``all_families_stage_sampler``) and cProfile see one thread, and in the FE cascades the main thread only waits for the stage workers (12 of 33 s at 100000 x 100). This one reads ``sys._current_frames()``
of every thread every ``SAMPLE_S`` seconds and records, per thread, the innermost frame that lies in an mlframe module (``where``) and whether the thread is blocked (the innermost frame of any module is a lock / event /
queue wait: the thread is idle, not working).

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.all_thread_sampler N P`` (env ``SS_NBINS`` default quantile, ``SAMPLE_S`` default 0.02, ``TOP`` default 30). A warm-up fit comes first.
Output: the wall, the thread-seconds by innermost mlframe function split into working and blocked, and the mean number of working threads. Thread-seconds above the wall mean the threads overlapped; a function with
many working thread-seconds and a low CPU / GPU load in the tick timeline is a host loop that holds a worker (the lever for the near-100 % occupancy goal).
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
import warnings
from mlframe.feature_selection._benchmarks.binning_label import binning_params

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)

_BLOCKING = ("threading.py", "queue.py", "_base.py", "thread.py", "selectors.py")
_BLOCKING_FUNCS = {"wait", "acquire", "_wait_for_tstate_lock", "get", "result", "join", "_worker", "select"}


def _is_blocked(frame) -> bool:
    """Whether the innermost frame is a lock, event, queue or executor wait."""
    name = os.path.basename(frame.f_code.co_filename)
    return name in _BLOCKING and frame.f_code.co_name in _BLOCKING_FUNCS


def _innermost_mlframe(frame) -> str:
    """``module:function`` of the innermost frame inside mlframe, or an empty string when the stack has none."""
    while frame is not None:
        path = frame.f_code.co_filename.replace("\\", "/")
        if "/mlframe/" in path:
            return f"{os.path.basename(path)}:{frame.f_code.co_name}"
        frame = frame.f_back
    return ""


def main() -> None:
    """Warm-up fit, then the measured fit with the sampler running; prints the ranking."""
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F
    from mlframe.feature_selection.filters.mrmr import MRMR

    n, p = int(sys.argv[1]), int(sys.argv[2])
    sample_s = float(os.environ.get("SAMPLE_S", "0.02"))
    top = int(os.environ.get("TOP", "30"))
    kw = F.all_families_kwargs()
    kw.update(binning_params(os.environ.get("SS_NBINS", "quantile")))
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
    data = F._make(n, p, 2)
    working: "collections.Counter[str]" = collections.Counter()
    blocked: "collections.Counter[str]" = collections.Counter()
    live_working: "list[int]" = []
    stop = threading.Event()
    me = threading.get_ident()

    def sample() -> None:
        """Sample every thread until ``stop``."""
        while not stop.wait(sample_s):
            n_work = 0
            for tid, frame in sys._current_frames().items():
                if tid == me or tid == threading.get_ident():
                    continue
                where = _innermost_mlframe(frame)
                if not where:
                    continue
                if _is_blocked(frame):
                    blocked[where] += 1
                else:
                    working[where] += 1
                    n_work += 1
            live_working.append(n_work)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    print(f"WALL {wall:.1f}  samples {len(live_working)}  mean working threads {sum(live_working) / max(1, len(live_working)):.2f}", flush=True)
    print("--- thread-seconds WORKING by innermost mlframe function")
    for where, c in working.most_common(top):
        print(f"{c * sample_s:7.2f}  {where}")
    print("--- thread-seconds BLOCKED by innermost mlframe function")
    for where, c in blocked.most_common(10):
        print(f"{c * sample_s:7.2f}  {where}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))
    main()
