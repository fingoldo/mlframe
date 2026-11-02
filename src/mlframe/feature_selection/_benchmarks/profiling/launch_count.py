"""Count the launches of our compiled CUDA kernels in one every-family fit, by kernel and by calling function.

``python -m mlframe.feature_selection._benchmarks.profiling.launch_count 100000 100``: a warm fit at 20000 x 30, then a counted fit. Only ``cupy.RawKernel`` / ``RawModule`` kernels are counted (cupy's own elementwise kernels, sorts and
reductions are cdef types whose ``__call__`` cannot be patched). Use it instead of an nsys export of a whole fit, which writes a multi-GB sqlite (see ``audits/2026-10-22/profiles/launch_counts_100k_x_100.md``).
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
import warnings
from typing import Any

_STRICT_ENV = {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}


def _install(cnt: "collections.Counter", where: "collections.Counter") -> None:
    """Patch ``RawKernel`` and ``RawModule.get_function`` so that every launch is counted by kernel name and by the nearest mlframe frame."""
    import cupy as cp
    from cupy._core import raw

    lock = threading.Lock()

    def caller() -> str:
        """The mlframe function that issued the launch."""
        frame = sys._getframe(3)
        while frame is not None and "mlframe" not in frame.f_code.co_filename:
            frame = frame.f_back
        return (frame.f_code.co_filename.split("filters")[-1].lstrip(r"\/") + ":" + frame.f_code.co_name) if frame else "?"

    class CountingKernel:
        """Proxy that counts the launches of a compiled kernel."""

        def __init__(self, kernel: Any, name: str) -> None:
            self._k = kernel
            self._n = name

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            with lock:
                cnt[self._n] += 1
                where[caller()] += 1
            return self._k(*args, **kwargs)

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._k, attr)

    base_kernel = cp.RawKernel

    class CountedRawKernel(base_kernel):  # type: ignore[misc, valid-type]
        """``RawKernel`` whose launches are counted."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            with lock:
                cnt[str(getattr(self, "name", "?"))] += 1
                where[caller()] += 1
            return super().__call__(*args, **kwargs)

    cp.RawKernel = CountedRawKernel
    raw.RawKernel = CountedRawKernel
    base_module = raw.RawModule

    class CountedRawModule(base_module):  # type: ignore[misc, valid-type]
        """``RawModule`` whose functions count their launches."""

        def get_function(self, name: str) -> Any:
            return CountingKernel(super().get_function(name), name)

    cp.RawModule = CountedRawModule
    raw.RawModule = CountedRawModule


def main(n: int = 100000, p: int = 100) -> int:
    """Run the counted fit and print the totals; returns a process exit code."""
    warnings.simplefilter("ignore")
    os.environ.update(_STRICT_ENV)
    cnt: collections.Counter = collections.Counter()
    where: collections.Counter = collections.Counter()
    _install(cnt, where)
    from mlframe.feature_selection._benchmarks.profiling import family_scale as scale
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = scale.all_families_kwargs()
    kwargs["nbins_strategy"] = "mdlp"
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kwargs).fit(*scale._make(20000, 30, 1))
    cnt.clear()
    where.clear()
    start = time.perf_counter()
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kwargs).fit(*scale._make(n, p, 0))
    print(f"WALL {time.perf_counter() - start:.1f}")
    print("total launches", sum(cnt.values()))
    for name, count in cnt.most_common(14):
        print(f"{count:8d} {name[:70]}")
    print("--- by mlframe caller")
    for name, count in where.most_common(18):
        print(f"{count:8d} {name[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 100000, int(sys.argv[2]) if len(sys.argv) > 2 else 100))
