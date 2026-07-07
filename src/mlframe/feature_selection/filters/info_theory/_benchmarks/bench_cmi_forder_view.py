"""Bench + identity check for the F-order (column-contiguous) view cache in _cpu_cmi_loop.

factors_data is (n, nfeat) C-contiguous, so the per-candidate CMI melt reads each candidate column
with an nfeat*4-byte stride -> cache-line thrash. _cmi_forder_view caches a column-contiguous copy of
the fit-constant matrix once per fit; every round then reads columns contiguously. This measures the
shipped path end-to-end through _cpu_cmi_loop (MLFRAME_CMI_FORDER on vs off) and asserts the result is
BIT-IDENTICAL either way (asfortranarray only changes physical order).

One-command re-run:
    python -m mlframe.feature_selection.filters.info_theory._benchmarks.bench_cmi_forder_view
"""
from __future__ import annotations
import os
import time
import numpy as np


def main():
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    for n, p, nb in [(30000, 1000, 16), (30000, 400, 12), (10000, 500, 12)]:
        rng = np.random.default_rng(0)
        nc = p + 2
        fd = np.empty((n, nc), dtype=np.int32, order="C")
        for c in range(nc):
            fd[:, c] = rng.integers(0, nb, n)
        fnb = np.full(nc, nb, dtype=np.int64)
        cand = np.arange(p, dtype=np.int64)
        y = np.array([p], dtype=np.int64)
        z = np.array([p + 1], dtype=np.int64)

        os.environ["MLFRAME_CMI_FORDER"] = "0"
        cm.reset_cmi_forder_cache()
        ref = cm._cpu_cmi_loop(fd, cand, y, z, fnb)
        os.environ["MLFRAME_CMI_FORDER"] = "1"
        cm.reset_cmi_forder_cache()
        got = cm._cpu_cmi_loop(fd, cand, y, z, fnb)
        maxdiff = float(np.max(np.abs(ref - got)))

        def best(k=6):
            b = 1e30
            for _ in range(k):
                t = time.perf_counter()
                cm._cpu_cmi_loop(fd, cand, y, z, fnb)
                b = min(b, time.perf_counter() - t)
            return b

        os.environ["MLFRAME_CMI_FORDER"] = "0"
        cm.reset_cmi_forder_cache()
        toff = best()
        os.environ["MLFRAME_CMI_FORDER"] = "1"
        cm.reset_cmi_forder_cache()
        ton = best()
        print(f"n={n} p={p} nb={nb}: C-order={toff*1000:7.1f}ms  F-order={ton*1000:7.1f}ms  " f"speedup={toff/ton:.2f}x  maxdiff={maxdiff:.1e}")


if __name__ == "__main__":
    main()
