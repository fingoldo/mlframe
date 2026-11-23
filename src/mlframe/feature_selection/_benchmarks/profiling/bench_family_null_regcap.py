"""A/B of the register cap of the family null kernel ``family_null_bits_v3`` (80 registers, 64 threads per block, achieved occupancy 0.369 by ``nvprof`` on the 20000 x 43 clustering fit).

The occupancy is register-limited (65536 / 80 = 819 threads per SM, 12 blocks of 64 = 37.5 percent of 2048). The kernel is latency-bound (memory-dependency stalls 61 percent, DRAM read 8.7 GB/s of 112,
double-precision efficiency 1.4 percent), so a lower register count may help; a spill would not. Interleaved rounds through the real entry (``_FamilyOnDevice.null_max``), synchronised per call.

Usage: ``python bench_family_null_regcap.py --n 20000 --k 80 --perms 64 --caps 0,64,48,40 --rounds 7``  (0 = the default compile).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any

os.environ.setdefault("MLFRAME_FE_GPU_STRICT", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _family_maxt_cuda_bits as bits
from mlframe.feature_selection.filters import _family_maxt_gate_device as dev


def compile_with_cap(cap: int):
    """Replace the v3 kernel getter by one that compiles with ``--maxrregcount=cap`` (0 keeps the default) and forgets the cache."""
    bits._kernels_v3.clear()

    def getter(_cp, ny1, pb, cap_bins, real):
        """Compile one v3 variant with the register cap under test."""
        opts = (f"-DNY1={int(ny1)}", f"-DPB={int(pb)}", f"-DCAP={int(cap_bins)}") + ((f"--maxrregcount={cap}",) if cap else ())
        key = (ny1, pb, cap_bins, real, cap)
        kern = bits._kernels_v3.get(key)
        if kern is None:
            kern = cp.RawModule(code=bits._SOURCE_V3.replace("{REAL}", real).replace("{ACC}", "double"), options=opts).get_function("family_null_bits_v3")
            bits._kernels_v3[key] = kern
        return kern

    bits._get_kernel_v3 = getter


def main() -> None:
    """Interleaved rounds over the register caps; prints the median wall per call and the kernel's register count."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--perms", type=int, default=64)
    ap.add_argument("--caps", default="0,64,48,40")
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    codes = cp.asarray(rng.integers(0, a.bins, size=(a.n, a.k)).astype(np.int8))
    y = np.ascontiguousarray((rng.random(a.n) > 0.5).astype(np.int64))
    caps = [int(c) for c in a.caps.split(",")]
    times: "dict[int, list[float]]" = {c: [] for c in caps}
    ref: "dict[int, Any]" = {}
    for rnd in range(a.rounds + 1):
        for c in caps:
            compile_with_cap(c)
            fam = dev._FamilyOnDevice(cp, [codes], y, 2)
            cp.cuda.Device().synchronize()
            t = time.perf_counter()
            out = fam.null_max(a.perms, np.random.default_rng(1))
            cp.cuda.Device().synchronize()
            dt = time.perf_counter() - t
            if rnd:
                times[c].append(dt)
            ref.setdefault(c, out)
    base = statistics.median(times[caps[0]])
    for c in caps:
        same = bool(np.array_equal(ref[c], ref[caps[0]]))
        print(f"cap={c or 'default'} median {statistics.median(times[c]) * 1e3:.2f} ms  ratio {statistics.median(times[c]) / base:.3f}  identical_output={same}")


if __name__ == "__main__":
    main()
