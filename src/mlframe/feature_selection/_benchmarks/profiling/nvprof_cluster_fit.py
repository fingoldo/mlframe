"""One warm MRMR fit of the clustering fixture inside the CUDA profiler window (``nvprof --profile-from-start off``), the fixture of ``profile_cluster_correlated.py``.

Usage: ``nvprof --profile-from-start off --print-gpu-summary python nvprof_cluster_fit.py --n 20000 --k 20``  (``export PATH="$PATH:/d/CUDA/v12.8/extras/CUPTI/lib64"`` first). The first fit warms the numba / NVRTC caches and is not profiled.
"""

from __future__ import annotations

import argparse
import os
import sys

for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "PYTHONUNBUFFERED": "1"}.items():
    os.environ[_k] = _v
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cupy as cp

import profile_cluster_correlated as P


def main() -> None:
    """Warm fit, then one fit inside the profiler window."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--generator", default="two_groups")
    ap.add_argument("--noise", type=float, default=0.4)
    a = ap.parse_args()
    X, y = P.make_data(a.generator, a.n, a.k, 0, a.noise)
    P.fit_once(X, y, "on", 0)
    cp.cuda.Device().synchronize()
    cp.cuda.runtime.profilerStart()
    wall, names, _ = P.fit_once(X, y, "on", 1)
    cp.cuda.Device().synchronize()
    cp.cuda.runtime.profilerStop()
    print(f"PROFILED_FIT wall_s={wall:.1f} selected={len(names)} columns={X.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
