"""The profiled fit with the CUDA profiler window opened only around one warm fit (``nvprof --profile-from-start off``)."""
import cProfile, pstats, os, sys, time, io, warnings
import functools
import sys
from pathlib import Path

_root = next((q for q in Path(__file__).resolve().parents if (q / "mlframe").is_dir()), None)  # this checkout, not an installed copy; a generated copy outside the package has none and relies on PYTHONPATH
if _root is not None:
    sys.path.insert(0, str(_root))
warnings.simplefilter("ignore")
for k, v in {
    "MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1",
    "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1", "PYTHONUNBUFFERED": "1",
}.items():
    os.environ[k] = v
import numpy as np, pandas as pd, cupy as cp
from mlframe.feature_selection._benchmarks.profiling._fe_fit_fixture import fit as _fe_fit

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000


fit = functools.partial(_fe_fit, n=n)


fit(1)
cp.cuda.Device().synchronize()
cp.cuda.runtime.profilerStart()
fit(2)
cp.cuda.Device().synchronize()
cp.cuda.runtime.profilerStop()
