"""The profiled fit with the CUDA profiler window opened only around one warm fit (``nvprof --profile-from-start off``)."""
import cProfile, pstats, os, sys, time, io, warnings
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
from mlframe.feature_selection.filters.mrmr import MRMR

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000


def make(seed):
    """Novel data per call: MRMR memoises by content hash, so a repeated frame would time the cache replay."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.uniform(0.1, 1.1, n) for _ in range(6))
    df = pd.DataFrame({k: v for k, v in zip("abcde", (a, b, c, d, e))})
    y = a**2 / b + f / 5.0 + np.log(np.abs(c) + 1e-9) * np.sin(d)
    return df, y


def fit(seed):
    df, y = make(seed)
    return MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1, random_seed=seed).fit(df, y)



fit(1)
cp.cuda.Device().synchronize()
cp.cuda.runtime.profilerStart()
fit(2)
cp.cuda.Device().synchronize()
cp.cuda.runtime.profilerStop()
