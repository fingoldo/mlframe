"""cProfile of one MRMR fit in GPU strict-resident mode on novel data (the F2 example).

Usage: ``python profile_fit_gpu.py <n_rows>``. Runs a warm-up fit and two timed cold-data fits (prints their wall), then profiles a third and writes the stats to
``PROF_FILE`` (``MLFRAME_PROFILE_DIR``/gpu_fit.prof); the other profiling scripts read that file. Run it on a quiet box: a concurrent GPU job moves every number.
"""
import cProfile, pstats, os, sys, time, io, warnings
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # this checkout, not an installed copy
from mlframe.feature_selection._benchmarks.profiling._paths import PROF_FILE  # noqa: E402
warnings.simplefilter("ignore")
for k, v in {
    "MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1",
    "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1", "PYTHONUNBUFFERED": "1",
}.items():
    os.environ[k] = v
import cupy as cp
from mlframe.feature_selection._benchmarks.profiling._fe_fit_fixture import fit as _fe_fit

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000


fit = functools.partial(_fe_fit, n=n)


fit(1)  # warm: jit + KTC + cupy kernels
for sd in (2, 3):
    cp.cuda.Device().synchronize()
    t = time.perf_counter()
    fit(sd)
    cp.cuda.Device().synchronize()
    print("cold-data fit seed", sd, "wall", round(time.perf_counter() - t, 2), "s")
pr = cProfile.Profile()
pr.enable()
t = time.perf_counter()
fs = fit(4)
cp.cuda.Device().synchronize()
pr.disable()
pr.dump_stats(str(PROF_FILE))
print("profiled wall", round(time.perf_counter() - t, 2), "s", [str(s) for s in fs.get_feature_names_out()])
for sort in ("tottime", "cumtime"):
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(28)
    print("=== by", sort)
    print("\n".join(l[:200] for l in s.getvalue().splitlines()[5:40]))
