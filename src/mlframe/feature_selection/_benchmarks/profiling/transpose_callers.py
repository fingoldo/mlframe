"""Callers of the radix select's row-to-column-major transpose in one warm GPU fit, with the shapes and the wall of each call."""
import collections, inspect, os, sys, time, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
import cupy as cp
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters import _gpu_resident_select as S

kw = F.all_families_kwargs()
kw["nbins_strategy"] = os.environ.get("BENCH_NBINS", "mdlp")
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))

stats = collections.defaultdict(lambda: [0, 0.0, 0])
real = S._transpose_to_cm


def wrapped(x, *a, **k):
    frames = inspect.stack()
    chain = " < ".join(f"{os.path.basename(f.filename)}:{f.function}" for f in frames[1:4])
    cp.cuda.Stream.null.synchronize()
    t = time.perf_counter()
    out = real(x, *a, **k)
    cp.cuda.Stream.null.synchronize()
    d = time.perf_counter() - t
    st = stats[chain]
    st[0] += 1
    st[1] += d
    st[2] += int(x.size) * int(x.dtype.itemsize)
    return out


S._transpose_to_cm = wrapped
data = F._make(100000, 100, 0)
t0 = time.perf_counter()
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
print("WALL", round(time.perf_counter() - t0, 1))
for chain, (n, d, nb) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
    print(f"{n:4d} calls {d:6.2f} s {nb / n / 1e6:7.1f} MB each  {chain}")
