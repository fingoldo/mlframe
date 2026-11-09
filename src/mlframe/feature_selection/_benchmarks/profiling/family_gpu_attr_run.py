import os, sys, time, warnings, pickle
warnings.simplefilter("ignore")

os.environ["MLFRAME_FE_STAGE_CONCURRENCY"] = "0"
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items(): os.environ.setdefault(k, v)
import cupy as cp
from mlframe.feature_selection.filters import _fe_family_timing as FT
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
LOG = []
class Slot(list):
    def __init__(self, name): super().__init__([0.0, 0]); self.name = name
    def __setitem__(self, i, v):
        if i == 0:
            d = v - self[0]; LOG.append((self.name, time.perf_counter() - d, time.perf_counter()))
        super().__setitem__(i, v)
class Led(dict):
    def __missing__(self, k): s = self[k] = Slot(k); return s
FT._FE_FAMILY_WALL = Led()
MRMR(verbose=0, random_seed=0, n_jobs=1).fit(*F._make(20000, 30, 1))
data = F._make(100000, 100, 0)
MRMR._FIT_CACHE.clear(); LOG.clear()
anchor = cp.ElementwiseKernel("float32 x", "float32 y", "y = x + 1", "anchor_kernel_zz")
a = cp.ones(1000, cp.float32); anchor(a, a); cp.cuda.Device().synchronize(); t_anchor = time.perf_counter()
t = time.perf_counter(); m = MRMR(verbose=0, random_seed=0, n_jobs=1).fit(*data); w = time.perf_counter() - t
cp.cuda.Device().synchronize()
print("WALL", round(w, 1))
pickle.dump({"t_anchor": t_anchor, "log": LOG}, open("D:/Temp/fam_gpu_log.pkl", "wb"))
