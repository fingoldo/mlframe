"""Same-process A/B: the single-column resident Fourier detector in a loop against detect_fourier_freqs_batch (100 columns x 100000 rows)."""
import os, time, numpy as np
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_FE_GPU_STRICT_RESIDENT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1"}.items(): os.environ.setdefault(k, v)
import cupy as cp
from mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_detect_batch import detect_fourier_freqs_batch
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import _detect_fourier_freqs_for_col
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
n = int(os.environ.get("N", "100000")); ncol = int(os.environ.get("NCOL", "100")); r = np.random.default_rng(0)
grid = [round(0.25 * k, 3) for k in range(1, 33)]
dt = _crit_np_dtype()
jobs = []
for i in range(ncol):
    z = r.random(n).astype(dt)
    f = [3.0, 2.0, 4.25, 0.0][i % 4]
    y = (np.sin(2 * np.pi * f * z) if f else 0) + 0.5 * r.standard_normal(n)
    jobs.append((z, y.astype(dt), grid))
P = dict(min_val_corr=0.15, min_rows=500, max_freqs=6)
def loop(): return [_detect_fourier_freqs_for_col(z, y, f_grid=g, **P) for z, y, g in jobs]
def batch(): return detect_fourier_freqs_batch(jobs, **P)
res = {}
for name, f in (("loop", loop), ("batch", batch), ("loop", loop), ("batch", batch)):
    f(); cp.cuda.Stream.null.synchronize(); t = time.perf_counter(); out = f(); cp.cuda.Stream.null.synchronize(); dt_ = time.perf_counter() - t
    res[name] = out; print(name, round(dt_, 2), "s")
tol = 8 * float(np.finfo(dt).eps)
same = all(len(a) == len(b) and all(abs(x - y) <= max(tol, 1e-6) for x, y in zip(a, b)) for a, b in zip(res["loop"], res["batch"]))
print("same lists:", same, "| nonempty", sum(1 for a in res["loop"] if a), "of", ncol)
