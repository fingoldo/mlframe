"""cProfile the scene/mrmr_fe pathology (3.75h on 2407x299, CPU idle => serial GIL-bound Python hotspot).

scene is near-saturated (all-features 0.97), so MRMR's relevance screen does NOT prune -> it keeps ~all 299 features
-> the O(p^2) FE pair-search runs on the full kept set. CPU was almost idle during the 3.75h run, which means the
time is NOT in njit/numpy/GPU compute (those saturate cores) but in SERIAL Python overhead (per-pair object/dict ops,
recipe construction, pure-Python MI, etc.). n_jobs=1 so cProfile captures the true serial hotspot (the GIL-bound FE
search doesn't parallelise anyway -> that is WHY CPU looked idle). Keep all 299 features (the pathology); subsample
rows to keep the profile tractable (the per-pair Python overhead is largely n-independent, so the hotspot shows even
at small n -- and if the wall barely drops with n, that itself confirms the bottleneck is n-independent Python).
"""
from __future__ import annotations
import os, sys, time, cProfile, pstats, io
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_ROWS = int(os.environ.get("SCENE_N", "500"))

# Profile the FE SERIAL hotspot, not the kernel-tuning sweep. The CPU-vs-GPU sweep runs synchronously on a
# cold/invalidated cache and (worse) blocks on a cross-process tuning lock that a killed process leaves stale.
# Monkeypatch get_or_tune to return the measurement-backed fallback (no sweep, no lock). TunerSpec.choose routes
# through get_or_tune too, so this covers every kernel (batch_mi, batch_pair_mi, joint_hist, mi_classif, ...).
def _disable_kernel_tuning_sweep():
    try:
        import pyutilz.performance.kernel_tuning.cache as _M

        def _no_sweep(self, kernel_name, *, dims, tuner, axes, fallback, **kw):
            return fallback() if callable(fallback) else fallback
        _M.KernelTuningCache.get_or_tune = _no_sweep
        # The postgres-loaded disk makes the per-host cache load_or_create() block for MINUTES (disk + filelock).
        # Use a throwaway in-memory cache so profiling the FE hotspot is not gated on disk I/O (we already force the
        # fallback above, so the cache content is irrelevant here).
        _inmem = _M.KernelTuningCache(in_memory=True)
        _M.KernelTuningCache.load_or_create = classmethod(lambda cls: _inmem)
        print("[kernel-tuning sweep+disk DISABLED for profiling -> in-memory fallback]", flush=True)
    except Exception as e:
        print(f"[no-sweep patch failed: {e}]", flush=True)


_disable_kernel_tuning_sweep()


def load_scene(n_rows):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="scene", version=1, as_frame=True, parser="auto")
    X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    y = pd.Series(pd.factorize(d.target)[0]); y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if n_rows < len(X):
        idx = np.random.default_rng(0).choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y


def main():
    X, y = load_scene(N_ROWS)
    print(f"scene subsample: shape={X.shape} pos={float(y.mean()):.3f}", flush=True)
    from mlframe.feature_selection.filters import MRMR
    # WARM the JIT + imports on a tiny fit FIRST (discarded) so the profiled fit below is free of the ~30s one-time
    # numba-compile/import cost -> the profile then shows the TRUE per-fit hotspot (the broad-val process was warm).
    tw = time.time(); MRMR(verbose=0, fe_max_steps=1, n_jobs=1, random_seed=0).fit(X.iloc[:150], y.iloc[:150])
    print(f"[warm-up fit {time.time()-tw:.1f}s] now profiling the warm fit...", flush=True)
    m = MRMR(verbose=0, fe_max_steps=1, n_jobs=1, random_seed=0)   # n_jobs=1: profile the true serial hotspot
    pr = cProfile.Profile(); t0 = time.time(); pr.enable()
    m.fit(X, y)
    pr.disable(); dt = time.time() - t0
    out = list(m.transform(X.iloc[:5]).columns)
    print(f"fit {dt:.1f}s; selected n={len(out)} (raw+eng); engineered={sum(1 for c in out if c not in X.columns)}", flush=True)
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s)
    print("\n========== TOP 35 by TOTTIME (where the serial time is actually spent) ==========")
    ps.sort_stats("tottime").print_stats(35); print(s.getvalue())
    s2 = io.StringIO(); ps2 = pstats.Stats(pr, stream=s2)
    print("\n========== TOP 25 by CUMTIME ==========")
    ps2.sort_stats("cumulative").print_stats(25); print(s2.getvalue())


if __name__ == "__main__":
    main()
