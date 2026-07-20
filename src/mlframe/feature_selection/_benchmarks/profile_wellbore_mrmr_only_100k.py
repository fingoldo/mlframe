"""Isolated MRMR.fit profiling harness on the REAL wellbore frame -- for nsys/ncu GPU profiling and
paired CPU-vs-GPU A/B, without the surrounding training-suite phases (model fits, reporting, SHAP)
that would dominate an nsys trace and blur kernel attribution.

Run modes (env):
  WELLBORE_MRMR_MODE=gpu|cpu   -- gpu: MLFRAME_FE_GPU_STRICT=1; cpu: MLFRAME_FE_GPU_STRICT=0 (default gpu)
  WELLBORE_TARGET_ROWS=100000  -- same whole-well row cap as wellbore_train.py (default 100k)
  WELLBORE_MRMR_CPROFILE=1     -- wrap fit in cProfile, dump .prof + top-40 (default off: nsys/ncu prefer no tracer)

Under nsys (kernel/transfer timeline + H2D/D2H audit):
  nsys profile -o mrmr_gpu_trace --trace=cuda,nvtx,osrt python src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only.py
Under ncu (per-kernel occupancy/bandwidth; -c caps kernel launches profiled, --set full for all sections):
  ncu -c 200 --set full -o mrmr_gpu_kernels python src/mlframe/feature_selection/_benchmarks/profile_wellbore_mrmr_only.py

The MRMR config mirrors wellbore_train.py's mrmr_kwargs exactly (fe_max_steps=1, cat FE on, medium
presets) so hotspots found here transfer 1:1 to the production suite run.
"""
import os
import time
from os.path import join

MODE = os.environ.get("WELLBORE_MRMR_MODE", "gpu").strip().lower()
os.environ.setdefault("MLFRAME_FE_GPU_STRICT", "1" if MODE == "gpu" else "0")
TARGET_ROWS = int(os.environ.get("WELLBORE_TARGET_ROWS", "100000"))
CPROFILE = os.environ.get("WELLBORE_MRMR_CPROFILE", "0") == "1"

import numpy as np
import polars as pl

if os.environ.get("WELLBORE_DUMP_AUDIT", "0") == "1":
    # Log every large-ndarray dump joblib's memmapping reducer performs (shape/dtype/MB + a short stack
    # hint), to attribute the residual 48 x ~2.5s _pickle.dumps the cProfile top shows.
    import traceback
    import joblib._memmapping_reducer as _jmr
    _orig_reduce = _jmr.ArrayMemmapForwardReducer.__call__
    def _audited(self, a):
        try:
            import numpy as _np
            if isinstance(a, _np.ndarray) and a.nbytes > 1_000_000 and not isinstance(a, _np.memmap):
                frames = [f for f in traceback.extract_stack(limit=25) if "mlframe" in f.filename]
                hint = f"{frames[-1].filename.split('mlframe')[-1]}:{frames[-1].lineno}" if frames else "?"
                print(f"[dump-audit] shape={a.shape} dtype={a.dtype} {a.nbytes/1e6:.0f}MB from {hint}", flush=True)
        except Exception:
            pass
        return _orig_reduce(self, a)
    _jmr.ArrayMemmapForwardReducer.__call__ = _audited

DATA_DIR = r"C:\Users\Admin\Machine learning\data\Competitions\ROGII - Wellbore Geology Prediction"

_lf = pl.scan_parquet(join(DATA_DIR, "train_df.parquet")).filter(~pl.col("TVT").is_null())
_counts = _lf.group_by("well_id").agg(pl.len().alias("n")).sort("well_id").collect()
_counts = _counts.with_columns(pl.col("n").cum_sum().alias("cum"))
_keep = _counts.filter(pl.col("cum") <= TARGET_ROWS)["well_id"].to_list() or [_counts["well_id"][0]]
df = _lf.filter(pl.col("well_id").is_in(_keep)).with_columns(pl.col(pl.Float64).cast(pl.Float32)).collect()

_drop = ["TVT_input", "ANCC", "ASTNL", "ASTNU", "BUDA", "EGFDL", "EGFDU", "well_id"]
y = df["TVT"].to_pandas()
X = df.drop([c for c in _drop + ["TVT"] if c in df.columns]).to_pandas()
print(f"[mrmr-only] mode={MODE} X={X.shape} strict_env={os.environ['MLFRAME_FE_GPU_STRICT']}")

from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
from mlframe.feature_selection.filters.mrmr import MRMR

if MODE == "gpu":
    # Production hides the polynom-loky pool's ~38s cold-start behind the categorization phase via
    # maybe_prewarm_polynom_loky_pool (background thread). This isolated harness has no categorization
    # step to hang the prewarm off of, so without this the first run_polynom_pair_fe call pays a cold
    # 16-worker spawn inline -- inflating wall-time/loky-overhead numbers relative to the real suite run.
    from mlframe.feature_selection.filters._joblib_safe import maybe_prewarm_polynom_loky_pool
    _prewarm_t0 = time.perf_counter()
    _prewarm_thread = maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters=5, n_jobs=16)
    if _prewarm_thread is not None:
        _prewarm_thread.join()
        print(f"[mrmr-only] polynom loky pool prewarmed in {time.perf_counter() - _prewarm_t0:.1f}s")

sel = MRMR(
    fe_max_steps=1,
    cat_fe_config=CatFEConfig(enable=True, include_numeric=True),
    fe_npermutations=100,
    fe_ntop_features=15,
    fe_unary_preset="medium",
    fe_binary_preset="medium",
    fe_smart_polynom_iters=5,
    fe_smart_polynom_optimization_steps=100,
    fe_min_polynom_degree=3,
    fe_max_polynom_degree=6,
    fe_optimizer=os.environ.get("WELLBORE_FE_OPTIMIZER", "cupy_kernel"),
    random_seed=42,
)

t0 = time.perf_counter()
if CPROFILE:
    import cProfile
    import io
    import pstats

    prof = cProfile.Profile()
    prof.enable()
    sel.fit(X, y)
    prof.disable()
    out = f"mrmr_only_{MODE}_{X.shape[0]}rows.prof"
    prof.dump_stats(out)
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(40)
    print(s.getvalue())
    print(f"[mrmr-only] profile dumped to {out}")
else:
    sel.fit(X, y)
wall = time.perf_counter() - t0
_sup = getattr(sel, "support_", None)
print(f"[mrmr-only] mode={MODE} fit wall={wall:.1f}s selected={0 if _sup is None else int(np.asarray(_sup).sum())} features")
