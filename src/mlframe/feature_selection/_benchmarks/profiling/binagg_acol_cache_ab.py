"""Cost of the per-(group, agg) set-up of the aggregated column that the per-column cache removes, at the fit's size.

Before the cache every (group, agg) pair rebuilt the float64 host copy of the aggregated column, its resident operand (a content hash plus a cache lookup), the finite mask,
the zeroed copy and the float mask; with G group columns the same set-up ran G times per aggregated column. Timed here: one set-up, and the whole builder over G group
columns sharing A aggregated columns (the cache removes (G - 1) / G of the set-ups, not a toggle, so the saving is the measured set-up times G - 1 times A).
"""
import time
import warnings

import cupy as cp
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
from mlframe.feature_selection.filters import _binned_numeric_agg_resident as res
from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

rng = np.random.default_rng(0)
n, G, A = 100_000, 4, 6
df = pd.DataFrame({**{f"g{i}": rng.normal(size=n) for i in range(G)}, **{f"a{j}": rng.normal(size=n) for j in range(A)}})
df.loc[::37, "a0"] = np.nan


def setup_once():
    """The set-up the cache now does once per aggregated column."""
    av = np.asarray(df["a0"].to_numpy(), dtype=np.float64)
    av_g = resident_operand(np.ascontiguousarray(av), ("binagg_a", "a0"), dtype=cp.float64)
    finite_g = cp.isfinite(av_g)
    v_safe = cp.where(finite_g, av_g, 0.0)
    finite_f = finite_g.astype(cp.float64)
    cp.cuda.Stream.null.synchronize()
    return v_safe, finite_f


setup_once()
t = []
for _ in range(15):
    t0 = time.perf_counter()
    setup_once()
    t.append(time.perf_counter() - t0)
per = float(np.median(t))

specs = []
for g in range(G):
    edges = np.quantile(df[f"g{g}"], np.linspace(0, 1, 11)[1:-1])
    for j in range(A):
        for stat in ("mean", "std"):
            specs.append({"name": f"a{j}_{stat}_by_g{g}", "group_col": f"g{g}", "agg_col": f"a{j}", "stat": stat, "edges": edges, "global": float(df[f"a{j}"].mean())})
folds = res.binagg_fold_ids(n, 5, 0)
res.build_binagg_oof_matrix_gpu(cp, df, specs, folds, 5)
tb = []
for _ in range(5):
    t0 = time.perf_counter()
    out = res.build_binagg_oof_matrix_gpu(cp, df, specs, folds, 5)
    cp.cuda.Stream.null.synchronize()
    tb.append(time.perf_counter() - t0)
whole = float(np.median(tb))
saved = per * (G - 1) * A
print(f"one set-up {per * 1e3:.1f} ms; builder over {G} groups x {A} aggregated columns x 2 stats: {whole * 1e3:.0f} ms; set-ups removed {(G - 1) * A}, about {saved * 1e3:.0f} ms ({100 * saved / (whole + saved):.1f} percent of the old time)")
