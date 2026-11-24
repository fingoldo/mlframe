"""Does the target-mix second pass admit anything? One fit with ``fe_target_mix_enable`` on and one with it off per continuous dataset, each in this process, one after the other.

Usage: ``python bench_target_mix_yield.py [ROWS] [COLS]``. For every dataset it prints the wall of both fits, the selected columns and engineered features of both, the provenance of the mix (base, child fit wall,
candidates proposed) and which proposed candidates survive in the final engineered list. The point is to decide, on the selected set and not on one synthetic target, whether the child fits (about the cost of a
whole fit each) pay for themselves. Nothing else may run on the machine while it does.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)
import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR


def frame(rows: int, cols: int, seed: int) -> pd.DataFrame:
    """Standard-normal float32 columns ``x0..``."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((rows, cols), dtype=np.float32), columns=[f"x{i}" for i in range(cols)])


def datasets(rows: int, cols: int) -> dict:
    """``name -> (X, y)``: continuous targets with different structure left after the strongest term."""
    out = {}
    X = frame(rows, cols, 1)
    x = {c: X[c].to_numpy() for c in X.columns[:8]}
    rng = np.random.default_rng(11)
    noise = lambda s: s * rng.standard_normal(rows, dtype=np.float32)  # noqa: E731
    out["additive_linear"] = (X, (x["x0"] + 0.8 * x["x1"] - 0.6 * x["x2"] + noise(0.5)).astype(np.float32))
    out["interaction_only"] = (X, (x["x0"] * x["x1"] + noise(0.5)).astype(np.float32))
    out["interaction_plus_second"] = (X, (2.0 * x["x0"] * x["x1"] + x["x2"] * x["x3"] + noise(0.5)).astype(np.float32))
    out["level_plus_interaction"] = (X, (3.0 * x["x4"] + 1.5 * x["x2"] * x["x3"] + noise(0.3)).astype(np.float32))
    out["periodic_plus_trend"] = (X, (np.sin(2.0 * x["x0"]) + 0.7 * x["x1"] ** 2 + 0.4 * x["x5"] + noise(0.3)).astype(np.float32))
    out["pure_noise"] = (X, noise(1.0).astype(np.float32))
    return out


def run(X: pd.DataFrame, y: np.ndarray, mix: bool) -> dict:
    """One fit with the all-families configuration; the wall, the selection and the mix provenance."""
    kw = F.all_families_kwargs()
    kw["binning_scheme"] = "mdlp"
    kw["fe_target_mix_enable"] = mix
    m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw)
    t0 = time.perf_counter()
    m.fit(X, pd.Series(y, name="y"))
    wall = time.perf_counter() - t0
    return {
        "wall": wall,
        "support": sorted(int(v) for v in np.asarray(m.support_).ravel()),
        "engineered": [str(n) for n in (m._engineered_features_ or [])],
        "prov": list(getattr(m, "fe_target_mix_provenance_", []) or []),
    }


def main() -> None:
    """Fit every dataset with the mix on and off and print the comparison."""
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for name, (X, y) in datasets(rows, cols).items():
        on = run(X, y, True)
        off = run(X, y, False)
        proposed = [f for e in on["prov"] for f in e["features"]]
        kept = [f for f in proposed if f in on["engineered"]]
        same = on["support"] == off["support"] and on["engineered"] == off["engineered"]
        print(f"== {name} {rows}x{cols}: wall on {on['wall']:.1f}s off {off['wall']:.1f}s | support on {on['support']} off {off['support']} | same selection: {same}", flush=True)
        print(f"   engineered on {on['engineered']}", flush=True)
        print(f"   engineered off {off['engineered']}", flush=True)
        for e in on["prov"]:
            print(f"   base {e['base']} child {e['child_fit_s']:.1f}s rows {e['child_rows']} proposed {len(e['features'])}", flush=True)
        print(f"   proposed {len(proposed)}, admitted into the final engineered list {len(kept)}", flush=True)


if __name__ == "__main__":
    main()
