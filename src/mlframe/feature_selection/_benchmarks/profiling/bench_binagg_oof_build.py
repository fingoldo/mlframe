"""Time of the device build of the binned-aggregate OOF matrix (80 columns: 4 group columns x 5 aggregate columns x 4 stats) at 100k and 1M rows, and the largest difference to a reference build.

Usage: ``python bench_binagg_oof_build.py [path.py]`` - the optional path is a module with the same ``build_binagg_oof_matrix_gpu`` to compare against (for example the previous revision of
``_binned_numeric_agg_resident.py`` written next to it with ``git show``); without it only the current build is timed.
"""

from __future__ import annotations

import importlib.util
import sys
import time

import numpy as np
import pandas as pd

SIZES = (100_000, 1_000_000)
N_GROUP, N_AGG, N_FOLDS, REPEATS = 4, 5, 5, 5


def _load(path: str):
    """Import the module at ``path`` under a private name."""
    spec = importlib.util.spec_from_file_location("binagg_reference", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["binagg_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


def _median_ms(cp, mod, X, specs, folds) -> "tuple[float, np.ndarray]":
    """Median wall time (ms) of the build after one warm call, and its result."""
    mod.build_binagg_oof_matrix_gpu(cp, X, specs, folds, N_FOLDS)
    ts = []
    for _ in range(REPEATS):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out = mod.build_binagg_oof_matrix_gpu(cp, X, specs, folds, N_FOLDS)
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3, cp.asnumpy(out)


def main() -> None:
    """Print the timings per size."""
    import cupy as cp

    from mlframe.feature_selection.filters import _binned_numeric_agg_resident as current
    from mlframe.feature_selection.filters._binned_numeric_agg_fe import SUPPORTED_STATS, fit_binned_numeric_agg

    reference = _load(sys.argv[1]) if len(sys.argv) > 1 else None
    for n in SIZES:
        rng = np.random.default_rng(0)
        X = pd.DataFrame({f"g{i}": rng.uniform(0, 1, n) for i in range(N_GROUP)} | {f"a{i}": rng.normal(0, 1, n) for i in range(N_AGG)})
        y = rng.normal(0, 1, n)
        feat, rec = fit_binned_numeric_agg(
            X, y, group_num_cols=[f"g{i}" for i in range(N_GROUP)], agg_num_cols=[f"a{i}" for i in range(N_AGG)], stats=SUPPORTED_STATS, nbins_base=10, n_folds=N_FOLDS, random_state=0
        )
        specs = [{"name": c, "group_col": rec[c]["group_col"], "agg_col": rec[c]["agg_col"], "stat": rec[c]["stat"], "edges": rec[c]["edges"], "global": rec[c]["global"]} for c in feat.columns]
        folds = current.binagg_fold_ids(n, N_FOLDS, 0)
        new_ms, new_out = _median_ms(cp, current, X, specs, folds)
        line = f"n={n} columns={len(specs)} current {new_ms:.1f} ms"
        if reference is not None:
            old_ms, old_out = _median_ms(cp, reference, X, specs, folds)
            line += f"  reference {old_ms:.1f} ms ({old_ms / new_ms:.2f}x)  max|diff| {np.nanmax(np.abs(old_out - new_out)):.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
