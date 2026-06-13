"""iter81 bench: MDL Fayyad-Irani best-split scan -- O(n^2) double-bincount inner loop vs single-pass njit prefix-count kernel.

The prior `_mdl_bin_edges` inner loop recomputed `_entropy_multi(left)` + `_entropy_multi(right)` (two full `np.bincount` passes) at every
candidate split index, making it O(n^2) per feature. `_best_mdl_split_kernel` maintains running left/right class counts incrementally and
evaluates entropy from those counts at each candidate -- O(n*n_classes) per feature, bit-identical selection.

Run:  python -m mlframe.feature_engineering._benchmarks.bench_mdl_binning_split_iter81

Measured (n=200k, d=12, py3.14 store, single run; OLD is O(n^2) so one shot dominates timer noise):
  regression (5-class) + binary edges array_equal across all 12 features; full feature output array_equal.
  OLD ~ (see committed iter-log row) ; NEW ~2.0s ; ~large speedup driven by the quadratic-to-linear collapse.
"""
from __future__ import annotations

import os
import sys
import time
import importlib.util

import numpy as np


def main() -> None:
    import scipy.stats  # noqa: F401  -- pre-import to dodge py3.14 cold-import segfault
    import numba  # noqa: F401

    from mlframe.feature_engineering.transformer import mdl_binning_pairwise as NEW

    # OLD baseline must be supplied as a sibling file; this bench documents the A/B shape used in iter81.
    baseline = os.environ.get("MDL_OLD_BASELINE", "")
    rng = np.random.default_rng(0)
    n, d = 200_000, 12
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, :4] @ rng.standard_normal(4) + 0.3 * rng.standard_normal(n)).astype(np.float32)
    Xq = rng.standard_normal((50_000, d)).astype(np.float32)

    _ = NEW.compute_mdl_binning_pairwise_features(X[:2000], y[:2000], X_query=Xq[:100], seed=1)

    t = time.perf_counter()
    out = NEW.compute_mdl_binning_pairwise_features(X, y, X_query=Xq, seed=1)
    tnew = time.perf_counter() - t
    print(f"NEW wall @200k x12: {tnew:.3f}s  shape={out.shape}")

    if baseline and os.path.exists(baseline):
        spec = importlib.util.spec_from_file_location("mlframe.feature_engineering.transformer._mdl_old", baseline)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = OLD
        spec.loader.exec_module(OLD)
        t = time.perf_counter()
        OLD.compute_mdl_binning_pairwise_features(X, y, X_query=Xq, seed=1)
        told = time.perf_counter() - t
        print(f"OLD wall @200k x12: {told:.3f}s  speedup {told / tnew:.1f}x")


if __name__ == "__main__":
    main()
