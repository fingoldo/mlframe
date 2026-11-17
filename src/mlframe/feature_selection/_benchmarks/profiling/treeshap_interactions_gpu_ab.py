"""Interleaved A/B of the device TreeSHAP interaction kernels in one process, with the hardware state before and after.

``python -m mlframe.feature_selection._benchmarks.profiling.treeshap_interactions_gpu_ab ROWS FEATURES [DEPTH] [TREES]`` times, round-robin over ``--rounds`` rounds: the pair-table kernel with the DFS stack in
local memory, the same with the stack in shared memory, and the older conditioned-scan path kernel (``--path``, slow). Best and median of the rounds are printed per variant, with ``hardware_state.line()`` of both ends.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from mlframe.feature_selection._benchmarks.profiling import hardware_state


def _timed(cp, fn):
    cp.cuda.runtime.deviceSynchronize()
    t = time.perf_counter()
    fn()
    cp.cuda.runtime.deviceSynchronize()
    return (time.perf_counter() - t) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", type=int)
    ap.add_argument("features", type=int)
    ap.add_argument("depth", type=int, nargs="?", default=4)
    ap.add_argument("trees", type=int, nargs="?", default=200)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--path", action="store_true", help="also time the conditioned-scan path kernel")
    a = ap.parse_args()
    import cupy as cp
    from xgboost import XGBRegressor

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_table_gpu import interaction_tensor_table_gpu

    rng = np.random.default_rng(0)
    X = rng.normal(size=(a.rows, a.features)).astype(np.float32)
    y = np.nan_to_num(X[:, 0] * X[:, 1] + X[:, 2] + X[:, 3])
    ens = extract_ensemble(XGBRegressor(n_estimators=a.trees, max_depth=a.depth, tree_method="hist").fit(X, y))
    assert ens is not None

    def variant(stack):
        def run():
            os.environ["MLFRAME_SHAP_TREESHAP_STACK"] = stack
            return interaction_tensor_table_gpu(ens, X)

        return run

    variants = {"table_local": variant("local"), "table_shared": variant("shared")}
    if a.path:
        variants["path"] = lambda: interaction_tensor_gpu_path(ens, X)
    for fn in variants.values():
        fn()
    print("before:", hardware_state.line(hardware_state.snapshot()))
    times: dict[str, list[float]] = {k: [] for k in variants}
    for _ in range(a.rounds):
        for k, fn in variants.items():
            times[k].append(_timed(cp, fn))
    print("after: ", hardware_state.line(hardware_state.snapshot()))
    for k, v in times.items():
        print(f"{k:14s} best {min(v):9.1f} ms  median {sorted(v)[len(v) // 2]:9.1f} ms")


if __name__ == "__main__":
    main()
