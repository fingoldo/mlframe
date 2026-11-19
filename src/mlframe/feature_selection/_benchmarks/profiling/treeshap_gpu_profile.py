"""Timing and agreement harness of the SHAP-proxy GPU TreeSHAP kernel against the numba kernel.

Trains a LightGBM model of the given shape, extracts the flat ensemble, and reports per backend the warmed best-of wall (the GPU call synchronised, uploads included, then the kernel alone), the largest
absolute difference of the two phi matrices, the SHAP-rank agreement (Spearman of the mean |phi| per feature) and the hardware state (GPU temperature, clocks, throttle) the timings were taken under.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.treeshap_gpu_profile --n 20000 --f 100 --trees 100 --depth 6 [--block 128]``
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mlframe.feature_selection._benchmarks.profiling.hardware_state import line, snapshot


def build_ensemble(n: int, f: int, trees: int, depth: int, seed: int = 0, with_model: bool = False):
    """``(ensemble, X)`` (and the fitted model when ``with_model``) of a LightGBM regressor trained on ``n`` synthetic rows with ``f`` features."""
    import lightgbm as lgb

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f)).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] * X[:, 2] + np.sin(X[:, 3]) + 0.3 * rng.standard_normal(n)
    model = lgb.LGBMRegressor(n_estimators=trees, max_depth=depth, num_leaves=2**depth, verbose=-1, n_jobs=4).fit(X, y)
    return (extract_ensemble(model), X, model) if with_model else (extract_ensemble(model), X)


def best_of(fn, repeats: int = 5) -> float:
    """Smallest wall of ``repeats`` calls after one warm-up call."""
    import cupy as cp

    fn()
    cp.cuda.Device().synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()  # the timer stops at completion, not at the launch
        out.append(time.perf_counter() - t0)
    return min(out)


def main(argv: "list[str] | None" = None) -> int:
    """Run the comparison and print one line per measurement."""
    import cupy as cp

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_numba
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--f", type=int, default=100)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--shap", action="store_true", help="also time the public shap library (TreeExplainer, path-dependent) on the same model")
    ap.add_argument("--cprofile", action="store_true", help="print the cProfile table of one numba call")
    args = ap.parse_args(argv)
    ens, X, model = build_ensemble(args.n, args.f, args.trees, args.depth, with_model=True)
    print("hardware before:", line(snapshot()))
    phi_n, _ = treeshap_phi_base_numba(ens, X)
    t_numba = best_of(lambda: treeshap_phi_base_numba(ens, X), 2)

    def gpu():
        """One synchronised GPU call, uploads and download included."""
        out = treeshap_phi_base_gpu(ens, X)
        cp.cuda.Stream.null.synchronize()
        return out

    phi_g, _ = gpu()
    t_gpu = best_of(gpu, args.repeats)
    if args.shap:
        import shap

        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        phi_s = np.asarray(explainer.shap_values(X, check_additivity=False))
        t_shap = best_of(lambda: explainer.shap_values(X, check_additivity=False), 2)
        print(f"shap library {t_shap * 1e3:9.1f} ms | numba {t_numba * 1e3:9.1f} ms | numba vs shap {t_shap / t_numba:5.2f}x | max|d(numba,shap)| {float(np.max(np.abs(phi_n - phi_s))):.3e}")
    if args.cprofile:
        import cProfile
        import pstats

        prof = cProfile.Profile()
        prof.enable()
        treeshap_phi_base_numba(ens, X)
        prof.disable()
        pstats.Stats(prof).sort_stats("cumulative").print_stats(8)
    diff = float(np.max(np.abs(phi_n - phi_g)))
    rank_n, rank_g = np.argsort(-np.abs(phi_n).mean(0)), np.argsort(-np.abs(phi_g).mean(0))
    print(f"n={args.n} f={args.f} trees={args.trees} depth={args.depth} (ensemble depth {ens.max_depth})")
    print(f"numba {t_numba * 1e3:9.1f} ms | gpu {t_gpu * 1e3:9.1f} ms | speed-up {t_numba / t_gpu:6.2f}x | max|dphi| {diff:.3e} | same top-10 features {set(rank_n[:10]) == set(rank_g[:10])}")
    print("hardware after:", line(snapshot()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
