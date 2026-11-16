"""Public shap against mlframe's TreeSHAP (numba path, numba table, CUDA path, CUDA table): time, numerical closeness, rank agreement and additivity.

Trains a LightGBM or XGBoost regressor on synthetic data and prints one table: per implementation the best-of-N wall (GPU calls synchronised, uploads included), the speed-up over the public
``shap.TreeExplainer(..., feature_perturbation="tree_path_dependent")``, the largest and mean absolute difference to it, the Spearman correlation of the per-feature mean |phi| ranks, whether the top-10
features coincide, and the largest violation of ``base + sum(phi) == margin``.

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.treeshap_vs_shap <rows> <lightgbm|xgboost> [features trees depth]`` (a CUDA device is needed for the GPU rows).
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main(argv: "list[str] | None" = None) -> int:
    """Run the comparison for the given rows and model family and print the table."""
    import cupy as cp
    import shap

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba, treeshap_phi_base_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

    args = list(sys.argv[1:] if argv is None else argv)
    n = int(args[0])
    kind = args[1]
    f, trees, depth = (int(a) for a in args[2:5]) if len(args) >= 5 else (100, 100, 6)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, f)).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] * X[:, 2] + np.sin(X[:, 3]) + 0.3 * rng.standard_normal(n)
    if kind == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRegressor(n_estimators=trees, max_depth=depth, num_leaves=2**depth, verbose=-1, n_jobs=4).fit(X, y)
        margin = model.predict(X, raw_score=True)
    else:
        import xgboost as xgb

        model = xgb.XGBRegressor(n_estimators=trees, max_depth=depth, tree_method="hist", verbosity=0, n_jobs=4).fit(X, y)
        margin = model.predict(X, output_margin=True)
    ens = extract_ensemble(model)


    def best(fn, repeats):
        fn()
        out = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return min(out)


    def sync(fn):
        def run():
            r = fn()
            cp.cuda.Stream.null.synchronize()
            return r

        return run


    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    runs = {
        "shap (public)": lambda: np.asarray(explainer.shap_values(X, check_additivity=False)),
        "numba path": lambda: treeshap_phi_base_path(ens, X)[0],
        "numba table": lambda: treeshap_phi_base_numba(ens, X)[0],
        "gpu path (f32)": sync(lambda: treeshap_phi_base_gpu(ens, X, use_table=False)[0]),
        "gpu table (f32)": sync(lambda: treeshap_phi_base_gpu(ens, X)[0]),
    }
    reps = {"shap (public)": 1, "numba path": 1, "numba table": 3, "gpu path (f32)": 2, "gpu table (f32)": 5}
    phis, times = {}, {}
    for name, fn in runs.items():
        phis[name] = np.asarray(fn(), dtype=np.float64)
        times[name] = best(fn, reps[name])
    ref = phis["shap (public)"]
    scale = float(np.abs(ref).mean())
    rank_ref = np.argsort(-np.abs(ref).mean(0))


    def ranks(a):
        order = np.argsort(-np.abs(a).mean(0))
        r = np.empty(len(order))
        r[order] = np.arange(len(order))
        return r


    rr = ranks(ref)
    base = float(ens.base_offset)
    print(f"=== {kind} n={n} f={f} trees={trees} depth={depth} | shap {shap.__version__} | mean|phi| of shap {scale:.4f}")
    print(f"{'implementation':18s} {'time ms':>10s} {'x vs shap':>9s} {'max|d|':>10s} {'mean|d|':>10s} {'rel.max':>8s} {'spearman':>8s} {'top10==':>7s} {'|add. err|':>10s}")
    for name, a in phis.items():
        d = np.abs(a - ref)
        rho = float(np.corrcoef(ranks(a), rr)[0, 1])
        same10 = set(np.argsort(-np.abs(a).mean(0))[:10]) == set(rank_ref[:10])
        add = float(np.max(np.abs(base + a.sum(1) - margin)))
        print(f"{name:18s} {times[name] * 1e3:10.1f} {times['shap (public)'] / times[name]:9.2f} {d.max():10.2e} {d.mean():10.2e} {d.max() / scale:8.1e} {rho:8.4f} {same10!s:>7s} {add:10.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
