"""Profile composite-target discovery + wrap + ensemble end-to-end.

Bypasses ``train_mlframe_models_suite`` (which transitively imports
``mlframe.feature_selection.filters`` -- in flux during a parallel
refactor) and runs the composite-target code path directly:
``CompositeTargetDiscovery.fit`` -> per-spec inner training ->
``CompositeTargetEstimator.from_fitted_inner`` -> ensemble build ->
predict on test rows. cProfile output is sorted by cumulative time
and filtered to mlframe internals so the hotspot list points at our
code, not LightGBM / sklearn internals.

Usage::

    python -m mlframe.benchmarks.composite_profile
    python -m mlframe.benchmarks.composite_profile --n 5000 --top 60
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

# Insert repo root on path so this script works under either
# ``python -m mlframe.benchmarks.composite_profile`` or a direct
# ``python composite_profile.py`` invocation.
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository")


def make_data(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x = rng.normal(size=(n, 10))
    y = 0.95 * base + 0.5 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.3, size=n)
    cols = {"TVT_prev": base, "target": y}
    for i in range(10):
        cols[f"x{i}"] = x[:, i]
    return pd.DataFrame(cols)


def composite_pipeline(
    df: pd.DataFrame, *, screening: str = "hybrid",
    use_oof: bool = False,
    use_stack: bool = True,
) -> Tuple[float, dict]:
    """Run discovery -> per-spec training -> wrap -> ensemble. Returns
    (test_rmse, timings_dict)."""
    from mlframe.training.composite import (
        CompositeCrossTargetEnsemble,
        CompositeTargetDiscovery,
        CompositeTargetEstimator,
        compute_oof_holdout_predictions,
        get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    import lightgbm as lgb

    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), int(0.85 * n))
    test_idx = np.arange(int(0.85 * n), n)
    feature_cols = [c for c in df.columns if c != "target"]
    y_full = df["target"].to_numpy()

    timings: dict = {}

    # 1. Discovery.
    t0 = time.perf_counter()
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["TVT_prev"],
        transforms=["diff", "ratio", "logratio", "linear_residual"],
        mi_sample_n=min(2000, len(train_idx)),
        top_k_after_mi=4,
        eps_mi_gain=-1.0,
        screening=screening,
        tiny_model_n_estimators=30,
        tiny_model_sample_n=min(2000, len(train_idx)),
        top_m_after_tiny=3,
    )
    disc = CompositeTargetDiscovery(cfg).fit(
        df=df, target_col="target", feature_cols=feature_cols,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    timings["discovery_s"] = time.perf_counter() - t0

    # 2. Per-spec training (raw + composite).
    t0 = time.perf_counter()
    components: List = []
    component_names: List[str] = []

    train_X = df.loc[train_idx, feature_cols].reset_index(drop=True)
    test_X = df.loc[test_idx, feature_cols].reset_index(drop=True)
    train_y = y_full[train_idx]
    test_y = y_full[test_idx]

    raw_model = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
    raw_model.fit(train_X, train_y)
    components.append(raw_model)
    component_names.append("raw")

    for spec in disc.specs_:
        transform = get_transform(spec.transform_name)
        base_train = train_X[spec.base_column].to_numpy()
        valid = transform.domain_check(train_y, base_train)
        t_train = transform.forward(train_y[valid], base_train[valid], spec.fitted_params)
        inner = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
        inner.fit(train_X.iloc[valid].reset_index(drop=True), t_train)
        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=inner,
            transform_name=spec.transform_name,
            base_column=spec.base_column,
            transform_fitted_params=spec.fitted_params,
            y_train=train_y[valid],
        )
        components.append(wrapper)
        component_names.append(spec.name)
    timings["training_s"] = time.perf_counter() - t0

    # 3. Ensemble.
    t0 = time.perf_counter()
    if use_stack:
        pred_matrix = np.column_stack(
            [np.asarray(c.predict(train_X)).reshape(-1) for c in components]
        )
        ensemble = CompositeCrossTargetEnsemble.from_linear_stack(
            component_models=components,
            component_names=component_names,
            component_predictions=pred_matrix,
            y_train=train_y,
        )
    else:
        rmses = []
        for c in components:
            pred = np.asarray(c.predict(train_X)).reshape(-1)
            rmses.append(float(np.sqrt(np.mean((pred - train_y) ** 2))))
        ensemble = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=components,
            component_names=component_names,
            component_train_rmse=rmses,
        )
    timings["ensemble_s"] = time.perf_counter() - t0

    if use_oof:
        t0 = time.perf_counter()
        component_specs = []
        for name in component_names:
            if name == "raw":
                component_specs.append(None)
            else:
                matching = next(
                    (s for s in disc.specs_ if s.name == name), None,
                )
                component_specs.append(
                    {"base_column": matching.base_column,
                     "transform_name": matching.transform_name,
                     "fitted_params": matching.fitted_params}
                    if matching else None
                )
        base_per_spec = {
            spec.base_column: train_X[spec.base_column].to_numpy()
            for spec in disc.specs_
        }
        compute_oof_holdout_predictions(
            component_models=components,
            component_names=component_names,
            component_specs=component_specs,
            train_X=train_X,
            y_train_full=train_y,
            base_train_full_per_spec=base_per_spec,
            holdout_frac=0.2,
            random_state=42,
        )
        timings["oof_refit_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_pred = ensemble.predict(test_X)
    timings["test_predict_s"] = time.perf_counter() - t0

    test_rmse = float(np.sqrt(np.mean((test_pred - test_y) ** 2)))
    return test_rmse, timings


def profile_one(n: int, screening: str, use_oof: bool, top: int = 40) -> None:
    df = make_data(n=n)
    label = f"n={n} screening={screening} oof={use_oof}"
    print(f"\n{'=' * 80}\nPROFILE: {label}\n{'=' * 80}")

    prof = cProfile.Profile()
    prof.enable()
    rmse, timings = composite_pipeline(
        df, screening=screening, use_oof=use_oof,
    )
    prof.disable()

    print(f"\nresult test_rmse = {rmse:.4f}")
    print("phase timings (s):")
    for k, v in timings.items():
        print(f"  {k:20s} = {v:7.3f}")

    print(f"\nTop-{top} hotspots in mlframe.* by cumulative time:")
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats("mlframe", top)
    print(s.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--screening", type=str, default="hybrid")
    parser.add_argument("--oof", action="store_true",
                        help="Include OOF refit in the profile.")
    args = parser.parse_args()
    profile_one(n=args.n, screening=args.screening,
                use_oof=args.oof, top=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
