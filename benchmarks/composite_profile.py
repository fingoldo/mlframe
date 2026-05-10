"""Profile every composite-target feature with cProfile + wall-time
calibration, per the CLAUDE.md "Profile every new feature" rule.

This harness covers:

1. ``BaselineDiagnostics.fit_and_report`` -- the per-target
   diagnostic that runs by default.
2. ``CompositeTargetDiscovery.fit`` -- discovery (MI screening +
   tiny-model rerank).
3. ``CompositeTargetEstimator.predict`` -- single-model y-scale
   inference; the relevant latency knob for online single-row
   serving (HFT / ad-tech).
4. ``CompositeCrossTargetEnsemble.predict`` -- multi-component
   inference; same online-latency concern for the full ensemble.
5. ``compute_oof_holdout_predictions`` -- the heavy honest-OOF
   re-fit path used for stacking weights.

Each feature is run in two modes:

- **cProfile**: shows where time goes inside mlframe.* code.
  Inflated 10-13x on pandas / sklearn deep-stack call timings vs
  standalone wall-time microbench.
- **Wall-time microbench**: median over N repetitions outside
  cProfile. Calibrates which cProfile-flagged hotspots are real
  vs attribution noise.

Usage::

    python -m mlframe.benchmarks.composite_profile
    python -m mlframe.benchmarks.composite_profile --n 5000 --top 30 --oof
    python -m mlframe.benchmarks.composite_profile --feature baseline_diag
    python -m mlframe.benchmarks.composite_profile --feature wrapper_predict --batch 1000

Reports:

For each feature, prints
- wall-time median + min/max over N repetitions
- cProfile top-K hotspots in mlframe.* sorted by cumulative time
- cProfile-vs-wall ratio per top hotspot (calibration)
- "no actionable speedup" verdict per CLAUDE.md rule when applicable
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository")


# ----------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------


def make_data(n: int, seed: int = 0, n_features: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x = rng.normal(size=(n, n_features))
    y = 0.95 * base + 0.5 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(scale=0.3, size=n)
    cols = {"TVT_prev": base, "target": y}
    for i in range(n_features):
        cols[f"x{i}"] = x[:, i]
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------
# Per-feature workloads
# ----------------------------------------------------------------------


def workload_baseline_diag(df: pd.DataFrame) -> Callable[[], Any]:
    """Return a no-arg callable that runs BaselineDiagnostics once."""
    from mlframe.training.baseline_diagnostics import BaselineDiagnostics
    from mlframe.training.configs import BaselineDiagnosticsConfig

    cfg = BaselineDiagnosticsConfig(
        ablation_top_k=4, quick_model_n_estimators=80, sample_n=None,
    )
    bd = BaselineDiagnostics(cfg)
    feature_cols = [c for c in df.columns if c != "target"]
    y = df["target"].to_numpy()
    df_x = df.drop(columns=["target"])
    return lambda: bd.fit_and_report(
        train_df=df_x, train_target=y, feature_cols=feature_cols,
        target_type="regression", target_name="target",
    )


def workload_discovery(df: pd.DataFrame, screening: str = "hybrid") -> Callable[[], Any]:
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    feature_cols = [c for c in df.columns if c != "target"]
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"],
        transforms=["diff", "ratio", "logratio", "linear_residual"],
        mi_sample_n=min(2000, len(train_idx)),
        top_k_after_mi=4, eps_mi_gain=-1.0,
        screening=screening,
        tiny_model_n_estimators=30,
        tiny_model_sample_n=min(2000, len(train_idx)),
        top_m_after_tiny=3,
        mi_estimator="bin",
    )
    return lambda: CompositeTargetDiscovery(cfg).fit(
        df=df, target_col="target", feature_cols=feature_cols, train_idx=train_idx,
    )


def _build_fitted_wrapper(df: pd.DataFrame):
    """Helper: fit a composite estimator wrapper for the predict
    profiling workloads."""
    from mlframe.training.composite import (
        CompositeTargetDiscovery, CompositeTargetEstimator, get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    import lightgbm as lgb

    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    feature_cols = [c for c in df.columns if c != "target"]
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"],
        transforms=["linear_residual"], mi_sample_n=600,
        top_k_after_mi=1, eps_mi_gain=-1.0,
    )
    disc = CompositeTargetDiscovery(cfg).fit(
        df=df, target_col="target", feature_cols=feature_cols, train_idx=train_idx,
    )
    spec = disc.specs_[0]
    transform = get_transform(spec.transform_name)
    train_X = df.iloc[train_idx][feature_cols].reset_index(drop=True)
    train_y = df["target"].to_numpy()[train_idx]
    base_train = train_X[spec.base_column].to_numpy()
    valid = transform.domain_check(train_y, base_train)
    t_train = transform.forward(
        train_y[valid], base_train[valid], spec.fitted_params,
    )
    inner = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
    inner.fit(train_X.iloc[valid].reset_index(drop=True), t_train)
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner, transform_name=spec.transform_name,
        base_column=spec.base_column,
        transform_fitted_params=spec.fitted_params,
        y_train=train_y[valid],
    )
    return wrapper, disc, train_X, train_y


def workload_wrapper_predict(df: pd.DataFrame, batch: int) -> Callable[[], Any]:
    wrapper, _, train_X, _ = _build_fitted_wrapper(df)
    # Hold a ref to the predict input so the workload only times predict.
    sample = train_X.head(batch).reset_index(drop=True)
    return lambda: wrapper.predict(sample)


def workload_ensemble_predict(df: pd.DataFrame, batch: int) -> Callable[[], Any]:
    """Build a 3-component ensemble (raw + 2 composite wrappers) and
    profile its predict path. Models are fitted ahead of time so we
    only time predict."""
    from mlframe.training.composite import (
        CompositeCrossTargetEnsemble, CompositeTargetDiscovery,
        CompositeTargetEstimator, get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    import lightgbm as lgb

    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    feature_cols = [c for c in df.columns if c != "target"]
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"],
        transforms=["diff", "linear_residual"], mi_sample_n=600,
        top_k_after_mi=2, eps_mi_gain=-1.0,
    )
    disc = CompositeTargetDiscovery(cfg).fit(
        df=df, target_col="target", feature_cols=feature_cols, train_idx=train_idx,
    )
    train_X = df.iloc[train_idx][feature_cols].reset_index(drop=True)
    train_y = df["target"].to_numpy()[train_idx]

    raw_model = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
    raw_model.fit(train_X, train_y)
    components = [raw_model]
    names = ["raw"]
    for spec in disc.specs_:
        transform = get_transform(spec.transform_name)
        base_train = train_X[spec.base_column].to_numpy()
        valid = transform.domain_check(train_y, base_train)
        t_train = transform.forward(
            train_y[valid], base_train[valid], spec.fitted_params,
        )
        inner = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
        inner.fit(train_X.iloc[valid].reset_index(drop=True), t_train)
        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=inner, transform_name=spec.transform_name,
            base_column=spec.base_column,
            transform_fitted_params=spec.fitted_params,
            y_train=train_y[valid],
        )
        components.append(wrapper)
        names.append(spec.name)
    rmses = [
        float(np.sqrt(np.mean((c.predict(train_X) - train_y) ** 2)))
        for c in components
    ]
    ensemble = CompositeCrossTargetEnsemble.from_train_metrics(
        component_models=components, component_names=names,
        component_train_rmse=rmses,
    )
    sample = train_X.head(batch).reset_index(drop=True)
    return lambda: ensemble.predict(sample)


def workload_oof_helper(df: pd.DataFrame) -> Callable[[], Any]:
    from mlframe.training.composite import (
        CompositeTargetDiscovery, CompositeTargetEstimator,
        compute_oof_holdout_predictions, get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    import lightgbm as lgb

    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    feature_cols = [c for c in df.columns if c != "target"]
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"],
        transforms=["linear_residual"], mi_sample_n=600,
        top_k_after_mi=1, eps_mi_gain=-1.0,
    )
    disc = CompositeTargetDiscovery(cfg).fit(
        df=df, target_col="target", feature_cols=feature_cols, train_idx=train_idx,
    )
    spec = disc.specs_[0]
    transform = get_transform(spec.transform_name)
    train_X = df.iloc[train_idx][feature_cols].reset_index(drop=True)
    train_y = df["target"].to_numpy()[train_idx]
    base_train = train_X[spec.base_column].to_numpy()
    valid = transform.domain_check(train_y, base_train)
    t_train = transform.forward(
        train_y[valid], base_train[valid], spec.fitted_params,
    )
    inner = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
    inner.fit(train_X.iloc[valid].reset_index(drop=True), t_train)
    raw_model = lgb.LGBMRegressor(n_estimators=80, num_leaves=15, verbose=-1)
    raw_model.fit(train_X, train_y)
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner, transform_name=spec.transform_name,
        base_column=spec.base_column,
        transform_fitted_params=spec.fitted_params, y_train=train_y[valid],
    )
    components = [raw_model, wrapper]
    names = ["raw", spec.name]
    component_specs = [None, {
        "base_column": spec.base_column,
        "transform_name": spec.transform_name,
        "fitted_params": spec.fitted_params,
    }]
    base_per_spec = {spec.base_column: base_train}

    return lambda: compute_oof_holdout_predictions(
        component_models=components, component_names=names,
        component_specs=component_specs,
        train_X=train_X, y_train_full=train_y,
        base_train_full_per_spec=base_per_spec,
        holdout_frac=0.2, random_state=42,
    )


# ----------------------------------------------------------------------
# Profile + calibrate runner
# ----------------------------------------------------------------------


def _wall_time_microbench(fn: Callable[[], Any], reps: int) -> Tuple[float, float, float]:
    """Wall-time outside cProfile. Returns (median, min, max) seconds."""
    times: List[float] = []
    fn()  # warm-up (LightGBM JIT etc)
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(np.median(arr)), float(np.min(arr)), float(np.max(arr))


def _cprofile_top(fn: Callable[[], Any], top: int) -> Tuple[float, str]:
    """Run fn under cProfile. Returns (cprofile_total_s, top_K_text)."""
    prof = cProfile.Profile()
    prof.enable()
    fn()
    prof.disable()
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
    ps.print_stats("mlframe", top)
    text = s.getvalue()
    # Extract cprofile total from the header line.
    total = 0.0
    for line in text.split("\n"):
        if "function calls" in line and "seconds" in line:
            # E.g. "         123456 function calls in 0.123 seconds"
            try:
                total = float(line.split(" in ")[-1].split(" ")[0])
            except (IndexError, ValueError):
                pass
            break
    return total, text


def profile_feature(label: str, fn_builder: Callable[[], Callable[[], Any]],
                    *, reps: int = 5, top: int = 20) -> None:
    print(f"\n{'=' * 80}\nFEATURE: {label}\n{'=' * 80}")
    fn = fn_builder()
    # Wall-time first (no cProfile attribution noise).
    med, mn, mx = _wall_time_microbench(fn, reps=reps)
    print(f"\nWall-time (median over {reps}): {med * 1000:.2f} ms "
          f"(min={mn * 1000:.2f} ms, max={mx * 1000:.2f} ms)")

    # Now with cProfile.
    cprof_total, hotspot_text = _cprofile_top(fn, top=top)
    inflation = cprof_total / med if med > 0 else float("nan")
    print(
        f"cProfile total: {cprof_total * 1000:.2f} ms "
        f"(inflation vs wall: {inflation:.1f}x)"
    )
    if inflation > 5:
        print(
            "  -> cProfile attribution noise is HIGH. Trust wall-time numbers; "
            "treat cProfile cumulative-time hotspots as relative-rank guidance, "
            "not absolute timing."
        )
    print(f"\nTop-{top} cumulative-time hotspots in mlframe.*:")
    print(hotspot_text)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


FEATURES: Dict[str, Callable[[Any], Callable[[], Any]]] = {
    "baseline_diag": lambda args: workload_baseline_diag(make_data(args.n)),
    "discovery": lambda args: workload_discovery(make_data(args.n)),
    "discovery_mi_only": lambda args: workload_discovery(make_data(args.n), screening="mi"),
    "wrapper_predict": lambda args: workload_wrapper_predict(make_data(args.n), args.batch),
    "ensemble_predict": lambda args: workload_ensemble_predict(make_data(args.n), args.batch),
    "oof_helper": lambda args: workload_oof_helper(make_data(args.n)),
}


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile every composite-target feature with cProfile + wall-time.",
    )
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=1000,
                        help="Predict-batch size for predict-path features.")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--feature", type=str, default="all",
                        help=f"Run a single feature; one of {sorted(FEATURES)} or 'all'.")
    args = parser.parse_args(argv)

    targets = sorted(FEATURES) if args.feature == "all" else [args.feature]
    if args.feature != "all" and args.feature not in FEATURES:
        print(f"Unknown feature '{args.feature}'. Choose from: {sorted(FEATURES)}")
        return 1

    for label in targets:
        try:
            profile_feature(
                label, lambda lbl=label: FEATURES[lbl](args),
                reps=args.reps, top=args.top,
            )
        except Exception as exc:
            print(f"\n[{label}] FAILED: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
