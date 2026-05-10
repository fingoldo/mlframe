"""Composite-target full S1-S16 benchmark suite.

Runs each scenario through CompositeTargetDiscovery + final tiny LGBM
fit, measures per-scenario test RMSE on a held-out split, and emits a
leaderboard. Drives the choice of defaults that
``CompositeTargetDiscoveryConfig`` ships with.

Replaces the legacy 5-scenario benchmark that called
``train_mlframe_models_suite`` (heavy, currently broken by an API
drift on ``_build_configs_from_params``). The new version uses
``CompositeTargetDiscovery`` directly + LightGBM for the final fit;
no full suite invocation, no API-drift fragility.

Scenarios (16 total)
--------------------

- S1  pure_lag                       diff should win
- S2  lag_with_signal                linear_residual should win
- S3  multiplicative                 logratio should win
- S4  proportional                   ratio should win
- S5  power_skew                     univariate transforms (out of scope)
- S6  lognormal_y                    logratio should win
- S7  multi_base                     linear_residual on stronger base
- S8  no_dominant_base               raw-y baseline gate should fire
- S9  mixed_regimes                  ambiguous winner (ensemble territory)
- S10 noisy_base                     linear_residual + raw-y gate
- S11 user_data_proxy                AR(1) lag + spatial coords (TVT-like)
- S12 distribution_shift             stationarity check
- S13 outliers_in_base               domain-validity skips
- S14 heteroscedasticity             logratio stabilises variance
- S15 categorical_base               type-check skip diff/ratio
- S16 false_positive_autobase        composite shouldn't help

Per scenario we report:
- chosen base + transform (top-1 spec from discovery)
- composite test RMSE
- raw-y test RMSE (baseline)
- improvement % vs raw (negative = composite worse)

Usage::

    python -m mlframe.benchmarks.composite_target_benchmark
    python -m mlframe.benchmarks.composite_target_benchmark --fast
    python -m mlframe.benchmarks.composite_target_benchmark --scenario S3_multiplicative

Outputs JSON and a matplotlib heatmap to ``benchmarks/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository")


# ----------------------------------------------------------------------
# Synthetic data generators
# ----------------------------------------------------------------------


def s1_pure_lag(n: int = 4000, seed: int = 0):
    """y = base + small noise. diff is the canonical winner."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10, scale=3, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = base + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", "diff"


def s2_lag_with_signal(n: int = 4000, seed: int = 0):
    """y = 0.95*base + g(X) + epsilon. linear_residual should win."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10, scale=3, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (0.95 * base + 1.5 * x1 - 0.8 * np.sin(x2 * 2)
         + rng.normal(scale=0.1, size=n))
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", "linear_residual"


def s3_multiplicative(n: int = 4000, seed: int = 0):
    """y = base * exp(g(X) + eps). logratio should win."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = base * np.exp(0.5 * x1 + 0.3 * np.sin(x2 * 2)
                      + rng.normal(scale=0.05, size=n))
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", "logratio"


def s4_proportional(n: int = 4000, seed: int = 0):
    """y = base * (1 + 0.1*g(X)) + small noise. ratio should win."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(low=2.0, high=10.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = base * (1.0 + 0.1 * (x1 + 0.5 * np.sin(x2 * 3))
                + rng.normal(scale=0.005, size=n))
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", "ratio"


def s5_power_skew(n: int = 4000, seed: int = 0):
    """y = base^2 + g(X). Univariate sqrt_y / log_y is OUT of the
    default registry. Sanity: composite shouldn't outperform raw by
    much; raw-y gate should keep things sensible."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(low=1.0, high=5.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = base ** 2 + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", None


def s6_lognormal_y(n: int = 4000, seed: int = 0):
    """log(y) = a*log(base) + g(X), y > 0. logratio should win."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    log_y = (0.9 * np.log(base) + 0.4 * x1 - 0.2 * np.sin(x2 * 2)
             + rng.normal(scale=0.05, size=n))
    y = np.exp(log_y)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", "logratio"


def s7_multi_base(n: int = 4000, seed: int = 0):
    """y = beta1*base1 + beta2*base2 + g(X). Single-base mode:
    linear_residual on the stronger of base1/base2 wins."""
    rng = np.random.default_rng(seed)
    base1 = rng.normal(loc=10, scale=3, size=n)
    base2 = rng.normal(loc=5, scale=2, size=n)
    x1 = rng.normal(size=n)
    y = (1.5 * base1 + 0.8 * base2 + 0.5 * x1
         + rng.normal(scale=0.1, size=n))
    df = pd.DataFrame({"base1": base1, "base2": base2, "x1": x1, "y": y})
    return df, "y", ["base1", "base2", "x1"], "base1", "linear_residual"


def s8_no_dominant_base(n: int = 4000, seed: int = 0):
    """y = g(X) + noise. No dominant base. Composite shouldn't help
    raw-y; raw-y gate should reject every composite."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = (1.5 * x1 - 0.8 * np.sin(x2 * 2) + 0.5 * x3
         + rng.normal(scale=0.1, size=n))
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    return df, "y", ["x1", "x2", "x3"], None, None


def s9_mixed_regimes(n: int = 4000, seed: int = 0):
    """50% rows from S2 (additive lag), 50% from S3 (multiplicative).
    Single-spec winner: ambiguous; cross-target ensemble territory."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    additive_mask = rng.random(n) < 0.5
    y = np.zeros(n)
    y[additive_mask] = (
        0.95 * base[additive_mask]
        + 1.5 * x1[additive_mask]
        - 0.8 * np.sin(x2[additive_mask] * 2)
        + rng.normal(scale=0.1, size=int(additive_mask.sum()))
    )
    y[~additive_mask] = (
        base[~additive_mask] * np.exp(
            0.5 * x1[~additive_mask]
            + 0.3 * np.sin(x2[~additive_mask] * 2)
            + rng.normal(scale=0.05, size=int((~additive_mask).sum()))
        )
    )
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": y})
    return df, "y", ["base", "x1", "x2"], "base", None


def s10_noisy_base(n: int = 4000, seed: int = 0):
    """y = 0.95*base_clean + epsilon, observed base = base_clean +
    measurement noise. linear_residual fits alpha properly."""
    rng = np.random.default_rng(seed)
    base_clean = rng.normal(loc=10, scale=3, size=n)
    measurement_noise = rng.normal(scale=1.0, size=n)
    base = base_clean + measurement_noise
    x1 = rng.normal(size=n)
    y = 0.95 * base_clean + 0.5 * x1 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"], "base", "linear_residual"


def s11_user_data_proxy(n: int = 4000, seed: int = 0):
    """AR(1) lag + spatial coordinates (TVT production failure mode
    proxy until privacy review allows real data)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=n)
    Y = rng.uniform(-2, 2, size=n)
    Z = rng.uniform(-2, 2, size=n)
    spatial_trend = 1.5 * np.sin(X) + 1.0 * np.cos(Y) + 0.7 * Z
    y = np.zeros(n)
    y[0] = spatial_trend[0]
    for i in range(1, n):
        y[i] = (0.92 * y[i - 1]
                + 0.06 * spatial_trend[i]
                + rng.normal(scale=0.15))
    y_prev = np.r_[y[0], y[:-1]]
    f1 = rng.normal(size=n)
    df = pd.DataFrame({
        "X": X, "Y": Y, "Z": Z, "y_prev": y_prev, "f1": f1, "y": y,
    })
    return df, "y", ["X", "Y", "Z", "y_prev", "f1"], "y_prev", "linear_residual"


def s12_distribution_shift(n: int = 4000, seed: int = 0):
    """base distribution shifts between train and test halves."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([
        rng.normal(loc=10, scale=2, size=n // 2),
        rng.normal(loc=15, scale=3, size=n - n // 2),
    ])
    x1 = rng.normal(size=n)
    y = base + 0.5 * x1 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"], "base", "diff"


def s13_outliers_in_base(n: int = 4000, seed: int = 0):
    """5% near-zero in base: ratio/logratio should skip via
    domain_check; diff/linear_residual remain fine."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(low=2.0, high=10.0, size=n)
    near_zero_mask = rng.random(n) < 0.05
    base[near_zero_mask] = rng.uniform(low=-0.01, high=0.01,
                                        size=int(near_zero_mask.sum()))
    x1 = rng.normal(size=n)
    y = base + 0.5 * x1 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"], "base", "diff"


def s14_heteroscedasticity(n: int = 4000, seed: int = 0):
    """y variance scales with base. logratio stabilises log-domain."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    y = base * (1.0 + 0.3 * x1) + rng.normal(scale=base * 0.1, size=n)
    y = np.maximum(y, 1e-3)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"], "base", "logratio"


def s15_categorical_base(n: int = 4000, seed: int = 0):
    """Categorical base column. diff/ratio/logratio should skip via
    numeric type check; linear_residual via OHE not in default
    registry. Expected: NO composite kept; raw-y wins."""
    rng = np.random.default_rng(seed)
    cat_base = rng.choice(["a", "b", "c"], size=n)
    cat_offset = (
        pd.Series(cat_base).map({"a": 0.0, "b": 5.0, "c": 10.0}).to_numpy()
    )
    x1 = rng.normal(size=n)
    y = cat_offset + 0.5 * x1 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"cat_base": cat_base, "x1": x1, "y": y})
    return df, "y", ["cat_base", "x1"], None, None


def s16_false_positive_autobase(n: int = 4000, seed: int = 0):
    """y and base correlated only via shared time trend. Composite
    shouldn't help; raw-y gate should reject."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    base = 5.0 * t + rng.normal(scale=0.5, size=n)
    x1 = rng.normal(size=n)
    y = 5.0 * t + 0.5 * x1 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"], None, None


SCENARIOS: Dict[str, Tuple[Callable, Optional[str]]] = {
    "S1_pure_lag":              (s1_pure_lag, "diff"),
    "S2_lag_with_signal":       (s2_lag_with_signal, "linear_residual"),
    "S3_multiplicative":        (s3_multiplicative, "logratio"),
    "S4_proportional":          (s4_proportional, "ratio"),
    "S5_power_skew":            (s5_power_skew, None),
    "S6_lognormal_y":           (s6_lognormal_y, "logratio"),
    "S7_multi_base":            (s7_multi_base, "linear_residual"),
    "S8_no_dominant_base":      (s8_no_dominant_base, None),
    "S9_mixed_regimes":         (s9_mixed_regimes, None),
    "S10_noisy_base":           (s10_noisy_base, "linear_residual"),
    "S11_user_data_proxy":      (s11_user_data_proxy, "linear_residual"),
    "S12_distribution_shift":   (s12_distribution_shift, "diff"),
    "S13_outliers_in_base":     (s13_outliers_in_base, "diff"),
    "S14_heteroscedasticity":   (s14_heteroscedasticity, "logratio"),
    "S15_categorical_base":     (s15_categorical_base, None),
    "S16_false_positive_autobase": (s16_false_positive_autobase, None),
}


# ----------------------------------------------------------------------
# Run one scenario
# ----------------------------------------------------------------------


def _run_scenario(
    scenario_fn: Callable,
    *,
    n: int, seed: int,
    expected_transform: Optional[str],
) -> Dict:
    from mlframe.training.composite import (
        CompositeTargetDiscovery, get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from lightgbm import LGBMRegressor

    df, target_col, feature_cols, expected_base, _ = scenario_fn(
        n=n, seed=seed,
    )
    cut = int(len(df) * 0.8)
    train_idx = np.arange(cut)
    test_idx = np.arange(cut, len(df))

    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="hybrid",
        mi_sample_n=1500,
        tiny_model_sample_n=1200,
        tiny_model_n_estimators=60,
        tiny_model_cv_folds=3,
        eps_mi_gain=-1.0,
        auto_base_top_k=3,
        top_k_after_mi=8,
        top_m_after_tiny=2,
        require_beats_raw_baseline=True,
        raw_baseline_tolerance=1.05,
        random_state=seed,
        transforms=["diff", "ratio", "logratio", "linear_residual"],
        use_baseline_diagnostics_hint=False,
    )
    disc = CompositeTargetDiscovery(cfg)
    try:
        disc.fit(df=df, target_col=target_col, feature_cols=feature_cols,
                 train_idx=train_idx)
    except Exception as exc:
        return {
            "n_specs": 0, "top_base": None, "top_transform": None,
            "test_rmse": float("nan"), "raw_rmse": float("nan"),
            "expected_transform": expected_transform,
            "expected_base": expected_base,
            "discovery_error": str(exc),
        }

    # Build numeric feature matrix.
    numeric_x_cols: List[str] = []
    df_lgbm = df.copy()
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_lgbm[c]):
            df_lgbm[c] = pd.Categorical(df_lgbm[c]).codes.astype(np.float64)
        numeric_x_cols.append(c)
    y_train = df[target_col].to_numpy()[train_idx]
    y_test = df[target_col].to_numpy()[test_idx]
    inner = LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        random_state=seed, verbosity=-1,
    )
    raw_X_train = df_lgbm[numeric_x_cols].to_numpy()[train_idx]
    raw_X_test = df_lgbm[numeric_x_cols].to_numpy()[test_idx]
    inner.fit(raw_X_train, y_train)
    raw_rmse = float(np.sqrt(mean_squared_error(
        y_test, inner.predict(raw_X_test))))

    if not disc.specs_:
        return {
            "n_specs": 0, "top_base": None, "top_transform": None,
            "test_rmse": float("nan"), "raw_rmse": raw_rmse,
            "expected_transform": expected_transform,
            "expected_base": expected_base,
        }

    spec = disc.specs_[0]
    transform = get_transform(spec.transform_name)
    base_train = df[spec.base_column].to_numpy()[train_idx]
    base_test = df[spec.base_column].to_numpy()[test_idx]
    valid_train = transform.domain_check(y_train, base_train)
    if valid_train.sum() < 50:
        return {
            "n_specs": len(disc.specs_),
            "top_base": spec.base_column,
            "top_transform": spec.transform_name,
            "test_rmse": float("nan"), "raw_rmse": raw_rmse,
            "expected_transform": expected_transform,
            "expected_base": expected_base,
        }
    t_train = transform.forward(
        y_train[valid_train], base_train[valid_train], spec.fitted_params,
    )
    inner_c = LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        random_state=seed, verbosity=-1,
    )
    x_cols_no_base = [c for c in numeric_x_cols if c != spec.base_column]
    if not x_cols_no_base:
        return {
            "n_specs": len(disc.specs_),
            "top_base": spec.base_column,
            "top_transform": spec.transform_name,
            "test_rmse": float("nan"), "raw_rmse": raw_rmse,
            "expected_transform": expected_transform,
            "expected_base": expected_base,
        }
    X_train_c = df_lgbm[x_cols_no_base].to_numpy()[train_idx][valid_train]
    X_test_c = df_lgbm[x_cols_no_base].to_numpy()[test_idx]
    inner_c.fit(X_train_c, t_train)
    t_hat_test = inner_c.predict(X_test_c)
    y_hat_test = transform.inverse(t_hat_test, base_test, spec.fitted_params)
    finite = np.isfinite(y_hat_test - y_test)
    composite_rmse = (
        float(np.sqrt(mean_squared_error(
            y_test[finite], y_hat_test[finite])))
        if finite.any() else float("nan")
    )
    return {
        "n_specs": len(disc.specs_),
        "top_base": spec.base_column,
        "top_transform": spec.transform_name,
        "test_rmse": composite_rmse,
        "raw_rmse": raw_rmse,
        "expected_transform": expected_transform,
        "expected_base": expected_base,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--fast", action="store_true",
                        help="reps=1, n=2000")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()))
    args = parser.parse_args()

    if args.fast:
        args.reps = 1
        args.n = 2000

    scenarios = (
        [args.scenario] if args.scenario else list(SCENARIOS.keys())
    )

    print("Composite-target full S1-S16 benchmark")
    print(f"  reps      = {args.reps}")
    print(f"  n         = {args.n}")
    print(f"  scenarios = {len(scenarios)}\n")

    summary: List[Dict] = []

    for scenario_name in scenarios:
        gen, expected_transform = SCENARIOS[scenario_name]
        print(f"--- {scenario_name} (expected_transform={expected_transform}) ---")
        per_rep_results: List[Dict] = []
        for rep in range(args.reps):
            t0 = time.perf_counter()
            r = _run_scenario(
                gen, n=args.n, seed=rep,
                expected_transform=expected_transform,
            )
            dt = time.perf_counter() - t0
            r["elapsed_s"] = dt
            per_rep_results.append(r)
            ce = r.get("discovery_error")
            print(f"  rep={rep}: base={r['top_base']!s:>10s} "
                  f"transform={r['top_transform']!s:>16s} "
                  f"test_rmse={r['test_rmse']:.4f} "
                  f"raw_rmse={r['raw_rmse']:.4f} "
                  f"({dt:.1f}s)"
                  + (f" ERROR={ce}" if ce else ""))

        valid = [r for r in per_rep_results
                 if np.isfinite(r["test_rmse"])
                 and np.isfinite(r["raw_rmse"])]
        if valid:
            mean_composite = float(np.mean([r["test_rmse"] for r in valid]))
            mean_raw = float(np.mean([r["raw_rmse"] for r in valid]))
            improvement_pct = (
                (mean_raw - mean_composite) / mean_raw * 100.0
                if mean_raw > 0 else float("nan")
            )
        else:
            mean_composite = float("nan")
            mean_raw = float(np.mean(
                [r["raw_rmse"] for r in per_rep_results
                 if np.isfinite(r["raw_rmse"])]
            )) if per_rep_results else float("nan")
            improvement_pct = float("nan")

        if expected_transform:
            correct_count = sum(
                1 for r in per_rep_results
                if r["top_transform"] == expected_transform
            )
        else:
            correct_count = None

        any_specs = sum(1 for r in per_rep_results
                        if r.get("n_specs", 0) > 0)

        print(f"  Summary: composite RMSE={mean_composite:.4f} "
              f"raw RMSE={mean_raw:.4f} "
              f"improvement={improvement_pct:+.2f}%"
              + (f"  transform_correct={correct_count}/{args.reps}"
                 if correct_count is not None else "")
              + f"  any_specs={any_specs}/{args.reps}\n")

        summary.append({
            "scenario": scenario_name,
            "expected_transform": expected_transform,
            "mean_composite_rmse": mean_composite,
            "mean_raw_rmse": mean_raw,
            "improvement_pct": improvement_pct,
            "transform_correct_count": correct_count,
            "any_specs_count": any_specs,
            "n_reps": args.reps,
            "per_rep_results": per_rep_results,
        })

    print("=" * 92)
    print("LEADERBOARD")
    print("=" * 92)
    print(f"{'scenario':<32s} {'expected':>16s} {'composite':>10s} "
          f"{'raw':>10s} {'imprv%':>8s}  pick")
    for row in summary:
        et = row["expected_transform"] or "-"
        c = row["mean_composite_rmse"]
        r_ = row["mean_raw_rmse"]
        i = row["improvement_pct"]
        cc = (f"{row['transform_correct_count']}/{row['n_reps']}"
              if row["transform_correct_count"] is not None
              else "n/a")
        c_str = f"{c:.4f}" if np.isfinite(c) else "  nan"
        r_str = f"{r_:.4f}" if np.isfinite(r_) else "  nan"
        i_str = f"{i:+.2f}" if np.isfinite(i) else "  nan"
        print(f"{row['scenario']:<32s} {et:>16s} {c_str:>10s} "
              f"{r_str:>10s} {i_str:>8s}  {cc}")

    out_path = "benchmarks/composite_target_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Heatmap.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [row["scenario"] for row in summary]
        improvs = np.array([
            row["improvement_pct"] if np.isfinite(row["improvement_pct"])
            else 0.0
            for row in summary
        ]).reshape(1, -1)
        fig, ax = plt.subplots(figsize=(16, 3))
        im = ax.imshow(improvs, cmap="RdYlGn", aspect="auto",
                       vmin=-10, vmax=50)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels(["improvement % vs raw"])
        for i in range(len(names)):
            v = improvs[0, i]
            text_color = "white" if abs(v) > 25 else "black"
            ax.text(i, 0, f"{v:+.1f}", ha="center", va="center",
                    color=text_color, fontsize=8)
        fig.colorbar(im, ax=ax, label="improvement % vs raw-y")
        ax.set_title("Composite-target benchmark: per-scenario improvement vs raw")
        fig.tight_layout()
        plot_path = "benchmarks/composite_target_benchmark_heatmap.png"
        fig.savefig(plot_path, dpi=110)
        print(f"Heatmap written to {plot_path}")
    except Exception as plot_err:
        print(f"(heatmap skipped: {plot_err})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
