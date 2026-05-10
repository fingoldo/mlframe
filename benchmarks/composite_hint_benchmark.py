"""Benchmark: BaselineDiagnostics ablation hint vs MI-only auto-base.

Targets the production failure mode: a target ``y`` that has both
(a) a dominant lag-style feature (``TVT_prev``-analogue) AND
(b) high-MI-but-low-residual-signal features (spatial coordinates with
global trend).

Pairwise ``MI(y, x)`` ranking can prefer the spatial coords because
their pairwise MI happens to be larger on the screening sample, even
though their predictive contribution (measured by ablation) is small.
The ablation hint captures the right signal directly.

Metrics
-------
- ``base_choice_correctness``: did auto-base put the dominant lag in
  the top-1 slot? Binary per run.
- ``final_oof_rmse``: end-to-end OOF RMSE on a held-out test split,
  comparing the model trained on the *first* discovered composite
  (with hint) vs without hint vs raw-y baseline. Lower is better.

Usage::

    python -m mlframe.benchmarks.composite_hint_benchmark --reps 5
    python -m mlframe.benchmarks.composite_hint_benchmark --fast

Output: per-scenario table + verdict line (improvement %).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository")


# ----------------------------------------------------------------------
# Synthetic scenarios
# ----------------------------------------------------------------------


def make_lag_dominant_with_spatial_distractors(
    n: int = 4000, seed: int = 0,
) -> Tuple[pd.DataFrame, str, List[str], str]:
    """The production failure analogue.

    y = AR(1) lag dominant + small spatial trend + noise.
    Spatial coords X, Y, Z have moderate pairwise MI(y, x) due to the
    trend but contribute negligibly to RMSE on ablation.
    y_prev (lag-1) is the *true* dominant feature: dropping it
    multiplies RMSE by ~5x.
    """
    rng = np.random.default_rng(seed)
    # Spatial structure: y has a smooth trend over (X, Y, Z) coords
    # with moderate dependency. Pairwise MI(y, X) ~ 1.5 due to trend.
    X = rng.uniform(-2, 2, size=n)
    Y = rng.uniform(-2, 2, size=n)
    Z = rng.uniform(-2, 2, size=n)
    spatial_trend = 1.5 * np.sin(X) + 1.0 * np.cos(Y) + 0.7 * Z

    # AR(1) on spatial-conditioned residual. The lag-1 dominates.
    y = np.zeros(n)
    y[0] = spatial_trend[0]
    for i in range(1, n):
        y[i] = (0.92 * y[i - 1]                      # lag dominance
                + 0.06 * spatial_trend[i]            # weak spatial
                + rng.normal(scale=0.15))
    y_prev = np.r_[y[0], y[:-1]]
    f1 = rng.normal(size=n)  # noise feature
    f2 = rng.normal(size=n)  # noise feature

    df = pd.DataFrame({
        "X": X, "Y": Y, "Z": Z,
        "y_prev": y_prev,
        "f1": f1, "f2": f2,
        "y": y,
    })
    return df, "y", ["X", "Y", "Z", "y_prev", "f1", "f2"], "y_prev"


def make_multibase_with_distractor(
    n: int = 4000, seed: int = 0,
) -> Tuple[pd.DataFrame, str, List[str], str]:
    """Two contender bases: linear-residual-friendly ``base_a``
    (clean linear contribution) vs ``distractor`` (high non-linear
    MI with y but contributes nothing to a linear fit). Truth is
    base_a; auto-base by MI may rank distractor higher.
    """
    rng = np.random.default_rng(seed)
    base_a = rng.normal(size=n)
    distractor = rng.uniform(-3, 3, size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    # y depends linearly on base_a + small structural f1/f2 + non-
    # linear bond to distractor (raises MI but contributes ~0 in RMSE).
    y = (1.0 * base_a + 0.3 * f1 - 0.2 * f2
         + 0.4 * np.sin(distractor * 4)
         + rng.normal(scale=0.05))
    df = pd.DataFrame({
        "base_a": base_a, "distractor": distractor,
        "f1": f1, "f2": f2, "y": y,
    })
    return df, "y", ["base_a", "distractor", "f1", "f2"], "base_a"


def make_heavy_tail_unstable_mi(
    n: int = 4000, seed: int = 0,
) -> Tuple[pd.DataFrame, str, List[str], str]:
    """Heavy-tail y where pairwise MI estimation on the screening
    sample is unstable. The TRUE dominant base is ``base_clean``
    (small linear contribution, no tail interaction), but tail rows
    contain a misleading non-linear bond between ``base_misleading``
    and y that inflates pairwise MI when sampling lands on those
    rows. MI ranking flips between bases run-to-run; ablation is
    stable because it uses the full dataset.
    """
    rng = np.random.default_rng(seed)
    base_clean = rng.normal(size=n)
    # Mostly noise, with a tail (~5%) where it spikes y heavily.
    base_misleading = rng.normal(size=n)
    tail_mask = rng.random(n) < 0.05
    # base_misleading carries a heavy non-linear bond ONLY in tail rows.
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    y_clean = (1.5 * base_clean + 0.4 * f1 - 0.3 * f2
               + rng.normal(scale=0.1, size=n))
    # Tail-row bonus that inflates MI(y, base_misleading) on the
    # screening sample if tail rows happen to be over-represented.
    y = y_clean.copy()
    y[tail_mask] += 50.0 * np.sin(base_misleading[tail_mask] * 6)
    df = pd.DataFrame({
        "base_clean": base_clean,
        "base_misleading": base_misleading,
        "f1": f1, "f2": f2, "y": y,
    })
    return df, "y", ["base_clean", "base_misleading", "f1", "f2"], "base_clean"


SCENARIOS = {
    "lag_dominant_with_spatial_distractors":
        make_lag_dominant_with_spatial_distractors,
    "multibase_with_distractor":
        make_multibase_with_distractor,
    "heavy_tail_unstable_mi":
        make_heavy_tail_unstable_mi,
}


# ----------------------------------------------------------------------
# Run a single comparison
# ----------------------------------------------------------------------


def _run_with_config(
    df: pd.DataFrame, target_col: str, feature_cols: List[str],
    train_idx: np.ndarray, test_idx: np.ndarray,
    *, hint: List[str] | None,
    seed: int,
) -> Dict:
    """Run discovery with / without hint, fit a tiny model on the
    top-1 composite, return OOF RMSE on test_idx + chosen base."""
    from mlframe.training.composite import (
        CompositeTargetDiscovery,
        CompositeTargetEstimator,
        get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    # Small ``mi_sample_n`` is the regime where pairwise MI
    # estimation is noisiest and the hint matters most. Production
    # 4M-row runs use mi_sample_n=200_000 (1-5% sample); the synthetic
    # benchmark mirrors that ratio (~10% of n) plus a small-sample
    # variant in scenario "heavy_tail_unstable_mi" to expose noise.
    cfg_kwargs = dict(
        enabled=True, screening="hybrid",
        mi_sample_n=400,
        tiny_model_sample_n=800,
        tiny_model_n_estimators=60,
        tiny_model_cv_folds=3,
        eps_mi_gain=-1.0,
        auto_base_top_k=3,
        top_k_after_mi=8,
        top_m_after_tiny=2,
        require_beats_raw_baseline=True,
        raw_baseline_tolerance=1.05,
        random_state=seed,
        transforms=["diff", "linear_residual"],
    )
    if hint is not None:
        cfg_kwargs["dominant_features_hint"] = hint
    cfg = CompositeTargetDiscoveryConfig(**cfg_kwargs)
    disc = CompositeTargetDiscovery(cfg)
    disc.fit(df=df, target_col=target_col, feature_cols=feature_cols,
             train_idx=train_idx)
    if not disc.specs_:
        return {
            "kept_specs": 0, "top_base": None, "top_transform": None,
            "test_rmse": float("nan"),
            "raw_baseline_rmse": disc.raw_y_baseline_rmse_,
        }

    spec = disc.specs_[0]
    transform = get_transform(spec.transform_name)

    # Train a final small LGBM on the chosen composite over train_idx,
    # measure RMSE on test_idx.
    from lightgbm import LGBMRegressor
    base_train = df[spec.base_column].to_numpy()[train_idx]
    base_test = df[spec.base_column].to_numpy()[test_idx]
    y_train = df[target_col].to_numpy()[train_idx]
    y_test = df[target_col].to_numpy()[test_idx]
    x_cols = [c for c in feature_cols if c != spec.base_column]
    X_train = df[x_cols].to_numpy()[train_idx]
    X_test = df[x_cols].to_numpy()[test_idx]

    valid = transform.domain_check(y_train, base_train)
    if valid.sum() < 50:
        return {
            "kept_specs": len(disc.specs_),
            "top_base": spec.base_column,
            "top_transform": spec.transform_name,
            "test_rmse": float("nan"),
            "raw_baseline_rmse": disc.raw_y_baseline_rmse_,
        }
    t_train = transform.forward(
        y_train[valid], base_train[valid], spec.fitted_params,
    )
    inner = LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        random_state=seed, verbosity=-1,
    )
    inner.fit(X_train[valid], t_train)
    t_hat_test = inner.predict(X_test)
    y_hat_test = transform.inverse(t_hat_test, base_test, spec.fitted_params)
    finite = np.isfinite(y_hat_test - y_test)
    rmse = float(np.sqrt(mean_squared_error(
        y_test[finite], y_hat_test[finite])))
    return {
        "kept_specs": len(disc.specs_),
        "top_base": spec.base_column,
        "top_transform": spec.transform_name,
        "test_rmse": rmse,
        "raw_baseline_rmse": disc.raw_y_baseline_rmse_,
    }


def _run_raw_y_only(
    df: pd.DataFrame, target_col: str, feature_cols: List[str],
    train_idx: np.ndarray, test_idx: np.ndarray, seed: int,
) -> float:
    """Tiny LGBM trained directly on raw y; reference RMSE on test."""
    from lightgbm import LGBMRegressor
    X_train = df[feature_cols].to_numpy()[train_idx]
    X_test = df[feature_cols].to_numpy()[test_idx]
    y_train = df[target_col].to_numpy()[train_idx]
    y_test = df[target_col].to_numpy()[test_idx]
    inner = LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        random_state=seed, verbosity=-1,
    )
    inner.fit(X_train, y_train)
    y_hat = inner.predict(X_test)
    return float(np.sqrt(mean_squared_error(y_test, y_hat)))


def _ablation_top_k(
    df: pd.DataFrame, target_col: str, feature_cols: List[str],
    train_idx: np.ndarray, k: int = 3, seed: int = 0,
) -> List[str]:
    """Quick ablation: train baseline on all features, then drop one
    at a time and measure RMSE delta on a held-out slice. Return
    top-K by delta_pct (proxy for BaselineDiagnostics ablation)."""
    from lightgbm import LGBMRegressor
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx.size)
    cut = int(train_idx.size * 0.8)
    inner_train = train_idx[perm[:cut]]
    inner_val = train_idx[perm[cut:]]
    X_full_train = df[feature_cols].to_numpy()[inner_train]
    y_full_train = df[target_col].to_numpy()[inner_train]
    X_val = df[feature_cols].to_numpy()[inner_val]
    y_val = df[target_col].to_numpy()[inner_val]
    base_model = LGBMRegressor(
        n_estimators=80, num_leaves=15, learning_rate=0.1,
        random_state=seed, verbosity=-1,
    )
    base_model.fit(X_full_train, y_full_train)
    base_rmse = float(np.sqrt(mean_squared_error(
        y_val, base_model.predict(X_val))))
    deltas: List[Tuple[str, float]] = []
    for drop_col in feature_cols:
        kept = [c for c in feature_cols if c != drop_col]
        if not kept:
            continue
        X_drop_train = df[kept].to_numpy()[inner_train]
        X_drop_val = df[kept].to_numpy()[inner_val]
        m = LGBMRegressor(
            n_estimators=80, num_leaves=15, learning_rate=0.1,
            random_state=seed, verbosity=-1,
        )
        m.fit(X_drop_train, y_full_train)
        rmse = float(np.sqrt(mean_squared_error(
            y_val, m.predict(X_drop_val))))
        delta_pct = ((rmse - base_rmse) / base_rmse) * 100.0 if base_rmse > 0 else 0.0
        deltas.append((drop_col, delta_pct))
    deltas.sort(key=lambda t: -t[1])
    return [c for c, _ in deltas[:k]]


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--fast", action="store_true",
                        help="reps=2, n=2000")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()))
    args = parser.parse_args()

    if args.fast:
        args.reps = 2
        args.n = 2000

    scenarios = (
        [args.scenario] if args.scenario else list(SCENARIOS.keys())
    )

    print("Composite-target hint benchmark (BaselineDiagnostics ablation "
          "vs MI-only auto-base)\n")
    print(f"  reps         = {args.reps}")
    print(f"  n            = {args.n}")
    print(f"  scenarios    = {scenarios}\n")

    summary_rows: List[Dict] = []

    for scenario_name in scenarios:
        gen = SCENARIOS[scenario_name]
        print(f"--- Scenario: {scenario_name} ---")

        rmse_no_hint: List[float] = []
        rmse_with_hint: List[float] = []
        rmse_raw: List[float] = []
        base_correct_no_hint = 0
        base_correct_with_hint = 0

        for rep in range(args.reps):
            df, target_col, feature_cols, true_dominant_base = gen(
                n=args.n, seed=rep,
            )
            # Time-respecting split: first 80% train, last 20% test.
            cut = int(len(df) * 0.8)
            train_idx = np.arange(cut)
            test_idx = np.arange(cut, len(df))

            t0 = time.perf_counter()
            r_no_hint = _run_with_config(
                df, target_col, feature_cols, train_idx, test_idx,
                hint=None, seed=rep,
            )
            dt_no_hint = time.perf_counter() - t0

            # Compute hint via tiny ablation (proxy for BaselineDiagnostics).
            ablation_hint = _ablation_top_k(
                df, target_col, feature_cols, train_idx,
                k=3, seed=rep,
            )
            t0 = time.perf_counter()
            r_with_hint = _run_with_config(
                df, target_col, feature_cols, train_idx, test_idx,
                hint=ablation_hint, seed=rep,
            )
            dt_with_hint = time.perf_counter() - t0

            r_raw = _run_raw_y_only(
                df, target_col, feature_cols, train_idx, test_idx, rep,
            )

            rmse_no_hint.append(r_no_hint["test_rmse"])
            rmse_with_hint.append(r_with_hint["test_rmse"])
            rmse_raw.append(r_raw)
            if r_no_hint["top_base"] == true_dominant_base:
                base_correct_no_hint += 1
            if r_with_hint["top_base"] == true_dominant_base:
                base_correct_with_hint += 1
            print(f"  rep={rep}: "
                  f"no_hint base={r_no_hint['top_base']:>10s} "
                  f"rmse={r_no_hint['test_rmse']:.4f}; "
                  f"with_hint base={r_with_hint['top_base']:>10s} "
                  f"rmse={r_with_hint['test_rmse']:.4f}; "
                  f"raw rmse={r_raw:.4f} "
                  f"(ablation_hint={ablation_hint})")

        rmse_no_hint_arr = np.array([r for r in rmse_no_hint
                                      if np.isfinite(r)])
        rmse_with_hint_arr = np.array([r for r in rmse_with_hint
                                        if np.isfinite(r)])
        rmse_raw_arr = np.array([r for r in rmse_raw
                                  if np.isfinite(r)])
        no_hint_mean = (rmse_no_hint_arr.mean()
                        if rmse_no_hint_arr.size else float("nan"))
        with_hint_mean = (rmse_with_hint_arr.mean()
                          if rmse_with_hint_arr.size else float("nan"))
        raw_mean = (rmse_raw_arr.mean()
                    if rmse_raw_arr.size else float("nan"))

        if (np.isfinite(no_hint_mean) and np.isfinite(with_hint_mean)
                and no_hint_mean > 0):
            improvement_pct = ((no_hint_mean - with_hint_mean)
                               / no_hint_mean) * 100.0
        else:
            improvement_pct = float("nan")

        print(f"\n  Summary (mean across reps):")
        print(f"    raw-y baseline RMSE       = {raw_mean:.4f}")
        print(f"    composite no_hint RMSE    = {no_hint_mean:.4f} "
              f"(base correct: {base_correct_no_hint}/{args.reps})")
        print(f"    composite with_hint RMSE  = {with_hint_mean:.4f} "
              f"(base correct: {base_correct_with_hint}/{args.reps})")
        print(f"    Hint improvement on RMSE  = {improvement_pct:+.2f}%")
        print(f"    Composite vs raw          = "
              f"with_hint {((raw_mean - with_hint_mean) / raw_mean * 100.0):+.2f}% "
              f"of raw\n")
        summary_rows.append({
            "scenario": scenario_name,
            "raw_rmse": float(raw_mean),
            "no_hint_rmse": float(no_hint_mean),
            "with_hint_rmse": float(with_hint_mean),
            "improvement_pct": float(improvement_pct),
            "base_correct_no_hint": base_correct_no_hint,
            "base_correct_with_hint": base_correct_with_hint,
            "n_reps": args.reps,
        })

    print("=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    for row in summary_rows:
        print(
            f"{row['scenario']:>50s}: hint_improves={row['improvement_pct']:+.2f}%, "
            f"base_correct {row['base_correct_no_hint']} -> "
            f"{row['base_correct_with_hint']} (of {row['n_reps']})"
        )

    out_path = "benchmarks/composite_hint_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
