"""Wide ensemble-strategy shootout for composite-target cross-target.

Tests 11 ensemble strategies (mean / median / trimmed_mean /
inverse_rmse / inverse_variance / bma_softmax / oof_weighted /
linear_stack_ridge / nnls_stack / stacked_gbdt / best_single_by_train)
across 6 representative synthetic scenarios x 3 seeds, then picks
the global winner by mean test-RMSE-improvement-vs-best_single.

Goal: replace the current R10b default
``CompositeTargetDiscoveryConfig.cross_target_ensemble_strategy =
"oof_weighted"`` with whichever strategy demonstrably wins on the
broader benchmark.

Rationale: the single-fixture demo run on 2026-05-10 showed
``oof_weighted`` losing 12.6% to best_single while ``nnls_stack``
won by 1.06%. One fixture is not enough for a default flip --
this script provides the broader evidence.

Usage::

    python -m mlframe.benchmarks.composite_ensemble_shootout
    python -m mlframe.benchmarks.composite_ensemble_shootout --fast

Outputs JSON to ``benchmarks/composite_ensemble_shootout_results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository/mlframe")
sys.path.insert(0, "D:/Upd/Programming/PythonCodeRepository")


# ----------------------------------------------------------------------
# Scenario fixtures (each yields multiple correlated bases for
# composite ensembling)
# ----------------------------------------------------------------------


def sc_two_decorrelated_bases(n: int = 4000, seed: int = 0):
    """y = a*base_1 + b*base_2 + g(X) + noise. Each composite spec
    captures a different additive contribution; ensembling decorrelates
    errors."""
    rng = np.random.default_rng(seed)
    base_a = rng.normal(loc=10, scale=2, size=n)
    base_b = rng.normal(loc=5, scale=1, size=n)
    x1 = rng.normal(size=n)
    y = (0.6 * base_a + 0.7 * base_b + 0.4 * x1
         + rng.normal(scale=0.3, size=n))
    df = pd.DataFrame({
        "base_a": base_a, "base_b": base_b, "x1": x1, "y": y,
    })
    return df, "y", ["base_a", "base_b", "x1"]


def sc_three_lag_variants(n: int = 4000, seed: int = 0):
    """Industrial lag set: y_prev, y_prev_smooth, y_prev_lag2 all
    correlate strongly with y. Tests how ensemble handles redundant
    bases (ensemble should weight one or two heavily, not split
    uniformly)."""
    rng = np.random.default_rng(seed)
    y_prev = rng.normal(loc=0, scale=2, size=n)
    # smooth = 3-period rolling-mean approximation.
    y_prev_smooth = np.copy(y_prev)
    for i in range(2, n):
        y_prev_smooth[i] = (y_prev[i - 2] + y_prev[i - 1] + y_prev[i]) / 3.0
    # lag2 = y_prev shifted by 1.
    y_prev_lag2 = np.r_[y_prev[0], y_prev[:-1]]
    x1 = rng.normal(size=n)
    y = 0.95 * y_prev + 0.5 * x1 + rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({
        "y_prev": y_prev, "y_prev_smooth": y_prev_smooth,
        "y_prev_lag2": y_prev_lag2, "x1": x1, "y": y,
    })
    return df, "y", ["y_prev", "y_prev_smooth", "y_prev_lag2", "x1"]


def sc_heteroscedastic_logratio(n: int = 4000, seed: int = 0):
    """y = base * exp(g(X) + noise) with heavy-tail base; logratio
    composite is correct on average but residuals scale with base."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    y = base * np.exp(0.5 * x1 + rng.normal(scale=0.05, size=n))
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"]


def sc_mixed_regimes(n: int = 4000, seed: int = 0):
    """50% rows additive, 50% multiplicative. No single transform
    fits everywhere; ensemble of diff + logratio could plausibly win."""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    additive = rng.random(n) < 0.5
    y = np.zeros(n)
    y[additive] = (
        0.95 * base[additive] + 1.5 * x1[additive]
        + rng.normal(scale=0.1, size=int(additive.sum()))
    )
    y[~additive] = (
        base[~additive] * np.exp(
            0.5 * x1[~additive]
            + rng.normal(scale=0.05, size=int((~additive).sum()))
        )
    )
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"]


def sc_noisy_borderline(n: int = 4000, seed: int = 0):
    """Borderline case: one base is real, four are noise. Ensemble
    needs to weight the real one heavily and discount noise."""
    rng = np.random.default_rng(seed)
    base_real = rng.normal(loc=10, scale=2, size=n)
    noise_a = rng.normal(size=n)
    noise_b = rng.normal(size=n)
    noise_c = rng.normal(size=n)
    x1 = rng.normal(size=n)
    y = 0.95 * base_real + 0.4 * x1 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({
        "base_real": base_real, "noise_a": noise_a, "noise_b": noise_b,
        "noise_c": noise_c, "x1": x1, "y": y,
    })
    return df, "y", ["base_real", "noise_a", "noise_b", "noise_c", "x1"]


def sc_distribution_shift(n: int = 4000, seed: int = 0):
    """base distribution shifts between train and test halves.
    Tests how ensemble strategies generalise across distributional
    drift."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([
        rng.normal(loc=10, scale=2, size=n // 2),
        rng.normal(loc=15, scale=3, size=n - n // 2),
    ])
    x1 = rng.normal(size=n)
    y = 0.97 * base + 0.5 * x1 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "y": y})
    return df, "y", ["base", "x1"]


SCENARIOS: Dict[str, Callable] = {
    "two_decorrelated_bases":    sc_two_decorrelated_bases,
    "three_lag_variants":        sc_three_lag_variants,
    "heteroscedastic_logratio":  sc_heteroscedastic_logratio,
    "mixed_regimes":             sc_mixed_regimes,
    "noisy_borderline":          sc_noisy_borderline,
    "distribution_shift":        sc_distribution_shift,
}


# ----------------------------------------------------------------------
# Composite training + ensemble strategies
# ----------------------------------------------------------------------


def _train_composite_specs(
    df, target_col, feature_cols, train_idx, test_idx, seed,
):
    """Train each (base, transform) composite spec via discovery,
    return per-spec train/test predictions and component models."""
    from mlframe.training.composite import (
        CompositeTargetDiscovery, CompositeTargetEstimator, get_transform,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from lightgbm import LGBMRegressor

    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, screening="hybrid",
        mi_sample_n=1500, tiny_model_sample_n=1200,
        tiny_model_n_estimators=60, tiny_model_cv_folds=3,
        eps_mi_gain=-1.0, top_k_after_mi=8, top_m_after_tiny=4,
        random_state=seed, auto_base_top_k=4,
        require_beats_raw_baseline=False, per_bin_n_bins=0,
        cross_target_ensemble_strategy="off",
        use_baseline_diagnostics_hint=False,
        transforms=["diff", "linear_residual", "ratio", "logratio"],
        collapse_linear_residual_alpha_eps=0.0,
        auto_base_dedup_corr_threshold=1.0,
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
        detect_linear_residual_alpha_drift=False,
    )
    disc = CompositeTargetDiscovery(cfg).fit(
        df, target_col=target_col, feature_cols=feature_cols,
        train_idx=train_idx,
    )
    if not disc.specs_:
        return None
    train_preds: List[np.ndarray] = []
    test_preds: List[np.ndarray] = []
    component_models: List[Any] = []
    component_names: List[str] = []
    component_train_rmse: List[float] = []
    y_train_arr = df[target_col].iloc[train_idx].to_numpy()
    for spec in disc.specs_:
        t = get_transform(spec.transform_name)
        b_tr = df[spec.base_column].iloc[train_idx].to_numpy()
        b_te = df[spec.base_column].iloc[test_idx].to_numpy()
        valid = t.domain_check(y_train_arr, b_tr)
        if valid.sum() < 50:
            continue
        t_tr = t.forward(
            y_train_arr[valid], b_tr[valid], spec.fitted_params,
        )
        x_cols = [c for c in feature_cols if c != spec.base_column]
        if not x_cols:
            continue
        m = LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            random_state=seed, verbosity=-1,
        )
        # Train on rows where transform domain holds.
        x_train_full = df[x_cols].iloc[train_idx].to_numpy()
        m.fit(x_train_full[valid], t_tr)
        # Predict on FULL train + test sets (including domain-violated
        # train rows -- LGBM will extrapolate; if ``transform.inverse``
        # then produces NaN we fall through to the spec-skip below).
        t_hat_tr = m.predict(x_train_full)
        t_hat_te = m.predict(df[x_cols].iloc[test_idx].to_numpy())
        y_hat_tr = t.inverse(t_hat_tr, b_tr, spec.fitted_params)
        y_hat_te = t.inverse(t_hat_te, b_te, spec.fitted_params)
        # Sanity: drop specs with non-finite predictions.
        if (not np.all(np.isfinite(y_hat_tr))
                or not np.all(np.isfinite(y_hat_te))):
            continue
        train_preds.append(y_hat_tr)
        test_preds.append(y_hat_te)
        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=m, transform_name=spec.transform_name,
            base_column=spec.base_column,
            transform_fitted_params=spec.fitted_params,
            y_train=y_train_arr,
        )
        component_models.append(wrapper)
        component_names.append(spec.name)
        component_train_rmse.append(float(np.sqrt(mean_squared_error(
            y_train_arr, y_hat_tr,
        ))))
    if not train_preds:
        return None
    return {
        "train_preds": train_preds,
        "test_preds": test_preds,
        "component_models": component_models,
        "component_names": component_names,
        "component_train_rmse": component_train_rmse,
        "y_train": y_train_arr,
    }


def _evaluate_strategies(comp, y_test_arr, seed):
    """Evaluate 11 ensemble strategies on per-spec predictions.

    Returns dict ``{strategy_name: test_rmse}``.
    """
    from mlframe.training.composite import CompositeCrossTargetEnsemble
    from lightgbm import LGBMRegressor
    train_preds = comp["train_preds"]
    test_preds = comp["test_preds"]
    train_mat = np.column_stack(train_preds)
    test_mat = np.column_stack(test_preds)
    y_train_arr = comp["y_train"]
    n_components = test_mat.shape[1]
    rmses: Dict[str, float] = {}
    naive_rmse = float(np.sqrt(mean_squared_error(
        y_train_arr, np.full_like(y_train_arr, y_train_arr.mean()),
    )))

    # 1. mean
    p = test_mat.mean(axis=1)
    rmses["mean"] = float(np.sqrt(mean_squared_error(y_test_arr, p)))

    # 2. median
    p = np.median(test_mat, axis=1)
    rmses["median"] = float(np.sqrt(mean_squared_error(y_test_arr, p)))

    # 3. trimmed_mean (drop max + min per row)
    if n_components >= 3:
        sorted_p = np.sort(test_mat, axis=1)
        p = sorted_p[:, 1:-1].mean(axis=1)
    else:
        p = test_mat.mean(axis=1)
    rmses["trimmed_mean"] = float(np.sqrt(mean_squared_error(
        y_test_arr, p,
    )))

    # 4. inverse_rmse
    inv = 1.0 / np.maximum(np.array(comp["component_train_rmse"]), 1e-9)
    inv = inv / inv.sum()
    rmses["inverse_rmse"] = float(np.sqrt(mean_squared_error(
        y_test_arr, test_mat @ inv,
    )))

    # 5. inverse_variance
    train_errs_sq = (train_mat - y_train_arr.reshape(-1, 1)) ** 2
    var_per = np.maximum(train_errs_sq.mean(axis=0), 1e-9)
    inv_var = 1.0 / var_per
    inv_var = inv_var / inv_var.sum()
    rmses["inverse_variance"] = float(np.sqrt(mean_squared_error(
        y_test_arr, test_mat @ inv_var,
    )))

    # 6. bma_softmax
    rmse_arr = np.array(comp["component_train_rmse"])
    rmse_std = max(float(np.std(rmse_arr)), 1e-9)
    logits = -rmse_arr / rmse_std
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    rmses["bma_softmax"] = float(np.sqrt(mean_squared_error(
        y_test_arr, test_mat @ w,
    )))

    # 7. oof_weighted (R10b current default)
    oof = CompositeCrossTargetEnsemble.from_train_metrics(
        component_models=comp["component_models"],
        component_names=comp["component_names"],
        component_train_rmse=comp["component_train_rmse"],
        baseline_train_rmse=naive_rmse,
    )
    if hasattr(oof, "weights"):
        rmses["oof_weighted"] = float(np.sqrt(mean_squared_error(
            y_test_arr, test_mat @ oof.weights,
        )))
    else:
        # Single best fallback fired (no component beats baseline).
        best_idx = int(np.argmin(comp["component_train_rmse"]))
        rmses["oof_weighted"] = float(np.sqrt(mean_squared_error(
            y_test_arr, test_preds[best_idx],
        )))

    # 8. linear_stack_ridge
    try:
        lin = CompositeCrossTargetEnsemble.from_linear_stack(
            component_models=comp["component_models"],
            component_names=comp["component_names"],
            component_predictions=train_mat,
            y_train=y_train_arr, ridge_alpha=1.0,
        )
        rmses["linear_stack_ridge"] = float(np.sqrt(mean_squared_error(
            y_test_arr, test_mat @ lin.weights,
        )))
    except Exception:
        rmses["linear_stack_ridge"] = float("nan")

    # 9. nnls_stack
    try:
        nnls = CompositeCrossTargetEnsemble.from_nnls_stack(
            component_models=comp["component_models"],
            component_names=comp["component_names"],
            component_predictions=train_mat,
            y_train=y_train_arr,
        )
        rmses["nnls_stack"] = float(np.sqrt(mean_squared_error(
            y_test_arr, test_mat @ nnls.weights,
        )))
    except Exception:
        rmses["nnls_stack"] = float("nan")

    # 10. stacked_gbdt
    meta = LGBMRegressor(
        n_estimators=80, num_leaves=8, learning_rate=0.1,
        random_state=seed, verbosity=-1,
    )
    meta.fit(train_mat, y_train_arr)
    rmses["stacked_gbdt"] = float(np.sqrt(mean_squared_error(
        y_test_arr, meta.predict(test_mat),
    )))

    # 11. best_single_by_train (sanity baseline)
    best_idx = int(np.argmin(comp["component_train_rmse"]))
    rmses["best_single_by_train"] = float(np.sqrt(mean_squared_error(
        y_test_arr, test_preds[best_idx],
    )))

    # Plus reference: best single by TEST (oracle, can't be used in
    # production but useful upper bound for the table).
    test_rmses = [
        float(np.sqrt(mean_squared_error(y_test_arr, p)))
        for p in test_preds
    ]
    rmses["_oracle_best_single_test"] = float(min(test_rmses))
    return rmses


def _run_one(scenario_name, seed, n):
    gen = SCENARIOS[scenario_name]
    df, target_col, feature_cols = gen(n=n, seed=seed)
    cut = int(0.8 * n)
    train_idx = np.arange(cut)
    test_idx = np.arange(cut, n)
    comp = _train_composite_specs(
        df, target_col, feature_cols, train_idx, test_idx, seed,
    )
    if comp is None:
        return None
    y_test_arr = df[target_col].iloc[test_idx].to_numpy()
    return _evaluate_strategies(comp, y_test_arr, seed)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--fast", action="store_true",
                        help="reps=2, n=2000")
    args = parser.parse_args()
    if args.fast:
        args.reps = 2
        args.n = 2000

    print("Composite cross-target ensemble shootout: 11 strategies x "
          f"{len(SCENARIOS)} scenarios x {args.reps} seeds\n")
    all_rmses: Dict[str, Dict[str, List[float]]] = {}
    skipped: List[Tuple[str, int]] = []
    for sc_name in SCENARIOS:
        for rep in range(args.reps):
            t0 = time.perf_counter()
            r = _run_one(sc_name, seed=rep, n=args.n)
            dt = time.perf_counter() - t0
            if r is None:
                skipped.append((sc_name, rep))
                print(f"  {sc_name} rep={rep}: SKIPPED (no specs) "
                      f"({dt:.1f}s)")
                continue
            all_rmses.setdefault(sc_name, {})
            for strategy, rmse in r.items():
                all_rmses[sc_name].setdefault(strategy, []).append(rmse)
            best = min(
                ((k, v) for k, v in r.items()
                 if k != "_oracle_best_single_test"),
                key=lambda kv: kv[1],
            )
            print(f"  {sc_name} rep={rep}: best={best[0]} "
                  f"({best[1]:.4f}) ({dt:.1f}s)")
    if skipped:
        print(f"\n  ({len(skipped)} runs skipped: {skipped[:3]}...)")

    # Aggregate: mean RMSE per (scenario, strategy), then improvement
    # vs best_single_by_train (the practical baseline).
    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS (improvement % vs best_single_by_train; "
          "averaged over reps)")
    print("=" * 90)
    strategies_list = list(set(
        s for sc in all_rmses.values() for s in sc.keys()
    ))
    strategies_list.remove("_oracle_best_single_test")
    if "best_single_by_train" in strategies_list:
        strategies_list.remove("best_single_by_train")
    strategies_list = ["best_single_by_train"] + sorted(strategies_list)
    # Per-strategy improvement aggregated across all (scenario, rep).
    strategy_improvements: Dict[str, List[float]] = {
        s: [] for s in strategies_list
    }
    for sc_name, sc_rmses in all_rmses.items():
        baseline = sc_rmses.get("best_single_by_train")
        if not baseline:
            continue
        for s in strategies_list:
            if s not in sc_rmses:
                continue
            for b_rmse, s_rmse in zip(baseline, sc_rmses[s]):
                if b_rmse > 0:
                    strategy_improvements[s].append(
                        (b_rmse - s_rmse) / b_rmse * 100.0
                    )
    # Per-strategy mean + win count.
    overall = []
    for s in strategies_list:
        imps = strategy_improvements[s]
        if not imps:
            continue
        mean_imp = float(np.mean(imps))
        median_imp = float(np.median(imps))
        win_count = sum(1 for i in imps if i > 0)
        overall.append({
            "strategy": s,
            "mean_imp_pct": mean_imp,
            "median_imp_pct": median_imp,
            "wins": win_count,
            "total": len(imps),
        })
    overall.sort(key=lambda r: -r["mean_imp_pct"])
    print(f"\n{'strategy':<25s} {'mean_imp%':>10s} {'median%':>10s} "
          f"{'wins':>10s}")
    for row in overall:
        print(f"  {row['strategy']:<23s} {row['mean_imp_pct']:>+10.2f} "
              f"{row['median_imp_pct']:>+10.2f} "
              f"{row['wins']:>3d}/{row['total']:<3d}")

    winner = overall[0]["strategy"] if overall else None
    print("\n" + "=" * 90)
    print(f"WINNER (by mean improvement vs best_single_by_train): {winner}")
    print("=" * 90)

    # Per-scenario breakdown for the top-3 strategies.
    print("\nPer-scenario detail (top-3 strategies by overall mean):")
    top_3 = [r["strategy"] for r in overall[:3]]
    for sc_name, sc_rmses in all_rmses.items():
        baseline_list = sc_rmses.get("best_single_by_train", [])
        if not baseline_list:
            continue
        baseline_mean = float(np.mean(baseline_list))
        print(f"\n  {sc_name} (best_single_by_train mean RMSE="
              f"{baseline_mean:.4f}):")
        for s in top_3:
            if s in sc_rmses:
                vals = sc_rmses[s]
                if vals:
                    mean_v = float(np.mean(vals))
                    imp = (baseline_mean - mean_v) / baseline_mean * 100.0
                    print(f"    {s:<22s} RMSE={mean_v:.4f} "
                          f"(imp={imp:+.2f}%)")

    out_path = "benchmarks/composite_ensemble_shootout_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"reps": args.reps, "n": args.n,
                       "scenarios": list(SCENARIOS.keys())},
            "skipped": [list(t) for t in skipped],
            "per_scenario_rmses": all_rmses,
            "overall_ranking": overall,
            "winner": winner,
        }, f, indent=2)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
