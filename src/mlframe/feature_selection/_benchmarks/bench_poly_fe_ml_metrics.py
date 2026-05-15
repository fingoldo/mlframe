"""Real-ML-metrics benchmark for orthogonal-polynomial pair Feature
Engineering.

Earlier benches measured KSG mutual-information uplift -- a proxy. This
bench measures **end-task ML metrics** (MAE / RMSE / R^2 for regression,
accuracy / AUROC / log-loss for classification) on real datasets, with
proper train-fold-only feature engineering and 5-fold CV.

Pipeline per (dataset, fold):
1. Pick top-K candidate feature pairs by joint KSG MI on the train fold.
2. For each polynomial basis in {hermite, legendre, chebyshev, laguerre}:
   * fit ``optimise_hermite_pair`` on (X_train, y_train) -- coefficients
     optimised against MI(engineered, y_train);
   * apply learned ``HermiteResult.transform`` to X_val (no leakage --
     no y_val touched);
   * augment X with engineered columns, fit GBDT, score on val.
3. Multi-basis ensemble: concat engineered columns from ALL 4 bases as
   additional features, let GBDT pick.
4. Compare to baseline (raw features only).

Datasets:
* Regression: california_housing, diabetes, friedman1
* Classification: breast_cancer, wine (3-class), synth_xor

Run::

    python -m mlframe.feature_selection._benchmarks.bench_poly_fe_ml_metrics
    python -m mlframe.feature_selection._benchmarks.bench_poly_fe_ml_metrics --datasets diabetes,wine --top-k 2 --n-trials 30
"""
from __future__ import annotations

import argparse
import time
import warnings
from itertools import combinations
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")  # sklearn convergence + optuna experimental noise


def _load_california(n_subsample: Optional[int] = 4000, seed: int = 42):
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = data.data.astype(np.float64), data.target.astype(np.float64)
    if n_subsample and n_subsample < len(X):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_subsample, replace=False)
        X, y = X[idx], y[idx]
    return X, y, False, "california"


def _load_diabetes():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return data.data.astype(np.float64), data.target.astype(np.float64), False, "diabetes"


def _load_friedman1(n: int = 2000, seed: int = 42):
    """Smooth nonlinear regression target. y = 10*sin(pi*x_0*x_1) +
    20*(x_2 - 0.5)^2 + 10*x_3 + 5*x_4 + noise. Inputs Uniform[0, 1]."""
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=n, n_features=10, noise=1.0, random_state=seed)
    return X.astype(np.float64), y.astype(np.float64), False, "friedman1"


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return data.data.astype(np.float64), data.target.astype(np.int64), True, "breast_cancer"


def _load_wine():
    from sklearn.datasets import load_wine
    data = load_wine()
    return data.data.astype(np.float64), data.target.astype(np.int64), True, "wine"


def _load_synth_xor(n: int = 2000, seed: int = 42):
    """2 signal features (XOR) + 2 noise features. Target y = sign(x0*x1)."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    noise_a = rng.normal(size=n)
    noise_b = rng.uniform(-1, 1, size=n)
    X = np.column_stack([x0, x1, noise_a, noise_b])
    y = (np.sign(x0 * x1) > 0).astype(np.int64)
    return X, y, True, "synth_xor"


_DATASETS = {
    "california": _load_california,
    "diabetes": _load_diabetes,
    "friedman1": _load_friedman1,
    "breast_cancer": _load_breast_cancer,
    "wine": _load_wine,
    "synth_xor": _load_synth_xor,
}

_BASES = ["hermite", "legendre", "chebyshev", "laguerre"]


def _select_top_pairs(X_train, y_train, k, discrete_target,
                       max_candidates_pairs=60, n_subsample_for_mi=1500):
    """Rank candidate (i, j) pairs by joint KSG MI with y on the train
    fold. Cap candidates to ``C(top_features_by_single_mi, 2)`` to avoid
    quadratic blowup on wide datasets. Subsample to bound per-pair MI
    cost (sklearn KSG MI is O(n log n))."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    n_features = X_train.shape[1]
    mi_func = mutual_info_classif if discrete_target else mutual_info_regression
    # Subsample for MI estimation to bound wall-time on large train folds.
    if len(X_train) > n_subsample_for_mi:
        rng = np.random.default_rng(123)
        sub = rng.choice(len(X_train), size=n_subsample_for_mi, replace=False)
        Xs, ys = X_train[sub], y_train[sub]
    else:
        Xs, ys = X_train, y_train
    if n_features > 12:
        # Pre-rank by single-feature MI, take top-12 by MI, then form pairs.
        single_mi = mi_func(Xs, ys, n_neighbors=3, random_state=42,
                            discrete_features=False)
        top_idx = np.argsort(single_mi)[::-1][:12]
        pairs = list(combinations(sorted(top_idx.tolist()), 2))
    else:
        pairs = list(combinations(range(n_features), 2))
    if len(pairs) > max_candidates_pairs:
        pairs = pairs[:max_candidates_pairs]
    scores = []
    for i, j in pairs:
        Xij = Xs[:, [i, j]]
        # Avoid constant columns (e.g. centered+scaled diabetes 'sex').
        if np.std(Xij[:, 0]) < 1e-12 or np.std(Xij[:, 1]) < 1e-12:
            continue
        mi = mi_func(Xij, ys, n_neighbors=3, random_state=42,
                     discrete_features=False)
        scores.append((i, j, float(mi.max())))
    scores.sort(key=lambda x: -x[2])
    return scores[:k]


def _engineer_columns(X_train, y_train, X_val, top_pairs, basis,
                       discrete_target, n_trials, max_degree=3,
                       n_subsample_for_fe=1500):
    """Fit polynomial-pair FE on (X_train, y_train), apply to X_val.
    Returns (eng_train, eng_val) -- engineered columns only (no original
    features). Empty arrays of shape (n, 0) if no pair beats baseline.

    For wall-time control: when ``len(X_train) > n_subsample_for_fe`` we
    fit the polynomial coefficients on a random subsample, then apply
    the learned ``HermiteResult.transform`` to the FULL train + val
    sets. KSG MI estimator scales as O(n*log(n)*k), so a 4000 -> 1500
    cut is ~3x faster while still giving stable coefficient estimates
    (we only fit ``2 * (max_degree + 1) <= 8`` parameters)."""
    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair
    eng_tr, eng_va = [], []
    rng = np.random.default_rng(42)
    if len(X_train) > n_subsample_for_fe:
        sub_idx = rng.choice(len(X_train), size=n_subsample_for_fe, replace=False)
    else:
        sub_idx = None
    for i, j, _ in top_pairs:
        x_a_tr, x_b_tr = X_train[:, i], X_train[:, j]
        x_a_va, x_b_va = X_val[:, i], X_val[:, j]
        if np.std(x_a_tr) < 1e-12 or np.std(x_b_tr) < 1e-12:
            continue
        if sub_idx is not None:
            fit_a, fit_b, fit_y = x_a_tr[sub_idx], x_b_tr[sub_idx], y_train[sub_idx]
        else:
            fit_a, fit_b, fit_y = x_a_tr, x_b_tr, y_train
        try:
            res = optimise_hermite_pair(
                fit_a, fit_b, fit_y,
                discrete_target=discrete_target,
                max_degree=max_degree, min_degree=2,
                n_trials=n_trials, seed=42, basis=basis,
                # Lower threshold so we keep marginal-uplift features (the
                # downstream GBDT will gate them itself via tree splits).
                baseline_uplift_threshold=1.0,
                early_stop_no_improve=max(15, n_trials // 3),
            )
        except Exception:
            res = None
        if res is None:
            continue
        eng_tr.append(res.transform(x_a_tr, x_b_tr))
        eng_va.append(res.transform(x_a_va, x_b_va))
    n_train, n_val = len(X_train), len(X_val)
    if not eng_tr:
        return np.empty((n_train, 0)), np.empty((n_val, 0))
    return np.column_stack(eng_tr), np.column_stack(eng_va)


def _build_model(discrete_target, model_kind):
    """Construct the downstream estimator. ``gbdt`` = production-style
    HistGradientBoosting (captures nonlinearities natively, so FE
    benefit is often muted); ``linear`` = Ridge / LogisticRegression
    where FE matters more dramatically because the model itself is
    blind to interactions."""
    if model_kind == "gbdt":
        if discrete_target:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=42, max_iter=200,
                                                    early_stopping=False)
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=42, max_iter=200,
                                              early_stopping=False)
    elif model_kind == "linear":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        if discrete_target:
            from sklearn.linear_model import LogisticRegression
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(random_state=42, max_iter=2000, C=1.0,
                                    multi_class="auto"),
            )
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    raise ValueError(f"unknown model_kind={model_kind!r}")


def _fit_and_score(X_tr, y_tr, X_va, y_va, discrete_target, model_kind="gbdt"):
    """Fit estimator on (X_tr, y_tr), score on (X_va, y_va)."""
    est = _build_model(discrete_target, model_kind)
    est.fit(X_tr, y_tr)
    y_pred = est.predict(X_va)
    if discrete_target:
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        y_proba = est.predict_proba(X_va)
        classes = est.classes_ if hasattr(est, "classes_") else est[-1].classes_
        acc = float(accuracy_score(y_va, y_pred))
        ll = float(log_loss(y_va, y_proba, labels=classes))
        try:
            if y_proba.shape[1] == 2:
                auc = float(roc_auc_score(y_va, y_proba[:, 1]))
            else:
                auc = float(roc_auc_score(y_va, y_proba, multi_class="ovr",
                                           average="macro"))
        except Exception:
            auc = float("nan")
        return dict(acc=acc, log_loss=ll, auc=auc)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = float(mean_absolute_error(y_va, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_va, y_pred)))
    r2 = float(r2_score(y_va, y_pred))
    return dict(mae=mae, rmse=rmse, r2=r2)


def _eval_dataset(name, X, y, discrete_target, *,
                   n_splits, top_k, n_trials, max_degree, verbose,
                   model_kind="gbdt"):
    from sklearn.model_selection import KFold, StratifiedKFold
    cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
          if discrete_target
          else KFold(n_splits=n_splits, shuffle=True, random_state=42))
    methods = ["baseline"] + _BASES + ["ensemble"]
    results = {m: [] for m in methods}
    pair_log = []  # for diagnostics
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if verbose:
            print(f"  [{name}] fold {fold + 1}/{n_splits}: train={len(tr_idx)}, val={len(va_idx)}", flush=True)
        # Top-K pair selection on train fold only (no leakage)
        top_pairs = _select_top_pairs(X_tr_raw, y_tr, k=top_k,
                                       discrete_target=discrete_target)
        pair_log.append([(i, j) for i, j, _ in top_pairs])
        # Engineer per-basis FE columns (cache once per fold)
        eng_per_basis = {}
        for basis in _BASES:
            eng_per_basis[basis] = _engineer_columns(
                X_tr_raw, y_tr, X_va_raw, top_pairs, basis,
                discrete_target, n_trials, max_degree=max_degree,
            )
        # Score baseline + each basis + ensemble
        results["baseline"].append(_fit_and_score(X_tr_raw, y_tr, X_va_raw, y_va,
                                                    discrete_target, model_kind))
        for basis in _BASES:
            eng_tr, eng_va = eng_per_basis[basis]
            X_tr_aug = np.column_stack([X_tr_raw, eng_tr]) if eng_tr.size else X_tr_raw
            X_va_aug = np.column_stack([X_va_raw, eng_va]) if eng_va.size else X_va_raw
            results[basis].append(_fit_and_score(X_tr_aug, y_tr, X_va_aug, y_va,
                                                   discrete_target, model_kind))
        # Ensemble: concat engineered cols across all 4 bases
        all_eng_tr = [eng_per_basis[b][0] for b in _BASES if eng_per_basis[b][0].size]
        all_eng_va = [eng_per_basis[b][1] for b in _BASES if eng_per_basis[b][1].size]
        if all_eng_tr:
            X_tr_ens = np.column_stack([X_tr_raw] + all_eng_tr)
            X_va_ens = np.column_stack([X_va_raw] + all_eng_va)
            results["ensemble"].append(_fit_and_score(X_tr_ens, y_tr, X_va_ens, y_va,
                                                        discrete_target, model_kind))
        else:
            results["ensemble"].append(results["baseline"][-1])
    return results, pair_log


def _aggregate(results, discrete_target):
    """Return per-method (mean, std) for each metric."""
    metric_keys = (["acc", "log_loss", "auc"] if discrete_target
                   else ["mae", "rmse", "r2"])
    agg = {}
    for method, fold_dicts in results.items():
        agg[method] = {}
        for m in metric_keys:
            vals = np.array([d[m] for d in fold_dicts], dtype=np.float64)
            vals_finite = vals[np.isfinite(vals)]
            agg[method][m] = (
                float(np.mean(vals_finite)) if len(vals_finite) else float("nan"),
                float(np.std(vals_finite)) if len(vals_finite) else float("nan"),
            )
    return agg, metric_keys


def _print_table(name, agg, metric_keys, baseline_ref):
    print(f"\n  --- {name} ---")
    header = f"  {'method':>10s}  " + "  ".join(f"{m:>14s}" for m in metric_keys)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method, m_dict in agg.items():
        cells = []
        for m in metric_keys:
            mean, std = m_dict[m]
            base_mean, _ = baseline_ref[m]
            # Compute relative improvement vs baseline.
            if m in ("mae", "rmse", "log_loss"):  # lower better
                delta_pct = (base_mean - mean) / abs(base_mean) * 100 if base_mean != 0 else 0.0
                marker = "+" if delta_pct > 0.5 else ("-" if delta_pct < -0.5 else " ")
            else:  # acc, auc, r2 -- higher better
                delta_pct = (mean - base_mean) / abs(base_mean) * 100 if base_mean != 0 else 0.0
                marker = "+" if delta_pct > 0.5 else ("-" if delta_pct < -0.5 else " ")
            cells.append(f"{mean:7.4f}+/-{std:5.4f}{marker if method != 'baseline' else ' '}")
        print(f"  {method:>10s}  " + "  ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all",
                        help="comma-separated subset of: " + ",".join(_DATASETS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of feature pairs to engineer per fold")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="optuna trials per (pair, basis, degree)")
    parser.add_argument("--max-degree", type=int, default=3)
    parser.add_argument("--model", choices=["gbdt", "linear"], default="gbdt",
                        help="downstream model: gbdt (HistGradientBoosting) "
                              "or linear (Ridge/LogisticRegression). FE benefit "
                              "is typically larger with linear models -- GBDT "
                              "captures nonlinearities natively.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    datasets = (list(_DATASETS) if args.datasets == "all"
                else args.datasets.split(","))

    print(f"\n=== Polynomial-pair FE -- ML metrics bench ===")
    print(f"  model={args.model}, n_splits={args.n_splits}, top_k={args.top_k}, "
          f"n_trials={args.n_trials}, max_degree={args.max_degree}")
    print(f"  datasets: {', '.join(datasets)}\n")

    overall_t0 = time.perf_counter()
    summary_rows = []
    for d_name in datasets:
        if d_name not in _DATASETS:
            print(f"  unknown dataset {d_name!r}, skipping")
            continue
        t0 = time.perf_counter()
        X, y, discrete_target, name = _DATASETS[d_name]()
        if not args.quiet:
            print(f"\n  Loading {name}: X={X.shape}, "
                  f"target={'classification' if discrete_target else 'regression'}",
                  flush=True)
        results, pair_log = _eval_dataset(
            name, X, y, discrete_target,
            n_splits=args.n_splits, top_k=args.top_k,
            n_trials=args.n_trials, max_degree=args.max_degree,
            verbose=not args.quiet,
            model_kind=args.model,
        )
        agg, metric_keys = _aggregate(results, discrete_target)
        baseline_ref = agg["baseline"]
        _print_table(name, agg, metric_keys, baseline_ref)
        dt = time.perf_counter() - t0
        print(f"  ({dt:.1f}s; pairs picked across folds: {pair_log})")
        summary_rows.append((name, discrete_target, agg))

    print(f"\n  Total wall: {time.perf_counter() - overall_t0:.1f}s")

    # Final summary: which method has best primary metric per dataset
    print("\n  === Best method per dataset (primary metric) ===")
    print(f"  {'dataset':>15s}  {'metric':>10s}  {'baseline':>14s}  {'best method':>14s}  {'best value':>14s}  {'delta vs base':>14s}")
    print("  " + "-" * 95)
    for name, discrete, agg in summary_rows:
        primary = "auc" if discrete else "rmse"
        higher_better = primary != "rmse"
        candidates = [(m, agg[m][primary][0]) for m in agg if not np.isnan(agg[m][primary][0])]
        if higher_better:
            best_m, best_v = max(candidates, key=lambda x: x[1])
        else:
            best_m, best_v = min(candidates, key=lambda x: x[1])
        base_v = agg["baseline"][primary][0]
        if higher_better:
            delta = (best_v - base_v) / abs(base_v) * 100
        else:
            delta = (base_v - best_v) / abs(base_v) * 100
        print(f"  {name:>15s}  {primary:>10s}  {base_v:>14.4f}  {best_m:>14s}  "
              f"{best_v:>14.4f}  {delta:>+13.2f}%")


if __name__ == "__main__":
    main()
