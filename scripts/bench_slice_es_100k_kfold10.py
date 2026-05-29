"""Serious bench: n=100k, KFold(10), regression + binary classification, CB/XGB/LGB.

Regression metrics: RMSE, MAE, MaxError (worst-case absolute residual).
Classification metrics: ROC AUC, ICE (Integral Calibration Error from mlframe.metrics).

For each (model, task) we run three ES configs across 10 paired folds:
  - baseline (no slice ES)
  - slice-stable mean   (K=5 temporal shards, aggregate=mean)
  - slice-stable t-LCB  (K=5 temporal shards, aggregate=t_lcb conf=0.9)

The KFold split shuffles 100k rows once with seed=0; within each fold's training portion
we carve a 10% temporal-trailing val for ES. The temporal val ordering (row index) is what
``source="temporal"`` shards on. Each fold gives one paired observation per config; paired
Wilcoxon over 10 folds tests whether the slice configurations differ from baseline on the
test fold metric. JSON results land in ``benchmarks/slice_es_100k_kfold10.json`` incrementally
so partial results survive crashes.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold

from mlframe.metrics._classification_report import fast_ice_only
from mlframe.metrics._classification_extras import (
    ks_statistic,
    matthews_corrcoef_binary,
    cohen_kappa_binary,
    balanced_accuracy_binary,
    g_mean_binary,
    brier_skill_score,
    gini_from_auc,
    f_beta_score,
    spiegelhalter_z,
    lift_at_k,
    hosmer_lemeshow_test,
    accuracy_ratio,
)
from mlframe.metrics._core_auc_brier import fast_brier_score_loss
from mlframe.metrics._log_loss_and_separation import fast_log_loss_binary
from mlframe.metrics._regression_extras import (
    fast_rmsle, fast_mape_mean, fast_smape, fast_mdape, fast_wmape,
    fast_mean_bias_error, fast_cv_rmse, fast_nash_sutcliffe, fast_explained_variance,
    fast_huber_loss, fast_pearson_corr, fast_spearman_corr, fast_kendall_tau,
    fast_concordance_index,
)
from mlframe.training._callbacks import (
    CatBoostCallback,
    LightGBMCallback,
    XGBoostCallback,
)
from mlframe.training._data_helpers import _setup_eval_set
from mlframe.training._slice_helpers import build_slice_eval_sets


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _gen_regression(n: int = 100_000, d: int = 15, seed: int = 0) -> tuple:
    """Realistic regression: 8 informative + 7 noise features, moderate Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    # Informative feature contributions
    y = (
        np.sin(2 * np.pi * X[:, 0])
        + 0.7 * X[:, 1]
        - 0.5 * X[:, 2] ** 2
        + 0.4 * X[:, 3] * X[:, 4]
        + 0.3 * np.tanh(X[:, 5])
        + 0.2 * X[:, 6]
        - 0.2 * X[:, 7]
    )
    y = y + rng.normal(0, 0.5, n)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y.astype(np.float32)


def _gen_regression_drift(n: int = 100_000, d: int = 15, seed: int = 0) -> tuple:
    """Concept-drift regression: Y(X) gradually changes along the row order.

    Train rows (early) follow ``y = f_early(X) + noise``; test rows (late) follow
    ``y = f_late(X) + noise`` with a smooth interpolation in between. The interpolation
    weight on each row's signal shifts coefficients of two informative features by 100%
    across the dataset's length. This is the setup the user hypothesised would surface
    slice-stable value: temporal shards of val see different (X, y) conditionals, and
    late-iter overfit on the dominant early-regime hurts test (which is the late regime).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    # Drift parameter t in [0, 1] along row index. Train will be early-t, test late-t.
    t = np.linspace(0.0, 1.0, n)
    # Early regime: feature 0 dominates; late regime: feature 1 dominates + sign flip on feature 2.
    early = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1] - 0.5 * X[:, 2] ** 2
    late = 0.3 * np.sin(2 * np.pi * X[:, 0]) + 1.2 * X[:, 1] + 0.5 * X[:, 2] ** 2 + 0.4 * np.tanh(X[:, 3])
    y = (1 - t) * early + t * late
    # Add the stable contributions (same across drift) and Gaussian noise
    y = y + 0.3 * X[:, 4] - 0.2 * X[:, 5] + rng.normal(0, 0.5, n)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y.astype(np.float32)


def _gen_classification(n: int = 100_000, d: int = 15, seed: int = 0, pos_rate: float = 0.30) -> tuple:
    """Realistic binary classif: informative + noise features, controlled positive rate."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    logit = (
        1.2 * X[:, 0]
        + 0.8 * X[:, 1]
        - 0.6 * X[:, 2] ** 2
        + 0.5 * X[:, 3] * X[:, 4]
        + 0.4 * np.tanh(X[:, 5])
        + 0.3 * X[:, 6]
        - 0.3 * X[:, 7]
    )
    # Shift so that 1{logit > thr} gives ~pos_rate positives
    thr = np.quantile(logit, 1.0 - pos_rate)
    y = (logit + rng.normal(0, 0.3, n) > thr).astype(np.int32)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y


# ---------------------------------------------------------------------------
# Test metrics
# ---------------------------------------------------------------------------

def _safe(fn, *args, default=float("nan"), **kwargs):
    """Run a metric robustly: skip-and-NaN on any error (domain mismatch, etc)."""
    try:
        v = fn(*args, **kwargs)
        if isinstance(v, tuple):
            v = v[0]  # for ones that return (stat, p_value) etc.
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


def _metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    resid = y_pred - y_true
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse":              float(np.sqrt(np.mean(resid ** 2))),
        "mae":               float(np.mean(np.abs(resid))),
        "max_err":           float(np.max(np.abs(resid))),
        "r2":                1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmsle":             _safe(fast_rmsle, y_true, y_pred),
        "mape":              _safe(fast_mape_mean, y_true, y_pred),
        "smape":             _safe(fast_smape, y_true, y_pred),
        "mdape":             _safe(fast_mdape, y_true, y_pred),
        "wmape":             _safe(fast_wmape, y_true, y_pred),
        "mbe":               _safe(fast_mean_bias_error, y_true, y_pred),
        "cv_rmse":           _safe(fast_cv_rmse, y_true, y_pred),
        "nash_sutcliffe":    _safe(fast_nash_sutcliffe, y_true, y_pred),
        "explained_var":    _safe(fast_explained_variance, y_true, y_pred),
        "huber":             _safe(fast_huber_loss, y_true, y_pred),
        "pearson_corr":      _safe(fast_pearson_corr, y_true, y_pred),
        "spearman_corr":     _safe(fast_spearman_corr, y_true, y_pred),
        "kendall_tau":       _safe(fast_kendall_tau, y_true[:5000], y_pred[:5000]),  # O(N^2) -- subsample
        "concordance_idx":   _safe(fast_concordance_index, y_true[:5000], y_pred[:5000]),
    }


def _metrics_classification(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    y_true_int = np.asarray(y_true, dtype=np.int32)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)
    y_pred_bin = (y_prob_f >= 0.5).astype(np.int32)
    roc_auc = _safe(roc_auc_score, y_true_int, y_prob_f)
    return {
        "roc_auc":           roc_auc,
        "gini":              gini_from_auc(roc_auc) if not np.isnan(roc_auc) else float("nan"),
        "ice":               _safe(fast_ice_only, y_true=y_true_int, y_pred=y_prob_f),
        "log_loss":          _safe(fast_log_loss_binary, y_true_int, y_prob_f),
        "brier":             _safe(fast_brier_score_loss, y_true_int, y_prob_f),
        "ks":                _safe(ks_statistic, y_true_int, y_prob_f),
        "mcc":               _safe(matthews_corrcoef_binary, y_true_int, y_pred_bin),
        "cohen_kappa":       _safe(cohen_kappa_binary, y_true_int, y_pred_bin),
        "balanced_acc":      _safe(balanced_accuracy_binary, y_true_int, y_pred_bin),
        "g_mean":            _safe(g_mean_binary, y_true_int, y_pred_bin),
        "brier_skill":       _safe(brier_skill_score, y_true_int, y_prob_f),
        "f_beta_2":          _safe(f_beta_score, y_true_int, y_pred_bin, beta=2.0),
        "spiegelhalter_z":   _safe(spiegelhalter_z, y_true_int, y_prob_f),
        "lift_at_10pct":     _safe(lift_at_k, y_true_int, y_prob_f, k_frac=0.10),
        "hosmer_lemeshow":   _safe(hosmer_lemeshow_test, y_true_int, y_prob_f),
        "accuracy_ratio":    _safe(accuracy_ratio, y_true_int, y_prob_f),
    }


# ---------------------------------------------------------------------------
# Per-fold fitter dispatch
# ---------------------------------------------------------------------------

def _build_callback(*, model: str, task: str, slice_k: int, aggregate: str, confidence: float = 0.5):
    monitor_metric = "l2" if task in ("regr", "regr_drift") else "binary_logloss"
    if task == "classif":
        monitor_metric = "binary_logloss"
    mode = "min"

    common = dict(
        patience=40, min_delta=0.0, monitor_metric=monitor_metric, mode=mode,
        slice_k=slice_k, slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=confidence,
        slice_correlation_inflation=1.0,  # neutral (no NB inflation)
        slice_persist_history=False, verbose=0,
    )
    if model == "lgb":
        return LightGBMCallback(monitor_dataset="valid_0", **common)
    if model == "xgb":
        return XGBoostCallback(monitor_dataset="validation_0",
                                **{**common, "monitor_metric": "rmse" if task in ("regr", "regr_drift") else "logloss"})
    if model == "cb":
        return CatBoostCallback(monitor_dataset="validation",
                                 **{**common, "monitor_metric": "RMSE" if task in ("regr", "regr_drift") else "Logloss"})
    raise ValueError(model)


def _fit_and_score(
    *, X_tr_full: pd.DataFrame, y_tr_full: np.ndarray,
    X_te: pd.DataFrame, y_te: np.ndarray,
    model: str, task: str, slice_k: int, aggregate: str, confidence: float, seed: int,
) -> dict:
    """One paired observation: split tr_full into train+val, fit, score on X_te.

    val = trailing 10% (preserves the source="temporal" semantics on shards)."""
    n = len(y_tr_full)
    val_size = int(0.10 * n)
    train_end = n - val_size
    X_train = X_tr_full.iloc[:train_end].reset_index(drop=True)
    y_train = y_tr_full[:train_end]
    X_val = X_tr_full.iloc[train_end:].reset_index(drop=True)
    y_val = y_tr_full[train_end:]
    t_val = np.arange(len(y_val), dtype=np.float64)  # implicit time = row index

    cb_obj = _build_callback(model=model, task=task, slice_k=slice_k if slice_k > 0 else 0,
                              aggregate=aggregate, confidence=confidence)
    extra_eval_sets = None
    if slice_k > 0:
        extra_eval_sets = build_slice_eval_sets(
            X_val, y_val, source="temporal", k=slice_k, min_rows_per_shard=8,
            random_state=seed, time_values=t_val,
        )
        if not extra_eval_sets:
            cb_obj.slice_k = 0  # silent fallback

    fit_params: dict = {}
    if model == "lgb":
        m = lgb.LGBMRegressor if task in ("regr", "regr_drift") else lgb.LGBMClassifier
        booster = m(n_estimators=1000, learning_rate=0.05, num_leaves=31, verbose=-1,
                     random_state=seed, n_jobs=-1)
        fit_params["callbacks"] = [cb_obj]
        type_name = "LGBMRegressor" if task in ("regr", "regr_drift") else "LGBMClassifier"
        _setup_eval_set(type_name, fit_params, X_val, y_val, model_category="lgb",
                        extra_eval_sets=extra_eval_sets)
    elif model == "xgb":
        m = xgb.XGBRegressor if task in ("regr", "regr_drift") else xgb.XGBClassifier
        # XGB sklearn API: callbacks via constructor (not fit kwargs in 2.x).
        # verbose_eval=0 / verbosity=0 don't silence the per-iter eval_set log on .fit(eval_set=...);
        # only ``verbose=False`` on fit does. We pass it via fit_params below.
        booster = m(n_estimators=1000, learning_rate=0.05, max_depth=6, verbosity=0,
                     random_state=seed, tree_method="hist", n_jobs=-1, early_stopping_rounds=None,
                     callbacks=[cb_obj])
        fit_params["verbose"] = False
        # _setup_eval_set wiring for XGB list-of-tuples eval_set
        if extra_eval_sets:
            eval_list = [(X_val, y_val)] + [(s.X, s.y) for s in extra_eval_sets]
            fit_params["eval_set"] = eval_list
        else:
            fit_params["eval_set"] = [(X_val, y_val)]
    elif model == "cb":
        m = CatBoostRegressor if task in ("regr", "regr_drift") else CatBoostClassifier
        booster = m(iterations=1000, learning_rate=0.05, depth=6, verbose=False,
                     random_seed=seed, allow_writing_files=False, early_stopping_rounds=None)
        fit_params["callbacks"] = [cb_obj]
        type_name = "CatBoostRegressor" if task in ("regr", "regr_drift") else "CatBoostClassifier"
        _setup_eval_set(type_name, fit_params, X_val, y_val, model_category="cb",
                        extra_eval_sets=extra_eval_sets)
    else:
        raise ValueError(model)

    booster.fit(X_train, y_train, **fit_params)

    if task in ("regr", "regr_drift"):
        preds = booster.predict(X_te)
        return _metrics_regression(y_te, np.asarray(preds, dtype=np.float64))
    else:
        if hasattr(booster, "predict_proba"):
            proba = booster.predict_proba(X_te)
            probs = proba[:, 1] if proba.ndim == 2 else proba.ravel()
        else:
            probs = booster.predict(X_te)
        return _metrics_classification(y_te, np.asarray(probs, dtype=np.float64))


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------

def _wilcoxon(baseline: list[float], slice_es: list[float], direction: str) -> float:
    diffs = (np.array(baseline) - np.array(slice_es)) if direction == "min" else \
            (np.array(slice_es) - np.array(baseline))
    if np.all(diffs == 0):
        return 1.0
    return float(wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue)


def main():
    n_rows = 100_000
    n_folds = 10
    seed = 0

    Path("benchmarks").mkdir(exist_ok=True)
    out_path = Path("benchmarks/slice_es_100k_kfold10.json")

    print(f"=== n={n_rows}, KFold({n_folds}), {n_folds} paired folds per (model, task, config) ===")
    print("Generating data once (regression + drift + classification)...")
    X_reg, y_reg = _gen_regression(n=n_rows)
    X_drift, y_drift = _gen_regression_drift(n=n_rows)
    X_cls, y_cls = _gen_classification(n=n_rows)
    print(f"  regr      : y mean={y_reg.mean():.3f} std={y_reg.std():.3f}")
    print(f"  regr_drift: y mean={y_drift.mean():.3f} std={y_drift.std():.3f}")
    print(f"  classif   : pos rate={y_cls.mean():.3f}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_idx = list(kf.split(np.arange(n_rows)))

    # Test direction by metric (most are min; flags below are "max-is-better").
    HIGHER_IS_BETTER = {
        # regression
        "r2", "nash_sutcliffe", "explained_var", "pearson_corr", "spearman_corr",
        "kendall_tau", "concordance_idx",
        # classification
        "roc_auc", "gini", "ks", "mcc", "cohen_kappa", "balanced_acc", "g_mean",
        "brier_skill", "f_beta_2", "lift_at_10pct", "accuracy_ratio",
        # Spiegelhalter Z: closer to 0 is better, doesn't fit min/max -- treat as min(|z|) later
    }
    def _direction(metric: str) -> str:
        return "max" if metric in HIGHER_IS_BETTER else "min"

    configs = [
        ("baseline",   dict(slice_k=0, aggregate="mean", confidence=0.5)),
        ("slice_mean", dict(slice_k=5, aggregate="mean", confidence=0.5)),
        ("slice_tlcb", dict(slice_k=5, aggregate="t_lcb", confidence=0.9)),
    ]
    models = ["lgb", "xgb", "cb"]
    tasks = [
        ("regr",       X_reg,   y_reg,   None),  # metrics list inferred from fit result keys
        ("regr_drift", X_drift, y_drift, None),
        ("classif",    X_cls,   y_cls,   None),
    ]

    # Resume support: load existing rows if any (skip re-running completed (task, model) blocks)
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            print(f"Resumed: {len(results)} rows already in {out_path}")
        except Exception:
            results = []
    else:
        results = []

    def _flush():
        out_path.write_text(json.dumps(results, indent=2))

    def _block_done(task_name: str, model: str) -> bool:
        """A (task, model) block is done when all 3 configs are present with n_folds==10."""
        present = {r["config"] for r in results
                   if r["task"] == task_name and r["model"] == model and r["n_folds"] == n_folds}
        return present == {"baseline", "slice_mean", "slice_tlcb"}

    grand_t0 = time.perf_counter()
    for task_name, X_all, y_all, _ in tasks:
        for model in models:
            if _block_done(task_name, model):
                print(f"\n--- {task_name.upper()} | {model.upper()} ---  [resume: skipping, already done]", flush=True)
                continue
            print(f"\n--- {task_name.upper()} | {model.upper()} ---", flush=True)
            per_config_per_fold: dict[str, list[dict]] = {cfg_name: [] for cfg_name, _ in configs}
            for fi, (tr_idx, te_idx) in enumerate(fold_idx):
                X_tr_full = X_all.iloc[tr_idx].reset_index(drop=True)
                y_tr_full = y_all[tr_idx]
                X_te = X_all.iloc[te_idx].reset_index(drop=True)
                y_te = y_all[te_idx]
                fold_t0 = time.perf_counter()
                for cfg_name, cfg_kwargs in configs:
                    m = _fit_and_score(
                        X_tr_full=X_tr_full, y_tr_full=y_tr_full,
                        X_te=X_te, y_te=y_te,
                        model=model, task=task_name, seed=seed, **cfg_kwargs,
                    )
                    per_config_per_fold[cfg_name].append(m)
                headline = per_config_per_fold["baseline"][-1].get("rmse",
                            per_config_per_fold["baseline"][-1].get("roc_auc", 0.0))
                print(f"  fold {fi+1}/{n_folds} done in {time.perf_counter()-fold_t0:.1f}s "
                      f"(baseline headline={headline:.4f})", flush=True)

            # Metric list discovered dynamically from the first fold result
            metric_list = list(per_config_per_fold["baseline"][0].keys())
            for cfg_name, _ in configs:
                fold_scores = per_config_per_fold[cfg_name]
                summary = dict(task=task_name, model=model, config=cfg_name,
                                n_folds=n_folds, metrics_per_fold=fold_scores)
                for metric in metric_list:
                    vals = [f[metric] for f in fold_scores]
                    vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                    if len(vals_clean) < 2:
                        continue
                    summary[f"median_{metric}"] = round(float(np.median(vals_clean)), 6)
                    summary[f"mean_{metric}"] = round(float(np.mean(vals_clean)), 6)
                    if cfg_name != "baseline":
                        base_vals = [f[metric] for f in per_config_per_fold["baseline"]]
                        # paired only on rows where both sides are non-NaN
                        paired = [(b, s) for b, s in zip(base_vals, vals)
                                  if not (np.isnan(b) or np.isnan(s))]
                        if len(paired) < 2:
                            continue
                        b_arr = [p[0] for p in paired]; s_arr = [p[1] for p in paired]
                        d = _direction(metric)
                        p = _wilcoxon(b_arr, s_arr, direction=d)
                        med_b = float(np.median(b_arr)); med_s = float(np.median(s_arr))
                        gap = (med_b - med_s) if d == "min" else (med_s - med_b)
                        denom = max(abs(med_b), 1e-9)
                        summary[f"p_{metric}"] = round(p, 4)
                        summary[f"gap_pct_{metric}"] = round(gap / denom * 100.0, 3)
                results.append(summary)
                _flush()

    print(f"\nTotal wall: {(time.perf_counter()-grand_t0)/60:.1f} min")

    # Pretty summary: print every metric, mark significance
    print("\n=== SUMMARY (median + paired Wilcoxon one-sided vs baseline; gap as %% improvement) ===")
    for task_name, *_ in tasks:
        print(f"\n--- {task_name.upper()} ---")
        for model in models:
            rows = [r for r in results if r["task"] == task_name and r["model"] == model]
            if not rows:
                continue
            baseline = next(r for r in rows if r["config"] == "baseline")
            metric_keys = [k.replace("median_", "") for k in baseline.keys() if k.startswith("median_")]
            # Pretty-print: header row
            print(f"\n  [{model.upper()}]")
            print(f"  {'metric':18s} {'baseline':>10s}  " + "  ".join(
                f"{c:>22s}" for c, _ in configs if c != "baseline"))
            for metric in metric_keys:
                base_med = baseline.get(f"median_{metric}", float("nan"))
                row_parts = [f"  {metric:18s} {base_med:>10.4f}"]
                for cfg_name, _ in configs:
                    if cfg_name == "baseline":
                        continue
                    row = next((r for r in rows if r["config"] == cfg_name), None)
                    if row is None:
                        row_parts.append(f"{'-':>22s}")
                        continue
                    p = row.get(f"p_{metric}")
                    gap = row.get(f"gap_pct_{metric}")
                    if p is None or gap is None:
                        row_parts.append(f"{'n/a':>22s}")
                    else:
                        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else " "))
                        row_parts.append(f"  gap={gap:+6.2f}% p={p:.3f}{sig:<3}")
                print("  ".join(row_parts))

    print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
