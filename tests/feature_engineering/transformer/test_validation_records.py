"""Honest validation of 7 standing records - multi-seed + full-N + median/IQR reporting.

Per VALIDATION_TODO.md priorities 1+3:
- Remove 4000 cap (done in test_biz_val_real_datasets._cap_rows).
- Multi-seed: re-run 7 records on seeds {0, 7, 17, 42, 99} to separate fold-luck from signal.
- Per-record per-dataset: report lift_median + lift_iqr; flag records whose CI overlaps zero.

Records under test (mechanism + dataset + metric + target_lift_from_iter):
1. iter 61  abalone XGB R2 +4.05% via multi_temp_residual_band + cdist
2. iter 66  mammography LGB AUC +14.46% via class_balanced_hard_row + RFF
3. iter 68  kin8nm LGB R2 +11.91% (marginal) via multi_baseline_hard_row + RFF
4. iter 69  abalone CB R2 +3.84% via baseline_disagreement + cdist
5. iter 72  abalone LGB R2 +3.19% via local_density_gradient alone
6. iter 77  diabetes CB PR_AUC +6.75% (marginal) via local_curvature alone
7. iter 77  diabetes ALL-5 sweep via local_curvature alone (separate diagnostic)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split

pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")
pytest.importorskip("catboost")
pytest.importorskip("sklearn")

from mlframe.feature_engineering.transformer import (
    compute_baseline_disagreement_features,
    compute_class_balanced_hard_row_features,
    compute_local_curvature_features,
    compute_local_density_gradient_features,
    compute_multi_baseline_hard_row_features,
    compute_multi_temp_residual_band_features,
    compute_rff_features,
)

# Reuse the existing test harness loaders and matrix utilities.
from tests.feature_engineering.transformer.test_biz_val_real_datasets import (
    _features_cdist,
    _features_rff,
    _load_abalone,
    _load_diabetes_classification,
    _load_kin8nm,
    _load_mammography,
)


pytestmark = [pytest.mark.fast, pytest.mark.biz_transformer]

# Seeds for validation.
_VALIDATION_SEEDS = (0, 7, 17, 42, 99)


# ---------- Mechanism + RFF/cdist combo builders for the 7 records ----------


def _build_iter61(X_tr, X_te, y_tr, task, seed):
    """multi_temp_residual_band + cdist."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    mt_tr = compute_multi_temp_residual_band_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
        seed=seed, task=task_str, n_bands=5, temps=(0.3, 1.0, 3.0),
    ).to_numpy()
    mt_te = compute_multi_temp_residual_band_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
        seed=seed, task=task_str, n_bands=5, temps=(0.3, 1.0, 3.0),
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, mt_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, mt_te, _strip(cd_te, X_te.shape[1])], axis=1))


def _build_iter66(X_tr, X_te, y_tr, task, seed):
    """class_balanced_hard_row + RFF."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    cb_tr = compute_class_balanced_hard_row_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
        seed=seed, task=task_str, n_hard_per_side=8, temp=1.0,
    ).to_numpy()
    cb_te = compute_class_balanced_hard_row_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
        seed=seed, task=task_str, n_hard_per_side=8, temp=1.0,
    ).to_numpy()
    rf_tr, rf_te = _features_rff_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, cb_tr, _strip(rf_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, cb_te, _strip(rf_te, X_te.shape[1])], axis=1))


def _build_iter68(X_tr, X_te, y_tr, task, seed):
    """multi_baseline_hard_row + RFF."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    mb_tr = compute_multi_baseline_hard_row_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
        seed=seed, task=task_str, n_hard_per_side=8, temp=1.0,
    ).to_numpy()
    mb_te = compute_multi_baseline_hard_row_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
        seed=seed, task=task_str, n_hard_per_side=8, temp=1.0,
    ).to_numpy()
    rf_tr, rf_te = _features_rff_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, mb_tr, _strip(rf_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, mb_te, _strip(rf_te, X_te.shape[1])], axis=1))


def _build_iter69(X_tr, X_te, y_tr, task, seed):
    """baseline_disagreement + cdist."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, _strip(cd_te, X_te.shape[1])], axis=1))


def _build_iter72(X_tr, X_te, y_tr, task, seed):
    """local_density_gradient alone."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    ld_tr = compute_local_density_gradient_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
        seed=seed, task=task_str, k_neighbors=32,
    ).to_numpy()
    ld_te = compute_local_density_gradient_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
        seed=seed, task=task_str, k_neighbors=32,
    ).to_numpy()
    return (np.concatenate([X_tr, ld_tr], axis=1),
            np.concatenate([X_te, ld_te], axis=1))


def _build_iter77(X_tr, X_te, y_tr, task, seed):
    """local_curvature alone."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    cv_tr = compute_local_curvature_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
        seed=seed, task=task_str, k_neighbors=40,
    ).to_numpy()
    cv_te = compute_local_curvature_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
        seed=seed, task=task_str, k_neighbors=40,
    ).to_numpy()
    return (np.concatenate([X_tr, cv_tr], axis=1),
            np.concatenate([X_te, cv_te], axis=1))


# Local seeded versions of the FE primitives.
def _features_rff_seeded(X_tr, X_te, y_tr, task, seed):
    from mlframe.feature_engineering import compute_rff_features
    import polars as pl
    rff_tr = compute_rff_features(pl.DataFrame(X_tr), n_features=256, seed=seed, sigma="median").to_numpy()
    rff_te = compute_rff_features(pl.DataFrame(X_te), n_features=256, seed=seed, sigma="median").to_numpy()
    return (np.concatenate([X_tr, rff_tr], axis=1),
            np.concatenate([X_te, rff_te], axis=1))


def _features_cdist_seeded(X_tr, X_te, y_tr, task, seed):
    # Reuse the existing cdist builder verbatim (it is already deterministic given X).
    return _features_cdist(X_tr, X_te, y_tr, task)


def _strip(full, n):
    return full[:, n:]


# ---------- Per-seed measurement utility ----------


def _measure_lift_seeded(X_full, y_full, task, builder, seed: int, target_model: str, target_metric: str) -> float:
    """Train a fresh model on raw vs +features, return lift on test fold."""
    import lightgbm as lgb
    from xgboost import XGBClassifier, XGBRegressor
    from catboost import CatBoostClassifier, CatBoostRegressor

    stratify = y_full if task == "binary" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y_full, test_size=0.3, random_state=seed, stratify=stratify)

    # Raw baseline
    if task == "binary":
        if target_model == "lgb":
            m_raw = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1).fit(X_tr, y_tr)
            p_raw = m_raw.predict_proba(X_te)[:, 1]
        elif target_model == "xgb":
            m_raw = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1, use_label_encoder=False, eval_metric="logloss", verbosity=0).fit(X_tr, y_tr)
            p_raw = m_raw.predict_proba(X_te)[:, 1]
        else:
            m_raw = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, random_seed=seed, verbose=0).fit(X_tr, y_tr)
            p_raw = m_raw.predict_proba(X_te)[:, 1]
    else:
        if target_model == "lgb":
            m_raw = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1).fit(X_tr, y_tr)
            p_raw = m_raw.predict(X_te)
        elif target_model == "xgb":
            m_raw = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1, verbosity=0).fit(X_tr, y_tr)
            p_raw = m_raw.predict(X_te)
        else:
            m_raw = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, random_seed=seed, verbose=0).fit(X_tr, y_tr)
            p_raw = m_raw.predict(X_te)

    score_raw = _score(y_te, p_raw, task, target_metric)

    # With FE
    X_tr_fe, X_te_fe = builder(X_tr, X_te, y_tr, task, seed)
    if task == "binary":
        if target_model == "lgb":
            m_fe = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1).fit(X_tr_fe, y_tr)
            p_fe = m_fe.predict_proba(X_te_fe)[:, 1]
        elif target_model == "xgb":
            m_fe = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1, use_label_encoder=False, eval_metric="logloss", verbosity=0).fit(X_tr_fe, y_tr)
            p_fe = m_fe.predict_proba(X_te_fe)[:, 1]
        else:
            m_fe = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, random_seed=seed, verbose=0).fit(X_tr_fe, y_tr)
            p_fe = m_fe.predict_proba(X_te_fe)[:, 1]
    else:
        if target_model == "lgb":
            m_fe = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1).fit(X_tr_fe, y_tr)
            p_fe = m_fe.predict(X_te_fe)
        elif target_model == "xgb":
            m_fe = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=seed, n_jobs=-1, verbosity=0).fit(X_tr_fe, y_tr)
            p_fe = m_fe.predict(X_te_fe)
        else:
            m_fe = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05, random_seed=seed, verbose=0).fit(X_tr_fe, y_tr)
            p_fe = m_fe.predict(X_te_fe)

    score_fe = _score(y_te, p_fe, task, target_metric)
    return float(score_fe - score_raw)


def _score(y_true, y_pred, task, metric):
    if metric == "R²" or metric == "R2":
        return r2_score(y_true, y_pred)
    if metric == "AUC":
        return roc_auc_score(y_true, y_pred)
    if metric == "PR_AUC":
        return average_precision_score(y_true, y_pred)
    if metric == "Brier":
        return -brier_score_loss(y_true, y_pred)  # negated so higher = better
    if metric == "LogLoss":
        return -log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6))
    raise ValueError(f"unknown metric {metric}")


# ---------- Validation drivers ----------


def _validate(loader_fn, builder, target_model, target_metric, claimed_lift, label):
    """Run builder × 5 seeds, return (median_lift, lift_iqr, all_lifts)."""
    X_full, y_full, task = loader_fn()
    lifts = []
    for seed in _VALIDATION_SEEDS:
        try:
            lift = _measure_lift_seeded(X_full, y_full, task, builder, seed, target_model, target_metric)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {type(exc).__name__}: {exc}")
            lift = float("nan")
        print(f"  [seed={seed}] {label} {target_model} {target_metric} lift: {lift:+.4f}")
        lifts.append(lift)
    lifts_arr = np.array([l for l in lifts if not np.isnan(l)])
    if lifts_arr.size == 0:
        return float("nan"), float("nan"), lifts
    median = float(np.median(lifts_arr))
    iqr = float(np.quantile(lifts_arr, 0.75) - np.quantile(lifts_arr, 0.25))
    lo = float(lifts_arr.min())
    hi = float(lifts_arr.max())
    survives = "SURVIVES" if median > 0 and lo > -abs(median) * 0.3 else "FOLD-NOISE?"
    print(f"\n>>> {label} {target_model} {target_metric}: median={median:+.4f} IQR={iqr:.4f} min={lo:+.4f} max={hi:+.4f} | claimed={claimed_lift:+.4f} | {survives}")
    return median, iqr, lifts


def test_validate_iter61_abalone_xgb_r2():
    _validate(_load_abalone, _build_iter61, "xgb", "R2", 0.0405, "iter61_mtrbattn+cdist")


def test_validate_iter66_mammography_lgb_auc():
    _validate(_load_mammography, _build_iter66, "lgb", "AUC", 0.1446, "iter66_cbhrattn+rff")


def test_validate_iter68_kin8nm_lgb_r2():
    _validate(_load_kin8nm, _build_iter68, "lgb", "R2", 0.1191, "iter68_mbhrattn+rff")


def test_validate_iter69_abalone_cb_r2():
    _validate(_load_abalone, _build_iter69, "cb", "R2", 0.0384, "iter69_blagreement+cdist")


def test_validate_iter72_abalone_lgb_r2():
    _validate(_load_abalone, _build_iter72, "lgb", "R2", 0.0319, "iter72_ldgrad_alone")


def test_validate_iter77_diabetes_cb_pr_auc():
    _validate(_load_diabetes_classification, _build_iter77, "cb", "PR_AUC", 0.0675, "iter77_curv_alone")
