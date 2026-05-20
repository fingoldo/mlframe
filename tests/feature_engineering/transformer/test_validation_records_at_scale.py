"""Priority 2 of the multi-seed honesty pass: do the 2 SURVIVING records
(iter68 kin8nm RFF, iter69 abalone disagreement+cdist) still hold on a
LARGER regression dataset?

Test target: California Housing (~20640 rows x 8 numeric features), which is
~2.5x the size of kin8nm (8192) and ~5x the size of abalone (4177).

The 4000-row cap is gone (already in test_validation_records.py). Each seed
gets its own KFold + train/test split. We run 3 seeds {0, 17, 42} to keep
runtime under ~30min while still letting fold-noise become visible.

Verdict rule (same as test_validation_records.py): SURVIVES iff
``median > 0 AND min > -0.3 * median``; FOLD-NOISE? otherwise.

Caveat already documented in RESULTS.md: California Housing's +rff result on
its OWN test_biz_val matrix is negative (-3.6% LGB R2), so iter68 generalising
here would be informative. iter69's cdist component is currently seeded
internally with seed=42 (in test_biz_val_real_datasets._features_cdist), so
the seed variation here exercises baseline_disagreement + train/test split,
but not the cdist neighbours themselves.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

# Module docstring: "runtime under ~30min" with 3 seeds, real California Housing
# (20640 rows) + Year_100k loaded from disk. Each train fit + KFold cycle
# exceeds the fast-mode --timeout=60. Mark slow_only so --fast skips it.
pytestmark = [pytest.mark.slow_only, pytest.mark.biz_transformer]

from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_california
from tests.feature_engineering.transformer.test_validation_records import (
    _build_iter68,
    _build_iter69,
    _measure_lift_seeded,
)


# Earlier pytestmark = pytest.mark.fast was here -- removed because the
# slow_only marker at module top (line 35) is correct for this 30-min suite
# and the later assignment silently overrode it.


_SCALE_SEEDS = (0, 17, 42)


def _validate_scale(loader_fn, builder, target_model, target_metric, claimed_lift, label):
    X_full, y_full, task = loader_fn()
    print(f"\n  Dataset shape: {X_full.shape}  task: {task}")
    lifts = []
    for seed in _SCALE_SEEDS:
        try:
            lift = _measure_lift_seeded(X_full, y_full, task, builder, seed, target_model, target_metric)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {type(exc).__name__}: {exc}")
            lift = float("nan")
        print(f"  [seed={seed}] {label} {target_model} {target_metric} lift: {lift:+.4f}")
        lifts.append(lift)
    arr = np.array([l for l in lifts if not np.isnan(l)])
    if arr.size == 0:
        print(f">>> {label}: ALL ERROR")
        return
    median = float(np.median(arr))
    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)) if arr.size > 1 else 0.0
    lo = float(arr.min())
    hi = float(arr.max())
    survives = "SURVIVES" if median > 0 and lo > -abs(median) * 0.3 else "FOLD-NOISE?"
    print(f">>> {label} {target_model} {target_metric}: median={median:+.4f} IQR={iqr:.4f} min={lo:+.4f} max={hi:+.4f} | small-N claimed={claimed_lift:+.4f} | {survives}")


def test_scale_iter68_california_lgb_r2():
    """iter68 (multi_baseline_hard_row + RFF) was +11.42% median on kin8nm 8k. Does it hold on California 20k?"""
    _validate_scale(_load_california, _build_iter68, "lgb", "R2", 0.1142, "iter68_mbhrattn+rff_CA20k")


def test_scale_iter69_california_cb_r2():
    """iter69 (baseline_disagreement + cdist) was +2.26% median on abalone 4k. Does it hold on California 20k?"""
    _validate_scale(_load_california, _build_iter69, "cb", "R2", 0.0226, "iter69_blagreement+cdist_CA20k")


def _load_year_100k():
    """Year-prediction-MSD subsampled to 100k. Audio features (90) → song year (regression).

    Loads from local npz cache to avoid repeated OpenML ARFF-parser OOM.
    """
    import os
    cache_path = "D:/Temp/year_prediction_cache.npz"
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        X = d["X"]
        y = d["y"]
    else:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=44027, as_frame=False, parser="liac-arff")
        X = np.asarray(ds.data, dtype=np.float32)
        y = np.asarray(ds.target, dtype=np.float32)
    rng = np.random.default_rng(2026)
    idx = rng.choice(X.shape[0], 100_000, replace=False)
    return X[idx], y[idx], "regression"


def test_scale_iter69_year_100k_cb_r2():
    """iter69 (baseline_disagreement + cdist) survives at California 20k. Does it scale to year-prediction 100k?

    25x larger than abalone (the small-N record dataset). If iter69 still shows positive
    lift at 100k, the mechanism is genuinely general regression-FE, not a small-N artifact.
    """
    _validate_scale(_load_year_100k, _build_iter69, "cb", "R2", 0.0115, "iter69_blagreement+cdist_Year100k")


# ---------- iter102 - baseline_disagreement_v2 (iter69 + ExtraTrees orthogonal baseline) ----------


def _build_iter102(X_tr, X_te, y_tr, task, seed):
    """baseline_disagreement_v2 + cdist (iter102 - iter69 with ExtraTrees added)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import compute_baseline_disagreement_v2_features

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, _strip(cd_te, X_te.shape[1])], axis=1))


# iter102 vs iter69 head-to-head: same harness, same datasets, multi-seed-from-start.

def test_iter102_abalone_cb_r2():
    """iter102 on abalone (was iter69 +2.26% median CB R2). Target: match or beat."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter102, "cb", "R2", 0.0226, "iter102_blagreementv2+cdist_abalone")


def test_iter102_california_cb_r2():
    """iter102 on California (was iter69 +1.15% median CB R2 at 20k). Target: match or beat."""
    _validate_scale(_load_california, _build_iter102, "cb", "R2", 0.0115, "iter102_blagreementv2+cdist_CA20k")


def test_iter102_year_100k_cb_r2():
    """iter102 on year-prediction 100k (was iter69 +4.92% median CB R2). Target: match or beat."""
    _validate_scale(_load_year_100k, _build_iter102, "cb", "R2", 0.0492, "iter102_blagreementv2+cdist_Year100k")


# ---------- iter103 - residual_stratified_distance (alone, no cdist) ----------
# Structural shift from "baseline ensembles": expose LOCAL DENSITY of baseline-easy vs
# baseline-hard training rows around each query. Different signal from iter69/102.


def _build_iter103(X_tr, X_te, y_tr, task, seed):
    """residual_stratified_distance alone (iter103)."""
    from sklearn.model_selection import KFold
    from mlframe.feature_engineering.transformer import compute_residual_stratified_distance_features

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    rsd_tr = compute_residual_stratified_distance_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    rsd_te = compute_residual_stratified_distance_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    return (np.concatenate([X_tr, rsd_tr], axis=1),
            np.concatenate([X_te, rsd_te], axis=1))


def test_iter103_abalone_cb_r2():
    """iter103 on abalone (was iter69 +2.26%, iter102 +2.74%). Target: improve or differ structurally."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter103, "cb", "R2", 0.0274, "iter103_rsd_abalone")


def test_iter103_california_cb_r2():
    """iter103 on California 20k (was iter69 +1.15%)."""
    _validate_scale(_load_california, _build_iter103, "cb", "R2", 0.0115, "iter103_rsd_CA20k")


def test_iter103_year_100k_cb_r2():
    """iter103 on year-prediction 100k (was iter69 +4.92%, iter102 +4.93%)."""
    _validate_scale(_load_year_100k, _build_iter103, "cb", "R2", 0.0492, "iter103_rsd_Year100k")


# ---------- iter104 - iter69 + iter103 additive (does negative-alone iter103 help iter69?) ----------


def _build_iter104(X_tr, X_te, y_tr, task, seed):
    """iter69 (baseline_disagreement + cdist) + iter103 (residual_stratified_distance)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_residual_stratified_distance_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    rsd_tr = compute_residual_stratified_distance_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    rsd_te = compute_residual_stratified_distance_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, rsd_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, rsd_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter104_abalone_cb_r2():
    """iter104 on abalone (was iter69 +2.26%, iter102 +2.74%, iter103 -1.39% alone)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter104, "cb", "R2", 0.0274, "iter104_iter69+iter103_abalone")


def test_iter104_california_cb_r2():
    """iter104 on California 20k (was iter69 +1.15%, iter103 -0.48% alone)."""
    _validate_scale(_load_california, _build_iter104, "cb", "R2", 0.0115, "iter104_iter69+iter103_CA20k")


def test_iter104_year_100k_cb_r2():
    """iter104 on year-prediction 100k (was iter69 +4.92%, iter103 +0.96% alone)."""
    _validate_scale(_load_year_100k, _build_iter104, "cb", "R2", 0.0492, "iter104_iter69+iter103_Year100k")


# ---------- iter105 - triple: baseline_disagreement_v2 + cdist + residual_stratified_distance ----------
# Does abalone-helper (ExtraTrees baseline) compose with Year-100k-helper (RSD geometric density)?
# = best-of-breed across N regimes?


def _build_iter105(X_tr, X_te, y_tr, task, seed):
    """baseline_disagreement_v2 (iter102) + cdist + residual_stratified_distance (iter103)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_v2_features,
        compute_residual_stratified_distance_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    rsd_tr = compute_residual_stratified_distance_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    rsd_te = compute_residual_stratified_distance_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, rsd_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, rsd_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter105_abalone_cb_r2():
    """iter105 triple on abalone (best was iter102 +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter105, "cb", "R2", 0.0274, "iter105_triple_abalone")


def test_iter105_california_cb_r2():
    """iter105 triple on California 20k (best was iter69 +1.15%)."""
    _validate_scale(_load_california, _build_iter105, "cb", "R2", 0.0115, "iter105_triple_CA20k")


def test_iter105_year_100k_cb_r2():
    """iter105 triple on year-prediction 100k (best was iter104 +5.25%)."""
    _validate_scale(_load_year_100k, _build_iter105, "cb", "R2", 0.0525, "iter105_triple_Year100k")


# ---------- iter106 - y-quintile-conditioned baseline-prediction-at-kNN (alone, no cdist) ----------


def _build_iter106(X_tr, X_te, y_tr, task, seed):
    """y_quintile_baseline_knn alone (iter106)."""
    from sklearn.model_selection import KFold
    from mlframe.feature_engineering.transformer import compute_y_quintile_baseline_knn_features

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    yq_tr = compute_y_quintile_baseline_knn_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    yq_te = compute_y_quintile_baseline_knn_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    return (np.concatenate([X_tr, yq_tr], axis=1),
            np.concatenate([X_te, yq_te], axis=1))


def test_iter106_abalone_cb_r2():
    """iter106 on abalone (best so far iter102 +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter106, "cb", "R2", 0.0274, "iter106_yqbk_abalone")


def test_iter106_california_cb_r2():
    """iter106 on California 20k (best so far iter69 +1.15%)."""
    _validate_scale(_load_california, _build_iter106, "cb", "R2", 0.0115, "iter106_yqbk_CA20k")


def test_iter106_year_100k_cb_r2():
    """iter106 on year-prediction 100k (best so far iter104 +5.25%)."""
    _validate_scale(_load_year_100k, _build_iter106, "cb", "R2", 0.0525, "iter106_yqbk_Year100k")


# ---------- iter107 - bgmm_quantile_bands alone (existing iter56 mechanism, untested under multi-seed-from-start) ----------
# Structural family shift: Bayesian GMM density features instead of baseline-prediction or kNN-residual.


def _build_iter107(X_tr, X_te, y_tr, task, seed):
    """bgmm_quantile_bands alone (iter107)."""
    from sklearn.model_selection import KFold
    from mlframe.feature_engineering.transformer import compute_bgmm_quantile_bands_features

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bqb_tr = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bqb_te = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    return (np.concatenate([X_tr, bqb_tr], axis=1),
            np.concatenate([X_te, bqb_te], axis=1))


def test_iter107_abalone_cb_r2():
    """iter107 BGM alone on abalone."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter107, "cb", "R2", 0.0274, "iter107_bgm_abalone")


def test_iter107_california_cb_r2():
    """iter107 BGM alone on California 20k."""
    _validate_scale(_load_california, _build_iter107, "cb", "R2", 0.0115, "iter107_bgm_CA20k")


def test_iter107_year_100k_cb_r2():
    """iter107 BGM alone on year-prediction 100k."""
    _validate_scale(_load_year_100k, _build_iter107, "cb", "R2", 0.0525, "iter107_bgm_Year100k")


# ---------- iter108 - iter69 on BINARY classification (task-regime generalisation test) ----------


def _load_adult_binary():
    """Adult (49k rows, binary >50k income). Categorical columns one-hot encoded; total ~100 numeric features."""
    from sklearn.datasets import fetch_openml
    import pandas as pd
    ds = fetch_openml(data_id=1590, as_frame=True, parser="liac-arff")
    # Drop rows with any NaN
    df = ds.data.copy()
    df["__y__"] = (ds.target == ">50K").astype(np.int32)
    df = df.dropna()
    y = df["__y__"].to_numpy()
    X_df = df.drop(columns=["__y__"])
    # One-hot encode categoricals
    X_oh = pd.get_dummies(X_df, drop_first=True, dummy_na=False)
    X = X_oh.to_numpy(dtype=np.float32)
    return X, y, "binary"


def test_iter108_adult_lgb_auc():
    """iter69 on Adult 49k binary classification (LGB AUC). Does iter69 generalise from regression to binary?"""
    _validate_scale(_load_adult_binary, _build_iter69, "lgb", "AUC", 0.0, "iter108_iter69_Adult49k")


def test_iter108_adult_cb_auc():
    """iter69 on Adult 49k binary classification (CB AUC)."""
    _validate_scale(_load_adult_binary, _build_iter69, "cb", "AUC", 0.0, "iter108_iter69_Adult49k_cb")


# ---------- iter109 - iter69 on Higgs 98k binary (scale-up of iter108) ----------


def _load_higgs_binary():
    """Higgs (~98k rows, 28 numeric features, binary signal vs background)."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=23512, as_frame=False, parser="liac-arff")
    X = np.asarray(ds.data, dtype=np.float32)
    # Higgs target is string-encoded class label
    y = (np.asarray(ds.target) == "1").astype(np.int32)
    if y.sum() == 0:
        # alternative encoding
        unique_targets = np.unique(ds.target)
        y = (np.asarray(ds.target) == unique_targets[-1]).astype(np.int32)
    # Drop rows with any NaN
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask], "binary"


def test_iter109_higgs_cb_auc():
    """iter69 on Higgs 98k binary (CB AUC). Scale-up of iter108's Adult 49k +0.63%."""
    _validate_scale(_load_higgs_binary, _build_iter69, "cb", "AUC", 0.0063, "iter109_iter69_Higgs98k_cb")


def test_iter109_higgs_lgb_auc():
    """iter69 on Higgs 98k binary (LGB AUC). LGB barely benefited on Adult; does it scale?"""
    _validate_scale(_load_higgs_binary, _build_iter69, "lgb", "AUC", 0.0006, "iter109_iter69_Higgs98k_lgb")


# ---------- iter110 - iter69 + iter66 (class_balanced_hard_row + RFF) on Adult binary ----------
# Test whether iter66's retracted-on-mammography mechanism adds value as additive component
# at Adult 49k scale (4x the mammography 11k).


def _build_iter110(X_tr, X_te, y_tr, task, seed):
    """iter69 (baseline_disagreement + cdist) + iter66 (class_balanced_hard_row + RFF)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import (
        _features_cdist_seeded, _features_rff_seeded, _strip,
    )
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_class_balanced_hard_row_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cb_tr = compute_class_balanced_hard_row_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
        seed=seed, task=task_str, n_hard_per_side=8, temp=1.0,
    ).to_numpy()
    cb_te = compute_class_balanced_hard_row_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
        seed=seed, task=task_str, n_hard_per_side=8, temp=1.0,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    rf_tr, rf_te = _features_rff_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, cb_tr, _strip(cd_tr, X_tr.shape[1]), _strip(rf_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, cb_te, _strip(cd_te, X_te.shape[1]), _strip(rf_te, X_te.shape[1])], axis=1))


def test_iter110_adult_cb_auc():
    """iter110 = iter69 + iter66 on Adult 49k binary (was iter69 alone +0.63%)."""
    _validate_scale(_load_adult_binary, _build_iter110, "cb", "AUC", 0.0063, "iter110_iter69+iter66_Adult49k_cb")


# ---------- iter111 - iter69 on MAMMOGRAPHY 11k full-N (rare-positive binary stress test) ----------


def _load_mammography():
    """Mammography (~11183 rows, 6 numeric features, binary class label, 1.3% positive)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_mammography as _l
    return _l()


def test_iter111_mammography_cb_auc():
    """iter69 on mammography 11k full-N binary (rare-positive 1.3%, CB AUC)."""
    _validate_scale(_load_mammography, _build_iter69, "cb", "AUC", 0.0063, "iter111_iter69_Mammog11k_cb")


def test_iter111_mammography_lgb_auc():
    """iter69 on mammography 11k full-N binary (LGB AUC). iter66 retracted on this dataset under multi-seed."""
    _validate_scale(_load_mammography, _build_iter69, "lgb", "AUC", -0.0077, "iter111_iter69_Mammog11k_lgb")


# ---------- iter112 - iter69 with STRATIFIED KFold for binary (fix rare-positive bug) ----------


class _StratifiedKFoldYBound:
    """Wrapper around StratifiedKFold so its split(X) call (no y) works with existing FE primitives.

    Existing FE primitives call splitter.split(X) without y. This wrapper pre-binds y so the call
    works for stratified splits too.
    """
    def __init__(self, n_splits, shuffle, random_state, y):
        from sklearn.model_selection import StratifiedKFold
        self._inner = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._y = y

    def split(self, X, y=None, groups=None):
        return self._inner.split(X, self._y if y is None else y)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._inner.get_n_splits(X, y, groups)


def _build_iter69_stratified(X_tr, X_te, y_tr, task, seed):
    """Same as iter69 but uses StratifiedKFold for binary classification.

    Fixes the rare-positive bug where KFold may produce folds with 0 positives at <2% positive rate,
    causing baseline disagreement and cdist to compute meaningless statistics.
    """
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_class_distance_features,
    )

    if task == "binary":
        splitter = _StratifiedKFoldYBound(n_splits=5, shuffle=True, random_state=seed, y=y_tr)
        task_str = "binary"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        task_str = "regression"

    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    # cdist for binary needs stratified splits too.
    if task == "binary":
        cd_tr = compute_class_distance_features(
            X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
            seed=seed, task=task_str, standardize=True,
        ).to_numpy()
        cd_te = compute_class_distance_features(
            X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
            seed=seed, task=task_str, standardize=True,
        ).to_numpy()
    else:
        cd_tr_full, cd_te_full = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
        cd_tr = _strip(cd_tr_full, X_tr.shape[1])
        cd_te = _strip(cd_te_full, X_te.shape[1])

    return (np.concatenate([X_tr, bl_tr, cd_tr], axis=1),
            np.concatenate([X_te, bl_te, cd_te], axis=1))


def test_iter112_mammography_cb_auc():
    """iter69 with StratifiedKFold on mammography 11k (was iter111 -1.05% with regular KFold)."""
    _validate_scale(_load_mammography, _build_iter69_stratified, "cb", "AUC", -0.0105, "iter112_iter69strat_Mammog11k_cb")


def test_iter112_mammography_lgb_auc():
    """iter69 with StratifiedKFold on mammography 11k (was iter111 -0.36% with regular KFold)."""
    _validate_scale(_load_mammography, _build_iter69_stratified, "lgb", "AUC", -0.0036, "iter112_iter69strat_Mammog11k_lgb")


def test_iter112_adult_cb_auc():
    """iter69 with StratifiedKFold on Adult 49k (was iter108 +0.63%). Should match or beat."""
    _validate_scale(_load_adult_binary, _build_iter69_stratified, "cb", "AUC", 0.0063, "iter112_iter69strat_Adult49k_cb")


# ---------- iter113 - class-balanced baseline disagreement + cdist on rare-positive binary ----------


def _build_iter113(X_tr, X_te, y_tr, task, seed):
    """class-balanced baseline_disagreement (iter113) + cdist; with StratifiedKFold for binary."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_balanced_features,
        compute_class_distance_features,
    )

    if task == "binary":
        splitter = _StratifiedKFoldYBound(n_splits=5, shuffle=True, random_state=seed, y=y_tr)
        task_str = "binary"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        task_str = "regression"

    bl_tr = compute_baseline_disagreement_balanced_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_balanced_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    if task == "binary":
        cd_tr = compute_class_distance_features(
            X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
            seed=seed, task=task_str, standardize=True,
        ).to_numpy()
        cd_te = compute_class_distance_features(
            X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
            seed=seed, task=task_str, standardize=True,
        ).to_numpy()
    else:
        cd_tr_full, cd_te_full = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
        cd_tr = _strip(cd_tr_full, X_tr.shape[1])
        cd_te = _strip(cd_te_full, X_te.shape[1])

    return (np.concatenate([X_tr, bl_tr, cd_tr], axis=1),
            np.concatenate([X_te, bl_te, cd_te], axis=1))


def test_iter113_mammography_cb_auc():
    """iter113 (class-balanced baselines) on mammography 11k (was iter112 -0.70%)."""
    _validate_scale(_load_mammography, _build_iter113, "cb", "AUC", -0.0070, "iter113_balanced_Mammog11k_cb")


def test_iter113_mammography_lgb_auc():
    """iter113 on mammography 11k LGB AUC."""
    _validate_scale(_load_mammography, _build_iter113, "lgb", "AUC", -0.0184, "iter113_balanced_Mammog11k_lgb")


def test_iter113_adult_cb_auc():
    """iter113 on Adult 49k (was iter108 +0.63%); regression test that fix doesn't break balanced binary."""
    _validate_scale(_load_adult_binary, _build_iter113, "cb", "AUC", 0.0063, "iter113_balanced_Adult49k_cb")


# ---------- iter114 - SMOTE-augmented baselines for rare-positive binary ----------


def _build_iter114(X_tr, X_te, y_tr, task, seed):
    """SMOTE-augmented baseline_disagreement (iter114) + cdist."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_smote_features,
        compute_class_distance_features,
    )

    if task == "binary":
        splitter = _StratifiedKFoldYBound(n_splits=5, shuffle=True, random_state=seed, y=y_tr)
        task_str = "binary"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        task_str = "regression"

    bl_tr = compute_baseline_disagreement_smote_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_smote_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    if task == "binary":
        cd_tr = compute_class_distance_features(
            X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter,
            seed=seed, task=task_str, standardize=True,
        ).to_numpy()
        cd_te = compute_class_distance_features(
            X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter,
            seed=seed, task=task_str, standardize=True,
        ).to_numpy()
    else:
        cd_tr_full, cd_te_full = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
        cd_tr = _strip(cd_tr_full, X_tr.shape[1])
        cd_te = _strip(cd_te_full, X_te.shape[1])

    return (np.concatenate([X_tr, bl_tr, cd_tr], axis=1),
            np.concatenate([X_te, bl_te, cd_te], axis=1))


def test_iter114_mammography_cb_auc():
    """iter114 SMOTE on mammography 11k (was iter113 -0.55%; can SMOTE flip to positive?)."""
    _validate_scale(_load_mammography, _build_iter114, "cb", "AUC", -0.0055, "iter114_smote_Mammog11k_cb")


def test_iter114_mammography_lgb_auc():
    """iter114 SMOTE on mammography 11k LGB AUC."""
    _validate_scale(_load_mammography, _build_iter114, "lgb", "AUC", -0.0157, "iter114_smote_Mammog11k_lgb")


def test_iter114_adult_cb_auc():
    """iter114 on Adult 49k (regression test that SMOTE doesn't break balanced binary)."""
    _validate_scale(_load_adult_binary, _build_iter114, "cb", "AUC", 0.0063, "iter114_smote_Adult49k_cb")


# ---------- iter115 - IsolationForest anomaly-score features (pure-X, no labels) ----------


def _build_iter115_alone(X_tr, X_te, y_tr, task, seed):
    """anomaly_score alone (iter115). Pure-X mechanism."""
    from sklearn.model_selection import KFold
    from mlframe.feature_engineering.transformer import compute_anomaly_score_features

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    a_tr = compute_anomaly_score_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    a_te = compute_anomaly_score_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    return (np.concatenate([X_tr, a_tr], axis=1),
            np.concatenate([X_te, a_te], axis=1))


def _build_iter115_with_iter69(X_tr, X_te, y_tr, task, seed):
    """iter69 + anomaly_score (additive)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_anomaly_score_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    a_tr = compute_anomaly_score_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    a_te = compute_anomaly_score_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, a_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, a_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter115_mammography_cb_auc_alone():
    """iter115 anomaly score alone on mammography 11k binary (iter69 family failed here)."""
    _validate_scale(_load_mammography, _build_iter115_alone, "cb", "AUC", 0.0, "iter115_anom_alone_Mammog11k_cb")


def test_iter115_mammography_cb_auc_with_iter69():
    """iter115 anomaly + iter69 additive on mammography 11k binary."""
    _validate_scale(_load_mammography, _build_iter115_with_iter69, "cb", "AUC", 0.0, "iter115_anom+iter69_Mammog11k_cb")


def test_iter115_adult_cb_auc_with_iter69():
    """iter115 anomaly + iter69 additive on Adult 49k (does anomaly add to balanced binary?)."""
    _validate_scale(_load_adult_binary, _build_iter115_with_iter69, "cb", "AUC", 0.0063, "iter115_anom+iter69_Adult49k_cb")


# ---------- iter116 - test iter69 + iter104 (iter69+RSD) on kin8nm 8k (regression mechanism map completion) ----------
# kin8nm is the original RFF-winning dataset (iter68 +11.42% LGB R2 median, multi_baseline_hard_row + RFF).
# iter69-family was never tested here. If iter69 or iter104 works on kin8nm, regression boundary is fully mapped.


def _load_kin8nm():
    """kin8nm regression: ~8192 rows, 8 numeric features, robot arm dynamics target."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_kin8nm as _l
    return _l()


def test_iter116_kin8nm_lgb_r2_iter69():
    """iter69 alone (baseline_disagreement + cdist) on kin8nm 8k LGB R2 (vs kin8nm-record iter68 +11.42% via RFF)."""
    _validate_scale(_load_kin8nm, _build_iter69, "lgb", "R2", 0.1142, "iter116_iter69_kin8nm_lgb")


def test_iter116_kin8nm_cb_r2_iter69():
    """iter69 alone on kin8nm 8k CB R2."""
    _validate_scale(_load_kin8nm, _build_iter69, "cb", "R2", 0.0, "iter116_iter69_kin8nm_cb")


def test_iter116_kin8nm_lgb_r2_iter104():
    """iter104 (iter69 + RSD additive) on kin8nm 8k LGB R2. Best Year-100k mechanism; does it generalize?"""
    _validate_scale(_load_kin8nm, _build_iter104, "lgb", "R2", 0.1142, "iter116_iter104_kin8nm_lgb")


def test_iter116_kin8nm_cb_r2_iter104():
    """iter104 on kin8nm 8k CB R2."""
    _validate_scale(_load_kin8nm, _build_iter104, "cb", "R2", 0.0, "iter116_iter104_kin8nm_cb")


# ---------- iter117 - confirm iter104 LGB-target enhancement + iter102 generalisation tests ----------


def test_iter117_california_lgb_r2_iter69():
    """iter69 alone on California 20k LGB R2 (baseline; we only had CB R2 +1.15% before)."""
    _validate_scale(_load_california, _build_iter69, "lgb", "R2", 0.0, "iter117_iter69_CA20k_lgb")


def test_iter117_california_lgb_r2_iter104():
    """iter104 on California 20k LGB R2 (does the LGB-target enhancement pattern hold?)."""
    _validate_scale(_load_california, _build_iter104, "lgb", "R2", 0.0, "iter117_iter104_CA20k_lgb")


def test_iter117_year100k_lgb_r2_iter69():
    """iter69 alone on Year-100k LGB R2 (Year-100k had +5.25% CB R2; need LGB baseline)."""
    _validate_scale(_load_year_100k, _build_iter69, "lgb", "R2", 0.0, "iter117_iter69_Year100k_lgb")


def test_iter117_year100k_lgb_r2_iter104():
    """iter104 on Year-100k LGB R2."""
    _validate_scale(_load_year_100k, _build_iter104, "lgb", "R2", 0.0, "iter117_iter104_Year100k_lgb")


def test_iter117_kin8nm_cb_r2_iter102():
    """iter102 on kin8nm 8k CB R2 (ExtraTrees abalone-helper; does it generalise to kin8nm CB?)."""
    _validate_scale(_load_kin8nm, _build_iter102, "cb", "R2", 0.0, "iter117_iter102_kin8nm_cb")


# ---------- iter118 - cross-scale boundary test: iter102 large-N, iter104 small-N ----------


@pytest.mark.no_xdist
def test_iter118_year100k_cb_r2_iter102():
    """iter102 (+ExtraTrees) on Year-100k CB R2 (does it beat iter104's +5.25% at large N?).

    @no_xdist: Year-100k dataset (100k rows x 90 features) + 3 seeds + CB
    fits. Native-crashed the xdist worker under 14-parallel load on S:
    2026-05-20 (likely RAM ceiling: each CB fit holds the full 100k frame
    plus the ExtraTrees feature builder side-set). Run sequentially.
    """
    _validate_scale(_load_year_100k, _build_iter102, "cb", "R2", 0.0525, "iter118_iter102_Year100k_cb")


def test_iter118_abalone_cb_r2_iter104():
    """iter104 (+RSD) on abalone 4k CB R2 (does it beat iter102's +2.74% at small N?)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter104, "cb", "R2", 0.0274, "iter118_iter104_abalone_cb")


# ---------- iter119 - iter69 + boosting_leaf-index features (orthogonal signal source) ----------


def _build_iter119(X_tr, X_te, y_tr, task, seed):
    """iter69 + boosting_leaf-index features. Leaf membership is orthogonal to per-row prediction values."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_boosting_leaf_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    lf_tr = compute_boosting_leaf_features(
        X_train=X_tr, y_train=y_tr, X_query=None,
        seed=seed, task=task_str, n_estimators=30, max_depth=4,
    ).to_numpy()
    lf_te = compute_boosting_leaf_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te,
        seed=seed, task=task_str, n_estimators=30, max_depth=4,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, lf_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, lf_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter119_abalone_cb_r2():
    """iter119 (iter69 + leaf-index) on abalone 4k CB R2 (record iter102 +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter119, "cb", "R2", 0.0274, "iter119_iter69+leaf_abalone_cb")


def test_iter119_kin8nm_cb_r2():
    """iter119 on kin8nm 8k CB R2 (record iter102 +6.57%)."""
    _validate_scale(_load_kin8nm, _build_iter119, "cb", "R2", 0.0657, "iter119_iter69+leaf_kin8nm_cb")


def test_iter119_year100k_cb_r2():
    """iter119 on Year-100k CB R2 (record iter104 +5.25%)."""
    _validate_scale(_load_year_100k, _build_iter119, "cb", "R2", 0.0525, "iter119_iter69+leaf_Year100k_cb")


# ---------- iter120 - generalisation test of iter68 (multi_baseline_hard_row + RFF) ----------
# iter68 was the historical kin8nm LGB R2 record (+11.42% multi-seed). Does it generalise?


def test_iter120_abalone_lgb_r2_iter68():
    """iter68 on abalone 4k LGB R2 (test if iter68 generalises beyond kin8nm)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter68, "lgb", "R2", 0.0, "iter120_iter68_abalone_lgb")


def test_iter120_california_lgb_r2_iter68():
    """iter68 on California 20k LGB R2 (test generalisation; LGB is FLAT on CA for iter69-family)."""
    _validate_scale(_load_california, _build_iter68, "lgb", "R2", 0.0, "iter120_iter68_CA20k_lgb")


def test_iter120_year100k_lgb_r2_iter68():
    """iter68 on Year-100k LGB R2 (vs iter104 +3.09% record)."""
    _validate_scale(_load_year_100k, _build_iter68, "lgb", "R2", 0.0309, "iter120_iter68_Year100k_lgb")


# ---------- iter121 - iter69 + bgmm_quantile_bands additive (BGM as additive, not alone) ----------
# iter107 tested BGM alone (negative). Does BGM add value as additive component to iter69?
# This is the loop's original iter56 vision: multi-quantile-band BGM for regression.


def _build_iter121(X_tr, X_te, y_tr, task, seed):
    """iter69 + bgmm_quantile_bands."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_bgmm_quantile_bands_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bg_tr = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bg_te = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, bg_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, bg_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter121_abalone_cb_r2():
    """iter121 (iter69 + BGM) on abalone 4k CB R2 (vs iter102 record +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter121, "cb", "R2", 0.0274, "iter121_iter69+bgm_abalone_cb")


def test_iter121_kin8nm_cb_r2():
    """iter121 on kin8nm 8k CB R2 (vs iter102 record +6.57%)."""
    _validate_scale(_load_kin8nm, _build_iter121, "cb", "R2", 0.0657, "iter121_iter69+bgm_kin8nm_cb")


# ---------- iter122 - iter121 (iter69+BGM) scale test to Year-100k ----------


def _load_year_50k():
    """Year-prediction subsampled to 50k. Loads from local npz cache."""
    import os
    cache_path = "D:/Temp/year_prediction_cache.npz"
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        X = d["X"]
        y = d["y"]
    else:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=44027, as_frame=False, parser="liac-arff")
        X = np.asarray(ds.data, dtype=np.float32)
        y = np.asarray(ds.target, dtype=np.float32)
    rng = np.random.default_rng(2026)
    idx = rng.choice(X.shape[0], 50_000, replace=False)
    return X[idx], y[idx], "regression"


def test_iter122_year50k_cb_r2_iter121():
    """iter121 on Year-50k CB R2 (smaller subsample to avoid BGM OOM at 100k)."""
    _validate_scale(_load_year_50k, _build_iter121, "cb", "R2", 0.0525, "iter122_iter121_Year50k_cb")


# ---------- iter123 - baselines at Year-50k for fair comparison with iter122 ----------


def test_iter123_year50k_cb_r2_iter69():
    """iter69 alone on Year-50k CB R2 (fair scale-baseline for iter122 +4.03%)."""
    _validate_scale(_load_year_50k, _build_iter69, "cb", "R2", 0.0, "iter123_iter69_Year50k_cb")


def test_iter123_year50k_cb_r2_iter104():
    """iter104 (iter69+RSD) on Year-50k CB R2 (fair scale-baseline for iter122)."""
    _validate_scale(_load_year_50k, _build_iter104, "cb", "R2", 0.0, "iter123_iter104_Year50k_cb")


# ---------- iter124 - does iter121 (iter69+BGM) help binary too? ----------


def test_iter124_adult_cb_auc_iter121():
    """iter121 (iter69+BGM) on Adult 49k binary CB AUC (vs iter69 alone +0.63%). Does BGM generalise to binary?"""
    _validate_scale(_load_adult_binary, _build_iter121, "cb", "AUC", 0.0063, "iter124_iter121_Adult49k_cb")


# ---------- iter125 - compute_row_attention (original transformer-FE backbone) ----------
# Never validated under multi-seed-from-start protocol; structurally orthogonal to iter69.


def _build_iter125_alone(X_tr, X_te, y_tr, task, seed):
    """compute_row_attention alone (softmax-weighted kNN-target-encoding over random projections)."""
    from sklearn.model_selection import KFold
    from mlframe.feature_engineering.transformer import compute_row_attention

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    ra_tr = compute_row_attention(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed,
        n_heads=4, head_dim=8, k=32, softmax_temp=1.0, standardize=True,
    ).to_numpy()
    ra_te = compute_row_attention(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed,
        n_heads=4, head_dim=8, k=32, softmax_temp=1.0, standardize=True,
    ).to_numpy()
    return (np.concatenate([X_tr, ra_tr], axis=1),
            np.concatenate([X_te, ra_te], axis=1))


def test_iter125_kin8nm_cb_r2_alone():
    """row_attention alone on kin8nm 8k CB R2 (vs iter121 record +7.01%)."""
    _validate_scale(_load_kin8nm, _build_iter125_alone, "cb", "R2", 0.0701, "iter125_rowattn_kin8nm_cb")


def test_iter125_year50k_cb_r2_alone():
    """row_attention alone on Year-50k CB R2 (vs iter104 +4.32%)."""
    _validate_scale(_load_year_50k, _build_iter125_alone, "cb", "R2", 0.0432, "iter125_rowattn_Year50k_cb")


# ---------- iter126 - decision_region_depth: probe-based boundary distance features ----------


def _build_iter126_alone(X_tr, X_te, y_tr, task, seed):
    """decision_region_depth alone: 8 random-direction probes per row, distance to prediction flip."""
    from sklearn.model_selection import KFold
    from mlframe.feature_engineering.transformer import compute_decision_region_depth_features

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    d_tr = compute_decision_region_depth_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    d_te = compute_decision_region_depth_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    return (np.concatenate([X_tr, d_tr], axis=1),
            np.concatenate([X_te, d_te], axis=1))


def _build_iter126_with_iter69(X_tr, X_te, y_tr, task, seed):
    """iter69 + decision_region_depth additive."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_decision_region_depth_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    d_tr = compute_decision_region_depth_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    d_te = compute_decision_region_depth_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, d_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, d_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter126_abalone_cb_r2_alone():
    """decision_region_depth alone on abalone 4k CB R2."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter126_alone, "cb", "R2", 0.0, "iter126_drd_abalone_cb")


def test_iter126_abalone_cb_r2_with_iter69():
    """iter69 + decision_region_depth on abalone 4k CB R2 (vs iter102 +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter126_with_iter69, "cb", "R2", 0.0274, "iter126_iter69+drd_abalone_cb")


def test_iter126_kin8nm_cb_r2_with_iter69():
    """iter69 + decision_region_depth on kin8nm 8k CB R2 (vs iter121 +7.01%)."""
    _validate_scale(_load_kin8nm, _build_iter126_with_iter69, "cb", "R2", 0.0701, "iter126_iter69+drd_kin8nm_cb")


# ---------- iter127 - local_intrinsic_dim (geometric manifold-dim signal, orthogonal to predictions) ----------


def _build_iter127_with_iter69(X_tr, X_te, y_tr, task, seed):
    """iter69 + local_intrinsic_dim additive."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_local_intrinsic_dim_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    li_tr = compute_local_intrinsic_dim_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    li_te = compute_local_intrinsic_dim_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, li_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, li_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter127_abalone_cb_r2_with_iter69():
    """iter69 + local_intrinsic_dim on abalone 4k CB R2 (vs iter102 record +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter127_with_iter69, "cb", "R2", 0.0274, "iter127_iter69+lid_abalone_cb")


def test_iter127_year50k_cb_r2_with_iter69():
    """iter69 + local_intrinsic_dim on Year-50k CB R2 (vs iter104 +4.32%)."""
    _validate_scale(_load_year_50k, _build_iter127_with_iter69, "cb", "R2", 0.0432, "iter127_iter69+lid_Year50k_cb")


# ---------- iter128 - local_lift (kNN target rate; memory-light) ----------


def _build_iter128_with_iter69(X_tr, X_te, y_tr, task, seed):
    """iter69 + local_lift additive."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_local_lift_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    ll_tr = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    ll_te = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, ll_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, ll_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter128_abalone_cb_r2_with_iter69():
    """iter69 + local_lift on abalone 4k CB R2 (vs iter102 record +2.74%, iter69 baseline +2.26%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter128_with_iter69, "cb", "R2", 0.0274, "iter128_iter69+loclift_abalone_cb")


def test_iter128_kin8nm_cb_r2_with_iter69():
    """iter69 + local_lift on kin8nm 8k CB R2 (vs iter121 record +7.01%)."""
    _validate_scale(_load_kin8nm, _build_iter128_with_iter69, "cb", "R2", 0.0701, "iter128_iter69+loclift_kin8nm_cb")


# ---------- iter129 - scale local_lift additive to other datasets/targets ----------


def test_iter129_year50k_cb_r2_with_iter69():
    """iter69 + local_lift on Year-50k CB R2 (vs iter104 +4.32%). Does local_lift scale to medium-N?"""
    _validate_scale(_load_year_50k, _build_iter128_with_iter69, "cb", "R2", 0.0432, "iter129_iter69+loclift_Year50k_cb")


def test_iter129_kin8nm_lgb_r2_with_iter69():
    """iter69 + local_lift on kin8nm 8k LGB R2 (vs iter68 record +11.42%). Does local_lift help LGB-target too?"""
    _validate_scale(_load_kin8nm, _build_iter128_with_iter69, "lgb", "R2", 0.1142, "iter129_iter69+loclift_kin8nm_lgb")


# ---------- iter130 - iter69 + local_lift + BGM on kin8nm CB (does the triple compose?) ----------


def _build_iter130(X_tr, X_te, y_tr, task, seed):
    """iter69 + local_lift + BGM additive: 8 + 12 + 32 + 12 = 64 features."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_local_lift_features,
        compute_bgmm_quantile_bands_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    ll_tr = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    ll_te = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    bg_tr = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bg_te = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, ll_tr, bg_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, ll_te, bg_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter130_kin8nm_cb_r2():
    """iter69 + local_lift + BGM on kin8nm 8k CB R2 (vs iter128 record +8.06%)."""
    _validate_scale(_load_kin8nm, _build_iter130, "cb", "R2", 0.0806, "iter130_iter69+loclift+bgm_kin8nm_cb")


# ---------- iter131 - quadruple combo on kin8nm + iter102+local_lift on abalone ----------


def _build_iter131_quadruple(X_tr, X_te, y_tr, task, seed):
    """iter69 (v2 with ExtraTrees) + local_lift + BGM additive (quadruple)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_v2_features,
        compute_local_lift_features,
        compute_bgmm_quantile_bands_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    ll_tr = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    ll_te = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    bg_tr = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bg_te = compute_bgmm_quantile_bands_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, ll_tr, bg_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, ll_te, bg_te, _strip(cd_te, X_te.shape[1])], axis=1))


def _build_iter131_iter102_plus_loclift(X_tr, X_te, y_tr, task, seed):
    """iter102 (iter69 + ExtraTrees) + local_lift additive (combining abalone's iter102 record with kin8nm-helper local_lift)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_v2_features,
        compute_local_lift_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    ll_tr = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    ll_te = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, ll_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, ll_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter131_kin8nm_cb_r2_quadruple():
    """iter69_v2 (with ExtraTrees) + local_lift + BGM on kin8nm CB R2 (vs iter130 record +8.31%)."""
    _validate_scale(_load_kin8nm, _build_iter131_quadruple, "cb", "R2", 0.0831, "iter131_quadruple_kin8nm_cb")


def test_iter131_abalone_cb_r2_iter102plusloclift():
    """iter102 + local_lift on abalone CB R2 (vs iter102 record +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter131_iter102_plus_loclift, "cb", "R2", 0.0274, "iter131_iter102+loclift_abalone_cb")


# ---------- iter132 - does iter130's triple combo generalise beyond kin8nm? ----------


def test_iter132_abalone_cb_r2_triple():
    """iter130 triple (iter69+loclift+BGM) on abalone CB R2 (vs iter102 record +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter130, "cb", "R2", 0.0274, "iter132_triple_abalone_cb")


def test_iter132_year50k_cb_r2_triple():
    """iter130 triple on Year-50k CB R2 (vs iter104 record +4.32%)."""
    _validate_scale(_load_year_50k, _build_iter130, "cb", "R2", 0.0432, "iter132_triple_Year50k_cb")


# ---------- iter133 - iter130 triple on Year-100k CB R2 ----------


def test_iter133_year100k_cb_r2_triple():
    """iter130 triple on Year-100k CB R2 (vs iter104 record +5.25%)."""
    _validate_scale(_load_year_100k, _build_iter130, "cb", "R2", 0.0525, "iter133_triple_Year100k_cb")


# ---------- iter134 - iter128 (iter69+local_lift, NO BGM) on Year-100k CB R2 ----------


def test_iter134_year100k_cb_r2_iter128():
    """iter128 (iter69+local_lift, no BGM) on Year-100k CB R2 (vs iter104 +5.25%; memory-lighter than iter130 triple)."""
    _validate_scale(_load_year_100k, _build_iter128_with_iter69, "cb", "R2", 0.0525, "iter134_iter128_Year100k_cb")


# ---------- iter135 - iter102 + local_lift (no BGM) on kin8nm CB R2 ----------


def _build_iter135(X_tr, X_te, y_tr, task, seed):
    """iter102 (iter69+ExtraTrees) + local_lift additive (no BGM, memory-light)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_v2_features,
        compute_local_lift_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_v2_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    ll_tr = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    ll_te = compute_local_lift_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str, k=32,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, ll_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, ll_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter135_kin8nm_cb_r2():
    """iter102 + local_lift (no BGM) on kin8nm 8k CB R2 (vs iter130 record +8.31%)."""
    _validate_scale(_load_kin8nm, _build_iter135, "cb", "R2", 0.0831, "iter135_iter102+loclift_kin8nm_cb")


def test_iter135_abalone_cb_r2():
    """iter102 + local_lift on abalone 4k CB R2 (vs iter102 record +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter135, "cb", "R2", 0.0274, "iter135_iter102+loclift_abalone_cb")


# ---------- iter136 - does iter135 (iter102+loclift) generalise to Year-50k CB R2? ----------


def test_iter136_year50k_cb_r2():
    """iter135 mechanism on Year-50k CB R2 (vs iter104 record +4.32%). Does iter102+loclift beat iter104?"""
    _validate_scale(_load_year_50k, _build_iter135, "cb", "R2", 0.0432, "iter136_iter102+loclift_Year50k_cb")


def test_iter136_california_cb_r2():
    """iter135 mechanism on California 20k CB R2 (vs iter69 +1.15%). Does iter102+loclift beat iter69 on California?"""
    _validate_scale(_load_california, _build_iter135, "cb", "R2", 0.0115, "iter136_iter102+loclift_CA20k_cb")


# ---------- iter137 - iter69 + quantile_spread_fan additive (different uncertainty signal) ----------


def _build_iter137(X_tr, X_te, y_tr, task, seed):
    """iter69 + quantile_spread_fan additive (Q90-Q10 spread of LGB quantile regressors)."""
    from sklearn.model_selection import KFold
    from tests.feature_engineering.transformer.test_validation_records import _features_cdist_seeded, _strip
    from mlframe.feature_engineering.transformer import (
        compute_baseline_disagreement_features,
        compute_quantile_spread_fan_features,
    )

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    task_str = "binary" if task == "binary" else "regression"
    bl_tr = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    bl_te = compute_baseline_disagreement_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    qf_tr = compute_quantile_spread_fan_features(
        X_train=X_tr, y_train=y_tr, X_query=None, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    qf_te = compute_quantile_spread_fan_features(
        X_train=X_tr, y_train=y_tr, X_query=X_te, splitter=splitter, seed=seed, task=task_str,
    ).to_numpy()
    cd_tr, cd_te = _features_cdist_seeded(X_tr, X_te, y_tr, task, seed)
    return (np.concatenate([X_tr, bl_tr, qf_tr, _strip(cd_tr, X_tr.shape[1])], axis=1),
            np.concatenate([X_te, bl_te, qf_te, _strip(cd_te, X_te.shape[1])], axis=1))


def test_iter137_kin8nm_cb_r2():
    """iter69 + quantile_spread_fan on kin8nm CB R2 (vs iter130 +8.31%)."""
    _validate_scale(_load_kin8nm, _build_iter137, "cb", "R2", 0.0831, "iter137_iter69+qfan_kin8nm_cb")


def test_iter137_abalone_cb_r2():
    """iter69 + quantile_spread_fan on abalone CB R2 (vs iter102 +2.74%)."""
    from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_abalone
    _validate_scale(_load_abalone, _build_iter137, "cb", "R2", 0.0274, "iter137_iter69+qfan_abalone_cb")
