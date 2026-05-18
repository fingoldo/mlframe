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

from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_california
from tests.feature_engineering.transformer.test_validation_records import (
    _build_iter68,
    _build_iter69,
    _measure_lift_seeded,
)


pytestmark = pytest.mark.fast


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

    Cached after first download (OpenML cache). Subsample with fixed seed for reproducibility.
    """
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
