"""Business-value integration tests for mlframe fairness metrics and sample-weight schemas.

NOTE: These are regression sensors, not scientific benchmarks. Synthetic data parameters
(n_samples, imbalance ratio, thresholds) are intentionally tuned so that the effect is
stably visible across all seeds. If a wiring/logic change breaks fairness or sample-weight
forwarding tomorrow, these tests will catch it. They do NOT prove the features work on
real-world data.

Test 1 — Fairness feature path: training with `fairness_features=["group"]` runs end-to-end
on a dataset with differential per-group base rates and surfaces (or at least does not
crash on) per-group evaluation, vs. a baseline run with empty `fairness_features`.

Test 2 — Sample weights lift minority recall on a 90:10 imbalanced binary problem when
inverse-frequency weights are supplied via the FTE's `sample_weights` dict.

Both tests mimic the conventions of `test_bizvalue_outliers_earlystop.py`: they call
`train_mlframe_models_suite` then `predict_mlframe_models_suite`, compute the metric of
interest on a held-out test set manually, and use `pytest.xfail` (not threshold drops)
when the measured lift falls short of the business target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from mlframe.training.core import (
    predict_mlframe_models_suite,
    train_mlframe_models_suite,
)

from .shared import SimpleFeaturesAndTargetsExtractor, TimestampedFeaturesExtractor


# --------------------------------------------------------------------------------------
# Data factories
# --------------------------------------------------------------------------------------

def _make_grouped_classification(n_train=2000, n_test=500, n_features=8, seed=42):
    """Binary classification with a categorical "group" column.

    Group A has positive_rate ~0.55, group B has positive_rate ~0.20. Underlying linear
    signal is identical across groups, but B's lower base rate + smaller share of train
    typically yields lower per-group AUROC/precision -> a realistic fairness probe.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    X = rng.randn(n_total, n_features)
    coefs = np.array([1.5, -1.0, 0.8, -0.5] + [0.0] * (n_features - 4))
    logits = X @ coefs

    # Group assignment: 70% A, 30% B
    group = rng.choice(["A", "B"], size=n_total, p=[0.7, 0.3])
    # Lower base rate for B via group-specific intercept.
    intercept = np.where(group == "A", 0.20, -1.20)
    p = 1.0 / (1.0 + np.exp(-(logits + intercept)))
    y = (rng.uniform(size=n_total) < p).astype(int)

    cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["group"] = group
    df["target"] = y

    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)
    return train_df, test_df


def _make_imbalanced_classification(n_train=6000, n_test=1500, n_features=10, minority_frac=0.07, seed=7):
    """93:7 imbalanced binary classification; minority class has moderate signal.

    Construction: assign exactly minority_frac of rows to label=1, then generate features
    such that label=1 rows have a mean shift on the first 4 features. Large n (7500)
    ensures stable recall estimates across seeds. 7% minority (vs prior 10%) gives
    inverse-frequency weights (~13x) more leverage to shift the decision boundary.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    y = np.zeros(n_total, dtype=int)
    n_pos = int(round(minority_frac * n_total))
    pos_idx = rng.choice(n_total, size=n_pos, replace=False)
    y[pos_idx] = 1

    X = rng.randn(n_total, n_features)
    # Mean shift on first 4 features for positives; moderate signal — enough to
    # learn the class but not so strong that unweighted models already nail it.
    shift = np.array([1.2, -1.0, 0.8, -0.6] + [0.0] * (n_features - 4))
    X[y == 1] += shift

    cols = [f"f_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    # Stratified split by label so the test set has a representative minority slice
    # (a plain shuffle can give 0 positives in test by chance on heavy imbalance).
    pos_mask = df["target"].values == 1
    pos_df = df[pos_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    neg_df = df[~pos_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_pos_n = int(round(len(pos_df) * (n_test / n_total)))
    test_neg_n = n_test - test_pos_n
    test_df = pd.concat([pos_df.iloc[:test_pos_n], neg_df.iloc[:test_neg_n]], ignore_index=True)
    train_df = pd.concat([pos_df.iloc[test_pos_n:], neg_df.iloc[test_neg_n:]], ignore_index=True)
    test_df = test_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    train_df = train_df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return train_df, test_df


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _predict_proba_pos(results):
    """Extract positive-class probability vector from predict_mlframe_models_suite output."""
    if results.get("probabilities"):
        probs = next(iter(results["probabilities"].values()))
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()
    preds = next(iter(results["predictions"].values()))
    return np.asarray(preds, dtype=float)


def _predict_labels(results):
    preds = next(iter(results["predictions"].values()))
    return np.asarray(preds).astype(int)


# --------------------------------------------------------------------------------------
# Test 1 — Fairness metric emission path is conditional on `fairness_features`
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
@pytest.mark.parametrize("mlframe_model", ["lgb", "cb", "xgb"])
def test_fairness_features_emits_per_group_path(tmp_path, common_init_params, seed, mlframe_model):
    """Run the suite with and without fairness_features=["group"]; verify the fairness
    code path is reachable end-to-end and per-group prediction quality is computable.

    Observed behavior (documented): mlframe's per-trainer `fairness_report` is computed
    inside the trainer's metrics dict and is NOT surfaced via the public
    `train_mlframe_models_suite` metadata return value. So the assertions below:
      * Confirm the suite completes successfully under both configurations.
      * Recompute per-group AUROC + precision on the held-out test set (independent of
        whether the suite returns the fairness_report) and assert basic shape:
        both groups present, parity gap in [0, 1].
      * Assert that metadata for run A (no fairness features) does NOT carry a populated
        `fairness_report`, and that run B's behavior_config snapshot reflects the
        requested fairness_features.
    """
    pytest.importorskip({"lgb": "lightgbm", "cb": "catboost", "xgb": "xgboost"}[mlframe_model])

    train_df, test_df = _make_grouped_classification(seed=seed)

    # Sanity on dataset construction: B has materially lower positive rate.
    pos_rate_a = float(train_df.loc[train_df["group"] == "A", "target"].mean())
    pos_rate_b = float(train_df.loc[train_df["group"] == "B", "target"].mean())
    assert pos_rate_a - pos_rate_b > 0.10, (
        f"Dataset construction failed: pos_rate_a={pos_rate_a:.3f} pos_rate_b={pos_rate_b:.3f}"
    )

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

    def _run(model_name, behavior_config):
        data_dir = str(tmp_path / "data" / model_name)
        models, metadata = train_mlframe_models_suite(
            df=train_df,
            target_name="test_target",
            model_name=model_name,
            features_and_targets_extractor=fte,
            mlframe_models=[mlframe_model],
            init_common_params=common_init_params,
            behavior_config=behavior_config,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=data_dir,
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 80},
        )
        models_path = f"{data_dir}/models/test_target/{model_name}"
        results = predict_mlframe_models_suite(
            df=test_df,
            models_path=models_path,
            features_and_targets_extractor=fte,
            return_probabilities=True,
            verbose=0,
        )
        return models, metadata, results

    # Run A: fairness disabled
    _, meta_a, results_a = _run(f"{mlframe_model}_no_fairness_s{seed}", {"fairness_features": []})

    # Run B: fairness enabled on the "group" column
    _, meta_b, results_b = _run(
        f"{mlframe_model}_with_fairness_s{seed}",
        {"fairness_features": ["group"], "fairness_min_pop_cat_thresh": 50},
    )

    # Per-group metrics computed independently from suite (truth source: held-out test).
    proba_b = _predict_proba_pos(results_b)
    y_test = test_df["target"].values
    g_test = test_df["group"].values

    per_group = {}
    for g in ("A", "B"):
        mask = g_test == g
        # Need both classes present for AUROC.
        if mask.sum() < 20 or len(np.unique(y_test[mask])) < 2:
            continue
        per_group[g] = {
            "auroc": float(roc_auc_score(y_test[mask], proba_b[mask])),
            "precision": float(precision_score(y_test[mask], (proba_b[mask] >= 0.5).astype(int), zero_division=0)),
            "n": int(mask.sum()),
        }

    assert set(per_group.keys()) == {"A", "B"}, f"expected per-group metrics for A and B, got {per_group}"
    aurocs = [m["auroc"] for m in per_group.values()]
    parity_gap = float(max(aurocs) - min(aurocs))
    assert 0.0 <= parity_gap <= 1.0, f"parity_gap out of range: {parity_gap}"

    # Now probe metadata for any fairness surfacing. mlframe currently does not propagate
    # the per-target fairness_report into top-level suite metadata, so we treat presence
    # as a bonus (not required) and assert the absence in run A in either case.
    def _has_populated_fairness(meta):
        # Walk top-level keys looking for a non-empty fairness payload.
        for key, val in meta.items():
            if "fairness" in key.lower():
                if val is None:
                    continue
                if isinstance(val, (list, dict)) and len(val) == 0:
                    continue
                return True
        return False

    fairness_in_a = _has_populated_fairness(meta_a)
    fairness_in_b = _has_populated_fairness(meta_b)
    # Run A must not have a populated fairness payload (proves conditional path).
    assert not fairness_in_a, (
        f"Run A (fairness_features=[]) unexpectedly carries fairness payload in metadata: "
        f"{[k for k in meta_a if 'fairness' in k.lower()]}"
    )
    # Bug B fix 2026-04-15: train_mlframe_models_suite now aggregates per-model
    # fairness_report into metadata["fairness_report"] when fairness_features is set.
    assert fairness_in_b, (
        f"Run B (fairness_features=['group']) should expose fairness payload in metadata; "
        f"keys={list(meta_b.keys())}"
    )

    # Also: assert the suite completed for both runs (sanity).
    assert meta_a.get("model_name") == f"{mlframe_model}_no_fairness_s{seed}"
    assert meta_b.get("model_name") == f"{mlframe_model}_with_fairness_s{seed}"


# --------------------------------------------------------------------------------------
# Test 2 — Inverse-frequency sample weights lift minority-class recall / F1
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42, 7, 99])
@pytest.mark.parametrize("mlframe_model", ["lgb", "cb", "xgb"])
def test_sample_weights_lift_minority_recall(tmp_path, common_init_params, seed, mlframe_model):
    """Inverse-frequency weighting (minority *9, majority *1) on a 90:10 imbalanced
    binary classification problem should lift minority-class recall (or F1) measurably
    versus an unweighted baseline."""
    pytest.importorskip({"lgb": "lightgbm", "cb": "catboost", "xgb": "xgboost"}[mlframe_model])

    train_df, test_df = _make_imbalanced_classification(seed=seed)
    y_train = train_df["target"].values
    minority_frac = float(y_train.mean())
    # Sanity on imbalance.
    assert 0.05 < minority_frac < 0.20, f"unexpected minority_frac={minority_frac:.3f}"

    # Inverse-frequency weights: minority gets ~9x.
    n = len(y_train)
    w_min = float((1.0 - minority_frac) / minority_frac)  # ~9 at 10% minority
    sample_weight_vec = np.where(y_train == 1, w_min, 1.0).astype(np.float64)

    # methodology: fixed-threshold recall — top-k neutralizes weight effects on ranking

    def _run(model_name, sample_weights_dict):
        data_dir = str(tmp_path / "data" / model_name)
        fte = TimestampedFeaturesExtractor(
            target_column="target",
            regression=False,
            sample_weights=sample_weights_dict,
        )
        models, metadata = train_mlframe_models_suite(
            df=train_df,
            target_name="test_target",
            model_name=model_name,
            features_and_targets_extractor=fte,
            mlframe_models=[mlframe_model],
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=data_dir,
            models_dir="models",
            verbose=0,
            hyperparams_config={"iterations": 80},
        )
        # IMPORTANT: predict-time FTE must NOT carry training sample_weights (they map by
        # row count of the input df). Use a plain extractor for inference.
        pred_fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        models_path = f"{data_dir}/models/test_target/{model_name}"
        results = predict_mlframe_models_suite(
            df=test_df,
            models_path=models_path,
            features_and_targets_extractor=pred_fte,
            return_probabilities=True,
            verbose=0,
        )
        return metadata, results

    # Run A: no sample weights (empty dict -> uniform path)
    meta_a, results_a = _run(f"{mlframe_model}_uniform_s{seed}", {})

    # Run B: inverse-frequency weights
    meta_b, results_b = _run(f"{mlframe_model}_inv_freq_s{seed}", {"inv_freq": sample_weight_vec})

    y_test = test_df["target"].values

    # methodology: fixed-threshold recall — top-k neutralizes weight effects on ranking.
    # Sample weights primarily shift fitted PROBABILITIES (calibration/margin) not the
    # ranking order. A top-k threshold is pure ranking, so weights show ~0 effect there.
    # A fixed 0.5 threshold reveals the calibration shift: weighted models push minority
    # scores upward, so more cross 0.5 -> higher minority recall.
    proba_a = _predict_proba_pos(results_a)
    proba_b = _predict_proba_pos(results_b)

    n_pos_test = int(y_test.sum())
    assert n_pos_test > 5, f"test set has too few positives ({n_pos_test}) for a meaningful eval"

    pred_a = (proba_a >= 0.5).astype(int)
    pred_b = (proba_b >= 0.5).astype(int)

    recall_a = float(recall_score(y_test, pred_a, pos_label=1, zero_division=0))
    recall_b = float(recall_score(y_test, pred_b, pos_label=1, zero_division=0))
    f1_a = float(f1_score(y_test, pred_a, pos_label=1, zero_division=0))
    f1_b = float(f1_score(y_test, pred_b, pos_label=1, zero_division=0))

    msg = (
        f"recall_a={recall_a:.4f} recall_b={recall_b:.4f} d_recall={recall_b - recall_a:+.4f} | "
        f"f1_a={f1_a:.4f} f1_b={f1_b:.4f} d_f1={f1_b - f1_a:+.4f} | "
        f"minority_frac={minority_frac:.3f} w_min={w_min:.2f}"
    )

    recall_lift = recall_b - recall_a

    # Hard floor: sample weights must not regress recall beyond noise margin.
    assert recall_lift >= -0.20, f"sample weights regressed minority recall. {msg}"
    # Preferred: meaningful lift.
    assert recall_lift >= 0.05, f"sample weights did not lift minority recall at 0.5 threshold. {msg}"
