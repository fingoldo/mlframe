"""Adversarial-validation feature audit: empirically test the blanket-ban heuristic instead of trusting it.

A common but poorly-validated practice is to drop any feature that contributes strongly to train/test
separability under adversarial validation (:func:`mlframe.reporting.charts.drift.adversarial_auc`'s gain
importances), on the assumption that a distinguishable feature must hurt generalization. A 5th-place
IEEE-CIS-fraud writeup ran a controlled 3-way pseudo train/public/private split and found this assumption
false: "regardless of the value of AUC of Adversarial Validation, if OOF is high, public and private AUC tend
to be high" -- i.e. a feature's adversarial-AUC contribution is a poor predictor of whether it actually hurts
holdout generalization. This module operationalizes that check: rather than a binary ban, it ablates each
adversarial-flagged feature inside a pseudo train/public/private split (built from labeled train data only,
mimicking the leaderboard split structure) and reports whether dropping the feature actually helps.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def adversarial_validation_feature_audit(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    feature_names: Optional[Sequence[str]] = None,
    top_k_features: int = 10,
    pseudo_public_frac: float = 0.2,
    pseudo_private_frac: float = 0.2,
    seed: int = 0,
    lgbm_params: Optional[dict] = None,
    stability_folds: Optional[int] = None,
) -> dict:
    """Empirically test whether the top adversarial-AUC-contributing features actually hurt generalization.

    Parameters
    ----------
    X_train, y_train, X_test
        Labeled train features/target and unlabeled (or held-out) test features, same columns.
    feature_names
        Column names; inferred from ``X_train`` if it is a DataFrame.
    top_k_features
        How many of the highest adversarial-gain-importance features to ablate-test (bounded by the number
        of features actually present).
    pseudo_public_frac, pseudo_private_frac
        Fraction of ``X_train``/``y_train`` (after removing the pseudo-train portion) held out as a 3-way
        pseudo train/public/private split, simulating the competition leaderboard structure without needing
        real test labels.
    seed
        Controls both the adversarial classifier's CV and the pseudo-split (fold 0 when ``stability_folds``
        is set, so the returned ``audited_features`` are unchanged whether or not stability mode is used).
    lgbm_params
        Passed through to the LightGBM models fit inside the pseudo-split ablation.
    stability_folds
        Opt-in. When set to an int >= 2, repeats the pseudo-split (with ``stability_folds`` independent
        reshuffles seeded ``seed, seed+1, ..., seed+stability_folds-1``, each also re-fitting its own
        ablation models) and adds a ``"stability"`` block to every audited feature reporting how much its
        keep/drop call moves across reshuffles. A single random split can flip a borderline feature's
        recommendation by chance; this mode distinguishes robust calls (low delta variance, unanimous
        keep/drop vote) from noisy ones a caller should not trust from one split alone. Left ``None`` (the
        default), behavior and output are bit-identical to the pre-stability-mode implementation.

    Returns
    -------
    dict
        ``adversarial_auc`` (train-vs-test separability), ``audited_features`` (list of dicts, one per
        ablated feature: ``name``, ``adversarial_importance``, ``private_auc_delta_when_dropped`` (positive =
        dropping the feature IMPROVED pseudo-private AUC, i.e. the ban heuristic was right for this feature;
        negative = dropping HURT pseudo-private AUC, i.e. keep it despite the adversarial flag), and
        ``recommendation`` (``"drop"`` / ``"keep"``), plus a ``"stability"`` sub-dict when ``stability_folds``
        is set: ``delta_values`` (per-fold ``private_auc_delta_when_dropped``), ``delta_std``, ``keep_frac``
        (fraction of folds recommending "keep"), and ``stable`` (``True`` iff ``keep_frac`` is unanimous, i.e.
        0.0 or 1.0 -- the recommendation did not flip across any reshuffle)), ``importance_vs_generalization_correlation``
        (Pearson correlation between adversarial importance rank and ``private_auc_delta_when_dropped`` across
        the audited features -- low/near-zero replicates the source finding that adversarial contribution is a
        poor predictor of actual generalization harm).
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    from mlframe.metrics.core import fast_roc_auc
    from mlframe.reporting.charts.drift import adversarial_auc

    auc, _fpr, _tpr, importances, names = adversarial_auc(X_train, X_test, feature_names=feature_names, seed=seed, lgbm_params=lgbm_params)
    names_arr = np.asarray(names)
    order = np.argsort(importances)[::-1]
    k = min(top_k_features, len(order))
    top_indices = order[:k]

    if hasattr(X_train, "iloc"):
        X_train_df = X_train
    else:
        import pandas as pd

        X_train_df = pd.DataFrame(np.asarray(X_train), columns=list(names_arr))

    y_train = np.asarray(y_train)
    all_feature_cols = list(names_arr)
    rest_frac = pseudo_public_frac + pseudo_private_frac

    def _run_split(split_seed: int) -> list[dict]:
        """Run one pseudo-public/private train/eval split and score the AUC drop from removing each feature."""
        idx_train, idx_rest = train_test_split(np.arange(len(y_train)), test_size=rest_frac, random_state=split_seed, stratify=y_train)
        _idx_public, idx_private = train_test_split(idx_rest, test_size=pseudo_private_frac / rest_frac, random_state=split_seed, stratify=y_train[idx_rest])

        def _fit_auc(feature_cols: Sequence[str]) -> float:
            """Fit an LGBM classifier on ``feature_cols`` and return its pseudo-private AUC."""
            model = lgb.LGBMClassifier(**(lgbm_params or {"n_estimators": 100, "verbosity": -1}), random_state=split_seed)
            model.fit(X_train_df.iloc[idx_train][list(feature_cols)], y_train[idx_train])
            proba = np.asarray(model.predict_proba(X_train_df.iloc[idx_private][list(feature_cols)]))[:, 1]
            return float(fast_roc_auc(y_train[idx_private], proba))

        baseline_private_auc = _fit_auc(all_feature_cols)

        split_audited = []
        for idx in top_indices:
            feature_name = str(names_arr[idx])
            dropped_cols = [c for c in all_feature_cols if c != feature_name]
            dropped_private_auc = _fit_auc(dropped_cols)
            delta = dropped_private_auc - baseline_private_auc
            split_audited.append(
                {
                    "name": feature_name,
                    "adversarial_importance": float(importances[idx]),
                    "private_auc_delta_when_dropped": delta,
                    "recommendation": "drop" if delta > 0 else "keep",
                }
            )
        return split_audited

    audited = _run_split(seed)

    if stability_folds is not None:
        if stability_folds < 2:
            raise ValueError(f"stability_folds must be >= 2 to measure variance, got {stability_folds}")
        fold_results = [audited] + [_run_split(seed + fold_offset) for fold_offset in range(1, stability_folds)]
        for feature_pos, feature_entry in enumerate(audited):
            delta_values = [float(fold[feature_pos]["private_auc_delta_when_dropped"]) for fold in fold_results]
            keep_votes = sum(1 for fold in fold_results if fold[feature_pos]["recommendation"] == "keep")
            keep_frac = keep_votes / stability_folds
            feature_entry["stability"] = {
                "delta_values": delta_values,
                "delta_std": float(np.std(delta_values)),
                "keep_frac": keep_frac,
                "stable": keep_frac in (0.0, 1.0),
            }

    if len(audited) >= 2:
        imp_vals = np.array([a["adversarial_importance"] for a in audited])
        delta_vals = np.array([a["private_auc_delta_when_dropped"] for a in audited])
        if np.std(imp_vals) > 0 and np.std(delta_vals) > 0:
            correlation = float(np.corrcoef(imp_vals, delta_vals)[0, 1])
        else:
            correlation = 0.0
    else:
        correlation = float("nan")

    result = {
        "adversarial_auc": auc,
        "audited_features": audited,
        "importance_vs_generalization_correlation": correlation,
    }
    if stability_folds is not None:
        result["stability_folds"] = stability_folds
    return result


__all__ = ["adversarial_validation_feature_audit"]
