"""``AdversarialValidator``: a coherent facade over mlframe's existing adversarial-validation primitives.

Source: best-practice writeup -- build a classifier to distinguish train vs test rows; AUC near 1.0 signals
distribution shift; "select the training rows most similar to the test set... to use as your validation
fold."

Both underlying mechanisms already existed in mlframe as separate functions (search-for-reuse confirmed no
gap in the actual DIAGNOSTIC or SELECTION logic itself):
- :func:`mlframe.reporting.charts.drift.adversarial_auc` -- fits a train/test classifier, returns the AUC
  AND per-feature gain-importance (categoricals handled internally).
- :func:`mlframe.evaluation.adversarial_fold_selection.build_test_like_validation_fold` -- fits its own OOF
  train/test classifier, selects the train rows with the highest predicted is-test probability as a
  validation fold (expects already-numeric input, same contract as this class's ``select_validation_fold``).

What was genuinely missing, per the idea's own API sketch, was ONE coherent object tying both together (AUC
+ per-feature importance report + fold selection) rather than three separate function calls across two
modules -- this class is exactly that thin composition, not a reimplementation of either mechanism.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evaluation.adversarial_fold_selection import build_test_like_validation_fold
from ..reporting.charts.drift import adversarial_auc


class AdversarialValidator:
    """Facade: adversarial train/test AUC + per-feature importance report + test-like validation-fold selection.

    Parameters
    ----------
    max_rows_per_side, n_splits, seed, lgbm_params
        Forwarded to :func:`adversarial_auc` for the AUC/importance diagnostic (``fit``).

    Attributes
    ----------
    auc_
        Train-vs-test classifier AUC; near 0.5 means train/test are statistically indistinguishable (no
        detected shift), near 1.0 means a classifier can trivially separate them (strong shift).
    fpr_, tpr_
        ROC curve arrays from the diagnostic fit.
    importances_, feature_names_
        Per-feature gain importance from the separating classifier, aligned arrays.
    """

    def __init__(
        self,
        max_rows_per_side: Optional[int] = None,
        n_splits: int = 3,
        seed: int = 0,
        lgbm_params: Optional[dict] = None,
    ) -> None:
        self.max_rows_per_side = max_rows_per_side
        self.n_splits = n_splits
        self.seed = seed
        self.lgbm_params = lgbm_params

    def fit(self, X_train: Any, X_test: Any, feature_names: Optional[Sequence[str]] = None) -> "AdversarialValidator":
        """Diagnose train/test separability: fits the adversarial classifier and stores AUC + importances."""
        kwargs: dict = {"feature_names": feature_names, "n_splits": self.n_splits, "seed": self.seed}
        if self.max_rows_per_side is not None:
            kwargs["max_rows_per_side"] = self.max_rows_per_side
        if self.lgbm_params is not None:
            kwargs["lgbm_params"] = self.lgbm_params
        auc, fpr, tpr, importances, names = adversarial_auc(X_train, X_test, **kwargs)
        self.auc_: float = float(auc)
        self.fpr_: np.ndarray = fpr
        self.tpr_: np.ndarray = tpr
        self.importances_: np.ndarray = importances
        self.feature_names_: Tuple[str, ...] = names
        self._X_train = X_train
        self._X_test = X_test
        return self

    def report(self) -> pd.DataFrame:
        """Per-feature importance report from the separating classifier, sorted by importance descending."""
        if not hasattr(self, "auc_"):
            raise RuntimeError("AdversarialValidator.report() called before fit().")
        return pd.DataFrame({"feature": list(self.feature_names_), "importance": self.importances_}).sort_values("importance", ascending=False).reset_index(drop=True)

    def select_validation_fold(
        self,
        X_train: Optional[Any] = None,
        X_test: Optional[Any] = None,
        feature_names: Optional[Sequence[str]] = None,
        val_fraction: float = 0.2,
        n_splits: int = 5,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select the train rows most similar to the test distribution as a validation fold.

        ``X_train``/``X_test`` default to the frames passed to ``fit()``. Must be already-numeric (same
        contract as :func:`build_test_like_validation_fold` -- encode categoricals before calling this,
        unlike ``fit()``/``report()`` which handle them internally via ``adversarial_auc``).

        Returns
        -------
        tuple
            ``(val_idx, train_remainder_idx)`` -- see :func:`build_test_like_validation_fold`.
        """
        X_train = X_train if X_train is not None else getattr(self, "_X_train", None)
        X_test = X_test if X_test is not None else getattr(self, "_X_test", None)
        if X_train is None or X_test is None:
            raise ValueError("select_validation_fold: pass X_train/X_test, or call fit() first.")
        return build_test_like_validation_fold(
            X_train, X_test, feature_names=feature_names, val_fraction=val_fraction, n_splits=n_splits, seed=seed if seed is not None else self.seed,
        )

    def prune_drift_features(
        self,
        X_train: Optional[Any] = None,
        X_test: Optional[Any] = None,
        feature_names: Optional[Sequence[str]] = None,
        target_auc: float = 0.55,
        max_iterations: int = 10,
        features_per_iteration: int = 1,
    ) -> "AdversarialValidator":
        """Opt-in: iteratively drop the top drift-driving feature(s) and refit until AUC falls below ``target_auc``.

        Repeatedly fits the adversarial discriminator on the remaining feature set, drops the
        ``features_per_iteration`` highest-importance features, and refits -- isolating the minimal subset of
        features actually responsible for train/test separability. Stops when the discriminator AUC drops
        below ``target_auc``, ``max_iterations`` is hit, or no features remain to drop.

        ``X_train``/``X_test`` default to the frames passed to ``fit()``, same as ``select_validation_fold``.
        Does not mutate anything set by ``fit()``/``report()`` -- results live in their own
        ``pruned_features_``/``remaining_features_``/``pruning_history_`` attributes.

        Returns
        -------
        AdversarialValidator
            ``self``, for chaining. Sets ``pruned_features_`` (features removed, in drop order),
            ``remaining_features_`` (survivors), and ``pruning_history_`` (per-iteration ``(auc, dropped)`` list).
        """
        X_train = X_train if X_train is not None else getattr(self, "_X_train", None)
        X_test = X_test if X_test is not None else getattr(self, "_X_test", None)
        if X_train is None or X_test is None:
            raise ValueError("prune_drift_features: pass X_train/X_test, or call fit() first.")
        if feature_names is not None:
            remaining: List[str] = list(feature_names)
        elif hasattr(X_train, "columns"):
            remaining = list(X_train.columns)
        else:
            remaining = list(self.feature_names_) if hasattr(self, "feature_names_") else [f"f{i}" for i in range(np.asarray(X_train).shape[1])]

        kwargs: dict = {"n_splits": self.n_splits, "seed": self.seed}
        if self.max_rows_per_side is not None:
            kwargs["max_rows_per_side"] = self.max_rows_per_side
        if self.lgbm_params is not None:
            kwargs["lgbm_params"] = self.lgbm_params

        pruned: List[str] = []
        history: List[dict] = []
        auc = float("inf")
        for _ in range(max_iterations):
            if len(remaining) <= 1:
                break
            auc, _fpr, _tpr, importances, names = adversarial_auc(X_train, X_test, feature_names=remaining, **kwargs)
            auc = float(auc)
            if auc < target_auc:
                history.append({"auc": auc, "dropped": []})
                break
            order = np.argsort(importances)[::-1]
            n_drop = min(features_per_iteration, len(remaining) - 1)
            drop_now = [names[i] for i in order[:n_drop]]
            history.append({"auc": auc, "dropped": list(drop_now)})
            pruned.extend(drop_now)
            remaining = [n for n in remaining if n not in drop_now]
        else:
            # max_iterations exhausted without crossing target_auc: record the last-seen AUC for the survivors.
            pass

        self.pruned_features_: Tuple[str, ...] = tuple(pruned)
        self.remaining_features_: Tuple[str, ...] = tuple(remaining)
        self.pruning_history_: Tuple[dict, ...] = tuple(history)
        self.pruning_final_auc_: float = auc
        return self


__all__ = ["AdversarialValidator"]
