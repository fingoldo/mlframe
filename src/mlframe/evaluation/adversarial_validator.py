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

from typing import Any, Optional, Sequence, Tuple

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


__all__ = ["AdversarialValidator"]
