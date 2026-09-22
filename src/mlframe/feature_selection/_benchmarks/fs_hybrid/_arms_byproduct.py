"""Keeping the models a feature search already paid for, instead of throwing all but one away.

Every wrapper in this roster fits a model per candidate subset, compares the scores, keeps the best subset
and discards every model it fitted getting there. On a fifty-column bed that is dozens of fitted models on
overlapping feature sets -- exactly the diverse pool an ensemble wants -- thrown away to return a list of
column names.

This arm keeps them. It walks the same nested prefixes a wrapper walks, fits the same model on each, and
then hill-climbs a with-replacement ensemble over their out-of-fold predictions. The FITS are already paid
for by any wrapper doing this search; the only new cost is storing one prediction vector per subset.

Two things this does NOT claim, both worth stating because the framing invites them:

* **It is not a feature selector, and is scored as one only by courtesy.** Its `support` is the union of
  the subsets the ensemble actually leaned on, which is a real answer to "which columns does this use" and
  a poor answer to "which columns matter". The recovery numbers for this arm should be read with that in
  mind, which is why its provenance records the ensemble weights.
* **It does not make the search cheaper.** It makes the search's by-products useful. An arm that fits
  nothing extra but stores more is still paying the search's cost, and the cost axis charges it in full.

The prefixes come from a cheap ranking rather than from an expensive wrapper's own path. That keeps this
arm's cost comparable to the filters it sits beside in the roster, and it is the honest version of the
claim: if the ensemble only pays off when the underlying ranking is already good, that is worth knowing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from ._arms import BaseArm

logger = logging.getLogger(__name__)

__all__ = ["SUBSET_SIZES", "subset_predictions", "ByProductEnsembleArm"]

#: Prefix sizes the arm fits on. Geometric rather than every size: adjacent prefixes differing by one
#: column produce near-identical models, and an ensemble over near-identical members is one member.
SUBSET_SIZES: Sequence[int] = (2, 3, 5, 8, 13, 21)


def subset_predictions(X: pd.DataFrame, y: np.ndarray, subsets: Sequence[Sequence[str]], random_state: int = 0, n_splits: int = 3) -> List[np.ndarray]:
    """Return one out-of-fold prediction vector per feature subset.

    Out-of-fold rather than in-sample, because the ensemble weights are chosen against these vectors: an
    in-sample prediction makes a model that memorised the training rows look like the best ensemble member
    available, and the weights then chase the memorisation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(np.zeros(len(labels)), labels))

    out: List[np.ndarray] = []
    for columns in subsets:
        predictions = np.zeros(len(labels), dtype=np.float64)
        matrix = X[list(columns)].to_numpy(dtype=np.float64)
        for train_index, test_index in folds:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            model.fit(np.nan_to_num(matrix[train_index], nan=0.0), labels[train_index])
            predictions[test_index] = model.predict_proba(np.nan_to_num(matrix[test_index], nan=0.0))[:, 1]
        out.append(predictions)
    return out


class ByProductEnsembleArm(BaseArm):
    """Fit the nested prefixes a wrapper would fit, then ensemble them instead of discarding them.

    Declares ``score_kind='continuous'``: every column gets the total ensemble weight of the subsets
    containing it, which is a real per-feature number with full coverage -- a column in no chosen subset
    scores zero, which is a score and not an absence.
    """

    name = "byproduct-ensemble"
    score_kind = "continuous"

    def __init__(self, k: int = 10, random_state: int = 0, n_splits: int = 3):
        self.k = int(k)
        self.random_state = int(random_state)
        self.n_splits = int(n_splits)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Rank cheaply, fit the nested prefixes, hill-climb over them, and report what the ensemble used."""
        from sklearn.feature_selection import f_classif
        from sklearn.metrics import roc_auc_score

        from mlframe.feature_selection.varying_size_top_k_subsets import varying_size_top_k_subsets
        from mlframe.votenrank.hill_climb import hill_climb_ensemble

        names: List[str] = [str(column) for column in X.columns]
        scores, _p = f_classif(np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0), np.asarray(y))
        ranked = [names[index] for index in np.argsort(-np.nan_to_num(scores, nan=0.0))]

        sizes = [size for size in SUBSET_SIZES if size <= len(names)] or [min(2, len(names))]
        subsets = varying_size_top_k_subsets(ranked, sizes)
        predictions = subset_predictions(X, y, subsets, random_state=self.random_state, n_splits=self.n_splits)

        result = hill_climb_ensemble(predictions, np.asarray(y), roc_auc_score, maximize=True, max_iterations=20, random_state=self.random_state)
        weights = np.asarray(result.get("weights", np.zeros(len(subsets))), dtype=np.float64)

        # A column's score is the total weight of the subsets that contain it. A column the ensemble never
        # reached scores zero, which is a statement rather than a gap.
        per_column = {name: 0.0 for name in names}
        for weight, columns in zip(weights, subsets):
            for column in columns:
                per_column[column] += float(weight)
        aligned = np.asarray([per_column[name] for name in names], dtype=np.float64)

        budget = max(1, min(self.k, len(names)))
        keep = np.argsort(-aligned)[:budget]
        support = np.zeros(len(names), dtype=bool)
        support[keep] = True
        return {
            "support": support,
            "score": aligned,
            "selection_score": float(result["score"]) if result.get("score") is not None else None,
            "selection_metric": "roc_auc",
            # One fit per subset per fold, which is exactly what the search would have paid anyway. Counted
            # rather than assumed: the whole argument for this arm is that the fits are already paid for,
            # and an arm that under-reported them would be assuming its own conclusion.
            "n_model_fits": len(subsets) * self.n_splits,
            "provenance": {"subset_sizes": list(sizes), "n_subsets": len(subsets), "ensemble_weights": [round(float(value), 4) for value in weights], "n_members_used": int(np.count_nonzero(weights))},
        }
