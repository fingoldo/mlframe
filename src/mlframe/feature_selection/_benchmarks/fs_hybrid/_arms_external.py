"""Arms that are not this repository's own code, so "mlframe wins" can be falsified from outside.

A benchmark whose every arm comes from the package being benchmarked cannot produce a losing result for
that package: whichever arm wins, the headline is the same. The sklearn filters already in the roster are
one outside anchor; this module adds the other kind -- a selector from a gradient-boosting library, with a
search this repository does not implement.

CatBoost's ``select_features`` runs recursive elimination scored by three different criteria, and the
criteria are the interesting part rather than the library:

* ``RecursiveByShapValues`` -- eliminate by SHAP attribution, which is the criterion most of the modern
  literature uses and the one this repository's own ``BorutaShap`` and ``ShapProxiedFS`` arms are built
  on. It is the head-to-head comparison.
* ``RecursiveByLossFunctionChange`` -- eliminate by the measured change in the loss. Expensive and closest
  to what a wrapper should ideally do: it asks the question directly rather than through a proxy.
* ``RecursiveByPredictionValuesChange`` -- eliminate by how much predictions move. Cheap, and known to be
  the weakest of the three, which makes it the internal control: an arm family where all three score the
  same is not measuring what it thinks.

Each is a separate arm, because collapsing them into one would report whichever criterion happened to be
the default as "CatBoost", and the pre-registration's whole position on RFECV is that a method's internal
knobs are not a detail when they move the result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ._arms import BaseArm

logger = logging.getLogger(__name__)

__all__ = ["CATBOOST_ALGORITHMS", "CatBoostSelectArm", "catboost_available"]

#: The three elimination criteria, keyed by the short name each arm carries.
CATBOOST_ALGORITHMS: Dict[str, str] = {
    "shap": "RecursiveByShapValues",
    "loss": "RecursiveByLossFunctionChange",
    "predictions": "RecursiveByPredictionValuesChange",
}


def catboost_available() -> bool:
    """Return whether CatBoost can be imported here.

    Checked rather than assumed: CatBoost is an optional dependency, and an arm that raises on import
    would turn every cell of the grid into a recorded failure and give the arm a reliability score of zero
    for a reason that has nothing to do with selection.
    """
    try:
        import catboost  # noqa: F401  # presence is the whole question

        return True
    except Exception as exc:
        logger.info("catboost is not importable here, so its arms are unavailable: %s", exc)
        return False


class CatBoostSelectArm(BaseArm):
    """``catboost.select_features`` driven bare, one arm per elimination criterion.

    Declares ``score_kind='selection_order'``: the selector reports the order it eliminated features in,
    which is a genuine ranking over the columns it dropped and says nothing about the order of the ones it
    kept. That is exactly what ``selection_order`` means, and declaring ``continuous`` on the back of an
    elimination order would let an average-precision computation read a distance into it that is not there.
    """

    score_kind = "selection_order"

    def __init__(self, algorithm: str = "shap", k: int = 10, iterations: int = 120, steps: int = 3, random_state: int = 0):
        if algorithm not in CATBOOST_ALGORITHMS:
            raise ValueError(f"unknown catboost selection algorithm {algorithm!r}; expected one of {sorted(CATBOOST_ALGORITHMS)}")
        self.algorithm = algorithm
        self.k = int(k)
        self.iterations = int(iterations)
        self.steps = int(steps)
        self.random_state = int(random_state)
        self.name = f"catboost-{algorithm}"

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the selection and turn its elimination order into a ranked prefix."""
        from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, Pool

        names: List[str] = [str(column) for column in X.columns]
        budget = max(1, min(self.k, len(names)))
        model = CatBoostClassifier(iterations=self.iterations, verbose=0, random_seed=self.random_state, allow_writing_files=False)
        pool = Pool(X, np.asarray(y))

        summary = model.select_features(
            pool,
            features_for_select=list(range(len(names))),
            num_features_to_select=budget,
            algorithm=getattr(EFeaturesSelectionAlgorithm, CATBOOST_ALGORITHMS[self.algorithm]),
            steps=self.steps,
            train_final_model=False,
            verbose=0,
        )

        selected_indices = [int(index) for index in summary.get("selected_features", [])]
        eliminated = [int(index) for index in summary.get("eliminated_features", [])]
        support = np.zeros(len(names), dtype=bool)
        support[selected_indices] = True

        # The ranking is the reverse of the elimination order, with the survivors ahead of everything
        # eliminated: the LAST column dropped was the hardest to give up. Survivors carry no order among
        # themselves, which is what keeps this `selection_order` rather than a ranking it cannot supply.
        prefix: Tuple[int, ...] = tuple(selected_indices) + tuple(reversed(eliminated))
        return {
            "support": support,
            "ranked_prefix": prefix,
            "n_model_fits": max(1, self.steps),
            "provenance": {"algorithm": CATBOOST_ALGORITHMS[self.algorithm], "steps": self.steps, "iterations": self.iterations, "k": budget, "n_eliminated": len(eliminated)},
        }
