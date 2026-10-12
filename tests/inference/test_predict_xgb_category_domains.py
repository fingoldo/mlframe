"""Prediction must re-code categorical columns against the training domain, not against the scoring frame's own values.

With ``astype("category")`` alone, ``['a', 'c']`` codes ``c`` as 1 while ``['a', 'b', 'c']`` codes it as 2, so a model
fitted on the latter reads the scoring frame's ``c`` as its ``b``. Passing the training ``category_domains`` to
``get_models_raw_predictions`` keeps the codes aligned.
"""

import numpy as np
import pandas as pd

from mlframe.inference.predict import get_models_raw_predictions


class _CodeRecorder:
    """Stands in for a fitted XGBoost classifier: records the category codes it is handed at predict time."""

    def __init__(self):
        """Start with no recorded codes."""
        self.seen_codes = None

    def predict_proba(self, X):
        """Record the category codes of X and return a constant probability matrix."""
        self.seen_codes = X["c"].cat.codes.to_numpy().copy()
        return np.full((len(X), 2), 0.5)


def test_predict_recodes_categories_against_training_domain():
    """Scoring codes follow the training domain, so unseen-in-scoring categories keep their training code."""
    scoring = pd.DataFrame({"c": pd.Categorical(["a", "c"])})
    domains = {"c": ["a", "b", "c"]}

    model = _CodeRecorder()
    get_models_raw_predictions({"m": model}, scoring, allow_unvalidatable_models=True, category_domains=domains)

    # 'a' -> 0, 'c' -> 2 under the training domain.
    assert model.seen_codes.tolist() == [0, 2]


def test_predict_without_domain_keeps_scoring_frame_codes():
    """Without a domain the scoring frame's own value set sets the codes."""
    scoring = pd.DataFrame({"c": pd.Categorical(["a", "c"])})

    model = _CodeRecorder()
    get_models_raw_predictions({"m": model}, scoring, allow_unvalidatable_models=True)

    # Without the training domain the scoring frame's own categories are used: 'c' -> 1, the code a model
    # trained on ['a', 'b', 'c'] would read as 'b'.
    assert model.seen_codes.tolist() == [0, 1]


def test_predict_does_not_mutate_caller_frame():
    """Re-coding works on a copy; the caller's categories are unchanged."""
    scoring = pd.DataFrame({"c": pd.Categorical(["a", "c"])})
    before = scoring["c"].cat.categories.tolist()

    get_models_raw_predictions({"m": _CodeRecorder()}, scoring, allow_unvalidatable_models=True, category_domains={"c": ["a", "b", "c"]})

    assert scoring["c"].cat.categories.tolist() == before
