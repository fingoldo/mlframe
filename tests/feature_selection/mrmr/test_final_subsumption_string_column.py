"""The end-of-fit subsumption pass skips a string raw column instead of failing on it and cancelling the whole pass."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._final_subsumption_stage import run_final_subsumption_prune


def test_a_selected_string_column_does_not_cancel_the_pass(caplog):
    """A selected raw column holding strings is left alone and the pass completes without the 'failed' warning."""
    X = pd.DataFrame({"cat_user": ["u1", "u2", "u1", "u3"] * 50, "x": np.arange(200, dtype=float)})
    stub = SimpleNamespace(feature_names_in_=["cat_user", "x"], _engineered_continuous_={}, _engineered_recipes_=[], random_seed=0)
    with caplog.at_level(logging.WARNING):
        changed, selected = run_final_subsumption_prune(stub, X, [0, 1], np.zeros(200, dtype=int), False)
    assert (changed, selected) == (False, [0, 1])
    assert "failed" not in caplog.text
