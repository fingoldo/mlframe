"""A candidate the target-mix second pass builds over an engineered column replays from the input columns alone.

The second pass is fitted on a frame that already holds the engineered columns of the outer fit and treats them as raw inputs. A pair recipe over the target-encoding column was admitted, but
``transform`` has only the input columns, so the column was replaced by zeros (with a warning about an unresolved engineered source) and the downstream lift of the target encoding disappeared.
"""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters._mrmr_fit_impl import _residual_rescue_stage as stage
from tests.feature_selection.fe.test_fe_mechanisms_task_axis import _ISOLATE, _fx_cat_effect, _make_target


def _fit_and_transform(caplog):
    """Fit the kfold target-encoding mechanism on the cat-effect fixture and transform a disjoint frame, returning the output and the captured log text."""
    X, logit = _fx_cat_effect(0)
    y, _ = _make_target(logit, "regression", 0)
    model = MRMR(fe_kfold_te_enable=True, fe_kfold_te_cols=("cat_region",), **_ISOLATE).fit(X, pd.Series(y, name="y"))
    with caplog.at_level(logging.WARNING):
        out = model.transform(_fx_cat_effect(100)[0])
    return out, caplog.text


def test_no_output_column_is_replaced_by_zeros_at_transform(caplog):
    """No engineered column is unresolved at transform: no warning is logged and no output column is constant zero."""
    out, log_text = _fit_and_transform(caplog)
    assert "unresolved engineered" not in log_text
    values = np.asarray(out, dtype=np.float64)
    assert all(np.ptp(values[:, j]) > 0 for j in range(values.shape[1])), f"constant output columns: {[c for c, v in zip(out.columns, values.T) if np.ptp(v) == 0]}"


def test_a_candidate_over_an_engineered_column_without_a_parent_is_dropped():
    """A non-pair recipe that reads an engineered column cannot carry a parent and is dropped; a pair recipe whose engineered operand has no recipe is dropped too."""
    from mlframe.feature_selection.filters.engineered_recipes import build_unary_binary_recipe

    X = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)[::-1], "a__te": np.arange(50.0) * 2.0})
    recipe = build_unary_binary_recipe(
        name="add(a,a__te)", src_a_name="a", src_b_name="a__te", unary_a_name="identity", unary_b_name="identity", binary_name="add", unary_preset="minimal", binary_preset="minimal",
        quantization_nbins=10, quantization_method="quantile", quantization_dtype="int16", fit_values_for_edges=(X["a"] + X["a__te"]).to_numpy(),
    )
    assert stage._replayable_from_input(recipe, X, frozenset({"a", "b"}), {}) is None
    assert stage._replayable_from_input(recipe, X, frozenset({"a", "b", "a__te"}), {}) is recipe
