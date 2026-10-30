"""Every raw column that feeds a selected engineered recipe must also be selected in raw form.

A downstream model can learn from the raw operands things the composite does not express (a private linear term, a different
interaction), possibly beating what MRMR's feature engineering found. So whenever the final selection holds a recipe, the union of
that recipe's source columns must be present among the selected raw columns, and ``transform`` must emit them.

The invariant is asserted for ``redundancy_policy="emit_both"``, the policy that exists for this purpose. The default policy
``"drop"`` deliberately prunes subsumed operands (see ``test_biz_value_mrmr_fe_canonical``), so the default is checked separately and
only for the operands that are signal-bearing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(n: int = 6000, seed: int = 0):
    """y = a**2/b + log(c)*sin(d) with a pure-noise column e: two composite groups, one noise column."""
    rng = np.random.default_rng(seed)
    a, b, c = (rng.uniform(1.0, 5.0, n) for _ in range(3))
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(size=n)
    y = a**2 / b + np.log(c) * np.sin(d) + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), pd.Series(y, name="y")


def _fit(policy: str):
    """Fit MRMR with the given redundancy policy."""
    X, y = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return X, MRMR(verbose=0, random_seed=0, redundancy_policy=policy).fit(X, y)


def _recipe_sources(model) -> set[str]:
    """Union of source columns of the engineered recipes that are in the final selection."""
    selected = set(map(str, model.get_feature_names_out()))
    out: set[str] = set()
    for r in getattr(model, "_engineered_recipes_", None) or []:
        if getattr(r, "name", None) in selected:
            out |= set(getattr(r, "src_names", ()) or ())
    return out


@pytest.fixture(scope="module")
def emit_both_fit():
    """One emit_both fit shared by the assertions."""
    return _fit("emit_both")


def test_fixture_produces_a_composite(emit_both_fit):
    """Guard: without a selected recipe the invariant below would pass vacuously."""
    _X, model = emit_both_fit
    assert _recipe_sources(model), f"no engineered recipe selected: {list(model.get_feature_names_out())}"


def test_every_recipe_operand_is_selected_raw(emit_both_fit):
    """All unique source columns of all selected recipes appear as raw selected columns."""
    _X, model = emit_both_fit
    selected = set(map(str, model.get_feature_names_out()))
    missing = _recipe_sources(model) - selected
    assert not missing, f"raw operands of selected recipes missing from the selection: {sorted(missing)}; selected={sorted(selected)}"


def test_transform_emits_the_raw_operands(emit_both_fit):
    """The raw operands are present in the transformed frame, not only in the name list."""
    X, model = emit_both_fit
    cols = set(map(str, pd.DataFrame(model.transform(X)).columns))
    missing = _recipe_sources(model) - cols
    assert not missing, f"transform() lacks raw operands {sorted(missing)}"


def test_noise_column_is_still_rejected(emit_both_fit):
    """Retaining operands must not become retaining everything: the noise column stays out."""
    _X, model = emit_both_fit
    assert "e" not in set(map(str, model.get_feature_names_out()))


def test_default_policy_operand_retention_is_reported():
    """The default ``drop`` policy: record which recipe operands are missing; the BUG1 tests require some to be absent."""
    _X, model = _fit("drop")
    missing = _recipe_sources(model) - set(map(str, model.get_feature_names_out()))
    assert missing, "default policy now keeps every operand; the BUG1 'drop' tests and this contract must be reconciled"
