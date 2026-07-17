"""get_feature_names_out() width must equal transform() width, including with legacy non-replayable
recipes (audit P1-A, 2026-06-13).

transform's _append_engineered drops pre-D3 pickled k-way recipes lacking the chained-lookup payload
(``requires_refit_for_replay`` + no ``chain_lookups``). get_feature_names_out must apply the SAME
filter, else it advertises MORE columns than transform emits -> a width mismatch that breaks sklearn
Pipeline / ColumnTransformer / set_output on a legacy pickle.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _fit_small_mrmr(seed=0, n=1500):
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + np.log(c * 2) * np.sin(d / 3) + f / 5
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=seed).fit(X=df, y=pd.Series(y, name="y"))
    return fs, df


def test_names_out_width_equals_transform_width_baseline():
    fs, df = _fit_small_mrmr()
    assert len(fs.get_feature_names_out()) == fs.transform(df).shape[1]


def test_legacy_non_replayable_recipe_excluded_from_both_names_and_transform():
    """Inject a legacy recipe (requires_refit_for_replay, no chain_lookups) -- both
    get_feature_names_out and transform must DROP it, so widths stay equal. Pre-fix,
    get_feature_names_out counted it (width+1) while transform dropped it -> Pipeline-breaking mismatch."""
    from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

    fs, df = _fit_small_mrmr()
    base_w = len(fs.get_feature_names_out())
    # a legacy k-way recipe with NO chain payload -> transform's _append_engineered filters it out.
    legacy = EngineeredRecipe(
        name="__legacy_kway_no_chain__",
        kind="orth_triplet_cross",
        src_names=("a", "b", "c"),
        extra={"requires_refit_for_replay": True},
    )
    fs._engineered_recipes_ = [*list(getattr(fs, "_engineered_recipes_", [])), legacy]
    names_w = len(fs.get_feature_names_out())
    trans_w = fs.transform(df).shape[1]
    assert names_w == trans_w, (
        f"get_feature_names_out width {names_w} != transform width {trans_w} with a legacy "
        f"non-replayable recipe -- the legacy filter is not mirrored in get_feature_names_out"
    )
    # and the legacy recipe must NOT appear in the advertised names.
    assert "__legacy_kway_no_chain__" not in list(fs.get_feature_names_out())
    # baseline width unchanged (the legacy recipe added nothing on either side).
    assert names_w == base_w
