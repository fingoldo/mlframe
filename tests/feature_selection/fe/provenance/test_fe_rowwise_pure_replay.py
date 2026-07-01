"""FE candidates must be ROW-WISE PURE so recipe replay is slice-consistent (audit, 2026-06-13).

grad1/grad2 (np.gradient -- value depends on neighbouring ROWS) and logn (x - np.min(x) -- a
whole-column statistic recomputed at apply time) are NOT row-wise pure: a recipe built on them emits
DIFFERENT values on a row-slice / test frame (slice-replay corruption). They live only in the
non-default "maximal" preset; the fix excludes them from FE pair candidates so they are never
selected. This pins (a) no selected engineered feature uses them, and (b) every engineered column
replays byte-consistently full-vs-slice on a maximal-preset fit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_maximal_preset_excludes_non_rowwise_pure_and_replays_slice_consistent():
    from mlframe.feature_selection.filters import MRMR
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    n = 3000
    rng = np.random.default_rng(0)
    a, b, c, d, e = (rng.random(n) for _ in range(5))
    y = 0.3 * (a - 0.5) * (b - 0.5) + 0.4 * np.log(c * 2.0) + e / 5.0
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=0, fe_unary_preset="maximal",
                  fe_binary_preset="maximal").fit(X=df, y=pd.Series(y, name="y"))

    names = list(fs.get_feature_names_out())
    bad = [nm for nm in names if any(tok in nm for tok in ("grad1", "grad2", "logn"))]
    assert not bad, f"non-row-wise-pure ops leaked into the selection: {bad}"

    # every engineered recipe must replay byte-consistently full-vs-slice (the invariant grad/logn broke).
    for recipe in getattr(fs, "_engineered_recipes_", []):
        full = np.nan_to_num(np.asarray(apply_recipe(recipe, df), dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
        sl = df.iloc[500:900].reset_index(drop=True)
        part = np.nan_to_num(np.asarray(apply_recipe(recipe, sl), dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
        assert np.allclose(full[500:900], part, atol=1e-6, rtol=1e-5), (
            f"recipe {recipe.name!r} not slice-replay-consistent (row-wise-purity violation)"
        )
