"""Golden table of the dtype each surviving engineered-recipe kind returns on a fixed all-families fit, so a change of storage or precision cannot alter the output type unnoticed.

Every surviving kind returns the fit's float dtype (float32). The orth-univariate recipes still EVALUATE their basis in float64 and round the result once to the fit dtype, so fit and replay stay identical.
"""

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
from mlframe.feature_selection.filters.mrmr import MRMR

GOLDEN = {
    "grouped_quantile": {"float32"},
    "orth_univariate": {"float32"},
    "unary_binary": {"float32"},
}


@pytest.fixture(scope="module")
def fit_and_frame():
    """The all-families fit at 20000 x 40 and its input frame."""
    import os

    # the GPU-strict configuration of the profiling scripts: the dtype of some kinds depends on the device path
    for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
        os.environ.setdefault(k, v)
    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    X, y = F._make(20000, 40, 0)[:2]
    return MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(X, y), X


def test_recipe_output_dtypes_match_the_golden_table(fit_and_frame):
    """Every recipe kind present returns exactly the golden dtype; a kind outside the table is a reason to extend the table, not to ignore it."""
    sel, X = fit_and_frame
    seen: dict = {}
    for r in sel._engineered_recipes_:
        seen.setdefault(str(getattr(r, "kind", "")), set()).add(str(np.asarray(apply_recipe(r, X)).dtype))
    assert seen, "the fit produced no engineered recipe"
    for kind, dtypes in seen.items():
        assert kind in GOLDEN, f"recipe kind {kind!r} is not in the golden dtype table"
        assert dtypes == GOLDEN[kind], f"{kind}: {dtypes} != {GOLDEN[kind]}"


def test_orth_univariate_float32_output_is_the_rounded_float64_value(fit_and_frame, monkeypatch):
    """The default float32 column equals the float64 column rounded once, so fit-time and replay-time values are the same numbers."""
    sel, X = fit_and_frame
    recs = [r for r in sel._engineered_recipes_ if getattr(r, "kind", "") == "orth_univariate"]
    if not recs:
        pytest.skip("no orth_univariate recipe survived on this fixture")
    default = np.asarray(apply_recipe(recs[0], X))
    monkeypatch.setenv("MLFRAME_ORTH_UNIVARIATE_OUT_F64", "1")
    wide = np.asarray(apply_recipe(recs[0], X))
    assert default.dtype == np.float32 and wide.dtype == np.float64
    np.testing.assert_array_equal(default, wide.astype(np.float32))
