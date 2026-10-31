"""A unary/binary recipe replayed on an integer operand computes the unaries in the working float: numpy gives float16 for exp/sqrt of an int8 column, which lost precision and which the compiled binary kernels reject."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary import build_unary_binary_recipe


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int64, np.float16, np.bool_])
def test_replay_on_a_narrow_operand_equals_replay_on_its_float32_copy(dtype):
    """Same engineered column for the narrow-dtype operand and for the same values stored as float32."""
    rng = np.random.default_rng(0)
    n = 500
    a = rng.integers(0, 6, n)
    b = rng.standard_normal(n).astype(np.float32) + 3.0
    recipe = build_unary_binary_recipe(
        name="div(exp(a),sqrt(b))", src_a_name="a", src_b_name="b", unary_a_name="exp", unary_b_name="sqrt", binary_name="div", unary_preset="medium", binary_preset="minimal",
        quantization_nbins=None, quantization_method=None, quantization_dtype=np.int16,
    )
    narrow = pd.DataFrame({"a": a.astype(dtype), "b": b})
    wide = pd.DataFrame({"a": a.astype(dtype).astype(np.float32), "b": b})
    got = np.asarray(apply_recipe(recipe, narrow))
    want = np.asarray(apply_recipe(recipe, wide))
    assert np.array_equal(got, want)
