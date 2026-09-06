"""A mid-call GPU fallback must not change the precision of the columns a recipe replay emits.

`apply_unary_binary_gpu` materialises engineered columns in float32 under `MLFRAME_FE_VRAM_F32` and float64
otherwise. The numpy path is both the default and the fallback taken on ANY cupy runtime failure, and it
produced the operand's own dtype -- so with the flag on, a transient device fault part-way through one
`transform()` call changed the dtype, and therefore the values, of every column produced after it. Which
columns came back in which precision depended on when the GPU happened to fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


@pytest.fixture
def frame():
    """Two float64 operands the recipe below combines."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.uniform(1.0, 5.0, 256), "b": rng.uniform(1.0, 5.0, 256)})


def _recipe():
    """A minimal two-operand unary_binary recipe."""
    from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe

    return EngineeredRecipe(
        name="mul(a,b)",
        kind="unary_binary",
        src_names=("a", "b"),
        unary_names=("identity", "identity"),
        binary_name="mul",
        unary_preset="minimal",
        binary_preset="minimal",
    )


def test_the_numpy_replay_matches_the_gpu_paths_precision_when_f32_is_on(frame, monkeypatch):
    """With MLFRAME_FE_VRAM_F32 set, the fallback must emit float32 like the device path would."""
    gpu_mod = pytest.importorskip("mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary_gpu")
    monkeypatch.setattr(gpu_mod, "_vram_f32", lambda: True)

    out = apply_recipe(_recipe(), frame)
    assert out.dtype == np.float32, (
        f"the numpy replay emitted {out.dtype} while the GPU path under MLFRAME_FE_VRAM_F32 would have "
        "emitted float32; a mid-call fallback therefore changes the column's precision"
    )


def test_the_numpy_replay_stays_float64_when_f32_is_off(frame, monkeypatch):
    """The cast must be conditional -- with the flag off the device path is float64 too."""
    gpu_mod = pytest.importorskip("mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary_gpu")
    monkeypatch.setattr(gpu_mod, "_vram_f32", lambda: False)

    out = apply_recipe(_recipe(), frame)
    assert out.dtype == np.float64, f"the numpy replay downcast to {out.dtype} with MLFRAME_FE_VRAM_F32 off"


def test_the_values_still_match_to_float32_resolution(frame, monkeypatch):
    """The cast changes precision, not the computation."""
    gpu_mod = pytest.importorskip("mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary_gpu")

    monkeypatch.setattr(gpu_mod, "_vram_f32", lambda: False)
    f64 = apply_recipe(_recipe(), frame)
    monkeypatch.setattr(gpu_mod, "_vram_f32", lambda: True)
    f32 = apply_recipe(_recipe(), frame)

    np.testing.assert_allclose(f32.astype(np.float64), f64, rtol=1e-6)
