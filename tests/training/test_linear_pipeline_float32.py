"""The linear (scaling) pre-pipeline must emit float32, not float64.

SimpleImputer / StandardScaler upcast float32 -> float64 on older sklearn,
doubling the memory of the cached transformed frame (the prod 4.1M x 470
linear path sat at ~15 GB float64, RAM 111->128 GB). build_pipeline appends
a trailing float32 cast on the requires_scaling path so the persisted /
PipelineCache output stays float32 regardless of sklearn version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mlframe.training.strategies import ModelPipelineStrategy


class _ScalingStrategy(ModelPipelineStrategy):
    """Minimal concrete strategy on the requires_scaling (linear) path."""

    requires_scaling = True
    requires_imputation = True
    requires_encoding = False

    def build_estimator(self, *a, **k):
        """Build estimator."""
        return None

    def get_default_params(self, *a, **k):
        """Get default params."""
        return {}

    def cache_key(self, *a, **k):
        """Cache key."""
        return "k"


def _frame(dtype) -> pd.DataFrame:
    """Frame."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((300, 5)).astype(dtype), columns=list("abcde"))
    X.iloc[::11, 0] = np.nan  # exercise the imputer
    return X


def test_linear_pipeline_emits_float32_from_float32_input() -> None:
    """Linear pipeline emits float32 from float32 input."""
    st = _ScalingStrategy()
    pipe = st.build_pipeline(
        category_encoder=None,
        imputer=SimpleImputer(),
        scaler=StandardScaler(),
        base_pipeline=None,
        cat_features=[],
    )
    assert "to_float32" in [n for n, _ in pipe.steps], "float32 cast step missing"
    out = np.asarray(pipe.fit_transform(_frame(np.float32)))
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"


def test_linear_pipeline_downcasts_float64_input_too() -> None:
    # Even if upstream handed float64, the cached output must be float32.
    """Linear pipeline downcasts float64 input too."""
    st = _ScalingStrategy()
    pipe = st.build_pipeline(
        category_encoder=None,
        imputer=SimpleImputer(),
        scaler=StandardScaler(),
        base_pipeline=None,
        cat_features=[],
    )
    out = np.asarray(pipe.fit_transform(_frame(np.float64)))
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"


def test_cast_helper_is_idempotent_and_preserves_values() -> None:
    """Cast helper is idempotent and preserves values."""
    from mlframe.training.strategies.base import _cast_numeric_to_float32

    x = np.array([[1.5, 2.25], [3.125, 4.0]], dtype=np.float64)
    out = _cast_numeric_to_float32(x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, x.astype(np.float32))
    # idempotent on float32
    assert _cast_numeric_to_float32(out).dtype == np.float32
