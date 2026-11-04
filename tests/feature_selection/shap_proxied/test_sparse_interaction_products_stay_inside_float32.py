"""Sanitised interaction products must stay finite after the float32 cast the boosters apply."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from xgboost import XGBRegressor

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as explain_mod
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_interactions as mod


def test_overflowing_product_column_is_finite_in_float32(monkeypatch):
    """Operands near 1e20 multiply to ~1e40: finite in float64, inf in float32. The augmented frame handed to the model must be float32-finite."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n) * 1e20, "b": rng.normal(size=n) * 1e20, "c": rng.normal(size=n)})
    y = rng.normal(size=n)
    seen = {}

    def _spy(model_template, aug, y_arr, **kwargs):
        """Record the augmented frame instead of fitting."""
        seen["aug"] = aug.copy()
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(explain_mod, "compute_shap_matrix", _spy)
    with pytest.raises(RuntimeError, match="stop after capture"):
        mod.sparse_interaction_candidates(XGBRegressor(n_estimators=2), X, y, [(1.0, "a", "b")], classification=False)
    prod = next(c for c in seen["aug"].columns if c.startswith("_suprod_"))
    as_f32 = seen["aug"][prod].to_numpy().astype(np.float32)
    assert np.isfinite(as_f32).all()
