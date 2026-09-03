"""VOTENRANK-1 (2026-08-05 audit): ``shapley_blend``'s degenerate-fallback branch (every raw Shapley
value <= 0) picked the best-valued model via ``keep_mask`` but still multiplied it by its CLIPPED weight
(0.0 in this branch, since clipping value<=0 gives weight=0), silently returning an all-zero
``ensemble_pred`` while still reporting that model as "selected" -- a misleadingly valid-looking result.
Fixed by giving the sole fallback survivor weight 1.0 so ``ensemble_pred`` is that model's own real
predictions.
"""

from __future__ import annotations

import sys

import numpy as np

from mlframe.votenrank.shapley_blend import shapley_blend

# ``mlframe.votenrank/__init__.py`` does ``from .shapley_blend import shapley_blend``, which overwrites
# the package's ``shapley_blend`` ATTRIBUTE (the submodule reference) with the imported FUNCTION -- a
# classic name-collision gotcha. Re-resolve the real submodule via sys.modules (populated by the import
# above) rather than attribute access, so monkeypatching targets the module the function actually reads
# ``shapley_model_values`` from.
shapley_blend_mod = sys.modules["mlframe.votenrank.shapley_blend"]


def test_shapley_blend_degenerate_fallback_returns_nonzero_ensemble_pred(monkeypatch):
    """When every raw Shapley value is <= 0 (forced here via monkeypatching shapley_model_values, since
    reliably engineering real data/score_fn combinations that make every model's marginal contribution
    non-positive is not robust), ensemble_pred must be the selected fallback model's own predictions,
    not a silent all-zero array."""
    n_models, n_rows = 3, 10
    preds = np.arange(n_models * n_rows, dtype=np.float64).reshape(n_models, n_rows) + 1.0
    y = np.zeros(n_rows, dtype=np.float64)

    def _fake_shapley_model_values(preds, y, **kwargs):
        """Stand in for the real Shapley estimator: 3 models, all with negative raw values."""
        return np.array([-0.5, -0.1, -0.9]), {}

    monkeypatch.setattr(shapley_blend_mod, "shapley_model_values", _fake_shapley_model_values)

    result = shapley_blend(preds, y)

    assert not np.allclose(
        result["ensemble_pred"], 0.0
    ), "degenerate fallback must not silently return an all-zero ensemble_pred while reporting a model as selected"
    assert len(result["selected"]) == 1
    selected_idx = result["selected"][0]
    assert selected_idx == 1, "must pick the model with the least-negative (argmax) Shapley value"
    assert np.allclose(result["ensemble_pred"], preds[selected_idx]), "degenerate fallback's ensemble_pred must equal the selected model's own predictions"
    assert result["weights"][selected_idx] == 1.0
