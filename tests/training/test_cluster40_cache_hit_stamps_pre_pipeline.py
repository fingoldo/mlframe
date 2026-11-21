"""Audit 2026-11-20 finding 40-F13: a pre-pipeline cache hit stamps the fitted pre-pipeline on the model, as the fit path does."""

from types import SimpleNamespace

import pandas as pd

from mlframe.training.pipeline import _pipeline_helpers_apply as mod


def test_cache_hit_stamps_the_pre_pipeline_on_the_model(monkeypatch):
    """With a cache hit on an unfitted pipeline the model gets _mlframe_pre_pipeline before the function returns."""
    monkeypatch.setattr(mod, "_pre_pipeline_cache_key", lambda *a, **k: "key")
    monkeypatch.setattr(mod, "_pre_pipeline_cache_get", lambda *a, **k: "cached")
    monkeypatch.setattr(mod, "_is_fitted", lambda p: False)
    restored = (pd.DataFrame({"a": [1.0]}), None)
    monkeypatch.setattr(mod, "_apply_pre_pipeline_restore_cached", lambda *a, **k: restored)
    model = SimpleNamespace()
    pre = SimpleNamespace(name="pre")
    out = mod._apply_pre_pipeline_transforms(
        model, pre, pd.DataFrame({"a": [1.0]}), None, None, False, False, True, None, 0,
    )
    assert out == restored
    assert model._mlframe_pre_pipeline is pre
