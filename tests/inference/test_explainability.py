"""Smoke test for mlframe.inference.explainability (W5-4)."""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_import_explainability_module():
    """Module imports cleanly and exposes compute_shap_on_cv and init_model_instance.

    We deliberately skip running compute_shap_on_cv against a fitted CB model in this smoke
    test - it requires catboost+shap+imblearn+pandas all wired together; this would explode
    the per-file budget. The harder integration belongs in a slow tier; here we only verify
    the import surface is sane.
    """
    pytest.importorskip("shap")
    from mlframe.inference import explainability as ex

    assert callable(ex.init_model_instance)
    assert callable(ex.compute_shap_on_cv)
