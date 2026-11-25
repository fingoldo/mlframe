"""Per-instance SHAP skips a multi-output CatBoost model before shap's TreeExplainer parses it (audit 2026-11-20, finding 21-F3)."""
import numpy as np

import mlframe.reporting.charts.shap_per_instance as spi


def test_multioutput_catboost_never_reaches_the_tree_explainer(monkeypatch):
    """A multi-output CatBoost crashes shap's parser natively; the builder must return a skip without constructing the explainer."""
    def _no_explainer(*_a, **_k):
        """Fail loudly if the builder reaches the TreeExplainer."""
        raise AssertionError("TreeExplainer must not be built for a multi-output CatBoost")

    monkeypatch.setattr(spi, "is_tree_model", lambda _m: True)
    monkeypatch.setattr(spi, "_is_multi_output_catboost", lambda _m: True)
    monkeypatch.setattr(spi, "TreeExplainer", _no_explainer, raising=False)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 3))
    y = rng.standard_normal(120)
    result = spi.shap_worst_errors_explanation(object(), X, y, y + 0.1)
    assert result.paths == []
    assert "multi-output" in result.skipped
