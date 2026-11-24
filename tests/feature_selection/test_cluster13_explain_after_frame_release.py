"""Cluster 13 regression: explain() after the working frames were released raises a clear error (audit 2026-11-20, finding 13-F19).

``release_large_working_frames`` sets ``X_boruta_`` to None for a large fit. ``explain()`` then passed None into the explainer and failed deep
inside it. It now raises a RuntimeError that names the cause.
"""

from types import SimpleNamespace

import pytest

from mlframe.feature_selection.boruta_shap import _fit_explain as fe


def test_explain_after_release_raises_clear_error():
    """With the fitted frames released, explain() raises RuntimeError naming the release (13-F19)."""
    released = SimpleNamespace(X_boruta_=None, X_boruta_train_=None, sample=False)
    with pytest.raises(RuntimeError, match="released for a large fit"):
        fe.explain(released)
