"""HybridSelector._run_shap must drop structurally invalid shared MRMR artifacts instead of restricting them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.hybrid_selector import HybridSelector


def _captured_precomputed(monkeypatch, artifacts, n_rows):
    """Run _run_shap with a probe ShapProxiedFS and return the precomputed kwarg it received."""
    seen = {}

    class _Probe:
        """ShapProxiedFS stand-in that records its precomputed kwarg."""

        def __init__(self, **kw):
            """Record the precomputed argument."""
            seen["precomputed"] = kw["precomputed"]

        def fit(self, *a, **k):
            """Abort the fit once construction has been captured."""
            raise RuntimeError("stop after construction")

    monkeypatch.setattr("mlframe.feature_selection.shap_proxied_fs.ShapProxiedFS", _Probe)
    h = HybridSelector.__new__(HybridSelector)
    h.random_state = 0
    X = pd.DataFrame(np.zeros((n_rows, 2)), columns=["a", "b"])
    try:
        h._run_shap(X, np.zeros(n_rows), ["a", "b"], artifacts)
    except RuntimeError:
        pass
    return seen.get("precomputed")


def test_row_count_mismatch_drops_artifacts(monkeypatch):
    """A dict fitted on another row count is dropped."""
    art = {"su_to_target": np.array([0.1, 0.2]), "feature_names": ["a", "b"], "n_samples_at_fit": 99}
    assert _captured_precomputed(monkeypatch, art, n_rows=10) is None


def test_missing_su_drops_artifacts(monkeypatch):
    """A dict without su_to_target is dropped."""
    assert _captured_precomputed(monkeypatch, {"feature_names": ["a", "b"]}, n_rows=10) is None


def test_valid_artifacts_still_reach_the_shap_member(monkeypatch):
    """A valid dict is restricted and handed to the SHAP member."""
    art = {"su_to_target": np.array([0.1, 0.2]), "feature_names": ["a", "b"], "n_samples_at_fit": 10}
    monkeypatch.setattr("mlframe.feature_selection.shap_proxied_fs.restrict_artifacts", lambda a, idx: {"kept": list(idx)})
    assert _captured_precomputed(monkeypatch, art, n_rows=10) == {"kept": [0, 1]}
