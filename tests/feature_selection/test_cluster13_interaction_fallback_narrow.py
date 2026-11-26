"""Cluster 13 regression: the interaction proxy falls back to additive only on the expected failures (audit 2026-11-20, finding 13-F26).

A broad handler used to swallow programmer errors too. The fallback now catches only the expected model and tensor failures, so an unexpected
KeyError surfaces, while a ValueError from an unsupported model still falls back with a logged warning.
"""

import contextlib
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_interactions as inter
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_interactions import _augment_proxy_mode_interaction


def _run(monkeypatch, error):
    """Run the interaction augmentation with the tensor step raising ``error``; return the candidates and the report."""

    def _raise(*_args, **_kwargs):
        """Stand in for a tensor build that fails with ``error``."""
        raise error

    monkeypatch.setattr(inter, "compute_interaction_tensor", _raise)
    owner = SimpleNamespace(
        classification=False,
        min_features=1,
        max_features=3,
        top_n=5,
        interaction_proxy_top_k=3,
        metric="rmse",
        _rng=np.random.default_rng(0),
    )
    X_proxy = pd.DataFrame(np.zeros((20, 4)), columns=list("abcd"))
    candidates = [(0.5, (0,))]
    report = {}
    out = _augment_proxy_mode_interaction(
        owner, list(candidates), np.zeros((20, 4)), np.zeros(20), X_proxy, [0, 1, 2, 3], np.zeros(20), np.zeros(20), object(), report,
        lambda _name: contextlib.nullcontext(),
    )
    return out, report


def test_unexpected_error_in_the_interaction_step_propagates(monkeypatch):
    """A KeyError from the interaction step is a programmer error and is not swallowed by the fallback (13-F26)."""
    import pytest

    with pytest.raises(KeyError):
        _run(monkeypatch, KeyError("programmer error"))


def test_expected_failure_falls_back_to_additive_with_a_warning(monkeypatch, caplog):
    """A ValueError from an unsupported model falls back: candidates are returned unchanged, not applied, and one WARNING is logged (13-F26)."""
    with caplog.at_level(logging.WARNING):
        out, report = _run(monkeypatch, ValueError("unsupported model"))
    assert out == [(0.5, (0,))]
    assert report["proxy_mode_interaction"]["applied"] is False
    assert any("fell back to additive" in r.getMessage() for r in caplog.records)
