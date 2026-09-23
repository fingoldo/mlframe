"""The predict-time env-signature check warns on major/minor drift only (INT-19, PMT-27c).

``env_signature()`` records full versions, and the check compared the dicts whole, so a ``pandas 2.2.2 -> 2.2.3`` or a
Python security release fired the drift warning on every predict call. The documented policy - and the only difference
that says the models were built against another library - is major/minor.
"""

from __future__ import annotations

import logging

import pytest

from mlframe.training.core.predict import _validate_metadata_version_envelope, env_signature_drift


def test_a_patch_only_difference_is_not_drift():
    """Patch bumps across several libraries, including Python, leave the drift list empty."""
    saved = {"numpy": "2.1.3", "pandas": "2.2.2", "python": "3.11.8"}
    live = {"numpy": "2.1.9", "pandas": "2.2.3", "python": "3.11.9"}
    assert env_signature_drift(saved, live) == []


def test_a_minor_or_major_difference_is_drift():
    """A minor bump and a major bump are both reported, with both versions."""
    saved = {"numpy": "2.1.3", "pandas": "2.2.2", "polars": "1.9.0"}
    live = {"numpy": "2.2.0", "pandas": "3.0.1", "polars": "1.9.3"}
    assert env_signature_drift(saved, live) == [("numpy", "2.1.3", "2.2.0"), ("pandas", "2.2.2", "3.0.1")]


def test_an_absent_library_differs_from_any_version():
    """A library installed on one side only is drift, in both directions."""
    assert env_signature_drift({"catboost": None}, {"catboost": "1.2.5"}) == [("catboost", None, "1.2.5")]
    assert env_signature_drift({"catboost": "1.2.5"}, {}) == [("catboost", "1.2.5", None)]


@pytest.mark.parametrize("patch_only, expect_warning", [(True, False), (False, True)])
def test_the_predict_path_warns_only_on_major_minor_drift(caplog, monkeypatch, patch_only, expect_warning):
    """The bundle check itself: a patch-only saved signature is silent, a minor-level one warns with both versions."""
    from mlframe.training import composite as composite_pkg

    live = {"numpy": "2.1.3", "pandas": "2.2.3", "python": "3.11.9"}
    monkeypatch.setattr(composite_pkg, "env_signature", lambda: dict(live))
    saved = dict(live, pandas="2.2.2" if patch_only else "2.1.4")
    metadata = {"schema_version": 2, "composite_target_specs": {"regression": {"y": []}}, "composite_target_env_signature": saved}

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        _validate_metadata_version_envelope(metadata, "bundle")
    warned = [r.getMessage() for r in caplog.records if "env signature drift" in r.getMessage()]
    assert bool(warned) is expect_warning, warned
    if expect_warning:
        assert "pandas 2.1.4 -> 2.2.3" in warned[0]
