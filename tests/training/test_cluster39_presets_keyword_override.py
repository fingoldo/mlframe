"""Cluster 39 regression: preset factories accept a caller override of a key they set (audit 2026-11-20, finding 39-F46).

``tfidf_only(default_cat=...)`` used to raise a duplicate-keyword TypeError because the preset passed ``default_cat`` explicitly and again via
``**fhc_kwargs``. The caller's value now replaces the preset default. A plain call keeps the preset defaults.
"""

from mlframe.training.feature_handling.handlers import CatHandlerSpec, NoParams
from mlframe.training.feature_handling.presets import tfidf_only


def test_caller_override_of_preset_key_does_not_raise():
    """Passing default_cat to tfidf_only builds a config with the caller's value (39-F46)."""
    override = [CatHandlerSpec(method="native", params=NoParams(kind="native"))]
    cfg = tfidf_only(default_cat=override)
    assert cfg.default_cat == override


def test_plain_call_keeps_preset_defaults():
    """Control: without an override the preset's ordinal default_cat is used (39-F46)."""
    cfg = tfidf_only()
    assert cfg.default_cat[0].method == "ordinal"
