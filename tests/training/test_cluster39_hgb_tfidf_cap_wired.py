"""Cluster 39 regression: the HGB TF-IDF cap is applied on the feature-handling path (audit 2026-11-20, finding 39-F38).

``hgb_max_features_cap`` was exported and never called, so HGB densified a 5000-feature TF-IDF with no cap. The apply path now routes TF-IDF
params through ``_tfidf_params_for_model``, which caps them for ``hgb`` and leaves other model kinds unchanged.
"""

from mlframe.training.feature_handling.apply import _tfidf_params_for_model
from mlframe.training.feature_handling.handlers import TfidfParams
from mlframe.training.feature_handling.routing import HGB_TFIDF_MAX_FEATURES_CAP


def test_hgb_tfidf_is_capped():
    """An HGB run with max_features above the cap gets the cap (39-F38)."""
    out = _tfidf_params_for_model("tfidf", TfidfParams(max_features=5000), "hgb")
    assert out.max_features == HGB_TFIDF_MAX_FEATURES_CAP
    assert out.max_features < 5000


def test_sparse_capable_model_keeps_requested_features():
    """Control: a sparse-aware model keeps the requested max_features (39-F38)."""
    out = _tfidf_params_for_model("tfidf", TfidfParams(max_features=5000), "cb")
    assert out.max_features == 5000


def test_non_tfidf_method_is_untouched():
    """Control: non-TF-IDF methods pass through unchanged (39-F38)."""
    params = object()
    assert _tfidf_params_for_model("hashing", params, "hgb") is params
