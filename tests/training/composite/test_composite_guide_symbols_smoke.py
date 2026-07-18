"""Smoke test: every symbol the post-wave-10 composite guide documents must
import. Guards the guide against documenting a symbol that does not exist (or
got renamed). Pure import + attribute existence checks -- no fitting.
"""

from __future__ import annotations

import importlib

import pytest


# (module path, attribute) pairs, one per documented public symbol.
PUBLIC_SYMBOLS = [
    # estimator families
    ("mlframe.training.composite", "CompositeGLMEstimator"),
    ("mlframe.training.composite", "CompositeQuantileEstimator"),
    ("mlframe.training.composite", "CompositeMultiOutputEstimator"),
    ("mlframe.training.composite", "make_per_column_specs"),
    ("mlframe.training.composite", "CompositeDistributionEstimator"),
    ("mlframe.training.composite", "CompositeSurvivalEstimator"),
    ("mlframe.training.composite", "CompositePanelEstimator"),
    ("mlframe.training.composite", "CompositeRankEstimator"),
    ("mlframe.training.composite", "OrthogonalizedCompositeEstimator"),
    ("mlframe.training.composite", "BaggedCompositeEstimator"),
    ("mlframe.training.composite", "MissingAwareComposite"),
    ("mlframe.training.composite", "TailCompositeEstimator"),
    ("mlframe.training.composite", "CompositeOrRawStacker"),
    # workflow helpers
    ("mlframe.training.composite", "suggest_discovery_config"),
    ("mlframe.training.composite", "discover_and_wrap"),
    ("mlframe.training.composite", "DiscoverAndWrapResult"),
    ("mlframe.training.composite", "optimize_composite"),
    ("mlframe.training.composite", "stability_select_specs"),
    ("mlframe.training.composite", "engineer_temporal_bases"),
    ("mlframe.training.composite.discovery", "discover_incremental"),
    # interpretability
    ("mlframe.training.composite", "explain_prediction"),
    ("mlframe.training.composite", "attribution_summary"),
    ("mlframe.training.composite", "composite_report"),
    ("mlframe.training.composite", "composite_model_card"),
    # production
    ("mlframe.training.composite", "CompositeDriftMonitor"),
    ("mlframe.training.composite", "detect_base_target_leakage"),
    ("mlframe.training.composite", "PurgedTimeSeriesSplit"),
    ("mlframe.training.composite", "make_purged_cv"),
    ("mlframe.training.composite", "export_serving_spec"),
    ("mlframe.training.composite", "load_serving_spec"),
    ("mlframe.training.composite", "compare_models"),
    ("mlframe.training.composite", "should_promote"),
    ("mlframe.training.composite", "CompositeFeatureGenerator"),
    # ranking helper
    ("mlframe.training.composite.ranking", "ndcg_at_k"),
]


# (estimator class, method) pairs for the uncertainty menu -- methods rebound
# onto the estimator classes at import time.
BOUND_METHODS = [
    (
        "mlframe.training.composite",
        "CompositeTargetEstimator",
        [
            "init_aci",
            "predict_interval_online",
            "update_conformal",
            "get_aci_state",
            "calibrate_conformal",
            "calibrate_conformal_cqr",
            "calibrate_conformal_mondrian",
            "calibrate_conformal_weighted",
        ],
    ),
    (
        "mlframe.training.composite",
        "CompositeClassificationEstimator",
        ["calibrate_conformal_set", "predict_set", "calibrate_venn_abers", "predict_proba_interval", "predict_proba_venn_abers"],
    ),
    ("mlframe.training.composite", "CompositeGLMEstimator", ["calibrate_conformal_glm", "predict_interval_glm"]),
    ("mlframe.training.composite", "CompositeMultiOutputEstimator", ["calibrate_conformal", "predict_interval"]),
    ("mlframe.training.composite", "CompositeDistributionEstimator", ["predict_quantile", "predict_cdf", "sample", "crps"]),
    ("mlframe.training.composite", "TailCompositeEstimator", ["predict_tail_quantile", "tail_residual_offset"]),
]


@pytest.mark.parametrize("module_path,attr", PUBLIC_SYMBOLS)
def test_documented_symbol_imports(module_path, attr):
    """Documented symbol imports."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path}.{attr} is documented but missing"


@pytest.mark.parametrize("module_path,cls_name,methods", BOUND_METHODS)
def test_documented_bound_methods_exist(module_path, cls_name, methods):
    """Documented bound methods exist."""
    cls = getattr(importlib.import_module(module_path), cls_name)
    for m in methods:
        assert hasattr(cls, m), f"{cls_name}.{m}() is documented but missing"


def test_qrf_estimator_present_and_documented():
    # CompositeQRFEstimator landed as a real public estimator (own qrf.py module + __all__ + tests),
    # so the guide documents it and the symbol must resolve.
    """Qrf estimator present and documented."""
    mod = importlib.import_module("mlframe.training.composite")
    assert hasattr(mod, "CompositeQRFEstimator")
