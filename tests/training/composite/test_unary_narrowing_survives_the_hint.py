"""The unary narrowing and the dominant-feature hint compose; the hint must not undo the narrowing.

A production run logged "narrowed from 24 transform(s) to the 5 base-free unary one(s)" for all four regression targets,
then evaluated 59 candidates per target anyway: the hint was applied to the suite-level config instead of the narrowed
one. That cost 2 h 8 min of discovery whose single shipped spec was a unary transform the narrowed search finds too.
"""

from __future__ import annotations

from mlframe.training.composite.transforms import get_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig
from mlframe.training.core._phase_composite_discovery_gates import _per_target_discovery_config

_UNLIKELY = {
    "composite_recommendation": "unlikely_to_help",
    "composite_recommendation_reason": "init_score baseline matches raw",
    "ablation": [
        {"feature": "post_editing_time_ms", "delta_pct": 0.1},
        {"feature": "publish_delay_sec", "delta_pct": 0.05},
    ],
}


def _is_unary(name: str) -> bool:
    return not get_transform(name).requires_base


def test_hint_is_applied_on_top_of_the_narrowed_transforms():
    """The defect: the hint rebuilt the config from the suite level and every base-dependent transform came back."""
    base = CompositeTargetDiscoveryConfig()
    assert not all(_is_unary(t) for t in base.transforms), "the default search must contain base-dependent transforms"

    cfg, strengths = _per_target_discovery_config(base, _UNLIKELY, "target_total_charge", use_hint=True, hint_top_k=3)

    assert cfg.transforms and all(_is_unary(t) for t in cfg.transforms)
    assert list(cfg.dominant_features_hint) == ["post_editing_time_ms", "publish_delay_sec"]
    assert strengths == [0.1, 0.05]


def test_without_the_hint_the_narrowing_still_applies():
    """The hint switch only decides whether a hint is added."""
    cfg, strengths = _per_target_discovery_config(CompositeTargetDiscoveryConfig(), _UNLIKELY, "t", use_hint=False, hint_top_k=3)
    assert all(_is_unary(t) for t in cfg.transforms)
    assert strengths is None


def test_a_target_with_dominant_features_keeps_the_full_search():
    """Narrowing is only for the "unlikely to help" verdict; otherwise every family stays in."""
    base = CompositeTargetDiscoveryConfig()
    diag = dict(_UNLIKELY, composite_recommendation="high_potential")
    cfg, _ = _per_target_discovery_config(base, diag, "t", use_hint=True, hint_top_k=3)
    assert list(cfg.transforms) == list(base.transforms)
    assert list(cfg.dominant_features_hint) == ["post_editing_time_ms", "publish_delay_sec"]


def test_no_diagnostics_leaves_the_config_untouched():
    """Without BaselineDiagnostics there is nothing to narrow on and no hint to derive."""
    base = CompositeTargetDiscoveryConfig()
    cfg, strengths = _per_target_discovery_config(base, None, "t", use_hint=True, hint_top_k=3)
    assert cfg is base and strengths is None
