"""Meta-test: every nested-config field default equals the flat ``MRMR.__init__`` default it overwrites.

``apply_mrmr_config_objects`` copies EVERY field of a passed nested config onto its flat attribute, and it does
so unconditionally - so a config the caller left at its defaults still overwrites the flats. If a config field's
default drifts from the flat ctor default, then merely passing ``MRMR(hybrid_orth_config=HybridOrthConfig())``
silently changes behaviour, with nothing in the call to hint at it. That is exactly how the ensemble-scorer
roster was once emptied by an all-defaults config.

The gate enumerates ALL mapping sources, not just ``_CONFIG_ATTR_FIELD_MAPS``: the two hybrid-orth maps are
applied separately by ``apply_mrmr_config_objects``, and a gate that missed them would have missed the exact
field the original bug was about.
"""

from __future__ import annotations

import inspect

import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses import (
    _CONFIG_ATTR_FIELD_MAPS,
    _HYBRID_ORTH_FIELD_MAP,
    _HYBRID_ORTH_SCORERS_FIELD_MAP,
    DCDConfig,
    FastSearchConfig,
    GroupAwareConfig,
    HybridOrthConfig,
    HybridOrthScorersConfig,
    StabilitySelectionConfig,
    SynergyRedundancyConfig,
)

# config_attr name -> the dataclass that owns it, so the gate can instantiate each with no arguments.
_CONFIG_CLASSES = {
    "fast_search_config": FastSearchConfig,
    "stability_config": StabilitySelectionConfig,
    "synergy_config": SynergyRedundancyConfig,
    "group_aware_config": GroupAwareConfig,
    "dcd_config": DCDConfig,
}

# A field whose config default INTENTIONALLY differs from the flat ctor default, with the reason. Every entry
# is a claim that passing an all-defaults config SHOULD change behaviour, so each needs its justification here.
_FAST_SEARCH_REASON = (
    "FastSearchConfig mirrors MRMR._FAST_SEARCH_OVERRIDES (the fast-search PROFILE), not the ctor defaults: "
    "diverging from them is the entire point of the profile."
)
_INTENTIONAL_DEFAULT_DIVERGENCE: dict[str, str] = {
    "FastSearchConfig.fe_max_steps": _FAST_SEARCH_REASON,
    "FastSearchConfig.fe_pair_prewarp_enable": _FAST_SEARCH_REASON,
    "FastSearchConfig.fe_stability_vote_enable": _FAST_SEARCH_REASON,
    "FastSearchConfig.fe_escalation_underdelivery_enable": _FAST_SEARCH_REASON,
}

# Configs whose all-defaults instance is EXPECTED to move flat attrs (override profiles, not default mirrors).
_OVERRIDE_PROFILE_CONFIGS = {"fast_search_config"}


def _flat_defaults() -> dict:
    """``{param_name: default}`` for every defaulted ``MRMR.__init__`` parameter."""
    sig = inspect.signature(MRMR.__init__)
    return {name: p.default for name, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}


def _all_mappings() -> list[tuple[str, object, dict]]:
    """``[(label, config_instance, field_map), ...]`` covering every map apply_mrmr_config_objects consults."""
    out: list[tuple[str, object, dict]] = []
    for attr, field_map in _CONFIG_ATTR_FIELD_MAPS:
        out.append((attr, _CONFIG_CLASSES[attr](), field_map))
    # The hybrid-orth pair is applied outside _CONFIG_ATTR_FIELD_MAPS; enumerate both halves explicitly.
    out.append(("hybrid_orth_config", HybridOrthConfig(), _HYBRID_ORTH_FIELD_MAP))
    out.append(("hybrid_orth_config.scorers", HybridOrthScorersConfig(), _HYBRID_ORTH_SCORERS_FIELD_MAP))
    return out


def test_every_config_field_default_matches_its_flat_ctor_default():
    """No nested-config field default drifts from the flat MRMR ctor default it unconditionally overwrites."""
    flat = _flat_defaults()
    drifted: list[str] = []
    unmapped: list[str] = []
    for label, cfg, field_map in _all_mappings():
        for field_name, flat_attr in field_map.items():
            if flat_attr not in flat:
                unmapped.append(f"{label}.{field_name} -> MRMR.__init__ has no '{flat_attr}' parameter")
                continue
            if f"{type(cfg).__name__}.{field_name}" in _INTENTIONAL_DEFAULT_DIVERGENCE:
                continue
            cfg_default = getattr(cfg, field_name)
            if cfg_default != flat[flat_attr]:
                drifted.append(f"{label}.{field_name}={cfg_default!r} != MRMR(...{flat_attr}={flat[flat_attr]!r})")

    assert not unmapped, (
        "A config field maps to a flat attribute that MRMR.__init__ does not declare; the mapping or the "
        "signature is stale:\n  " + "\n  ".join(sorted(unmapped))
    )
    assert not drifted, (
        f"{len(drifted)} nested-config default(s) disagree with the flat MRMR ctor default they overwrite. "
        "Because apply_mrmr_config_objects copies every field unconditionally, passing an ALL-DEFAULTS config "
        "would silently change behaviour. Fix: make the config field default equal the ctor default (or, if the "
        "divergence is deliberate, add it to _INTENTIONAL_DEFAULT_DIVERGENCE with a reason).\n  " + "\n  ".join(sorted(drifted))
    )


@pytest.mark.parametrize("config_attr", sorted(set(_CONFIG_CLASSES) - _OVERRIDE_PROFILE_CONFIGS))
def test_all_defaults_config_does_not_change_any_flat_attr(config_attr):
    """Constructing MRMR with an all-defaults nested config must leave every flat attr at its bare-MRMR value.

    Override-profile configs are excluded (see ``_OVERRIDE_PROFILE_CONFIGS``): moving the flats is their job.
    """
    baseline = MRMR()
    with_cfg = MRMR(**{config_attr: _CONFIG_CLASSES[config_attr]()})
    field_map = dict(_CONFIG_ATTR_FIELD_MAPS)[config_attr]
    for field_name, flat_attr in field_map.items():
        assert getattr(with_cfg, flat_attr) == getattr(baseline, flat_attr), (
            f"MRMR({config_attr}=<all defaults>) changed {flat_attr}: "
            f"{getattr(baseline, flat_attr)!r} -> {getattr(with_cfg, flat_attr)!r} (via {field_name})"
        )
