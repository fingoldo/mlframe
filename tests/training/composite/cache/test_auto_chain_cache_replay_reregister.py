"""Auto-discovered chain transforms (chain_<residual>_<unary>) are registered in-process during a FRESH discovery;
on a CACHE replay that registration never runs, so a cached spec naming such a chain crashed get_transform /
predict-time inversion with UnknownTransformError (prod TVT 2026-06: cache HIT -> 'chain_linear_residual_yj' KeyError).
reregister_auto_chain_transforms rebuilds + registers them from the spec names."""

from __future__ import annotations

import pytest

from mlframe.training.composite import get_transform
from mlframe.training.composite.discovery._auto_chain import reregister_auto_chain_transforms
from mlframe.training.composite.transforms.naming import UnknownTransformError
from mlframe.training.composite.transforms.registry import _TRANSFORMS_REGISTRY


def test_reregister_rebuilds_missing_autochain_and_get_transform_resolves():
    name = "chain_linear_residual_yj"
    _TRANSFORMS_REGISTRY.pop(name, None)
    with pytest.raises(UnknownTransformError):
        get_transform(name)  # absent before re-registration (the crash the cache replay hit)
    done = reregister_auto_chain_transforms([name, "chain_monotonic_residual_cbrt"])
    assert name in done
    assert get_transform(name).name == name  # now resolves -> no UnknownTransformError on replay


def test_reregister_ignores_non_chain_and_already_registered_and_garbage():
    # linear_residual is a real registry transform (not a chain) -> skipped; bogus parses to nothing -> skipped.
    done = reregister_auto_chain_transforms(["linear_residual", "chain_not_a_real_unary", "", None, "chain_bogus_zzz"])
    assert done == []
