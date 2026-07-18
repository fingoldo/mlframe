"""DX6: curated public surface — __all__ resolves, read-only registry proxy
re-exported, star-import does not leak stdlib/submodule noise."""

from __future__ import annotations

from types import MappingProxyType

import mlframe.training.composite as composite


def test_all_names_resolve() -> None:
    """All names resolve."""
    missing = [n for n in composite.__all__ if not hasattr(composite, n)]
    assert missing == [], f"__all__ lists unresolved names: {missing}"


def test_transforms_registry_reexported_read_only() -> None:
    """Transforms registry reexported read only."""
    assert hasattr(composite, "TRANSFORMS_REGISTRY")
    assert isinstance(composite.TRANSFORMS_REGISTRY, MappingProxyType)
    import pytest

    with pytest.raises(TypeError):
        composite.TRANSFORMS_REGISTRY["x"] = None  # read-only


def test_star_import_excludes_noise() -> None:
    """Star import excludes noise."""
    ns: dict = {}
    exec("from mlframe.training.composite import *", ns)  # nosec B102 -- exec of a literal, locally-authored import string, never untrusted input
    for noise in ("logging", "annotations", "logger"):
        assert noise not in ns, f"star-import leaked {noise!r}"
    # The headline classes ARE exported.
    for sym in ("CompositeTargetEstimator", "CompositeClassificationEstimator", "conformal_quantile", "TRANSFORMS_REGISTRY"):
        assert sym in ns
