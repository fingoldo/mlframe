"""MODELS-15 regression test: mlframe.models.tuning must not leak third-party imports through star-import.

The bug (fixed): tuning.py defined no ``__all__`` (unlike its sibling ``rf_proximity.py``), so
``mlframe.models.__init__``'s ``from mlframe.models.tuning import *`` leaked 13 third-party symbols
(pd, np, db, uniform, loguniform, randint, KFold, train_test_split, cross_validate, check_scoring, Enum,
auto) plus the mutable global ``trained_models`` into mlframe.models's public API. Added an explicit
``__all__`` listing only the genuine public API (functions/classes/the documented ``trained_models``
global), matching the convention rf_proximity.py already follows.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast

_LEAKED_THIRD_PARTY_NAMES = (
    "pd", "np", "db", "uniform", "loguniform", "randint", "KFold",
    "train_test_split", "cross_validate", "check_scoring", "Enum", "auto",
)


def test_tuning_has_explicit_all():
    """tuning.py defines __all__ (matching the rf_proximity.py sibling convention)."""
    from mlframe.models import tuning

    assert hasattr(tuning, "__all__"), "tuning.py should define __all__ to control its star-import surface"
    assert isinstance(tuning.__all__, list)
    assert len(tuning.__all__) > 0


def test_tuning_all_excludes_third_party_imports():
    """None of tuning.py's third-party imports appear in its __all__."""
    from mlframe.models import tuning

    leaked = [name for name in _LEAKED_THIRD_PARTY_NAMES if name in tuning.__all__]
    assert not leaked, f"tuning.__all__ should not leak third-party imports, found: {leaked}"


def test_tuning_all_includes_genuine_public_api():
    """The genuine public API (documented functions/classes, the trained_models global) is still exported."""
    from mlframe.models import tuning

    for name in ("MLTaskType", "trained_models", "ParamsOptimizer", "CatboostParamsOptimizer", "get_model", "justify_estimator"):
        assert name in tuning.__all__, f"{name} should remain in tuning.__all__"
        assert hasattr(tuning, name)
