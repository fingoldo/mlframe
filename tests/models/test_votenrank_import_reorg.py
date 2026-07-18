"""Regression: votenrank-leaderboard lazy import must resolve after the ensembling subpackage reorg.

`ensembling.py` became `ensembling/__init__.py`, so `score_flavours.py`'s lazy `from .ensembling import ...`
(a submodule path that no longer exists) raised ModuleNotFoundError -- silently downgrading every votenrank
leaderboard build to a logged warning. The import must point at the package `__init__` (`from . import ...`).
"""

from __future__ import annotations

import logging


def test_votenrank_symbol_importable_from_package():
    """Votenrank symbol importable from package."""
    from mlframe.models.ensembling import _build_votenrank_leaderboard_from_results

    assert callable(_build_votenrank_leaderboard_from_results)


def test_maybe_build_votenrank_lazy_import_resolves(caplog):
    """Maybe build votenrank lazy import resolves."""
    from mlframe.models.ensembling.score_flavours import maybe_build_votenrank_leaderboard

    with caplog.at_level(logging.WARNING):
        maybe_build_votenrank_leaderboard({}, is_regression=False, build_votenrank_leaderboard_flag=True)
    # The fatal symptom was a "No module named 'mlframe.models.ensembling.ensembling'" warning; it must be gone.
    assert not any(
        "No module named" in r.getMessage() and "ensembling.ensembling" in r.getMessage() for r in caplog.records
    ), "votenrank lazy import still points at the stale .ensembling submodule path"
