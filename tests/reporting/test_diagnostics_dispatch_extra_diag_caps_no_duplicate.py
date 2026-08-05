"""REPORTING_A-6 regression test: _diagnostics_dispatch_extra's DIAG_ROW_CAP/DIAG_MAX_FEATURES must not be
an independently-maintained duplicate of diagnostics_dispatch.py's values -- they must resolve to the
SAME object/value as the actual consumer, so the two modules can never silently drift.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


def test_attribute_access_matches_parent_module():
    """mod.DIAG_ROW_CAP / mod.DIAG_MAX_FEATURES on _diagnostics_dispatch_extra equal diagnostics_dispatch's."""
    from mlframe.reporting import _diagnostics_dispatch_extra as extra_mod
    from mlframe.reporting import diagnostics_dispatch as parent_mod

    assert extra_mod.DIAG_ROW_CAP == parent_mod.DIAG_ROW_CAP
    assert extra_mod.DIAG_MAX_FEATURES == parent_mod.DIAG_MAX_FEATURES


def test_from_import_matches_parent_module():
    """`from _diagnostics_dispatch_extra import DIAG_ROW_CAP` resolves the same value as the parent."""
    from mlframe.reporting._diagnostics_dispatch_extra import DIAG_MAX_FEATURES, DIAG_ROW_CAP
    from mlframe.reporting.diagnostics_dispatch import DIAG_MAX_FEATURES as parent_max_features
    from mlframe.reporting.diagnostics_dispatch import DIAG_ROW_CAP as parent_row_cap

    assert DIAG_ROW_CAP == parent_row_cap
    assert DIAG_MAX_FEATURES == parent_max_features


def test_unknown_attribute_still_raises_attributeerror():
    """The module __getattr__ hook must not swallow genuinely missing attributes."""
    from mlframe.reporting import _diagnostics_dispatch_extra as extra_mod

    with pytest.raises(AttributeError):
        getattr(extra_mod, "NOT_A_REAL_ATTRIBUTE")
