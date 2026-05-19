"""Smoke test for mlframe.votenrank.iia_exp (W5-4)."""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_import_iia_exp_module():
    """Module imports cleanly and exposes its public callables."""
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    from mlframe.votenrank import iia_exp

    for name in ("fine_sorted_ranking", "compute_iia_for_fixed_models", "compute_iia"):
        assert callable(getattr(iia_exp, name)), f"{name} not callable"
