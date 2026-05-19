"""Smoke test for mlframe.votenrank.stability_exp (W5-4)."""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_import_stability_exp_module():
    """Module imports cleanly and exposes its public callables."""
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("seaborn")
    pytest.importorskip("matplotlib")
    from mlframe.votenrank import stability_exp

    for name in ("spearman_exp", "count_and_plot", "get_res_df", "create_exp_pic"):
        assert callable(getattr(stability_exp, name)), f"{name} not callable"
