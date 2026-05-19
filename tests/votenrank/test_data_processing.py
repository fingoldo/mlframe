"""Smoke test for mlframe.votenrank.data_processing (W5-4)."""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_import_data_processing_module():
    """Module imports cleanly and exposes its public callables."""
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    from mlframe.votenrank import data_processing as dp

    for name in ("preprocess_glue", "preprocess_sglue", "preprocess_value"):
        assert callable(getattr(dp, name)), f"{name} not callable"
