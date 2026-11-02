"""Shared fixtures of the discretization tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def honest_hooks(monkeypatch):
    """The honest-null hooks (``_honest_null_codes_fn``) the fits of the test install, in order.

    ``MRMR.fit`` releases the closure when it ends (it holds a float64 copy of the frame), so a test that wants to call the hook takes it here, at the moment the fit installs it."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    captured: list = []
    real_setattr = MRMR.__setattr__

    def spy(self, name, value):
        """Remember every hook a fit installs, then set it as usual."""
        if name == "_honest_null_codes_fn" and value is not None:
            captured.append(value)
        real_setattr(self, name, value)

    monkeypatch.setattr(MRMR, "__setattr__", spy)
    MRMR.clear_fit_cache()  # a cached fit would skip the installation
    return captured
