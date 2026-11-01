"""A leakage guard that cannot run says so instead of passing silently (FOP4-2)."""

import logging

import numpy as np
import pandas as pd

from mlframe.calibration import _post_train_calibrators as mod


def test_unhashable_row_index_skip_is_warned(caplog):
    """An index holding unhashable labels cannot be compared; the skip is a WARNING naming the check."""
    target = pd.Series([0, 1], index=pd.Index([[1], [2]], dtype=object))
    caplog.set_level(logging.WARNING)
    mod._check_index_overlap("m", target, pd.Index([[3], [4]], dtype=object), 0.9, 0.5)
    assert any("row-index overlap check" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_incomparable_target_skip_is_warned(caplog, monkeypatch):
    """When the equality comparison raises, the identity check logs that it was skipped."""
    def boom(*a, **k):
        """Fail like an incomparable dtype."""
        raise TypeError("cannot compare")

    monkeypatch.setattr(mod.np, "array_equal", boom)
    caplog.set_level(logging.WARNING)
    mod._check_target_identity("m", np.zeros(3), np.ones(3))
    assert any("identity check" in r.getMessage() for r in caplog.records)
