"""Member predictions of different lengths are stacked to the shortest, with a WARNING that names the lengths.

The stack cut every member to the shortest length without a word, so a caller could not tell that rows had been dropped.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling import process_method as pm


def test_ragged_member_predictions_warn_and_stack_to_the_shortest():
    """Two members of 10 and 7 rows stack to 7 rows and the WARNING names both lengths."""
    ns = SimpleNamespace()
    records: list = []

    class _Handler(logging.Handler):
        """Collect warnings from the ensembling logger."""

        def emit(self, record):
            """Keep each record."""
            records.append(record)

    handler = _Handler(level=logging.WARNING)
    pm.logger.addHandler(handler)
    try:
        pm._stack_member_predictions([np.arange(10.0), np.arange(7.0)], (ns,))
    finally:
        pm.logger.removeHandler(handler)
    assert ns.member_test_preds.shape == (7, 2)
    assert any("different lengths" in r.getMessage() and "10" in r.getMessage() for r in records)
