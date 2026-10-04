"""Known product gaps recorded as measured, strict expectations rather than unconditional imperative xfails."""

from __future__ import annotations

import pytest


def known_gap(reason: str, *, gap_closed: bool) -> None:
    """Record an open product gap, or fail loudly once the measurement shows it closed.

    ``gap_closed`` is the verdict of the real contract assertion evaluated on the measured value. While the gap is open the test xfails with ``reason``;
    when the contract starts holding, the test fails so the entry gets removed from the gap registry instead of silently hiding the fix.
    """
    if gap_closed:
        pytest.fail(f"known gap is closed, remove it from the gap registry: {reason}", pytrace=False)
    pytest.xfail(reason)
