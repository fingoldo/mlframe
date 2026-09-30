"""Turn a benchmark cell that was skipped by the platform guard into a skip of the test that ran it.

``run_cell`` records ANY ``BaseException`` as the cell's error so a crashed cell is data rather than an absence, and that includes the
``Skipped`` the macOS LightGBM guard in ``tests/conftest.py`` raises from ``Dataset.construct``. The test then reads a failed record and
reports a failure for a platform limitation it never got to exercise.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

_SKIPPED_PREFIX = "Skipped: "


def skip_if_the_platform_skipped(record: Dict[str, Any]) -> None:
    """Skip the calling test when ``record`` failed only because a pytest skip was raised inside the cell."""
    error = str(record.get("error") or "")
    if record.get("status") != "ok" and error.startswith(_SKIPPED_PREFIX):
        pytest.skip(error[len(_SKIPPED_PREFIX) :])
