"""Audit 2026-11-20 finding 29-F22: a tracker directory for a later run is not read as run 0.

It used to be parsed as run 0 and overwrite run 0's data. The test puts a run-1 directory in the tracker folder and checks it is named in
a WARNING as skipped.
"""

import logging

import pandas as pd
import pytest

pytest.importorskip("experiment_impact_tracker")

from mlframe.votenrank.utils import get_tracker_table


def test_later_run_directory_is_skipped_and_reported(tmp_path, caplog):
    """A modelA_task_1 directory is named in a WARNING and not read."""
    (tmp_path / "modelA_task_1").mkdir()
    data = pd.DataFrame({"task.x": [1.0]}, index=["modelA"])
    with caplog.at_level(logging.WARNING):
        get_tracker_table(data, str(tmp_path))
    assert any("modelA_task_1" in r.getMessage() for r in caplog.records)
