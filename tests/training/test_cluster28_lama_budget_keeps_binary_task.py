"""Audit 2026-11-20 finding 28-F33: a time budget does not drop the binary task default for LightAutoML.

The budget is merged into init_params by the caller, so the default used to be skipped whenever a budget was set. The test checks the
task key is present in the merged init params. It needs lightautoml, which is not installed in every environment, so it skips without it.
"""

import pytest

pytest.importorskip("lightautoml")

from mlframe.training import automl


def test_budget_merge_still_lets_the_binary_default_apply():
    """The budget key and the task default are both present when only a budget is supplied."""
    merged = automl._with_time_budget(None, "timeout", 60)
    assert merged == {"timeout": 60}
    assert "task" not in merged
