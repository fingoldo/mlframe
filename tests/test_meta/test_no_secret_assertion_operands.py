"""No test assertion has an environment value or a resolved DSN as an operand, because pytest prints operands.

``assert "K" not in os.environ`` prints ``repr(os.environ)`` when it fails, every value of the environment included. The safe shape folds the value to a bool before the assert
(``unset = "K" not in os.environ`` then ``assert unset``). The check is py-ci-shared's ``secret_assertion_operands``; the allowlist is empty, so nothing is excused.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_MIN_TEST_FILES = 3000  # the scanned tree has about 5600 files; the floor stops a moved tests/ from passing vacuously


def test_no_assertion_prints_an_environment_value():
    """Every assertion under ``tests/`` is free of ``os.environ`` and DSN-resolver operands."""
    from py_ci_shared.secret_assertion_operands import assert_no_secret_assertion_operands

    assert_no_secret_assertion_operands([REPO_ROOT / "tests"], root=REPO_ROOT, min_files=_MIN_TEST_FILES)
