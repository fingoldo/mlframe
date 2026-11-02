"""The pre-push kernel-tuning hook rejects unknown arguments and reads its skip flag like its sibling flag (tracker 2026-11-02 wave 15, CLI15-7)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepush_kernel_tuning.py"


def _load():
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location("prepush_kernel_tuning_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_argument_is_an_error_not_a_tuning_run():
    """--help or a typo used to be ignored and a full GPU tuning run started."""
    hook = _load()
    calls = []
    with pytest.raises(SystemExit):
        hook.main(["--bogus"], environ={}, run=lambda argv: calls.append(argv) or 0)
    assert calls == []


@pytest.mark.parametrize("value,skipped", [("1", True), ("true", True), ("0", False), ("false", False), ("no", False), ("", False)])
def test_skip_flag_false_values_do_not_skip(value, skipped):
    """MLFRAME_SKIP_KERNEL_TUNING_HOOK=false meant 'do not skip' to the user but disabled the hook."""
    hook = _load()
    calls = []
    hook.main([], environ={"MLFRAME_SKIP_KERNEL_TUNING_HOOK": value, "MLFRAME_PREPUSH_TUNE": "0"}, run=lambda argv: calls.append(argv) or 0)
    assert (calls == []) is skipped
