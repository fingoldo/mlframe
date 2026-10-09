"""The pre-push kernel-tuning step: tunes what is stale by default, blocks on a failed sweep, can be downgraded to a report or skipped."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepush_kernel_tuning.py"


@pytest.fixture(scope="module")
def script():
    """The script loaded by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("prepush_kernel_tuning", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_with(script, env):
    """Run main() with a recording stand-in for the CLI; returns (exit code, argv the CLI got or None)."""
    calls = []

    def fake_cli(argv):
        """Record the CLI call and succeed."""
        calls.append(list(argv))
        return 0

    code = script.main([], environ=env, run=fake_cli)
    return code, (calls[0] if calls else None)


def test_by_default_it_tunes_what_is_stale_within_a_time_limit(script):
    """The default does the work (a warning nobody reads fixes nothing): ensure, nothing on a host without CUDA, a 20-minute budget."""
    code, argv = _run_with(script, {})
    assert code == 0
    assert argv == ["ensure", "--if-cuda", "--max-minutes", "20"]
    _, argv = _run_with(script, {"MLFRAME_PREPUSH_TUNE_MINUTES": "5"})
    assert argv[-2:] == ["--max-minutes", "5"]


def test_a_failed_sweep_blocks_the_push_but_a_spent_budget_does_not(script):
    """A kernel that cannot be tuned is a kernel bug and stops the push (exit 2); running out of time only means the rest is tuned next time (exit 3 -> 0)."""
    assert script.main([], environ={}, run=lambda argv: 2) == 2
    assert script.main([], environ={}, run=lambda argv: 3) == 0
    assert script.main([], environ={}, run=lambda argv: 0) == 0


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "OFF"])
def test_tuning_can_be_downgraded_to_a_report(script, value):
    """MLFRAME_PREPUSH_TUNE=0 (or false/off/no) only reports and can never fail."""
    code, argv = _run_with(script, {"MLFRAME_PREPUSH_TUNE": value})
    assert code == 0 and argv == ["ensure", "--if-cuda", "--check", "--advisory"]


def test_the_step_can_be_skipped(script):
    """MLFRAME_SKIP_KERNEL_TUNING_HOOK=1 makes it do nothing at all."""
    code, argv = _run_with(script, {"MLFRAME_SKIP_KERNEL_TUNING_HOOK": "1"})
    assert code == 0 and argv is None


def test_the_hook_is_registered_for_pre_push_only():
    """.pre-commit-config.yaml wires the script as a pre-push hook scoped to kernel code, not to every commit."""
    import yaml

    config = yaml.safe_load((Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [h for repo in config["repos"] for h in repo.get("hooks", []) if h["id"] == "kernel-tuning-ensure"]
    assert len(hooks) == 1
    hook = hooks[0]
    assert hook["stages"] == ["pre-push"]
    assert hook["entry"] == "python scripts/prepush_kernel_tuning.py"
    assert hook["pass_filenames"] is False and "src/mlframe" in hook["files"]
