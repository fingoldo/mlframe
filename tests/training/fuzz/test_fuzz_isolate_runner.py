"""Smoke / regression tests for the combo-isolated fuzz runner.

These pin the isolation-runner contract WITHOUT running the (slow) fuzz suite:
the native-crash classifier, the process-tree kill (verified against a real
sleeping child + its child), and the driver-side timeout/native_crash row
logging. The full end-to-end isolation behaviour (c0004 clean-in-isolation,
the wall-kill on a heavy combo) is verified manually via
``run_fuzz_10k.py --isolate-combos`` -- see the module docstring there.
"""

from __future__ import annotations

import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys

import orjson
import pytest

import tests.training.run_fuzz_10k as R


def test_classify_native_crash_windows_access_violation():
    """Classify native crash windows access violation."""
    assert R._classify_native_crash(3221225477) is True  # 0xC0000005 unsigned
    assert R._classify_native_crash(-1073741819) is True  # signed two's complement


def test_classify_native_crash_posix_signal():
    """Classify native crash posix signal."""
    assert R._classify_native_crash(-11) is True  # SIGSEGV


def test_classify_native_crash_clean_pytest_exits():
    """Classify native crash clean pytest exits."""
    assert R._classify_native_crash(0) is False  # all passed
    assert R._classify_native_crash(1) is False  # tests failed
    assert R._classify_native_crash(5) is False  # no tests collected


def test_kill_process_tree_terminates_child_and_grandchild():
    """A child that spawns its own grandchild must be fully reaped: the old
    per-seed timeout failed precisely because it left descendants alive."""
    psutil = pytest.importorskip("psutil")
    # Parent python sleeps after spawning a grandchild python that also sleeps.
    code = "import subprocess,sys,time;g=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);print(g.pid,flush=True);time.sleep(120)"
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
    grandchild_pid = int(p.stdout.readline().strip())
    parent = psutil.Process(p.pid)
    assert parent.is_running()
    assert psutil.pid_exists(grandchild_pid)

    grandchild = psutil.Process(grandchild_pid)
    R._kill_process_tree(p.pid)
    p.wait(timeout=15)

    # Blocks until the orphan is gone; raises psutil.TimeoutExpired if the tree-kill left it running.
    grandchild.wait(timeout=15)
    assert not grandchild.is_running(), "grandchild orphaned after tree-kill"
    assert not parent.is_running()


class _StuckProcess:
    """Stand-in for a child that never exits: every bounded wait expires."""

    def __init__(self):
        """Record the timeout each wait is asked for."""
        self.timeouts = []

    def wait(self, timeout=None):
        """Raise TimeoutExpired as a still-running process would under a bounded wait."""
        self.timeouts.append(timeout)
        raise subprocess.TimeoutExpired(cmd="stuck", timeout=timeout)


@pytest.mark.parametrize("bound", [1.0, 3.0])
def test_reap_bounded_waits_exactly_its_bound_then_reports_undead(bound):
    """An undead child makes ``_reap_bounded`` wait out exactly the timeout it was given and report False,
    so the driver can orphan + continue rather than wedge on an OS-undead access-violation'd process."""
    stuck = _StuckProcess()
    assert R._reap_bounded(stuck, bound) is False
    assert stuck.timeouts == [bound]


def test_reap_bounded_reports_live_process_as_not_reaped():
    """A real still-running child is reported as not reaped within a short bound, and the driver can still kill it."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
    try:
        assert R._reap_bounded(p, 0.5) is False
        assert p.poll() is None
    finally:
        R._kill_process_tree(p.pid)
        R._reap_bounded(p, 10)


def test_reap_bounded_returns_true_for_exited_process():
    """Reap bounded returns true for exited process."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
    assert R._reap_bounded(p, 10) is True


@pytest.mark.hang_guard
def test_run_one_combo_does_not_wedge_on_child_holding_pipe_open_after_kill(monkeypatch):
    """Regression for the seed-0 AV wedge: an undead child whose stdout pipe
    stays open after the tree-kill (here simulated by a grandchild that inherits
    the pipe and outlives the parent) must NOT hang the driver's pipe drain. The
    background-drain + bounded-reap design must return within a bounded wall time
    even though the inherited pipe never closes.

    We monkeypatch the spawned command so no real pytest/fuzz suite runs: the
    fake child prints one line, spawns a grandchild that holds stdout open and
    sleeps, then itself hangs -- mimicking an AV'd parent with a live orphan."""
    pytest.importorskip("psutil")

    # Child: spawn a grandchild that INHERITS stdout (so the pipe stays open even
    # after the parent dies / is killed), print a marker, then hang forever.
    child_code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"  # inherits stdout
        "print('FAKE_CHILD_MARKER',flush=True);"
        "time.sleep(120)"
    )
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        """Fake popen."""
        return real_popen([sys.executable, "-c", child_code], **kwargs)

    monkeypatch.setattr(R.subprocess, "Popen", fake_popen)

    per_combo_timeout_s = 2
    _rc, tail, timed_out = R._run_one_combo(seed=0, short_id="cFAKE", per_combo_timeout_s=per_combo_timeout_s)

    # Returning at all proves the driver was not wedged by the inherited pipe; the watchdog's kill path is what sets timed_out.
    assert timed_out is True, "wall-timeout combo must be flagged timed_out"
    assert "FAKE_CHILD_MARKER" in tail, "drain thread should have captured child output"


def test_log_driver_row_writes_schema(tmp_path, monkeypatch):
    """Driver-logged timeout/native_crash rows reuse the JSONL schema and carry
    the combo fields + master_seed + error_class + isolated marker."""
    from tests.training._fuzz_combo import enumerate_combos

    combo = enumerate_combos(target=5, master_seed=4)[0]

    log = tmp_path / "_fuzz_results.jsonl"
    monkeypatch.setattr(R, "RESULTS_LOG", log)
    R._log_driver_row(combo, seed=4, outcome="timeout", dur=12.3, error_class="WallTimeout", error_summary="killed", tail="line1\nline2")

    rows = [orjson.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["outcome"] == "timeout"
    assert row["master_seed"] == 4
    assert row["error_class"] == "WallTimeout"
    assert row["short_id"] == combo.short_id()
    assert row["extra"]["isolated"] is True
    assert row["extra"]["driver_logged"] is True
