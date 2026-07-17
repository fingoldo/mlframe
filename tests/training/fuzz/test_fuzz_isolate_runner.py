"""Smoke / regression tests for the combo-isolated fuzz runner.

These pin the isolation-runner contract WITHOUT running the (slow) fuzz suite:
the native-crash classifier, the process-tree kill (verified against a real
sleeping child + its child), and the driver-side timeout/native_crash row
logging. The full end-to-end isolation behaviour (c0004 clean-in-isolation,
the wall-kill on a heavy combo) is verified manually via
``run_fuzz_10k.py --isolate-combos`` -- see the module docstring there.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import orjson
import pytest

import tests.training.run_fuzz_10k as R


def test_classify_native_crash_windows_access_violation():
    assert R._classify_native_crash(3221225477) is True  # 0xC0000005 unsigned
    assert R._classify_native_crash(-1073741819) is True  # signed two's complement


def test_classify_native_crash_posix_signal():
    assert R._classify_native_crash(-11) is True  # SIGSEGV


def test_classify_native_crash_clean_pytest_exits():
    assert R._classify_native_crash(0) is False  # all passed
    assert R._classify_native_crash(1) is False  # tests failed
    assert R._classify_native_crash(5) is False  # no tests collected


def test_kill_process_tree_terminates_child_and_grandchild():
    """A child that spawns its own grandchild must be fully reaped: the old
    per-seed timeout failed precisely because it left descendants alive."""
    psutil = pytest.importorskip("psutil")
    # Parent python sleeps after spawning a grandchild python that also sleeps.
    code = "import subprocess,sys,time;g=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);print(g.pid,flush=True);time.sleep(120)"
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    grandchild_pid = int(p.stdout.readline().strip())
    parent = psutil.Process(p.pid)
    assert parent.is_running()
    assert psutil.pid_exists(grandchild_pid)

    R._kill_process_tree(p.pid)
    p.wait(timeout=15)

    deadline = time.time() + 10
    while time.time() < deadline and psutil.pid_exists(grandchild_pid):
        time.sleep(0.1)
    assert not psutil.pid_exists(grandchild_pid), "grandchild orphaned after tree-kill"
    assert not parent.is_running()


def test_reap_bounded_returns_false_for_undead_process_within_timeout():
    """A child that ignores kill (simulated by a long sleep we never kill) must
    NOT block ``_reap_bounded`` past its timeout -- it returns False promptly so
    the driver can orphan + continue rather than wedge waiting for an OS-undead
    access-violation'd process to reap."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        t0 = time.time()
        alive_reaped = R._reap_bounded(p, 1.0)
        elapsed = time.time() - t0
        assert alive_reaped is False, "still-running process must report not-reaped"
        assert elapsed < 5.0, f"_reap_bounded blocked {elapsed:.1f}s past its 1s bound"
    finally:
        R._kill_process_tree(p.pid)
        R._reap_bounded(p, 10)


def test_reap_bounded_returns_true_for_exited_process():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    assert R._reap_bounded(p, 10) is True


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
        return real_popen([sys.executable, "-c", child_code], **kwargs)

    monkeypatch.setattr(R.subprocess, "Popen", fake_popen)

    t0 = time.time()
    rc, tail, timed_out = R._run_one_combo(seed=0, short_id="cFAKE", per_combo_timeout_s=2)
    elapsed = time.time() - t0

    # The whole call must return shortly after the 2s wall-timeout + bounded
    # reaps (2 + 30 ceiling is only hit if the process won't die; the watchdog
    # kills at 2s so the parent dies and reap returns fast). Generous ceiling.
    assert elapsed < 45.0, f"_run_one_combo wedged {elapsed:.1f}s on pipe-holding child"
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
