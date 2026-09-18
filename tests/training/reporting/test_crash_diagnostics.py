"""Crash diagnostics: an abrupt process death must leave evidence in the log / a persistent file."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

import pytest

import mlframe.training.crash_diagnostics as cd


def test_sys_excepthook_routes_traceback_to_logger(caplog):
    try:
        raise ValueError("boom-main")
    except ValueError:
        et, ev, tb = sys.exc_info()
    seen = []
    with caplog.at_level(logging.CRITICAL, logger=cd.logger.name):
        cd._sys_excepthook(et, ev, tb, _prev=lambda *a: seen.append(a))
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]
    assert msgs and "boom-main" in msgs[0] and "Traceback" in msgs[0]
    assert seen, "previous hook must still be chained"
    cd._UNCAUGHT["exc"] = None


def test_thread_excepthook_routes_to_logger(caplog, monkeypatch):
    prev_seen = []
    monkeypatch.setattr(threading, "excepthook", lambda a: cd._thread_excepthook(a, _prev=lambda x: prev_seen.append(x)))

    def _worker():
        raise RuntimeError("boom-thread")

    with caplog.at_level(logging.CRITICAL, logger=cd.logger.name):
        t = threading.Thread(target=_worker, name="worker-x")
        t.start()
        t.join()
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]
    assert any("worker-x" in m and "boom-thread" in m for m in msgs)
    assert prev_seen


def test_atexit_line_normal_and_after_exception(caplog):
    cd._UNCAUGHT["exc"] = None
    with caplog.at_level(logging.INFO, logger=cd.logger.name):
        cd._atexit_handler()
    assert any("exiting normally" in r.getMessage() for r in caplog.records)
    caplog.clear()
    cd._UNCAUGHT["exc"] = "ValueError: x"
    with caplog.at_level(logging.INFO, logger=cd.logger.name):
        cd._atexit_handler()
    assert any("uncaught exception" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    cd._UNCAUGHT["exc"] = None


def test_heartbeat_emits_and_stops(caplog):
    with caplog.at_level(logging.INFO, logger=cd.logger.name):
        hb = cd.Heartbeat(0.05, line_fn=lambda: "[heartbeat] test-line").start()
        deadline = time.time() + 5
        while hb.beats < 2 and time.time() < deadline:
            time.sleep(0.02)
        hb.stop()
    assert hb.beats >= 2
    assert not hb.alive
    assert any("test-line" in r.getMessage() for r in caplog.records)


def test_heartbeat_line_reports_phase_from_other_thread():
    from mlframe.training.phases import phase

    entered, release = threading.Event(), threading.Event()

    def _worker():
        with phase("unit_test_phase_xyz"):
            entered.set()
            release.wait(5)

    t = threading.Thread(target=_worker, name="phase-worker")
    t.start()
    try:
        assert entered.wait(5)
        line = cd.heartbeat_line()
    finally:
        release.set()
        t.join()
    assert "unit_test_phase_xyz" in line
    assert "rss=" in line or "mem=n/a" in line


def test_heartbeat_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MLFRAME_CRASH_HEARTBEAT_S", "0")
    monkeypatch.setattr(cd, "_HEARTBEAT", None)
    assert cd.start_heartbeat() is None


def test_faulthandler_file_created(tmp_path, monkeypatch):
    import faulthandler

    monkeypatch.setattr(cd, "_FAULT_FILE", None)
    monkeypatch.setattr(cd, "_FAULT_PATH", None)
    path = cd.open_faulthandler_file(str(tmp_path))
    try:
        assert path and os.path.isfile(path) and os.path.dirname(path) == str(tmp_path)
        faulthandler.dump_traceback(file=cd._FAULT_FILE)
        cd._FAULT_FILE.flush()
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "pid=" in content and "test_faulthandler_file_created" in content
    finally:
        faulthandler.enable()  # back to stderr before the file is closed
        cd._FAULT_FILE.close()


def test_resolve_crash_dir_prefers_log_file_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_CRASH_LOG_DIR", raising=False)
    h = logging.FileHandler(str(tmp_path / "run.log"), delay=True)
    root = logging.getLogger()
    root.addHandler(h)
    try:
        assert cd.resolve_crash_dir() == str(tmp_path)
    finally:
        root.removeHandler(h)
        h.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows commit accounting")
def test_windows_commit_status_sane():
    cs = cd.windows_commit_status()
    assert cs and cs["commit_limit_gb"] >= cs["phys_total_gb"] * 0.5 and cs["commit_avail_gb"] > 0
