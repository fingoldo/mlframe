"""Unit tests for `training.phases` registry helpers.

Covers the production phase registry (`reset_phase_registry`, `record_phase`,
`phase_snapshot`, `phase_ram_snapshot`, `format_phase_summary`, `phase`).
Pre-existing `test_phases_phases.py` exercises a different `phases` module
(validate_input_columns / dataset_reuse_cache_key); the registry helpers had
no direct test coverage.
"""

from __future__ import annotations

import logging
import threading

import pytest

from mlframe.training.phases import (
    format_phase_summary,
    phase,
    phase_ram_snapshot,
    phase_snapshot,
    record_phase,
    reset_phase_registry,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    # Snapshot+restore semantics: clear before each test, clear after so
    # parallel test files do not see leaked phase rows.
    reset_phase_registry()
    yield
    reset_phase_registry()


@pytest.fixture
def fake_phase_timer(monkeypatch):
    """Replace the phase registry's wall-clock timer with a strictly-increasing fake clock.

    The ``phase`` context manager brackets each block with ``_timer()`` at entry and exit; a fake clock that
    advances 1.0 per read makes recorded durations deterministic (one tick per ``_timer`` call) instead of
    relying on flaky ``time.sleep`` to coax a measurable elapsed wall-time."""
    import mlframe.training.phases as phases_mod

    counter = {"t": 0.0}

    def _tick() -> float:
        counter["t"] += 1.0
        return counter["t"]

    monkeypatch.setattr(phases_mod, "_timer", _tick)
    return counter


def test_reset_phase_registry_clears_state():
    record_phase("p1", 0.5)
    record_phase("p2", 1.0)
    assert len(phase_snapshot()) == 2
    reset_phase_registry()
    assert phase_snapshot() == []


def test_record_phase_aggregates_by_name():
    record_phase("io", 0.5)
    record_phase("io", 1.5)
    record_phase("io", 2.0)
    snap = phase_snapshot()
    assert len(snap) == 1
    name, total, count = snap[0]
    assert name == "io"
    assert total == pytest.approx(4.0)
    assert count == 3


def test_phase_snapshot_sorted_by_total_desc():
    record_phase("short", 0.1)
    record_phase("long", 10.0)
    record_phase("medium", 2.0)
    snap = phase_snapshot()
    totals = [row[1] for row in snap]
    assert totals == sorted(totals, reverse=True)
    assert snap[0][0] == "long"
    assert snap[-1][0] == "short"


def test_record_phase_with_ram_delta_accumulates():
    record_phase("alloc", 0.1, ram_delta_gb=0.5)
    record_phase("alloc", 0.1, ram_delta_gb=0.7)
    record_phase("noalloc", 0.1, ram_delta_gb=0.0)
    ram = phase_ram_snapshot()
    # noalloc never recorded (delta==0 short-circuits in record).
    assert "alloc" in ram
    assert ram["alloc"] == pytest.approx(1.2)
    assert "noalloc" not in ram


def test_phase_ram_snapshot_sorted_by_abs_magnitude():
    record_phase("a", 0.1, ram_delta_gb=-2.0)
    record_phase("b", 0.1, ram_delta_gb=0.5)
    record_phase("c", 0.1, ram_delta_gb=3.0)
    ram = phase_ram_snapshot()
    keys = list(ram.keys())
    # |c|=3 > |a|=2 > |b|=0.5
    assert keys == ["c", "a", "b"]


def test_format_phase_summary_empty_registry():
    out = format_phase_summary()
    assert "no timings recorded" in out


def test_format_phase_summary_truncates_to_top_n():
    for i in range(20):
        record_phase(f"phase_{i:02d}", float(i + 1))
    out = format_phase_summary(top=5)
    # 5 data rows + 2 (header + sep) = 7 lines total.
    assert len(out.splitlines()) == 7
    # Top entry must be the largest total.
    assert "phase_19" in out


def test_format_phase_summary_renders_phase_names_and_totals():
    record_phase("fit_cb", 12.34, ram_delta_gb=0.0)
    record_phase("predict", 5.67, ram_delta_gb=0.0)
    out = format_phase_summary()
    assert "fit_cb" in out
    assert "predict" in out
    assert "12.34" in out
    assert "5.67" in out


def test_phase_contextmanager_records_duration(fake_phase_timer):
    with phase("ctx_test"):
        pass
    snap = phase_snapshot()
    assert len(snap) == 1
    name, total, count = snap[0]
    assert name == "ctx_test"
    assert total >= 1.0  # one fake-clock tick between enter and exit
    assert count == 1


def test_phase_contextmanager_records_on_exception(fake_phase_timer):
    # On an in-block exception, the phase must STILL be recorded with a
    # finite duration so the operator can see which phase crashed.
    with pytest.raises(ValueError):
        with phase("crash_test"):
            raise ValueError("boom")
    snap = phase_snapshot()
    assert len(snap) == 1
    assert snap[0][0] == "crash_test"
    assert snap[0][1] >= 0.0
    assert snap[0][2] == 1


def test_phase_nested_context_managers_record_separately(fake_phase_timer):
    with phase("outer"):
        with phase("inner"):
            pass
    snap = phase_snapshot()
    names = {row[0] for row in snap}
    assert names == {"outer", "inner"}
    # Outer total includes inner.
    outer_total = next(row[1] for row in snap if row[0] == "outer")
    inner_total = next(row[1] for row in snap if row[0] == "inner")
    assert outer_total >= inner_total


def test_phase_emits_start_and_done_log_lines(caplog):
    # Logs at DEBUG by default; set caplog to capture both START and DONE.
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.phases"):
        with phase("logged", model="cb"):
            pass
    messages = [r.message for r in caplog.records]
    assert any("logged START" in m for m in messages)
    assert any("logged DONE" in m for m in messages)
    # context kwargs travel through the log line.
    assert any("model=cb" in m for m in messages)


def test_phase_emits_warning_on_exception(caplog):
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.phases"):
        with pytest.raises(RuntimeError):
            with phase("warn_test"):
                raise RuntimeError("explode")
    # RAISED line must be at WARNING.
    raised_records = [r for r in caplog.records if "RAISED" in r.message]
    assert len(raised_records) >= 1
    assert raised_records[0].levelno == logging.WARNING


def test_phase_registry_thread_safety():
    # Two threads both writing distinct phases must not lose rows.
    def worker(thread_id: int):
        for i in range(50):
            record_phase(f"t{thread_id}_p{i}", 0.001)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = phase_snapshot()
    # 4 threads x 50 distinct names = 200 rows.
    assert len(snap) == 200


def test_phase_registry_thread_safety_same_name():
    # 4 threads all writing the SAME name must accumulate without lost
    # increments (lock around totals + counts).
    def worker():
        for _ in range(100):
            record_phase("shared", 0.001)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = phase_snapshot()
    assert len(snap) == 1
    name, total, count = snap[0]
    assert count == 400
    assert total == pytest.approx(0.4)
