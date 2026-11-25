"""The stage tools of the profiling directory: the dumps of the MainThread laid over the windows of the stage ledger, and two ledgers compared."""

import json

from mlframe.feature_selection._benchmarks.profiling import ledger_compare as lc
from mlframe.feature_selection._benchmarks.profiling import stage_dump_profile as sp


def _ledger(path, rows):
    """Write ledger rows ``(name, t0, t1)`` as the harness does."""
    with open(path, "w", encoding="utf-8") as handle:
        for name, t0, t1 in rows:
            handle.write(json.dumps({"name": name, "t0": t0, "t1": t1, "thread_name": "MainThread"}) + "\n")


def test_dumps_inside_the_windows_of_a_stage_are_ranked_by_their_innermost_frame(tmp_path):
    """Only the dumps inside the stage's windows count; the innermost frame and the chain of the three innermost frames are reported."""
    ledger = tmp_path / "led.ledger.jsonl"
    _ledger(ledger, [("stage:a", 100.0, 110.0), ("stage:b", 110.0, 130.0), ("stage:a", 140.0, 150.0)])
    dump = tmp_path / "dump.txt"
    dump.write_text(
        "101.0 wait (pkg\\x.py:10) > run (pkg\\y.py:20) > fit (pkg\\z.py:30) > \n"
        "103.0 wait (pkg\\x.py:10) > run (pkg\\y.py:20) > fit (pkg\\z.py:30) > \n"
        "120.0 other (pkg\\q.py:5) > fit (pkg\\z.py:30) > \n"
        "145.0 sort (pkg\\s.py:7) > run (pkg\\y.py:20) > \n"
        "151.0 ended\n",
        encoding="utf-8",
    )
    windows = sp.read_windows(str(ledger), "stage:a")
    assert windows == [(100.0, 110.0), (140.0, 150.0)]
    report = sp.profile(sp.read_dumps(str(dump)), windows, 5)
    assert report.startswith("3 dumps")
    assert "  2  wait (x.py:10)" in report and "  1  sort (s.py:7)" in report and "other" not in report
    assert "wait (x.py:10) < run (y.py:20) < fit (z.py:30)" in report


def test_a_stretch_between_stages_is_read_in_seconds_since_the_first_ledger_row(tmp_path, capsys):
    """``--between`` counts seconds from the start of the ledger, ``--whole`` takes every dump."""
    ledger = tmp_path / "led.ledger.jsonl"
    _ledger(ledger, [("stage:a", 1000.0, 1010.0)])
    dump = tmp_path / "dump.txt"
    dump.write_text("1012.0 idle (m.py:1) > \n1030.0 busy (m.py:2) > \n", encoding="utf-8")
    assert sp.main([str(ledger), str(dump), "--between", "10", "20"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("1 dumps") and "idle (m.py:1)" in out and "busy" not in out
    assert sp.main([str(ledger), str(dump), "--whole"]) == 0
    assert capsys.readouterr().out.startswith("2 dumps")


def test_two_ledgers_are_compared_by_the_stage_that_moved_the_most(tmp_path):
    """Windows of one name are summed; the report is ordered by the size of the change."""
    old, new = tmp_path / "old.ledger.jsonl", tmp_path / "new.ledger.jsonl"
    _ledger(old, [("stage:x", 0.0, 10.0), ("stage:x", 20.0, 25.0), ("stage:y", 0.0, 4.0)])
    _ledger(new, [("stage:x", 0.0, 6.0), ("stage:y", 0.0, 4.5), ("stage:z", 0.0, 2.0)])
    lines = lc.compare(lc.stage_walls(str(old)), lc.stage_walls(str(new)), 3)
    assert [line.split()[1] for line in lines] == ["stage:x", "stage:z", "stage:y"]
    assert lines[0].split()[0] == "-9.0" and "15.0 ->     6.0" in lines[0]
