"""Unit tests for scripts/ci_failure_digest.py using inline log fixtures and a fake gh runner."""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci_failure_digest.py"


def _load():
    """Import the script module from its file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("ci_failure_digest", _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_failure_digest"] = module
    spec.loader.exec_module(module)
    return module


cfd = _load()

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
TS = "2026-10-08T12:00:01.1234567Z "
PYTEST_LOG = "\n".join(
    TS + s
    for s in [
        "=================================== FAILURES ===================================",
        "_______________________ TestFoo.test_bar[a-1] _______________________",
        "    def test_bar(self):",
        ">       assert f() == 3",
        "E       AssertionError: assert 2 == 3",
        "E        +  where 2 = f()",
        "E       extra1",
        "E       extra2",
        "_______________________ test_timeout_case _______________________",
        "+++++++++++++++++++++++++++++++++++ Timeout ++++++++++++++++++++++++++++++++++++",
        "E       Failed: Timeout (>900.0s) from pytest-timeout.",
        "=========================== short test summary info ============================",
        "FAILED tests/a/test_x.py::TestFoo::test_bar[a-1] - AssertionError: assert 2 == 3",
        "FAILED tests/a/test_x.py::test_timeout_case - Failed: Timeout (>900.0s) from pytest-timeout.",
        "ERROR tests/b/test_y.py - ModuleNotFoundError: No module named 'torch'",
        "ModuleNotFoundError: No module named 'torch'",
        'src/mlframe/m.py:12: error: Returning Any from function declared to return "int"  [no-any-return]',
        "src/mlframe/m.py:30: error: Incompatible types  [assignment]",
        "INTERNALERROR> Traceback",
        "##[error]The operation was canceled.",
    ]
)


def test_strip_line_removes_timestamp_and_ansi():
    """Timestamp prefix, ANSI colour codes and newline are removed."""
    assert cfd.strip_line(TS + "\x1b[31mFAILED x\x1b[0m\r\n") == "FAILED x"


def test_parse_log_extracts_failed_ids_and_assertion_lines():
    """FAILED/ERROR node ids are found and the first E lines of the matching section are attached, capped at three."""
    f = cfd.parse_log(PYTEST_LOG)
    node = "tests/a/test_x.py::TestFoo::test_bar[a-1]"
    assert set(f.tests) == {node, "tests/a/test_x.py::test_timeout_case", "tests/b/test_y.py"}
    assert f.tests[node][0] == "AssertionError: assert 2 == 3"
    assert "+  where 2 = f()" in f.tests[node]
    assert len(f.tests[node]) <= 3
    assert "extra2" not in f.tests[node]


def test_parse_log_mypy_grouped_by_file():
    """mypy errors keep file:line, message and the error-code suffix, grouped per file."""
    f = cfd.parse_log(PYTEST_LOG)
    assert list(f.mypy) == ["src/mlframe/m.py"]
    assert f.mypy["src/mlframe/m.py"][0].startswith("src/mlframe/m.py:12 Returning Any")
    assert f.mypy["src/mlframe/m.py"][0].endswith("[no-any-return]")
    assert len(f.mypy["src/mlframe/m.py"]) == 2


def test_parse_log_infra_classes():
    """Missing module, pytest-timeout, INTERNALERROR and cancellation are detected as infrastructure noise."""
    f = cfd.parse_log(PYTEST_LOG)
    assert f.infra == {"ModuleNotFoundError: torch", "pytest-timeout", "INTERNALERROR", "cancelled"}


def test_parse_log_clean_text_has_no_findings():
    """A log without failures yields nothing."""
    f = cfd.parse_log(TS + "all good\n" + TS + "5 passed in 1.0s")
    assert not f.tests and not f.mypy and not f.infra


def _grp(wf: str, kind: str, key: str, job: str = "j"):
    """Build a one-job group."""
    return cfd.Group(wf, kind, key, jobs=[job])


def test_classify_new_persisting_fixed():
    """Keys only in the current run are NEW, in both PERSISTING, only in the previous FIXED."""
    cur = {("test", "a"): _grp("W", "test", "a"), ("test", "b"): _grp("W", "test", "b")}
    prev = {("test", "b"): _grp("W", "test", "b"), ("mypy", "m.py"): _grp("W", "mypy", "m.py")}
    status = {g.key: g.status for g in cfd.classify(cur, prev)}
    assert status == {"a": "NEW", "b": "PERSISTING", "m.py": "FIXED"}


def test_merge_groups_unions_jobs_and_details():
    """The same test failing in two matrix jobs becomes one group listing both jobs."""
    into = {("test", "a"): cfd.Group("W", "test", "a", jobs=["j1"], details=["d1"])}
    cfd.merge_groups(into, {("test", "a"): cfd.Group("W", "test", "a", jobs=["j2"], details=["d1", "d2"])})
    assert into[("test", "a")].jobs == ["j1", "j2"]
    assert into[("test", "a")].details == ["d1", "d2"]


def test_sort_and_counts_and_markdown_order():
    """NEW groups render before PERSISTING before FIXED and per-workflow counts are right."""
    groups = [
        cfd.Group("W1", "test", "z", status="FIXED", jobs=["j"]),
        cfd.Group("W1", "test", "p", status="PERSISTING", jobs=["j"]),
        cfd.Group("W2", "mypy", "m.py", status="NEW", jobs=["j"], details=["m.py:1 boom"]),
    ]
    assert [g.status for g in cfd.sort_groups(groups)] == ["NEW", "PERSISTING", "FIXED"]
    assert cfd.count_by_workflow(groups)["W1"] == {"NEW": 0, "PERSISTING": 1, "FIXED": 1}
    md = cfd.render_markdown(groups, "master", "o/r", ["note"])
    assert md.index("## NEW") < md.index("## PERSISTING") < md.index("## FIXED")
    assert "| W2 | 1 | 0 | 0 |" in md
    assert "m.py:1 boom" in md and "- note" in md


def test_render_markdown_empty():
    """No groups still yields a valid digest."""
    assert "(no failures)" in cfd.render_markdown([], "master", "o/r")


def test_retry_fails_twice_then_succeeds():
    """Two failures followed by success returns the output after exactly three calls with two sleeps."""
    calls: List[int] = []
    sleeps: List[float] = []

    def once(args: Sequence[str]) -> str:
        """Fail twice, then succeed."""
        calls.append(1)
        if len(calls) < 3:
            raise cfd.GhError("error connecting to api.github.com")
        return "ok"

    assert cfd.retry_gh(["api", "x"], once, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2


def test_retry_gives_up_after_eight_attempts():
    """Eight consecutive failures raise GhError mentioning the attempt count."""
    calls: List[int] = []

    def once(args: Sequence[str]) -> str:
        """Always fail."""
        calls.append(1)
        raise cfd.GhError("TLS handshake timeout")

    with pytest.raises(cfd.GhError, match="8 attempts"):
        cfd.retry_gh(["api", "x"], once, sleep=lambda s: None)
    assert len(calls) == 8


def _fake_gh(responses: Dict[str, str]) -> Callable[[Sequence[str]], str]:
    """Build a run_gh fake that answers by the first response key contained in the joined args."""

    def run(args: Sequence[str]) -> str:
        """Return the canned response for the command."""
        joined = " ".join(args) + " "
        for key, val in responses.items():
            if key in joined:
                return val
        raise AssertionError(f"unexpected gh call: {joined}")

    return run


def _jobs(*pairs):
    """Build a jobs API payload from (id, name, conclusion) tuples."""
    return json.dumps({"jobs": [{"id": i, "name": n, "conclusion": c} for i, n, c in pairs]})


def _run(run_id: int, conclusion: str, created: str) -> dict:
    """Build a run list entry."""
    return {"databaseId": run_id, "conclusion": conclusion, "headSha": "abc", "createdAt": created}


def test_build_digest_end_to_end_with_fake_gh():
    """Latest vs previous run produce NEW, PERSISTING and FIXED groups; a green workflow and a stale one are ignored."""
    cur_log = TS + "FAILED t/a.py::test_old - boom\n" + TS + "FAILED t/a.py::test_new - boom\n"
    prev_log = TS + "FAILED t/a.py::test_old - boom\n" + TS + "FAILED t/a.py::test_gone - boom\n"
    gh = _fake_gh(
        {
            "workflow list": json.dumps([{"id": 1, "name": "CI"}, {"id": 2, "name": "Green"}, {"id": 3, "name": "Stale"}]),
            "--workflow 1 ": json.dumps([_run(20, "failure", "2026-10-09T01:00:00Z"), _run(10, "failure", "2026-10-08T01:00:00Z")]),
            "--workflow 2 ": json.dumps([_run(30, "success", "2026-10-09T01:00:00Z")]),
            "--workflow 3 ": json.dumps([_run(40, "failure", "2026-01-01T00:00:00Z")]),
            "runs/20/jobs": _jobs((201, "py312", "failure"), (202, "lint", "success")),
            "runs/10/jobs": _jobs((101, "py312", "failure")),
            "jobs/201/logs": cur_log,
            "jobs/101/logs": prev_log,
        }
    )
    groups, notes = cfd.build_digest(gh, "o/r", "master", 2, now=NOW)
    status = {g.key: (g.workflow, g.status, g.jobs) for g in groups}
    assert status == {
        "t/a.py::test_old": ("CI", "PERSISTING", ["py312"]),
        "t/a.py::test_new": ("CI", "NEW", ["py312"]),
        "t/a.py::test_gone": ("CI", "FIXED", ["py312"]),
    }
    assert notes == []


def test_green_latest_marks_previous_failures_fixed():
    """When the latest run succeeded, everything from the failed previous run is FIXED."""
    gh = _fake_gh(
        {
            "workflow list": json.dumps([{"id": 1, "name": "CI"}]),
            "--workflow 1 ": json.dumps([_run(2, "success", "2026-10-09T01:00:00Z"), _run(1, "failure", "2026-10-08T01:00:00Z")]),
            "runs/1/jobs": _jobs((11, "j", "failure")),
            "jobs/11/logs": TS + "src/x.py:3: error: bad  [misc]\n",
        }
    )
    groups, _ = cfd.build_digest(gh, "o/r", "master", 2, now=NOW)
    assert [(g.kind, g.key, g.status) for g in groups] == [("mypy", "src/x.py", "FIXED")]


def test_cancelled_job_becomes_infra_group():
    """A cancelled job in a failed run is reported as the cancelled infrastructure class even with an empty log."""
    gh = _fake_gh(
        {
            "workflow list": json.dumps([{"id": 1, "name": "CI"}]),
            "--workflow 1 ": json.dumps([_run(5, "failure", "2026-10-09T01:00:00Z")]),
            "runs/5/jobs": _jobs((51, "slow", "cancelled")),
            "jobs/51/logs": "",
        }
    )
    groups, notes = cfd.build_digest(gh, "o/r", "master", 2, now=NOW)
    assert [(g.kind, g.key, g.status) for g in groups] == [("infra", "cancelled", "NEW")]
    assert len(notes) == 1


def test_main_writes_markdown_and_json(tmp_path):
    """The CLI writes both outputs using the injected runner."""
    gh = _fake_gh({"workflow list": "[]"})
    md, js = tmp_path / "d.md", tmp_path / "d.json"
    rc = cfd.main(["--branch", "master", "--out", str(md), "--json", str(js)], run_gh=gh)
    assert rc == 0
    assert "(no failures)" in md.read_text(encoding="utf-8")
    assert json.loads(js.read_text(encoding="utf-8")) == []
