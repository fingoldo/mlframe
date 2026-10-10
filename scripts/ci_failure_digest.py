"""Summarise red GitHub Actions runs of a branch into a NEW / PERSISTING / FIXED Markdown digest.

Usage: python scripts/ci_failure_digest.py --branch master [--repo fingoldo/mlframe] [--days 2] [--out digest.md] [--json digest.json]

Parsing, grouping, classification and rendering are pure functions; all network access goes through an injectable ``run_gh`` callable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

RunGh = Callable[[Sequence[str]], str]
GroupKey = Tuple[str, str]  # (kind, identifier); kind is "test", "mypy" or "infra"

STATUS_ORDER = ("NEW", "PERSISTING", "FIXED")
GH_RETRIES = 8
GH_TIMEOUT_S = 180

_TS_RE = re.compile(r"^﻿?\d{4}-\d\d-\d\dT[\d:.]+Z ?")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_RESULT_RE = re.compile(r"^(FAILED|ERROR) (\S+?)(?: - (.*))?$")
_MYPY_RE = re.compile(r"^(?P<file>[^\s:]+\.pyi?):(?P<line>\d+)(?::\d+)?: error: (?P<msg>.*)$")
_SECTION_RE = re.compile(r"^_{3,} (.+?) _{3,}$")
_NOMODULE_RE = re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")
_TIMEOUT_RE = re.compile(r"Timeout \(>[\d.]+s\) from pytest-timeout|\+{3,} Timeout \+{3,}")
_CANCEL_RE = re.compile(r"The operation was cancell?ed|##\[error\]The run was cancell?ed", re.IGNORECASE)
MAX_E_LINES = 3


class GhNotFound(RuntimeError):
    """A 404 from the GitHub API: retrying cannot help and the caller decides whether the missing object matters."""


class GhError(RuntimeError):
    """Raised when a gh invocation keeps failing after all retries."""


@dataclass
class LogFindings:
    """Everything extracted from one job log."""

    tests: Dict[str, List[str]] = field(default_factory=dict)
    mypy: Dict[str, List[str]] = field(default_factory=dict)
    infra: Set[str] = field(default_factory=set)


@dataclass
class Group:
    """One failure group: a test id, a mypy file or an infrastructure class within one workflow."""

    workflow: str
    kind: str
    key: str
    status: str = "NEW"
    jobs: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)


def strip_line(line: str) -> str:
    """Remove the Actions timestamp prefix, ANSI escapes and the trailing newline."""
    return _ANSI_RE.sub("", _TS_RE.sub("", line.rstrip("\r\n")))


def parse_log(text: str) -> LogFindings:
    """Extract failed pytest ids, mypy errors, first assertion lines per failed test and infrastructure noise from a job log."""
    found = LogFindings()
    section: Optional[str] = None
    section_e: Dict[str, List[str]] = {}
    for raw in text.splitlines():
        line = strip_line(raw)
        sec = _SECTION_RE.match(line)
        if sec:
            section = sec.group(1)
            continue
        if line.startswith("E ") and section is not None:
            bucket = section_e.setdefault(section, [])
            if len(bucket) < MAX_E_LINES and line[1:].strip():
                bucket.append(line[1:].strip())
            continue
        res = _RESULT_RE.match(line)
        if res:
            node = res.group(2)
            found.tests.setdefault(node, [])
            if res.group(3) and res.group(3) not in found.tests[node]:
                found.tests[node].append(res.group(3)[:200])
            continue
        my = _MYPY_RE.match(line)
        if my:
            found.mypy.setdefault(my.group("file"), []).append(f"{my.group('file')}:{my.group('line')} {my.group('msg')}")
            continue
        for mod in _NOMODULE_RE.findall(line):
            found.infra.add(f"ModuleNotFoundError: {mod}")
        if _TIMEOUT_RE.search(line):
            found.infra.add("pytest-timeout")
        if "INTERNALERROR" in line:
            found.infra.add("INTERNALERROR")
        if _CANCEL_RE.search(line):
            found.infra.add("cancelled")
    for node, details in found.tests.items():
        tail = node.split("::")[-1]
        for name, e_lines in section_e.items():
            if name == tail or name.endswith("." + tail) or tail in name:
                details[:] = (e_lines + [d for d in details if d not in e_lines])[:MAX_E_LINES]
                break
    return found


def findings_to_groups(workflow: str, job_name: str, findings: LogFindings) -> Dict[GroupKey, Group]:
    """Turn one job's findings into groups keyed by (kind, identifier)."""
    out: Dict[GroupKey, Group] = {}
    for node, details in findings.tests.items():
        out[("test", node)] = Group(workflow, "test", node, jobs=[job_name], details=list(details))
    for path, lines in findings.mypy.items():
        out[("mypy", path)] = Group(workflow, "mypy", path, jobs=[job_name], details=lines[:MAX_E_LINES])
    for cls in sorted(findings.infra):
        out[("infra", cls)] = Group(workflow, "infra", cls, jobs=[job_name])
    return out


def merge_groups(into: Dict[GroupKey, Group], more: Dict[GroupKey, Group]) -> None:
    """Merge job-level groups into a run-level mapping, unioning jobs and details."""
    for key, grp in more.items():
        cur = into.get(key)
        if cur is None:
            into[key] = grp
            continue
        for job in grp.jobs:
            if job not in cur.jobs:
                cur.jobs.append(job)
        for det in grp.details:
            if det not in cur.details and len(cur.details) < MAX_E_LINES:
                cur.details.append(det)


def classify(current: Dict[GroupKey, Group], previous: Dict[GroupKey, Group]) -> List[Group]:
    """Label groups NEW (absent before), PERSISTING (in both) or FIXED (only in the previous run)."""
    result: List[Group] = []
    for key, grp in current.items():
        grp.status = "PERSISTING" if key in previous else "NEW"
        result.append(grp)
    for key, grp in previous.items():
        if key not in current:
            grp.status = "FIXED"
            result.append(grp)
    return result


def sort_groups(groups: Sequence[Group]) -> List[Group]:
    """Sort NEW first, then PERSISTING, then FIXED; stable by workflow, kind and key inside a status."""
    return sorted(groups, key=lambda g: (STATUS_ORDER.index(g.status), g.workflow, g.kind, g.key))


def count_by_workflow(groups: Sequence[Group]) -> Dict[str, Dict[str, int]]:
    """Count groups per workflow and status."""
    counts: Dict[str, Dict[str, int]] = {}
    for grp in groups:
        row = counts.setdefault(grp.workflow, {s: 0 for s in STATUS_ORDER})
        row[grp.status] += 1
    return counts


def render_markdown(groups: Sequence[Group], branch: str, repo: str, notes: Sequence[str] = ()) -> str:
    """Render the digest: per-workflow counts, then groups sorted NEW first."""
    ordered = sort_groups(groups)
    lines = [f"# CI failure digest for {repo}@{branch}", ""]
    lines += [f"- {n}" for n in notes] + ([""] if notes else [])
    lines += ["| Workflow | NEW | PERSISTING | FIXED |", "| --- | --- | --- | --- |"]
    for wf, row in sorted(count_by_workflow(ordered).items()):
        lines.append(f"| {wf} | {row['NEW']} | {row['PERSISTING']} | {row['FIXED']} |")
    if not ordered:
        lines.append("| (no failures) | 0 | 0 | 0 |")
    for status in STATUS_ORDER:
        part = [g for g in ordered if g.status == status]
        if not part:
            continue
        lines += ["", f"## {status} ({len(part)})", ""]
        for grp in part:
            jobs = ", ".join(grp.jobs[:4]) + (f" +{len(grp.jobs) - 4} more" if len(grp.jobs) > 4 else "")
            lines.append(f"- [{grp.workflow}] {grp.kind} `{grp.key}` (jobs: {jobs})")
            lines += [f"    - {d}" for d in grp.details[:MAX_E_LINES]]
    return "\n".join(lines) + "\n"


def groups_to_json(groups: Sequence[Group]) -> str:
    """Serialise the sorted groups to JSON."""
    return json.dumps([g.__dict__ for g in sort_groups(groups)], indent=1)


def retry_gh(
    args: Sequence[str],
    exec_once: Callable[[Sequence[str]], str],
    retries: int = GH_RETRIES,
    delay: float = 4.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Call ``exec_once(args)`` up to ``retries`` times, sleeping between failures, and raise GhError when all attempts fail."""
    last = ""
    for attempt in range(retries):
        try:
            return exec_once(args)
        except GhNotFound:
            raise
        except (GhError, subprocess.TimeoutExpired) as exc:  # noqa: PERF203
            last = str(exc)
            if attempt < retries - 1:
                sleep(delay)
    raise GhError(f"gh {' '.join(args[:3])} failed after {retries} attempts: {last[:300]}")


def _exec_gh_once(args: Sequence[str]) -> str:
    """Run one gh command via subprocess (no shell) and return stdout, raising GhError on a non-zero exit."""
    proc = subprocess.run(["gh", *args], capture_output=True, encoding="utf-8", errors="replace", timeout=GH_TIMEOUT_S, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        # Newer gh refuses to print an API response (a job log) that holds terminal escape sequences unless told to; older gh has no such flag.
        if "--allow-escape-sequences" in err and args and args[0] == "api" and "--allow-escape-sequences" not in args:
            return _exec_gh_once([*args, "--allow-escape-sequences"])
        if "HTTP 404" in err:
            raise GhNotFound(err)
        raise GhError(err or f"exit code {proc.returncode}")
    return proc.stdout


def default_run_gh(args: Sequence[str]) -> str:
    """Production ``run_gh``: gh through the retry helper."""
    return retry_gh(args, _exec_gh_once)


def gh_json(run_gh: RunGh, args: Sequence[str]) -> Any:
    """Run gh and parse stdout as JSON (parsed in Python so no --jq quoting is needed)."""
    return json.loads(run_gh(args))


def list_workflows(run_gh: RunGh) -> List[Tuple[int, str]]:
    """Return (id, name) of every workflow in the repo."""
    data = gh_json(run_gh, ["workflow", "list", "--all", "--json", "name,id"])
    return [(int(w["id"]), str(w["name"])) for w in data]


def latest_completed_runs(run_gh: RunGh, repo: str, workflow_id: int, branch: str) -> List[Dict[str, Any]]:
    """Return up to two latest completed runs (newest first) of a workflow on the branch."""
    try:
        data = gh_json(
            run_gh,
            ["run", "list", "-R", repo, "--workflow", str(workflow_id), "--branch", branch, "--status", "completed", "--limit", "2",
             "--json", "databaseId,conclusion,headSha,createdAt"],
        )
    except GhNotFound:
        return []  # a workflow that no longer exists on the default branch has no runs to report
    return list(data)


def collect_run_groups(run_gh: RunGh, repo: str, workflow: str, run: Dict[str, Any]) -> Dict[GroupKey, Group]:
    """Download every failed job log of a run and return its groups; non-failure runs yield no groups."""
    groups: Dict[GroupKey, Group] = {}
    if run.get("conclusion") != "failure":
        return groups
    jobs = gh_json(run_gh, ["api", f"repos/{repo}/actions/runs/{run['databaseId']}/jobs?per_page=100"])
    for job in jobs.get("jobs", []):
        if job.get("conclusion") not in ("failure", "cancelled", "timed_out"):
            continue
        text = run_gh(["api", f"repos/{repo}/actions/jobs/{job['id']}/logs"])
        findings = parse_log(text)
        if job.get("conclusion") == "cancelled":
            findings.infra.add("cancelled")
        if job.get("conclusion") == "timed_out":
            findings.infra.add("job timed out")
        merge_groups(groups, findings_to_groups(workflow, str(job.get("name", job.get("id"))), findings))
    return groups


def build_digest(run_gh: RunGh, repo: str, branch: str, days: int, now: Optional[datetime] = None) -> Tuple[List[Group], List[str]]:
    """Fetch, parse and classify the failures of every workflow; returns (groups, notes)."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    all_groups: List[Group] = []
    notes: List[str] = []
    for wf_id, wf_name in list_workflows(run_gh):
        runs = latest_completed_runs(run_gh, repo, wf_id, branch)
        if not runs:
            continue
        created = datetime.fromisoformat(str(runs[0]["createdAt"]).replace("Z", "+00:00"))
        if created < cutoff:
            continue
        current = collect_run_groups(run_gh, repo, wf_name, runs[0])
        previous = collect_run_groups(run_gh, repo, wf_name, runs[1]) if len(runs) > 1 else {}
        if len(runs) == 1 and runs[0].get("conclusion") == "failure":
            notes.append(f"{wf_name}: only one completed run, everything is NEW")
        all_groups += classify(current, previous)
    return all_groups, notes


def main(argv: Optional[Sequence[str]] = None, run_gh: Optional[RunGh] = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--repo", default="fingoldo/mlframe")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ns = ap.parse_args(argv)
    groups, notes = build_digest(run_gh or default_run_gh, ns.repo, ns.branch, ns.days)
    md = render_markdown(groups, ns.branch, ns.repo, notes)
    if ns.out:
        with open(ns.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)
    if ns.json_out:
        with open(ns.json_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(groups_to_json(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
