"""How far behind the code a published benchmark result has fallen, and when that stops being acceptable.

A benchmark result is a claim about code at one commit. The code moves; the result does not. After enough
commits to the selectors it measures, the atlas is describing something that no longer exists, and nothing
about the document says so -- it reads exactly as it did on the day it was true.

This is the failure that killed the two hundred `bench_*` scripts already in this repository. They were
not wrong; they went stale, silently, and then nobody could tell which of them still meant anything.

The measurement is deliberately narrow. What matters is not how old a result is in days, nor how many
commits the repository has taken, but how many commits landed on the PATHS the result is about. A month of
work on the reporting layer changes nothing about which features a selector picks; ten commits to
`feature_selection/` might change all of it.

Two thresholds, because there are two different statements to make:

* **warn** -- the result is drifting and should be re-run soon.
* **stale** -- the result no longer describes the current code, and a document quoting it has to say so.

Nothing here re-runs anything or edits a document. It reports, and the meta-test that consumes it fails
the build, because a staleness check nobody is forced to read is the same as no check.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404 - reads git history; every invocation below is a fixed argument list
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["WATCHED_PATHS", "WARN_COMMITS", "STALE_COMMITS", "Staleness", "commits_since", "assess", "format_staleness"]

#: The paths a benchmark result is a claim about. A commit outside these changes nothing the atlas says.
WATCHED_PATHS: Tuple[str, ...] = (
    "src/mlframe/feature_selection",
    "src/mlframe/data/datasets",
)

#: Drifting. Chosen so that a normal week of work on the selectors trips it, which is the cadence the
#: nightly tier is meant to run at anyway.
WARN_COMMITS = 10

#: No longer describing the current code. A document quoting a result this far behind must say so.
STALE_COMMITS = 40


@dataclass(frozen=True)
class Staleness:
    """How far one published result has fallen behind the code it describes."""

    manifest_sha: Optional[str]
    head_sha: Optional[str]
    commits_behind: Optional[int]
    verdict: str
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        """Return the record shape a report stores."""
        return {"manifest_sha": self.manifest_sha, "head_sha": self.head_sha, "commits_behind": self.commits_behind, "verdict": self.verdict, "reason": self.reason}


def _git(args: Sequence[str], repo: str) -> Optional[str]:
    """Run one read-only git command, returning its output or ``None`` when git cannot answer.

    Read-only by construction: every caller below passes a fixed argument list of query subcommands. A
    missing git, a missing repository or an unknown revision all return ``None`` rather than raising,
    because "cannot tell how stale this is" is a legitimate state -- a source tarball has no history --
    and it is reported as such instead of failing a build that has nothing wrong with it.
    """
    try:
        out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30, check=False)  # nosec B603 B607 - fixed read-only git query, no shell
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("git is not usable for a staleness check here: %s", exc)
        return None
    if out.returncode != 0:
        logger.info("git %s failed: %s", " ".join(args), out.stderr.strip()[:200])
        return None
    return out.stdout.strip()


def commits_since(sha: str, repo: str, paths: Sequence[str] = WATCHED_PATHS) -> Optional[int]:
    """Return how many commits touched ``paths`` between ``sha`` and HEAD, or ``None`` when git cannot say."""
    if not sha:
        return None
    out = _git(["rev-list", "--count", f"{sha}..HEAD", "--", *paths], repo)
    if out is None:
        return None
    try:
        return int(out)
    except ValueError:
        logger.info("git rev-list returned something that is not a count: %r", out[:80])
        return None


def assess(manifest_path: str, repo: Optional[str] = None, warn_at: int = WARN_COMMITS, stale_at: int = STALE_COMMITS) -> Staleness:
    """Read a run manifest and report how far behind the code its result has fallen.

    Args:
        manifest_path: The manifest written beside a results file.
        repo: Repository root; defaults to the one this module lives in.
        warn_at: Commits after which the result is drifting.
        stale_at: Commits after which the result no longer describes the code.

    Returns:
        A :class:`Staleness` whose ``verdict`` is one of ``current``, ``drifting``, ``stale`` or
        ``unknown``. ``unknown`` is a real answer: a manifest with no commit recorded, or a checkout with
        no history, cannot be aged, and guessing would be worse than saying so.
    """
    root = repo or os.path.dirname(os.path.abspath(__file__))
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        return Staleness(manifest_sha=None, head_sha=None, commits_behind=None, verdict="unknown", reason=f"no readable manifest at {manifest_path}: {exc}")

    sha = str(manifest.get("git_sha") or manifest.get("git_commit") or "")
    head = _git(["rev-parse", "HEAD"], root)
    if not sha:
        return Staleness(manifest_sha=None, head_sha=head, commits_behind=None, verdict="unknown", reason="the manifest records no commit, so the result cannot be dated against the code")

    behind = commits_since(sha, root)
    if behind is None:
        return Staleness(manifest_sha=sha, head_sha=head, commits_behind=None, verdict="unknown", reason="git could not count the commits between the manifest and HEAD")

    if behind >= stale_at:
        verdict, reason = "stale", f"{behind} commits have touched {', '.join(WATCHED_PATHS)} since this result was produced; it no longer describes the current code"
    elif behind >= warn_at:
        verdict, reason = "drifting", f"{behind} commits have touched the measured paths since this result was produced"
    else:
        verdict, reason = "current", f"{behind} commits have touched the measured paths since this result was produced"
    return Staleness(manifest_sha=sha, head_sha=head, commits_behind=behind, verdict=verdict, reason=reason)


def format_staleness(values: Dict[str, Staleness]) -> List[str]:
    """Render one line per published result, for a report that has to carry its own expiry date."""
    lines = ["", "=" * 100, "HOW FAR BEHIND THE CODE EACH PUBLISHED RESULT HAS FALLEN", "=" * 100, ""]
    if not values:
        lines.append("no manifests found, so nothing here is dated")
        return lines
    for name, value in sorted(values.items()):
        behind = "unknown" if value.commits_behind is None else str(value.commits_behind)
        lines.append(f"{name:<28}{value.verdict:<12}{behind:>6} commits behind  {value.reason}")
    return lines
