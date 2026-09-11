"""Ratchet on the tracked mccabe (C901) complexity debt.

``[tool.ruff.lint.mccabe]``'s comment describes the threshold-40 findings as "real, tracked debt". Nothing
tracked them: C901 is ``--ignore``d in every blocking invocation (a recorded, correct decision -- the debt
is not fixable in one pass), and the advisory job's output is nobody's assigned reading. The 2026-09-08
review measured the count at 92 against a comment that said 70, so the debt had grown ~31% unnoticed.

A ratchet costs nothing and has no false positives: the count may fall freely, and only a rise fails. The
number here is a measurement, not a target -- lower it whenever the debt drains.
"""

from __future__ import annotations

import orjson
import re
import subprocess  # nosec B404 - runs the repo's own pinned ruff on the repo's own source, no external input
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Measured 2026-09-08 on ruff 0.16.1 (the exact pin in pyproject.toml's dev extra). Ratchet DOWN only.
C901_CEILING = 92


def _c901_findings() -> list[str]:
    """Every C901 finding in src/mlframe, as "path:line" strings, via the repo's own ruff."""
    src = REPO_ROOT / "src" / "mlframe"
    assert src.is_dir(), f"src tree not found at {src}; this gate cannot run and must not report green"
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell, no user input
        [sys.executable, "-m", "ruff", "check", str(src), "--select", "C901", "--output-format", "json"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode not in (0, 1):
        pytest.skip(f"ruff unavailable or failed to run ({proc.returncode}): {proc.stderr.strip()[:200]}")
    try:
        findings = orjson.loads(proc.stdout or "[]")
    except orjson.JSONDecodeError:  # pragma: no cover - only on a ruff output-format change
        pytest.skip("could not parse ruff JSON output")
    return [f"{Path(f['filename']).as_posix()}:{f['location']['row']}" for f in findings]


def test_c901_complexity_debt_does_not_grow():
    """The number of functions over the mccabe threshold must not rise above the recorded ceiling."""
    findings = _c901_findings()
    assert len(findings) <= C901_CEILING, (
        f"mccabe C901 debt rose to {len(findings)} from the recorded ceiling of {C901_CEILING}. "
        f"Either simplify the new offender or, if the growth is deliberate, raise C901_CEILING in this "
        f"file with the reason in the commit message. Current findings:\n" + "\n".join(f"  {f}" for f in sorted(findings))
    )


def test_c901_ceiling_is_not_stale():
    """A ceiling far above the real count stops being a ratchet, so require it to track measured debt."""
    findings = _c901_findings()
    assert len(findings) >= C901_CEILING - 10, (
        f"mccabe C901 debt has drained to {len(findings)}, well under the recorded ceiling of "
        f"{C901_CEILING}. Lower C901_CEILING to {len(findings)} so the ratchet keeps its teeth."
    )


def test_mccabe_comment_matches_the_measured_count():
    """pyproject's mccabe note quotes a finding count; a stale number there is what hid this drift."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[tool.ruff.lint.mccabe]", 1)
    assert len(section) == 2, "expected a [tool.ruff.lint.mccabe] section in pyproject.toml"
    quoted = re.findall(r"threshold=40 keeps\s*#?\s*(\d+) findings", section[1][:2000])
    if not quoted:
        pytest.skip("the mccabe comment no longer quotes a finding count")
    assert int(quoted[0]) == C901_CEILING, (
        f"pyproject.toml's mccabe comment says {quoted[0]} findings while the ratchet records "
        f"{C901_CEILING}. Update the comment; a number nobody rechecks is what let the debt grow silently."
    )
