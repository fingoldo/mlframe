"""X_CICD_DEPENDENCIES-6 regression test: pre_test_gate.ps1's full-suite worker count must match the
documented quarter-cores policy (CLAUDE.md / memory), not half-cores.

Half physical cores was found to trip a known Windows paging-file exhaustion failure mode under joblib
fan-out on this machine; the quarter-cores divisor (16 physical -> -n 4) is the safe, documented value.
This is a source-inspection test (the divisor is a PowerShell-computed constant, not something a Python
test can invoke and assert on behaviorally without running PowerShell) -- pinned via regex so a future
edit accidentally reverting the divisor is caught.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_full_suite_worker_count_uses_quarter_cores_divisor():
    """The Gate 3 worker-count expression must divide physical cores by 4, not 2."""
    script = REPO_ROOT / "pre_test_gate.ps1"
    src = script.read_text(encoding="utf-8")
    match = re.search(r"Measure-Object -Property NumberOfCores -Sum\)\.Sum\s*/\s*(\d+)", src)
    assert match is not None, "could not locate the worker-count divisor expression in pre_test_gate.ps1"
    assert match.group(1) == "4", f"expected physical-cores divisor 4 (quarter-cores policy), got {match.group(1)}"
