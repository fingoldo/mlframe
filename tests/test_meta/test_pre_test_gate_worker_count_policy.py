"""X_CICD_DEPENDENCIES-6 regression test: pre_test_gate.ps1's full-suite worker count must match the
documented quarter-cores policy (CLAUDE.md / memory), not half-cores.

Half physical cores was found to trip a known Windows paging-file exhaustion failure mode under joblib
fan-out on this machine; the quarter-cores divisor (16 physical -> -n 4) is the safe, documented value.
The expression is a PowerShell constant, so PowerShell itself parses the script, finds the worker-count assignment and evaluates it against a
stubbed processor inventory; the test asserts on the number that comes back.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - runs the local PowerShell on an expression taken from this repo's own script
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")

_EVAL_SCRIPT = """
$ErrorActionPreference = 'Stop'
function Get-CimInstance { param($ClassName) [pscustomobject]@{NumberOfCores=@@CORES@@} }
$ast = [System.Management.Automation.Language.Parser]::ParseFile('@@PATH@@', [ref]$null, [ref]$null)
$found = @($ast.FindAll({ param($a) $a -is [System.Management.Automation.Language.AssignmentStatementAst] -and $a.Right.Extent.Text -like '*NumberOfCores*' }, $true))
if ($found.Count -ne 1) { Write-Output ('COUNT=' + $found.Count); exit 0 }
$n = Invoke-Expression $found[0].Right.Extent.Text
Write-Output ('WORKERS=' + $n)
"""


def _workers_in_script(script_path: Path, physical_cores: int) -> int | None:
    """Worker count the script's own ``NumberOfCores`` assignment yields on ``physical_cores``; None unless there is exactly one such assignment."""
    assert _POWERSHELL is not None
    script = _EVAL_SCRIPT.replace("@@CORES@@", str(physical_cores)).replace("@@PATH@@", str(script_path))
    proc = subprocess.run(  # nosec B603 - fixed argv, the evaluated expression comes from this repo's pre_test_gate.ps1
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=120, check=True
    )
    kind, _, value = proc.stdout.strip().partition("=")
    return int(value) if kind == "WORKERS" else None


@pytest.mark.skipif(_POWERSHELL is None, reason="no PowerShell on PATH to evaluate the gate expression")
def test_worker_count_evaluator_distinguishes_half_and_quarter_cores_and_a_missing_assignment(tmp_path):
    """The evaluator reports 8 workers for a half-cores script on 16 cores, 4 for a quarter-cores one, and None when no assignment matches."""
    expression = "[int]([Math]::Max(1, [Math]::Floor((Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum / {div})))"
    half, quarter, none = tmp_path / "half.ps1", tmp_path / "quarter.ps1", tmp_path / "none.ps1"
    half.write_text("$n = " + expression.format(div=2) + "\n", encoding="utf-8")
    quarter.write_text("$n = " + expression.format(div=4) + "\n", encoding="utf-8")
    none.write_text("$n = 3\n", encoding="utf-8")
    assert _workers_in_script(half, 16) == 8
    assert _workers_in_script(quarter, 16) == 4
    assert _workers_in_script(none, 16) is None


@pytest.mark.skipif(_POWERSHELL is None, reason="no PowerShell on PATH to evaluate the gate expression")
def test_full_suite_worker_count_uses_quarter_cores_divisor():
    """The Gate 3 worker-count expression yields a quarter of the physical cores, never fewer than one."""
    script = REPO_ROOT / "pre_test_gate.ps1"
    assert _workers_in_script(script, 16) == 4
    assert _workers_in_script(script, 2) == 1
