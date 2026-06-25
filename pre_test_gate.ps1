#requires -Version 5.1
param([switch]$ContinueOnCollectErrors, [switch]$GatesOnly)

# Pre-test gate for the mlframe full suite. Three escalating gates that short-circuit on the cheap checks, so a
# syntax / import / collection error never burns the multi-hour run:
#   Gate 1  ruff          (~3s)   HARD-fails on real-bug lint classes (bad format strings, is-with-literal, assert-on-tuple,
#                                 return/break/yield outside function, used-before-assignment, syntax). The 6 cosmetic /
#                                 dynamic-false-positive classes are REPORTED, not blocked: F401 unused-import, F811
#                                 redefinition, F821 (annotation forward-ref FPs on this numba/typing codebase), F822
#                                 stale-__all__, F841 unused-variable, F541 f-string-without-placeholder.
#   Gate 2  collect-only  (~50s)  HARD-fails on ANY import/collection error across tests/ (e.g. a test referencing a
#                                 missing module) -- the class of failure that otherwise aborts the full run partway.
#   Gate 3  full pytest -n        runs only if Gates 1-2 pass.
# Pass -ContinueOnCollectErrors to downgrade Gate 2 to a warning and add --continue-on-collection-errors to the full run
# (use when a collection error is external / owned by another session and you want the full results anyway).

$ErrorActionPreference = 'Continue'
$log = "$env:TEMP\pytest_master.log"
Set-Location "C:\Users\Admin\Machine learning\mlframe"
$env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
$ok = $true

# ---- Gate 1: ruff ----
Write-Host "== Gate 1: ruff (real-bug classes block; cosmetic classes reported) ==" -ForegroundColor Cyan
$gateIgnore = 'F401,F811,F821,F822,F841,F541'
python.exe -m ruff check src/mlframe tests --select F,E9 --ignore $gateIgnore --output-format concise
if ($LASTEXITCODE -ne 0) {
    Write-Host "RUFF GATE FAILED: real-bug lint above -- fix before the slow run." -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "ruff gate clean. Cosmetic findings (informational, not blocking):" -ForegroundColor DarkGray
    python.exe -m ruff check src/mlframe tests --select $gateIgnore --statistics 2>&1 |
        Select-String -Pattern '^\s*\d+\s+F' | ForEach-Object { "   " + $_.Line.Trim() }
}

# ---- Gate 2: collect-only ----
if ($ok) {
    Write-Host "== Gate 2: pytest --collect-only (import/collection errors across tests/) ==" -ForegroundColor Cyan
    python.exe -m pytest tests/ --collect-only -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {
        if ($ContinueOnCollectErrors) {
            Write-Host "COLLECT errors above (continuing anyway: -ContinueOnCollectErrors)." -ForegroundColor Yellow
        } else {
            Write-Host "COLLECT GATE FAILED: a test module can't import (see ERROR above) -- fix, or re-run with -ContinueOnCollectErrors." -ForegroundColor Red
            $ok = $false
        }
    }
}

# ---- Gate 3: full suite ----
if ($ok -and $GatesOnly) { Write-Host "Gates passed (-GatesOnly: not running the full suite)." -ForegroundColor Green }
elseif ($ok) {
    $n = [int]([Math]::Max(1, [Math]::Floor((Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum / 2)))
    Write-Host "== Gate 3: full suite -n $n (physical cores / 2) ==" -ForegroundColor Cyan
    $extra = @()
    if ($ContinueOnCollectErrors) { $extra += '--continue-on-collection-errors' }
    python.exe -m pytest tests/ -n $n --fast --dist=worksteal --max-worker-restart=20 --instafail --show-progress -p no:randomly -p no:cacheprovider --no-cov --tb=short --timeout=600 --maxfail=0 -ra -v --color=no @extra 2>&1 | Tee-Object -FilePath $log
    Write-Host "Full-suite log: $log" -ForegroundColor Cyan
} else {
    Write-Host "Skipped the full suite because a cheap gate failed (saved ~hours)." -ForegroundColor Yellow
}
