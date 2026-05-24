# Numba-disabled coverage entry point.
#
# Disables numba JIT so @njit kernel bodies become visible to coverage.py / sys.settrace, then runs pytest with explicit
# coverage over the kernel-heavy packages. Slow path -- not for daily CI; run weekly / on RC branches.
#
# Per memory `reference_numba_coverage_blind`: the default daily test suite intentionally measures dispatch coverage
# only; kernel-body coverage requires this script.
#
# Usage:
#   .\scripts\run_numba_coverage.ps1
#   .\scripts\run_numba_coverage.ps1 -TestSelector "-m numba_disabled"

param(
    [string]$TestSelector = ""
)

$env:NUMBA_DISABLE_JIT = "1"
$env:NUMBA_DISABLE_INTEL_SVML = "1"
$env:PYTHONUNBUFFERED = "1"

$kernelPackages = @(
    "src/mlframe/feature_selection/filters",
    "src/mlframe/feature_engineering",
    "src/mlframe/metrics",
    "src/mlframe/core",
    "src/mlframe/preprocessing/outliers.py"
)

$covArgs = @()
foreach ($pkg in $kernelPackages) {
    $covArgs += "--cov=$pkg"
}

$resultsDir = "_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

$pytestArgs = @(
    "-m", "pytest",
    "--timeout=300",
    "-s"
) + $covArgs + @(
    "--cov-report=xml:_results/coverage_numba_disabled.xml",
    "--cov-report=term-missing"
)

if ($TestSelector) {
    $pytestArgs += $TestSelector.Split(" ")
}

Write-Host "[numba-coverage] NUMBA_DISABLE_JIT=1 active; running pytest..."
Write-Host "[numba-coverage] args: $($pytestArgs -join ' ')"

& "D:/ProgramData/anaconda3/python.exe" @pytestArgs
$exit = $LASTEXITCODE
Write-Host "[numba-coverage] pytest exit code: $exit"
exit $exit
