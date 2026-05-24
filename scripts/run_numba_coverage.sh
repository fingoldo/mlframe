#!/usr/bin/env bash
# Numba-disabled coverage entry point (POSIX).
#
# Disables numba JIT so @njit kernel bodies become visible to coverage.py / sys.settrace, then runs pytest with
# explicit coverage over the kernel-heavy packages. Slow path -- not for daily CI; run weekly / on RC branches.
#
# Per memory `reference_numba_coverage_blind`: the default daily test suite intentionally measures dispatch coverage
# only; kernel-body coverage requires this script.
#
# Usage:
#   ./scripts/run_numba_coverage.sh
#   ./scripts/run_numba_coverage.sh -m numba_disabled

set -euo pipefail

export NUMBA_DISABLE_JIT=1
export NUMBA_DISABLE_INTEL_SVML=1
export PYTHONUNBUFFERED=1

mkdir -p _results

KERNEL_PACKAGES=(
    "src/mlframe/feature_selection/filters"
    "src/mlframe/feature_engineering"
    "src/mlframe/metrics"
    "src/mlframe/core"
    "src/mlframe/preprocessing/outliers.py"
)

COV_ARGS=()
for pkg in "${KERNEL_PACKAGES[@]}"; do
    COV_ARGS+=("--cov=$pkg")
done

echo "[numba-coverage] NUMBA_DISABLE_JIT=1 active; running pytest..."

python -m pytest \
    --timeout=300 \
    -s \
    "${COV_ARGS[@]}" \
    --cov-report=xml:_results/coverage_numba_disabled.xml \
    --cov-report=term-missing \
    "$@"
