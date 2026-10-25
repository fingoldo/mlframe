#!/usr/bin/env bash
# Run the independent pre-push hooks side by side and print every finding in one pass, instead of finding them one failed push at a time.
#
# pre-commit runs the hooks of one push strictly in series, but mypy-full, vulture-blocking and the hook-config parity check do not depend on each
# other (pass_filenames is false for all three), so they can share the idle cores. kernel-tuning-ensure is require_serial (it may run GPU sweeps)
# and goes last, alone. This only reports; the push itself still runs the hooks.
#
# Usage: scripts/prepush_parallel.sh        exit 0 = every hook passed, 1 = at least one failed (logs under ${TMPDIR:-/tmp}/prepush_<hook>.log)
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
logdir="${TMPDIR:-${TEMP:-/tmp}}"
parallel_hooks=(mypy-full vulture-blocking mlframe-hook-config-parity-on-push)
declare -A pids
for hook in "${parallel_hooks[@]}"; do
  pre-commit run "$hook" --hook-stage pre-push --all-files > "$logdir/prepush_${hook}.log" 2>&1 &
  pids[$hook]=$!
done
failed=0
for hook in "${parallel_hooks[@]}"; do
  if wait "${pids[$hook]}"; then
    echo "PASS  $hook"
  else
    echo "FAIL  $hook   (full log: $logdir/prepush_${hook}.log)"
    grep -E "error:|: error|redundant|unused|Unused|FAIL|mismatch" "$logdir/prepush_${hook}.log" | head -30
    failed=1
  fi
done
if pre-commit run kernel-tuning-ensure --hook-stage pre-push --all-files > "$logdir/prepush_kernel-tuning-ensure.log" 2>&1; then
  echo "PASS  kernel-tuning-ensure"
else
  echo "FAIL  kernel-tuning-ensure   (full log: $logdir/prepush_kernel-tuning-ensure.log)"
  failed=1
fi
exit "$failed"
