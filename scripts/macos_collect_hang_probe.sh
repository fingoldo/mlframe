#!/usr/bin/env bash
# Where does test collection hang on macOS?
#
# A full macOS shard prints `created: 1/1 worker` and then nothing until the job-level cap cancels it: collection never finishes, so
# the hang is in an import, a conftest or a fixture at collection time, not in a test. The in-process timeout cannot fire there, so this
# probe collects with a wall-clock limit and, if collection is still running when it ends, dumps the Python stack of every pytest process
# (the xdist worker included) with py-spy before killing it.
set -u

LIMIT_S="${COLLECT_LIMIT_S:-360}"
PYTEST_ARGS=(tests --collect-only -q --no-cov -p no:randomly -m "not slow and not gpu and not multigpu" --ignore=tests/training/test_core.py)

python -m pip install --quiet py-spy

probe() {
  local label="$1"
  shift
  local log="collect_${label}.log"
  echo "=== ${label}: pytest ${PYTEST_ARGS[*]} $*"
  PYTHONFAULTHANDLER=1 python -u -m pytest "${PYTEST_ARGS[@]}" "$@" >"$log" 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$LIMIT_S" ]; then
      echo "still collecting after ${LIMIT_S}s -- stack of every pytest process:"
      for p in $(pgrep -f 'pytest|execnet|xdist' || true); do
        echo "--- pid $p: $(ps -o command= -p "$p" | cut -c1-160)"
        sudo py-spy dump --pid "$p" 2>&1 | head -80 || true
      done
      pkill -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
      echo "--- last lines of ${log}"
      tail -n 15 "$log"
      return
    fi
    sleep 5
    waited=$((waited + 5))
  done
  wait "$pid"
  echo "collection ended with exit code $? after about ${waited}s"
  tail -n 6 "$log"
}

probe single -n 0
probe xdist -n 1 --dist=loadgroup
exit 0
