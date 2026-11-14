#!/usr/bin/env bash
# What happens to a macOS shard: does it collect, and does it then run?
#
# A full macOS shard used to print `created: 1/1 worker` and then nothing until the job-level cap cancelled it. The cause was a full MRMR fit (and
# two small model fits) executed while test modules were imported; with that gone, collection takes minutes but ends. This probe keeps the wall-clock
# limit and the py-spy dump of every pytest process for whatever still hangs, prints the collection errors, and then runs shard 1 of the real
# 10-way split with a limit, so a hang or a crash inside a test shows up as a stack or a native-crash report instead of a cancelled job.
set -u

python -m pip install --quiet py-spy

SELECT=(-m "not slow and not gpu and not multigpu" --ignore=tests/training/test_core.py)

# probe <label> <limit seconds> <pytest arguments...>: run in the background, dump the stacks and stop it when the limit passes.
probe() {
  local label="$1"
  local limit="$2"
  shift 2
  local log="probe_${label}.log"
  echo "=== ${label} (limit ${limit}s): pytest $*"
  PYTHONFAULTHANDLER=1 python -u -m pytest "$@" >"$log" 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$limit" ]; then
      echo "still running after ${limit}s -- stack of every pytest process:"
      for p in $(pgrep -f 'pytest|execnet|xdist' || true); do
        echo "--- pid $p: $(ps -o command= -p "$p" | cut -c1-160)"
        sudo py-spy dump --pid "$p" 2>&1 | head -80 || true
      done
      pkill -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  wait "$pid" 2>/dev/null
  echo "${label} ended with exit code $? after about ${waited}s"
  echo "--- errors and failures in ${log}"
  grep -E "^(ERROR|FAILED)|Fatal Python error|Segmentation fault|Aborted|OMP: Error|Interrupted:" "$log" | head -40 | cut -c1-240
  echo "--- the first timeout dump in ${log}"
  grep -n -A90 -m1 "Timeout (0:" "$log" | cut -c1-240
  echo "--- last lines of ${log}"
  tail -n 25 "$log" | cut -c1-240
}

probe collect_errors 300 tests/integrations/test_embed_website_fd_not_leaked.py tests/integrations/test_mlflow_embed_refuses_script_schemes.py --collect-only -q --no-cov -p no:randomly -n 0
probe core 900 tests/training/test_biz_val_training_core.py "${SELECT[@]}" -n 0 --no-cov -p no:randomly --timeout=300 --timeout-method=thread   -o faulthandler_timeout=240 -x -rfE -s
probe shard1 1500 tests "${SELECT[@]}" --splits 10 --group 1 -n 1 --dist=loadgroup --no-cov -p no:randomly --timeout=900 --timeout-method=thread \
  -o faulthandler_timeout=600 --durations=25 -rfE
exit 0
