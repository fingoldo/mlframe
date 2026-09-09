#!/bin/bash
# Round 5 of the macOS abort probe (see .github/workflows/macos-abort-probe.yml and
# audits/ci_review_2026-09-08/_TRACKER.md, X5). Round 4's one-by-one leg named a concrete repro:
# each of test_biz_val_training_suite_{classification,regression}_completes and
# ..._mlframe_models_subset[model_list0] segfaults alone, after collection, inside the test body.
# All three call train_mlframe_models_suite(..., mlframe_models=["lgb"], ...) on a small real fit.
# pytest's own traceback stops at threading.py frames -- a Python-level trace cannot see further
# into a native crash. This asks two questions a Python trace cannot answer.
set -uo pipefail

MODE="${1:?usage: macos_lgb_crash_repro.sh <lldb|env-mitigation>}"

TEST_ID="tests/training/test_biz_val_training_core.py::test_biz_val_training_suite_classification_completes"

if [ "$MODE" = "lldb" ]; then
  # lldb ships with Xcode command line tools, already present on GitHub's macOS runners. Running the
  # whole pytest invocation under it and asking for every thread's backtrace on crash is the only way
  # to see past the point a Python-level traceback (faulthandler's, in the logs already collected)
  # goes dark: a segfault inside native code (numba's JIT output, LightGBM's C++, or their shared
  # OpenMP runtime) has no Python frame to report in the first place.
  cat > /tmp/lldb_commands.txt <<'LLDB_EOF'
run
thread backtrace all
bt all
quit
LLDB_EOF
  echo "=== running under lldb ==="
  lldb --batch -s /tmp/lldb_commands.txt -- \
    python -m pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio -s --timeout=300 --timeout-method=thread
  exit 0
fi

if [ "$MODE" = "env-mitigation" ]; then
  # Tests the duplicate-OpenMP-runtime hypothesis directly rather than in isolation: numba's threading
  # layer, LightGBM's own OpenMP, and anything sklearn pulls in can each initialise their own copy of
  # libomp/libiomp in one process, and macOS's pthread_mutex_init failing (seen under xdist in earlier
  # rounds) is that failure's textbook shape. Forcing everything to one thread, and telling Apple's
  # dyld-level runtime to tolerate a duplicate rather than abort on it, isolates whether the crash is
  # the multi-runtime interaction or something else entirely.
  #
  # Each variant is a separate process so a crash in one does not stop the sweep or explain away the
  # next; `|| true` keeps that true even if `set -e` were added later.
  echo "=== baseline (no mitigation, expected to crash) ==="
  python -m pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio -s --timeout=300 --timeout-method=thread || echo "baseline: crashed as expected (exit $?)"

  echo "=== KMP_DUPLICATE_LIB_OK=TRUE (Intel's own escape hatch for exactly this class of crash) ==="
  KMP_DUPLICATE_LIB_OK=TRUE python -m pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio -s --timeout=300 --timeout-method=thread \
    && echo "KMP_DUPLICATE_LIB_OK=TRUE: PASSED" || echo "KMP_DUPLICATE_LIB_OK=TRUE: still crashed (exit $?)"

  echo "=== OMP_NUM_THREADS=1 (removes OpenMP's own internal parallelism, not just duplication) ==="
  OMP_NUM_THREADS=1 python -m pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio -s --timeout=300 --timeout-method=thread \
    && echo "OMP_NUM_THREADS=1: PASSED" || echo "OMP_NUM_THREADS=1: still crashed (exit $?)"

  echo "=== both together ==="
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio -s --timeout=300 --timeout-method=thread \
    && echo "both: PASSED" || echo "both: still crashed (exit $?)"

  echo "=== NUMBA_NUM_THREADS=1 (numba's own thread pool, independent of OMP_NUM_THREADS) ==="
  NUMBA_NUM_THREADS=1 python -m pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio -s --timeout=300 --timeout-method=thread \
    && echo "NUMBA_NUM_THREADS=1: PASSED" || echo "NUMBA_NUM_THREADS=1: still crashed (exit $?)"

  exit 0
fi

echo "unknown mode: $MODE" >&2
exit 2
