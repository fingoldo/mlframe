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
  # First attempt used a bare `run` and stopped at the wrong place: lldb halts on every `exec` event,
  # and `python -m pytest` re-execs through its console-script wrapper before reaching the test at all,
  # so `thread backtrace all` fired at dyld's startup stop instead of the crash -- the log showed only
  # `dyld_start`. Fixed two ways: invoke python directly (`python -c "import pytest; ..."`, one process,
  # one exec) rather than through the pytest wrapper, and `continue` past any stop that is not the fatal
  # signal instead of assuming the first `run` is the last stop needed.
  cat > /tmp/lldb_commands.txt <<'LLDB_EOF'
run
continue
continue
continue
continue
continue
bt all
register read
quit
LLDB_EOF
  echo "=== running under lldb ==="
  lldb --batch -s /tmp/lldb_commands.txt -- \
    python -c "import pytest, sys; sys.exit(pytest.main(['$TEST_ID', '--no-cov', '-p', 'no:randomly', '-p', 'no:anyio', '-s', '--timeout=300', '--timeout-method=thread']))"
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
