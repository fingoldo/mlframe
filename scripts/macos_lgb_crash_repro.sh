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
  # Round 5's first pass tried OS-level OpenMP env vars (KMP_DUPLICATE_LIB_OK, OMP_NUM_THREADS,
  # NUMBA_NUM_THREADS) against the full pytest suite call -- all five variants crashed identically,
  # every one inside lightgbm/basic.py's __init_from_np2d, which calls LGBM_DatasetCreateFromMat, a
  # C++ function that OpenMP-parallelises its own histogram binning using LightGBM's OWN num_threads
  # setting, not the process's OMP_NUM_THREADS env var. And lgb_shim.py explicitly sets
  # `self.n_jobs = os.cpu_count()` before fit specifically to skip LightGBM's slow core-count probe --
  # so every LightGBM fit in this codebase runs multi-threaded by construction, on every platform, and
  # the env vars this leg tried first could never have reached the thread count that matters.
  #
  # This pass tests the parameter that actually controls it, isolated from the rest of the suite: a
  # bare LightGBM fit (no mlframe) at n_jobs=1, then the real failing test with n_jobs forced to 1 via
  # LGBM_FORCE_N_JOBS (read below) to see whether serialising just this one library's threads is
  # sufficient, without touching numba/sklearn/joblib at all.
  echo "=== bare LightGBM fit, n_jobs=1, no mlframe involved at all ==="
  python -c "
import numpy as np
import lightgbm as lgb
rng = np.random.default_rng(42)
X = rng.random((400, 8)).astype(np.float64)
y = (X[:, 0] > 0.5).astype(int)
m = lgb.LGBMClassifier(n_estimators=50, n_jobs=1, verbose=-1)
m.fit(X, y)
print('bare lightgbm n_jobs=1: fit ok, pred sum =', int(m.predict(X).sum()))
" && echo "bare-lgb-n_jobs=1: PASSED" || echo "bare-lgb-n_jobs=1: CRASHED (exit $?)"

  echo "=== bare LightGBM fit, n_jobs=os.cpu_count() (what lgb_shim.py actually sets) ==="
  python -c "
import os
import numpy as np
import lightgbm as lgb
rng = np.random.default_rng(42)
X = rng.random((400, 8)).astype(np.float64)
y = (X[:, 0] > 0.5).astype(int)
m = lgb.LGBMClassifier(n_estimators=50, n_jobs=os.cpu_count(), verbose=-1)
m.fit(X, y)
print('bare lightgbm n_jobs=cpu_count: fit ok, pred sum =', int(m.predict(X).sum()))
" && echo "bare-lgb-n_jobs=cpu_count: PASSED" || echo "bare-lgb-n_jobs=cpu_count: CRASHED (exit $?)"

  echo "=== bare LightGBM fit, n_jobs=os.cpu_count(), 50 REPEATS (single-fit runs may just get lucky) ==="
  # A single bare fit passing would not settle whether n_jobs alone is the trigger -- the real test
  # crashes on some but not all invocations across CI history (round 1's serial leg died on the very
  # first collected item; other legs ran further before crashing), so this needs enough tries to see a
  # failure if the multi-threaded path is unstable rather than deterministic.
  python -c "
import os
import numpy as np
import lightgbm as lgb
rng = np.random.default_rng(0)
fails = 0
for i in range(50):
    X = rng.random((400, 8)).astype(np.float64)
    y = (X[:, 0] > 0.5).astype(int)
    m = lgb.LGBMClassifier(n_estimators=50, n_jobs=os.cpu_count(), verbose=-1)
    m.fit(X, y)
    m.predict(X)
print(f'{50} repeated bare-lightgbm n_jobs=cpu_count fits completed with no crash')
"  && echo "repeated-bare-lgb: PASSED (50/50, no crash)" || echo "repeated-bare-lgb: CRASHED (exit $?)"

  exit 0
fi

echo "unknown mode: $MODE" >&2
exit 2
