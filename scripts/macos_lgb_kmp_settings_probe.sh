#!/bin/bash
# Round 11 of the macOS abort probe (see .github/workflows/macos-abort-probe.yml and
# audits/ci_review_2026-09-08/_TRACKER.md, X5). Rounds 9 and 10 shipped the two most direct Python-side
# thread-count levers (LightGBM's own n_jobs=1, then the real OS env var OMP_NUM_THREADS=1 set before any
# submodule import) and BOTH failed identically -- lldb caught two threads concurrently inside the same
# libomp crash frame, proving the OpenMP team was never actually constrained to 1 by either lever. Rather
# than guess a third lever blind, this asks libomp itself what it resolved: KMP_SETTINGS=1 makes libomp's
# runtime print every ICV (thread count, affinity, the works) to stderr the first time ANY parallel region
# opens in the process -- this names whichever value actually won.
set -uo pipefail

export KMP_SETTINGS=1
export OMP_DISPLAY_ENV=TRUE
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

TEST_ID="tests/training/test_biz_val_training_core.py::test_biz_val_training_suite_classification_completes"

echo "=== env going in ==="
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS KMP_SETTINGS=$KMP_SETTINGS OMP_DISPLAY_ENV=$OMP_DISPLAY_ENV"

echo "=== bare LightGBM fit first, to see libomp's settings dump BEFORE mlframe/numba/sklearn import anything ==="
python -c "
import numpy as np
import lightgbm as lgb
rng = np.random.default_rng(42)
X = rng.random((400, 8)).astype(np.float64)
y = (X[:, 0] > 0.5).astype(int)
m = lgb.LGBMClassifier(n_estimators=5, n_jobs=1, verbose=-1)
m.fit(X, y)
print('bare lightgbm fit ok')
" 2>&1

echo "=== now the real failing test, same env, libomp's settings dump for the actual crash path ==="
pytest "$TEST_ID" --no-cov -p no:randomly -p no:anyio --timeout=300 --timeout-method=thread -s 2>&1
echo "pytest exit=$?"
