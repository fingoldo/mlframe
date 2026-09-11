#!/bin/bash
# Round 3 of the macOS abort probe (see .github/workflows/macos-abort-probe.yml and
# audits/ci_review_2026-09-08/_TRACKER.md, X5). No mlframe code, no pytest: one process per native
# library, so a crash names the library instead of its neighbours.
#
# Lives in its own file rather than inline in the workflow's `run:` block. A dozen of these
# one-liners in one YAML scalar made actionlint's shellcheck integration hang indefinitely on this
# machine (confirmed by bisection: removing half of them made it return instantly, and every
# one-liner passes shellcheck fine in isolation) -- a parser cost that scales with how many
# semicolon-heavy one-liners sit in one block, not a defect in any single line. A real file sidesteps
# it and is also more useful on its own: it can be run locally without a runner.
set -uo pipefail

run_probe() {
  echo "=== $1 ==="
  python -c "$2" && echo "OK: $1" || echo "CRASH: $1 (exit $?)"
}

run_probe numpy 'import numpy as np; print(np.__version__, np.linalg.svd(np.random.rand(64,64))[1][0])'
run_probe scipy 'import scipy, scipy.linalg as sl, numpy as np; print(scipy.__version__, sl.lu_factor(np.random.rand(64,64))[0][0,0])'
run_probe sklearn-openmp 'import sklearn; from sklearn.ensemble import HistGradientBoostingClassifier as H; import numpy as np; X=np.random.rand(512,8); y=(X[:,0]>0.5).astype(int); H(max_iter=5).fit(X,y); print(sklearn.__version__, "fit ok")'
run_probe numba-serial 'import numba, numpy as np; f=numba.njit(lambda a: a.sum())(np.ones(1000)); print(numba.__version__, f)'
# chr(10) rather than a multi-line body or backslash escapes: the kernel needs real newlines, and
# neither a YAML block scalar (when this lived inline) nor shell quoting survives them on one line.
run_probe numba-parallel 'import numba, numpy as np; N=chr(10); exec("from numba import njit, prange"+N+"@njit(parallel=True)"+N+"def s(a):"+N+" t=0.0"+N+" for i in prange(a.size): t+=a[i]"+N+" return t"); print(numba.__version__, s(np.ones(100000)))'
run_probe lightgbm 'import lightgbm as lgb, numpy as np; X=np.random.rand(512,8); y=(X[:,0]>0.5).astype(int); lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X,y); print(lgb.__version__, "fit ok")'
run_probe catboost 'import catboost as cb, numpy as np; X=np.random.rand(512,8); y=(X[:,0]>0.5).astype(int); cb.CatBoostClassifier(iterations=5, verbose=0).fit(X,y); print(cb.__version__, "fit ok")'
run_probe xgboost 'import xgboost as xgb, numpy as np; X=np.random.rand(512,8); y=(X[:,0]>0.5).astype(int); xgb.XGBClassifier(n_estimators=5).fit(X,y); print(xgb.__version__, "fit ok")'
run_probe torch 'import torch; print(torch.__version__, torch.randn(64,64).matmul(torch.randn(64,64)).sum().item())'
run_probe joblib-threads 'from joblib import Parallel, delayed; import numpy as np; print(sum(Parallel(n_jobs=4, backend="threading")(delayed(lambda i: float(np.random.rand(256,256).sum()))(i) for i in range(8))) > 0)'
# The combination, in the order a real run loads them. If every library above survives alone and
# this crashes, the interaction is the subject rather than any one library.
run_probe combined-import 'import numpy, scipy, sklearn, numba, lightgbm, catboost, xgboost, torch, pyarrow, polars, shap; print("combined import ok")'

# How many OpenMP runtimes ended up mapped into one process. More than one is the classic macOS
# crash shape and is worth knowing regardless of what else this sweep finds.
echo "=== loaded libomp copies ==="
python -c "import sklearn, lightgbm, catboost, numba, torch, subprocess, os; print(subprocess.run(['/usr/sbin/lsof','-p',str(os.getpid())],capture_output=True,text=True).stdout)" 2>/dev/null | grep -icE "libomp|libiomp|libgomp" || echo "lsof unavailable"
