"""Self-time of a cProfile run bucketed by library (cupy, mlframe, numpy, pandas, ...): where the wall goes by who spends it. Reads ``PROF_FILE`` written by ``profile_fit_gpu.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # this checkout, not an installed copy
from mlframe.feature_selection._benchmarks.profiling._paths import PROF_FILE  # noqa: E402

import pstats, collections
p = pstats.Stats(str(PROF_FILE)); st = p.stats
b = collections.Counter(); tot = 0.0
for k, v in st.items():
    fn = k[0].replace(chr(92), '/')
    tt = v[2]; tot += tt
    if 'cupy' in fn or (fn == '~' and 'cupy' in k[2]): key = 'cupy (python-side + sync waits)'
    elif 'mlframe' in fn: key = 'mlframe python/njit'
    elif 'numpy' in fn or (fn == '~' and 'numpy' in k[2]): key = 'numpy'
    elif 'pandas' in fn: key = 'pandas'
    elif 'scipy' in fn: key = 'scipy'
    elif 'numba' in fn or 'llvmlite' in fn: key = 'numba/llvmlite (jit/dispatch)'
    elif fn == '~': key = 'builtins/other C: ' + k[2][:40]
    else: key = 'other'
    b[key] += tt
print("total tottime", round(tot, 2))
for k, v in b.most_common(14): print(f"{v:6.2f}  {k}")
