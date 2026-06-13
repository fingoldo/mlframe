import os, cProfile, pstats, io, numpy as np
os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")
import scipy.stats  # noqa: F401  (ABI-order: import before mlframe to avoid native segfault on py3.14)
import numba  # noqa: F401
from mlframe.preprocessing.cleaning import is_variable_truly_continuous

rng = np.random.default_rng(0)
N = 200000
cols = {
    "cont": rng.normal(0, 1, N).astype(np.float64),
    "cont2": (rng.normal(100, 30, N)).astype(np.float64),
    "frac": np.round(rng.uniform(0, 1000, N), 3),
    "intlike": rng.integers(0, 500, N).astype(np.float64),
    "wide": rng.uniform(-1e6, 1e6, N).astype(np.float64),
}
for v in cols.values():
    is_variable_truly_continuous(values=v, use_quantile=0.1)

pr = cProfile.Profile()
pr.enable()
for _ in range(20):
    for v in cols.values():
        is_variable_truly_continuous(values=v, use_quantile=0.1)
pr.disable()
st = pstats.Stats(pr, stream=(s := io.StringIO()))
st.sort_stats("tottime").print_stats(25)
print(s.getvalue())
