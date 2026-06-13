import os, sys, cProfile, pstats, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlframe.feature_engineering.numerical import compute_numaggs

rng = np.random.default_rng(42)
N = 200_000
# Mixed: continuous with some repeats (typical numeric column).
arr = rng.normal(size=N).astype(np.float64)
arr[::7] = arr[1]  # inject repeats so unique/modes path is realistic

# warm numba
for _ in range(2):
    compute_numaggs(arr[:1000])

pr = cProfile.Profile()
pr.enable()
for _ in range(20):
    compute_numaggs(arr)
pr.disable()
st = pstats.Stats(pr, stream=sys.stdout)
st.sort_stats("tottime").print_stats(40)
