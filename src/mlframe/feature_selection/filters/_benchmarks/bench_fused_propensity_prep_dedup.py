"""Bench for critique P-11: fused_propensity shared-prep dedup (bit-identical).

fused_propensity called second_moment_propensity (which derives V/V2/yf/classes) and then RE-derived V/yf/classes
for its main-effect channel -- ~15% of the call spent on a duplicated contiguous-float64 + factorise + unique.
The dedup shares the prep once. Run: python -m mlframe.feature_selection.filters._benchmarks.bench_fused_propensity_prep_dedup
"""
import time
import numpy as np
from mlframe.feature_selection.filters._fe_interaction_prerank import fused_propensity

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n, p in [(100_000, 200), (300_000, 300), (500_000, 300)]:
        V = rng.standard_normal((n, p)); y = rng.integers(0, 3, n)
        fused_propensity(V, y, use_gbm=False)  # warm
        t = time.perf_counter()
        for _ in range(3):
            fused_propensity(V, y, use_gbm=False)
        print(f"n={n} p={p}: fused_propensity {(time.perf_counter()-t)/3*1000:.0f} ms/call")
