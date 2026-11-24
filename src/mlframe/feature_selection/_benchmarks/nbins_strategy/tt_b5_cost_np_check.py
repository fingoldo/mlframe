"""tt_b5_cost_np against the numba kernels of tt_b5_cost (same data, same permutations): max abs difference of s and of the null maxima, and the seconds of both at n=100000, p=64, K=9."""
import time, warnings
import numpy as np
warnings.simplefilter("ignore")
import tt_b5_cost as C, tt_b5_cost_np as N

prov = C.GenProvider(20000, 12, seed=3, chunk_rows=2000)
beta = np.random.default_rng(1).standard_normal(12) * 0.2
for K in (9, 10):
    a = C.feature_criterion(prov, K=K, B=12, seed=5, row_block=700, beta=beta)
    b = N.feature_criterion_np(prov, K=K, B=12, seed=5, row_block=700, beta=beta, col_block=5)
    print(f"K={K} all rows : max|s np-numba| = {np.abs(a['s'] - b['s']).max():.2e} (scale {a['s'].max():.3f}); max|T np-numba| = {np.abs(a['null_max'] - b['null_max']).max():.2e}; same best_k {np.array_equal(a['best_k'], b['best_k'])}; same accepted {np.array_equal(a['accepted'], b['accepted'])}")
a = C.feature_criterion(prov, K=9, B=12, null_rows=400, seed=5, row_block=400, beta=beta)
b = N.feature_criterion_np_sub(prov, K=9, B=12, null_rows=400, seed=5, row_block=400, beta=beta, col_block=5)
print(f"null_rows=400: max|s| = {np.abs(a['s'] - b['s']).max():.2e}; max|T| = {np.abs(a['null_max'] - b['null_max']).max():.2e}; same accepted {np.array_equal(a['accepted'], b['accepted'])}")

p2 = C.GenProvider(100000, 64, seed=1, chunk_rows=5000)
p2.target()
for name, fn in (("numba kernels", lambda: C.feature_criterion(p2, K=9, B=40)), ("numpy SIMD blocks, 1 thread", lambda: N.feature_criterion_np(p2, K=9, B=40, workers=1)), ("numpy SIMD blocks, 2 threads", lambda: N.feature_criterion_np(p2, K=9, B=40, workers=2))):
    g0 = p2.gen_seconds; t0 = time.perf_counter(); c0 = time.process_time(); r = fn(); w = time.perf_counter() - t0 - (p2.gen_seconds - g0)
    print(f"{name:30s} n=1e5 p=64 K=9 B=40: compute {w:6.2f} s wall, {time.process_time() - c0:6.2f} cpu-s; stages {({k: round(v, 2) for k, v in r['timing'].items()})}")
