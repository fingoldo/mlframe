"""Extended polynom-pair optimizer A/B: harder synthetic targets x 4 polynomial bases (chebyshev/hermite/
legendre/laguerre) x 4 variants (cma_batch, random_batch, numba_kernel, cupy_kernel). Extends
bench_polynom_optimizer_variants_ab.py (which fixed basis=chebyshev) after the ncu-driven GPU kernel round:
per-variant wall + median-over-seeds best MI per (case, basis). optuna/cma dropped -- already established
strictly slower at parity in the 6x6 table.

bench-attempt-rejected (numba cuda.jit coder twin, 2026-07-15): a cuda.jit port of the folded-key qbin coder
measured 12.59ms vs the RawKernel's 8.57ms (1.47x) on the 99401x617 timing twin -- the RawKernel > numba.cuda
ladder holds on cc 8.9 as it did on cc 6.1; RawKernel stays the only shipped GPU coder.
"""
import time

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

N = 20000
_r = np.random.default_rng(7)
CASES = {}
a = _r.standard_normal(N); b = _r.standard_normal(N)
CASES["cubic_inner"] = (a, b, ((a**3 - 2 * a) * b > 0).astype(np.int64))
g = _r.standard_normal(N); h = _r.standard_normal(N) * (1 + 2 * (_r.random(N) < 0.1))
CASES["ratio_regime"] = (g, h, ((g / np.where(np.abs(h) < 0.05, 0.05, h)) > 1.0).astype(np.int64))
c = _r.standard_normal(N); d = _r.standard_normal(N)
CASES["logmult_heavy"] = (np.exp(c), np.exp(d), ((c + d) > 0.5).astype(np.int64))
e = _r.standard_normal(N); f = _r.standard_normal(N)
CASES["cross_cheb"] = (e, f, (((4 * e**3 - 3 * e) * (2 * f**2 - 1)) > 0).astype(np.int64))
# --- new hard targets ---
i, j = _r.standard_normal(N), _r.standard_normal(N)
CASES["ring_band"] = (i, j, ((i**2 + j**2 > 1.0) & (i**2 + j**2 < 2.5)).astype(np.int64))  # radial band: needs even polys both sides
k, l = _r.standard_normal(N), _r.standard_normal(N)
CASES["xor_quadrant"] = (k, l, ((k > 0) ^ (l > 0)).astype(np.int64))  # pure sign interaction: mul should nail it
p_, q_ = _r.standard_normal(N), _r.standard_normal(N)
CASES["quintic_mix"] = (p_, q_, (((p_**5 - 4 * p_**3 + 2 * p_) * (q_**3 - q_)) > 0).astype(np.int64))  # degree-5 x degree-3
u_, v_ = _r.standard_normal(N) * 0.7, _r.standard_normal(N) * 0.7
CASES["atan2_sector"] = (u_, v_, ((np.arctan2(u_, v_) % (2 * np.pi / 3)) < (np.pi / 3)).astype(np.int64))  # angular sectors: atan2 bf

BASES = ("chebyshev", "hermite", "legendre", "laguerre")
VARIANTS = ("cma_batch", "random_batch", "numba_kernel", "cupy_kernel")
SEEDS = (42, 43, 44)
RESTARTS = 5
BASE = dict(min_degree=3, max_degree=6, coef_range=(-2.0, 2.0), l2_penalty=0.05,
            sweep_degrees=True, mi_estimator="plugin", discrete_target=True,
            warm_start=True, multi_fidelity=False)

if __name__ == "__main__":
    print(f"| variant | basis | wall_s | " + " | ".join(CASES) + " |", flush=True)
    print("|---|---|---|" + "---|" * len(CASES), flush=True)
    for opt in VARIANTS:
        for basis in BASES:
            t0 = time.perf_counter()
            row = []
            for name, (xa, xb, y) in CASES.items():
                med = []
                for seed in SEEDS:
                    best = 0.0
                    for ro in range(RESTARTS):
                        res = optimise_hermite_pair(x_a=xa, x_b=xb, y=y, seed=seed + ro, optimizer=opt,
                                                    n_trials=100, basis=basis, **BASE)
                        if res is not None and res.mi > best:
                            best = res.mi
                    med.append(best)
                row.append(float(np.median(med)))
            wall = time.perf_counter() - t0
            print(f"| {opt} | {basis} | {wall:.1f} | " + " | ".join(f"{v:.4f}" for v in row) + " |", flush=True)
