"""Extended polynom-pair optimizer A/B: harder synthetic targets x 4 polynomial bases (chebyshev/hermite/
legendre/laguerre) x 4 variants (cma_batch, random_batch, numba_kernel, cupy_kernel). Extends
bench_polynom_optimizer_variants_ab.py (which fixed basis=chebyshev) after the ncu-driven GPU kernel round.
optuna/cma dropped -- already established strictly slower at parity in the 6x6 table.

Metrics per (variant, basis, case, seed): best MI over restarts, total wall, and TIME-TO-FIRST-BEST --
the wall-clock offset at which the eventual best MI was FIRST reached (restart granularity: each restart's
completion timestamp; a variant that finds the winner on restart 1 of 3 beats one that needs restart 3 even
at equal final MI). Everything is persisted INCREMENTALLY to results/bench_polynom_optimizer_bases.json --
rerunning skips already-computed combos, so results are never recomputed.

bench-attempt-rejected (numba cuda.jit coder twin, 2026-07-15): a cuda.jit port of the folded-key qbin coder
measured 12.59ms vs the RawKernel's 8.57ms (1.47x) on the 99401x617 timing twin -- the RawKernel > numba.cuda
ladder holds on cc 8.9 as it did on cc 6.1; RawKernel stays the only shipped GPU coder.
"""
import json
import os
import time
from os.path import dirname, join

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair

N = 20000
N_TRIALS = 60
RESTARTS = 3
SEEDS = (42, 43, 44)
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
i, j = _r.standard_normal(N), _r.standard_normal(N)
CASES["ring_band"] = (i, j, ((i**2 + j**2 > 1.0) & (i**2 + j**2 < 2.5)).astype(np.int64))
k, l = _r.standard_normal(N), _r.standard_normal(N)
CASES["xor_quadrant"] = (k, l, ((k > 0) ^ (l > 0)).astype(np.int64))
p_, q_ = _r.standard_normal(N), _r.standard_normal(N)
CASES["quintic_mix"] = (p_, q_, (((p_**5 - 4 * p_**3 + 2 * p_) * (q_**3 - q_)) > 0).astype(np.int64))
u_, v_ = _r.standard_normal(N) * 0.7, _r.standard_normal(N) * 0.7
CASES["atan2_sector"] = (u_, v_, ((np.arctan2(u_, v_) % (2 * np.pi / 3)) < (np.pi / 3)).astype(np.int64))

BASES = ("chebyshev", "hermite", "legendre", "laguerre")
VARIANTS = ("cma_batch", "random_batch", "numba_kernel", "cupy_kernel")
BASE = dict(min_degree=3, max_degree=6, coef_range=(-2.0, 2.0), l2_penalty=0.05,
            sweep_degrees=True, mi_estimator="plugin", discrete_target=True,
            warm_start=True, multi_fidelity=False)
RESULTS_PATH = join(dirname(__file__), "results", "bench_polynom_optimizer_bases.json")


def _load() -> dict:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save(db: dict) -> None:
    os.makedirs(dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=1, sort_keys=True)


def run_combo(opt: str, basis: str, case: str, seed: int) -> dict:
    xa, xb, y = CASES[case]
    t0 = time.perf_counter()
    best_mi, best_first_at, best_restart = 0.0, float("nan"), -1
    restarts = []
    for ro in range(RESTARTS):
        res = optimise_hermite_pair(x_a=xa, x_b=xb, y=y, seed=seed + ro, optimizer=opt,
                                    n_trials=N_TRIALS, basis=basis, **BASE)
        t_done = time.perf_counter() - t0
        mi = float(res.mi) if res is not None else 0.0
        restarts.append({"restart": ro, "mi": mi, "done_at_s": round(t_done, 3)})
        if mi > best_mi:
            best_mi, best_first_at, best_restart = mi, t_done, ro
    return {"mi": round(best_mi, 6), "time_to_best_s": round(best_first_at, 3),
            "best_restart": best_restart, "wall_s": round(time.perf_counter() - t0, 3),
            "restarts": restarts, "n_trials": N_TRIALS}


if __name__ == "__main__":
    db = _load()
    for opt in VARIANTS:
        for basis in BASES:
            for case in CASES:
                for seed in SEEDS:
                    key = f"{opt}|{basis}|{case}|{seed}|{N_TRIALS}x{RESTARTS}"
                    if key in db:
                        continue
                    db[key] = run_combo(opt, basis, case, seed)
                    _save(db)  # incremental: a killed run resumes where it stopped
            done = [k for k in db if k.startswith(f"{opt}|{basis}|")]
            print(f"[{opt} x {basis}] {len(done)} combos in db", flush=True)
    # summary table: median-over-seeds MI (time_to_best in parens), one row per variant x basis
    print("\n| variant | basis | wall_s | " + " | ".join(CASES) + " |", flush=True)
    print("|---|---|---|" + "---|" * len(CASES), flush=True)
    for opt in VARIANTS:
        for basis in BASES:
            cells, wall = [], 0.0
            for case in CASES:
                rows = [db[f"{opt}|{basis}|{case}|{s}|{N_TRIALS}x{RESTARTS}"] for s in SEEDS]
                wall += sum(r["wall_s"] for r in rows)
                mis = sorted(r["mi"] for r in rows)
                ttb = sorted(r["time_to_best_s"] for r in rows)
                cells.append(f"{mis[len(mis)//2]:.4f} ({ttb[len(ttb)//2]:.1f}s)")
            print(f"| {opt} | {basis} | {wall:.0f} | " + " | ".join(cells) + " |", flush=True)
