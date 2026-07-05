"""Task C bench: large-n COST of fe_synergy_screen_max_features=250 (was 60).

At p~200: cap=60 SKIPS the O(p^2) synergy joint-MI sweep (200>60); cap=250 ENABLES
it (200<=250). So minimal vs medium here isolates the synergy-sweep wall-time as a
function of n. Bench MRMR.fit wall-time at p=200 across n in {5000,20000,100000}.

Frugal: n_jobs=2, one fit per (n,cap). The frame has a genuine zero-marginal synergy
pair (x0*x1) so the bootstrap has something real to find at cap=250.
"""
from __future__ import annotations
import time, warnings, gc
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from mlframe.feature_selection.filters.mrmr import MRMR

N_JOBS = 2
P = 200
NS = [5000, 20000, 100000]
CAPS = [60, 250]


def make(n, p=P, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    # zero-marginal synergy (x0*x1) + a couple linear signals; rest noise
    logit = 1.0 * X[:, 0] * X[:, 1] + 0.8 * X[:, 2] + 0.6 * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def fit_time(df, ys, cap):
    sel = MRMR(
        fe_synergy_screen_max_features=cap,
        fe_max_steps=1,
        fe_binary_preset="minimal",
        fe_unary_preset="minimal",
        verbose=0, random_seed=42, n_jobs=N_JOBS,
    )
    t0 = time.perf_counter()
    sel.fit(df, ys)
    ft = time.perf_counter() - t0
    _eng = getattr(sel, "_engineered_features_", None)
    n_eng = len(_eng) if _eng is not None else 0
    _sup = getattr(sel, "support_", None)
    n_sel = len(_sup) if _sup is not None else 0
    return ft, n_eng, n_sel


def main():
    print(f"p={P} caps={CAPS} ns={NS} n_jobs={N_JOBS}")
    print(f"{'n':>8} {'cap':>5} {'fit_s':>8} {'n_eng':>5} {'n_sel':>5}")
    results = {}
    for n in NS:
        df, ys = make(n)
        for cap in CAPS:
            ft, ne, nsel = fit_time(df, ys, cap)
            results[(n, cap)] = ft
            print(f"{n:>8} {cap:>5} {ft:>8.2f} {ne:>5} {nsel:>5}", flush=True)
        del df, ys; gc.collect()
    print("\n=== synergy-sweep cost: cap=250 vs cap=60 ===")
    for n in NS:
        t60, t250 = results[(n, 60)], results[(n, 250)]
        d = t250 - t60
        print(f"  n={n:>7}: cap60={t60:.2f}s cap250={t250:.2f}s  " f"delta=+{d:.2f}s ({100*d/max(t60,1e-9):+.0f}%)")


if __name__ == "__main__":
    main()
