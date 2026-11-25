"""Wall and CPU time of recommend_feature_transforms at n=100000 and p in {100, 1000} (research B4, direction (1)).

The panel is generated column by column straight into a float32 matrix (mixed marginals, blocks of 5 correlated at 0.5, 4 informative columns with log / square /
sqrt / abs links) so p=1000 fits in memory. Reports wall seconds, process CPU seconds (robust to other jobs on the machine), the screening share, the peak
working set, the number of accepted columns, and the float32-vs-float64 agreement of the gain statistic on a smaller case.
Usage: python tt_b4_timing.py --p 100 --n 100000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
from scipy.special import ndtr

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_reference import recommend_feature_transforms


def rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return float("nan")


def make_big(n, p, seed):
    r = np.random.default_rng(seed)
    X = np.empty((n, p), dtype=np.float32)
    sig = np.zeros(n)
    links = {0: "log", 1: "square", 2: "sqrt", 3: "abs"}
    common = None
    for j in range(p):
        if j % 5 == 0:
            common = r.standard_normal(n)
        z = np.sqrt(0.5) * common + np.sqrt(0.5) * r.standard_normal(n)
        if j in links:
            lk = links[j]
            if lk == "log":
                x = np.exp(0.8 * z); g = np.log(x)
            elif lk == "square":
                x = z; g = z ** 2
            elif lk == "sqrt":
                x = -np.log1p(-np.clip(ndtr(z), 0, 1 - 1e-12)); g = np.sqrt(x)
            else:
                x = z; g = np.abs(z)
            sig += (g - g.mean()) / g.std()
        else:
            kind = r.choice(7, p=(0.34, 0.14, 0.14, 0.10, 0.10, 0.10, 0.08))
            if kind == 0:
                x = z
            elif kind == 1:
                x = np.exp(0.8 * z)
            elif kind == 2:
                x = r.standard_t(3, n) * 0.5 + z
            elif kind == 3:
                x = r.poisson(3.0, n).astype(float)
            elif kind == 4:
                x = (1 + np.digitize(z, [-1.0, -0.3, 0.3, 1.0])).astype(float)
            elif kind == 5:
                x = -np.log1p(-np.clip(ndtr(z), 0, 1 - 1e-12))
            else:
                x = ndtr(z)
        X[:, j] = x
    y = sig + sig.std() * np.sqrt(0.3 / 0.7) * r.standard_normal(n)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_null", type=int, default=50)
    ap.add_argument("--agree", action="store_true", help="float32 vs float64 agreement on a 20000 x 100 case, then exit")
    a = ap.parse_args()
    if a.agree:
        X, y = make_big(20000, 100, 1)
        r32 = recommend_feature_transforms(X, y, n_null=a.n_null, seed=0, dtype=np.float32, replicate_half=False)
        r64 = recommend_feature_transforms(X, y, n_null=a.n_null, seed=0, dtype=np.float64, replicate_half=False)
        d = np.abs(r32.detail["T_obs"] - r64.detail["T_obs"])
        dn = np.abs(r32.detail["Tn"] - r64.detail["Tn"])
        print(f"float32 vs float64 (20000 x 100, no split): max |dT_obs| = {d.max():.2e}, max |dT_null| = {dn.max():.2e}, smallest 95% null gain threshold "
              f"= {np.quantile(r64.detail['Tn'].max(axis=1), 0.95):.2e}, same accepted set = {bool((r32.table['accepted'] == r64.table['accepted']).all())}")
        return
    t = time.perf_counter(); time.process_time()
    X, y = make_big(a.n, a.p, a.seed)
    print(f"generated {X.shape} in {time.perf_counter() - t:.1f} s wall, rss {rss_mb():.0f} MB", flush=True)
    for label, kw in (("split (discover on half A, replicate on half B)", dict(replicate_half=True)), ("no split (all rows for discovery)", dict(replicate_half=False))):
        w0, c0 = time.perf_counter(), time.process_time()
        res = recommend_feature_transforms(X, y, n_null=a.n_null, seed=0, rule="gated", **kw)
        w, cp = time.perf_counter() - w0, time.process_time() - c0
        acc = np.flatnonzero(res.table["accepted"])
        print(f"p={a.p} n={a.n} {label}: wall {w:.1f} s, cpu {cp:.1f} s, screening {res.detail['screen_seconds']:.1f} s, rss now {rss_mb():.0f} MB, "
              f"accepted {[(int(j), res.table['best'][j]) for j in acc]}, relevant (p_rel<=0.05) {int((res.table['p_rel'] <= 0.05).sum())}", flush=True)


if __name__ == "__main__":
    main()
