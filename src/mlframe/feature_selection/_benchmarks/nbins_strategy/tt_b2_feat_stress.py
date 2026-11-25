"""B2 feature-side STRESS fixtures that the zoo does not cover: off-centre square, a link outside the candidate set (sin), a step, and a correlated covariate z (the conditional
residual of y after OLS on z). Judge = ridge on [z, f(x)] on 50000 fresh rows; the spline ceiling (ridge on [z, B-splines of x]) shows how much a unary candidate can not reach.

python tt_b2_feat_stress.py --out FILE [--n 1000,5000] [--seeds 20]
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np

import tt_b2_core as C
import tt_b2_crit as K
import tt_b2_fix as FX
from tt_b2_run_feat import judge_feature

STRESS = ("offset_square", "sin2", "step", "corr_log", "corr_none", "outlier_x")


def gen(kind, n, rng):
    z = rng.standard_normal(n)
    e = rng.standard_normal(n)
    if kind == "offset_square":
        x = rng.standard_normal(n)
        g = (x - 1.5) ** 2
    elif kind == "sin2":
        x = rng.standard_normal(n)
        g = np.sin(2 * x)
    elif kind == "step":
        x = rng.standard_normal(n)
        g = (x > 0.5).astype(float)
    elif kind == "corr_log":  # x lognormal, link log, covariate z strongly correlated with log x
        lx = rng.standard_normal(n)
        x = np.exp(lx)
        z = 0.8 * lx + 0.6 * z
        g = lx
    elif kind == "corr_none":  # x lognormal, LINEAR link, z correlated with x on the log scale (identity is the truth)
        lx = rng.standard_normal(n)
        x = np.exp(lx)
        z = 0.8 * lx + 0.6 * z
        g = x
    elif kind == "outlier_x":  # linear link, but 0.5 % of x are huge (x * 30)
        x = rng.standard_normal(n)
        big = rng.random(n) < 0.005
        x = np.where(big, x * 30, x)
        g = x
    else:
        raise ValueError(kind)
    g = (g - g.mean()) / (g.std() + 1e-12)
    return x, z, g + 0.5 * z + 0.7 * e


def spline_ceiling(x, z, y, xt, zt, yt):
    from sklearn.preprocessing import SplineTransformer

    s = SplineTransformer(n_knots=10, degree=3, knots="quantile").fit(x[:, None])
    D = np.c_[np.ones(len(y)), z, s.transform(x[:, None])]
    Dt = np.c_[np.ones(len(yt)), zt, s.transform(xt[:, None])]
    W = C._ridge_multi(D, y[:, None])[:, 0]
    return float(np.sqrt(np.mean((yt - Dt @ W) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", default="1000,5000")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--B", type=int, default=100)
    a = ap.parse_args()
    t0 = time.perf_counter()
    with open(a.out, "w", encoding="utf-8") as fh:
        for kind in STRESS:
            for n in map(int, a.n.split(",")):
                for s in range(a.seeds):
                    ss = zlib.crc32(f"stress|{kind}|{n}|{s}".encode())
                    x, z, y = gen(kind, n, np.random.default_rng(ss + 1))
                    xt, zt, yt = gen(kind, 50000, np.random.default_rng(ss + 2))
                    d = K.feat_decide(x, z[:, None], y, np.random.default_rng(ss), B=a.B)
                    j = judge_feature(x, z, y, xt, zt, yt)
                    dp = FX.feat_decide_partial(x, z[:, None], y, np.random.default_rng(ss), B=a.B)
                    ceil = spline_ceiling(x, z, y, xt, zt, yt)
                    fh.write(json.dumps(dict(dist=kind, link="stress", n=n, amp=1.0, seed=s, pick=d["pick"], pick_partial=dp["pick"], best=d["best"], gain=d["gain"], tau=d["tau"], pval=d["pval"],
                                             rmse={k: v[0] for k, v in j.items()}, mae={k: v[1] for k, v in j.items()}, ceil=ceil)) + "\n")
                    fh.flush()
    print("DONE", f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
