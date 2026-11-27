"""tt_a5: variance-stabilising / residual-scale criterion for target transforms (agent 5, 2026-11-23).

For a candidate f: z=f(y) (standardised), cross-fitted closed-form additive-spline ridge fit on the features, residual r. H(f) = mean_j MI_rank(x_j; |r|) + MI_rank(fitted mean; |r|)
(rank bins, so both variables are on one scale and the MI is invariant). Pick argmin H; accept over identity only if the drop exceeds a permutation-null threshold.
Judge: held-out RMSE on the original scale, ridge (linear features) and HistGradientBoosting; inverse clipped to the training z range.
Usage: python tt_a5_resid_scale.py [--seeds 4] [--rows 4000]
"""
from __future__ import annotations

import logging
import argparse
import time
import warnings
from collections import Counter

import numpy as np
from scipy.stats import rankdata, skew

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)


def rcodes(v, k):
    return np.minimum((rankdata(v, method="ordinal") - 1) * k // len(v), k - 1).astype(np.int64)


def mi(a, b, k):
    j = np.bincount(a * k + b, minlength=k * k).reshape(k, k) / len(a)
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    m = j > 0
    return float((j[m] * np.log(j[m] / (pa @ pb)[m])).sum())


def cands(ytr):
    from sklearn.preprocessing import PowerTransformer
    c = {"identity": (lambda t: t, lambda z: z), "cbrt": (np.cbrt, lambda z: z ** 3)}
    if ytr.min() >= 0:
        c["sqrt"] = (np.sqrt, lambda z: np.maximum(z, 0) ** 2)
        c["log1p"] = (np.log1p, np.expm1)
    if ytr.min() > 0:
        c["log"] = (np.log, np.exp)
    pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(ytr.reshape(-1, 1))
    c["yeo_johnson"] = (lambda t: pt.transform(np.asarray(t).reshape(-1, 1)).ravel(),
                        lambda z: pt.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel())
    return c


def crossfit_resid(X, z, lam=1e-2):
    """2-fold cross-fitted additive-spline ridge residuals and fitted values (closed form)."""
    from sklearn.preprocessing import SplineTransformer
    n = len(z)
    idx = np.arange(n) % 2
    r = np.empty(n)
    m = np.empty(n)
    for f in (0, 1):
        tr, te = idx != f, idx == f
        st = SplineTransformer(n_knots=5, degree=3, extrapolation="linear").fit(X[tr])
        B = np.c_[np.ones(tr.sum()), st.transform(X[tr])]
        w = np.linalg.solve(B.T @ B + lam * n * np.eye(B.shape[1]), B.T @ z[tr])
        m[te] = np.c_[np.ones(te.sum()), st.transform(X[te])] @ w
        r[te] = z[te] - m[te]
    return r, m


def het_score(X, z, k, rng=None, perm=0):
    z = (z - z.mean()) / (z.std() + 1e-12)
    r, m = crossfit_resid(X, z)
    ca = rcodes(np.abs(r), k)
    cx = [rcodes(X[:, j], k) for j in range(X.shape[1])] + [rcodes(m, k)]
    h = float(np.mean([mi(c, ca, k) for c in cx]))
    null = []
    for _ in range(perm):
        cp = rng.permutation(ca)
        null.append(float(np.mean([mi(c, cp, k) for c in cx])))
    return h, float(skew(r)), null


def rmse_models(Xtr, ytr, Xte, yte, f, finv):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    ztr = f(ytr)
    lo, hi = ztr.min(), ztr.max()
    out = {}
    for nm, mdl in (("ridge", Ridge(1.0)), ("hgb", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=0))):
        p = mdl.fit(Xtr, ztr).predict(Xte)
        out[nm] = float(np.sqrt(np.mean((finv(np.clip(p, lo, hi)) - yte) ** 2)))
    return out


def make(kind, n, seed, p=6):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    g = 0.8 * X[:, 0] + 0.5 * X[:, 1]
    if kind == "mult_lognormal":
        y = np.exp(0.5 * g + 0.5 * rng.standard_normal(n))
    elif kind == "poisson":
        y = rng.poisson(np.exp(1.5 + 0.5 * g)).astype(float)
    elif kind == "gamma_mult":
        y = np.exp(1 + 0.4 * g) * rng.gamma(4, 1 / 4, n)
    elif kind == "linear_homosc":
        y = 10 + 3 * g + rng.standard_normal(n)
    elif kind == "linear_pos_homosc":
        y = 40 + 4 * g + 2 * rng.standard_normal(n)
    elif kind == "cube_homosc":
        y = (g + 0.3 * rng.standard_normal(n)) ** 3
    elif kind == "hetero_linear":
        y = 20 + 3 * g + (1 + 1.2 * np.abs(X[:, 0])) * rng.standard_normal(n)
    else:
        raise ValueError(kind)
    return X, y


def real():
    from sklearn.datasets import fetch_california_housing, load_diabetes
    out = {}
    d = load_diabetes()
    out["diabetes"] = (d.data, d.target.astype(float))
    try:
        c = fetch_california_housing(download_if_missing=False)
        out["california"] = (c.data, c.target.astype(float))
    except Exception as e:
        logging.getLogger(__name__).debug("tt_a5_resid_scale: ignored %s: %s", type(e).__name__, e)
        print("california NOT cached:", e)
    return out


def evaluate(X, y, seed, k=10, perm=20):
    rng = np.random.default_rng(seed)
    n = len(y)
    tr_mask = rng.permutation(n) < int(0.6 * n)
    Xtr, ytr, Xte, yte = X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask]
    k = int(min(k, max(4, len(ytr) // 60)))
    res = {}
    nullid = []
    for nm, (f, finv) in cands(ytr).items():
        t = time.perf_counter()
        h, sk, null = het_score(Xtr, f(ytr), k, rng, perm if nm == "identity" else 0)
        dt = time.perf_counter() - t
        if nm == "identity":
            nullid = null
        res[nm] = dict(H=h, skew=sk, t=dt, **rmse_models(Xtr, ytr, Xte, yte, f, finv))
    sd0 = float(np.std(nullid))
    thr = 1.645 * np.sqrt(2) * sd0
    best = min(res, key=lambda v: res[v]["H"])
    ok = res["identity"]["H"] - res[best]["H"] > thr and res["identity"]["H"] > np.quantile(nullid, 0.95)
    pick = best if ok else "identity"
    pick_sk = min(res, key=lambda v: abs(res[v]["skew"]))
    return res, pick, pick_sk, thr, float(np.mean(nullid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4000)
    a = ap.parse_args()
    kinds = ["mult_lognormal", "poisson", "gamma_mult", "linear_homosc", "linear_pos_homosc", "cube_homosc", "hetero_linear"]
    jobs = [(k, [make(k, a.rows, s) for s in range(a.seeds)]) for k in kinds]
    for nm, (X, y) in real().items():
        jobs.append((nm, [(X, y)] * (a.seeds if len(y) < 5000 else 2)))
    for kind, lst in jobs:
        picks, sks, br, bh, rr, rh, ts, Hs = [], [], [], [], [], [], [], {}
        for s, (X, y) in enumerate(lst):
            res, pick, pk_sk, thr, nm_ = evaluate(X, y, s)
            picks.append(pick)
            sks.append(pk_sk)
            br.append(min(res, key=lambda v: res[v]["ridge"]))
            bh.append(min(res, key=lambda v: res[v]["hgb"]))
            rr.append(res[pick]["ridge"] / res["identity"]["ridge"])
            rh.append(res[pick]["hgb"] / res["identity"]["hgb"])
            ts.append(np.mean([v["t"] for v in res.values()]) * 1e3)
            for v in res:
                Hs.setdefault(v, []).append(res[v]["H"])
        print(kind, "H:", {v: round(float(np.mean(h)), 4) for v, h in Hs.items()}, "null:", round(nm_, 4), "thr:", round(thr, 4))
        print("   pick", dict(Counter(picks)), "| skewpick", dict(Counter(sks)), "| best ridge", dict(Counter(br)), "best hgb", dict(Counter(bh)),
              "| ratio(pick/identity) ridge %.3f hgb %.3f | %.0f ms/cand" % (np.mean(rr), np.mean(rh), np.mean(ts)), flush=True)


if __name__ == "__main__":
    main()
