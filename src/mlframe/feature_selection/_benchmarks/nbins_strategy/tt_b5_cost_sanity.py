"""Do the fast criteria still DECIDE correctly?  Target: lognormal -> log, cube -> cbrt, linear -> identity (n=20000, p=8, B=40, 5 seeds).
Feature: p=12 columns, n=20000, y = log(x3) + 0.7 sqrt(x5) + 0.5 x7 + 0.3 e with positive columns 3 and 5 (lognormal / chi2-like), x7 linear, the rest noise; the criterion is applied to the
residual of y on a linear fit of ALL raw columns (so a raw column that explains y linearly is credited by the OLS, the transform gain is what is left)."""
import warnings
import numpy as np
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
import tt_b5_cost as C


def tgt_fix(kind, n, seed):
    rng = np.random.default_rng(100 + seed)
    X = rng.standard_normal((n, 8)).astype(np.float32); e = rng.standard_normal(n)
    if kind == "lognormal": y = np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.3 * e)
    elif kind == "cube": y = (3 + 0.6 * X[:, 0] + 0.4 * X[:, 1] + 0.25 * X[:, 2] + 0.15 * e) ** 3
    elif kind == "linear": y = 10 + 2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + e
    elif kind == "hetero_lognormal": y = np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2]) * (1 + 0.3 * np.abs(X[:, 3])) * np.exp(0.3 * e)
    return np.ascontiguousarray(X.T), y


for kind in ("lognormal", "cube", "linear", "hetero_lognormal"):
    res = []
    for seed in range(5):
        Xt, y = tgt_fix(kind, 20000, seed)
        prov = C.ArrayProvider(Xt, y, chunk_rows=2000)
        f = C.target_criterion(prov, K=10, B=40, seed=seed)
        res.append((f["best"], round(f["best_gain"], 3), round(f["tau"], 3), f["accepted"]))
    print("TARGET", kind, res, flush=True)

for seed in range(5):
    rng = np.random.default_rng(500 + seed)
    n, p = 20000, 12
    X = rng.standard_normal((p, n)).astype(np.float32)
    X[3] = np.exp(0.8 * rng.standard_normal(n)); X[5] = rng.chisquare(3, n)
    y = np.log(X[3]) + 0.7 * np.sqrt(X[5]) + 0.5 * X[7] + 0.3 * rng.standard_normal(n)
    D = np.hstack([np.ones((n, 1)), X.T.astype(np.float64)])
    coef = np.linalg.lstsq(D, y, rcond=None)[0]
    r = y - D @ coef
    prov = C.ArrayProvider(X, r, chunk_rows=2000)
    f = C.feature_criterion(prov, K=10, B=40, seed=seed, beta=coef[1:])
    acc = {int(j): (C.FEATURE_CANDS[f["best_k"][j]], round(float(f["best_gain"][j]), 4), round(float(f["tau"][j]), 4)) for j in range(p) if f["accepted"][j]}
    print("FEATURE seed", seed, "accepted columns (cand, gain, tau):", acc, "| n accepted", int(f["accepted"].sum()), flush=True)
