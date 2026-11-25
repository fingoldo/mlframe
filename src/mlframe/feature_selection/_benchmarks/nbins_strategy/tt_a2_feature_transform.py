"""Model-free criteria that recognise which unary transform of a FEATURE helps a linear / additive model (research agent 2, 2026-11-23).

Fixtures: y = g(x) + 0.5*z + noise (z a second, linear feature, used for the conditional criteria). Judge: held-out R2 of ridge on [z, f(x)] (and a spline GAM ceiling).
Criteria (no model fitted): Pearson info -0.5 log(1-rho^2) of f(x) vs y; same vs rank-gauss y; equal-width MI(f(x); y); distance correlation; each in a marginal and a conditional
(y minus OLS on z) form. Decision: best candidate accepted only if its gain over identity exceeds the 95th percentile of the permutation null of max_f(crit_f - crit_id).
Usage: python tt_a2_feature_transform.py [--seeds 7] [--rows 20000]
"""
from __future__ import annotations

import argparse
import collections
import time
import warnings

import numpy as np
from scipy import stats

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)


def make(kind, n, seed):
    r = np.random.default_rng(seed)
    z = r.standard_normal(n)
    e = r.standard_normal(n)
    if kind == "log":
        x = np.exp(r.standard_normal(n)); g = np.log(x)
    elif kind == "recip":
        x = 0.5 + 0.5 * np.exp(r.standard_normal(n)); g = 2.0 / x
    elif kind == "sqrt":
        x = r.exponential(1.0, n); g = 1.5 * np.sqrt(x)
    elif kind == "square":
        x = r.standard_normal(n); g = x ** 2
    elif kind == "abs":
        x = r.standard_normal(n); g = np.abs(x)
    elif kind == "sigmoid":
        x = 2 * r.standard_normal(n); g = 2.0 * np.tanh(2 * x)
    elif kind == "ctl_linear":
        x = r.standard_normal(n); g = x
    elif kind == "ctl_lognormal_linear":
        x = np.exp(r.standard_normal(n)); g = 0.5 * x
    elif kind == "ctl_noise":
        x = np.exp(r.standard_normal(n)); g = 0 * x
    else:
        raise ValueError(kind)
    return x, z, g + 0.5 * z + 0.7 * e


KINDS = ["log", "recip", "sqrt", "square", "abs", "sigmoid", "ctl_linear", "ctl_lognormal_linear", "ctl_noise"]


def cands(x):
    med = np.median(x)
    sc = np.subtract(*np.percentile(x, [75, 25])) + 1e-12
    c = {"id": x}
    if x.min() > 0:
        c.update(log=np.log(x), sqrt=np.sqrt(x), recip=1.0 / x)
    c.update(square=(x - med) ** 2, abs=np.abs(x - med), tanh=np.tanh((x - med) / sc), cbrt=np.cbrt(x))
    return c


def info(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    r = 0.0 if d == 0 else (a * b).sum() / d
    return -0.5 * np.log(max(1 - r * r, 1e-12))


def rankgauss(y):
    return stats.norm.ppf((stats.rankdata(y) - 0.5) / len(y))


def ew_mi(f, ycodes, k=16):
    e = np.linspace(f.min(), f.max(), k + 1)[1:-1]
    a = np.searchsorted(e, f, side="right")
    j = np.zeros((k, ycodes.max() + 1)); np.add.at(j, (a, ycodes), 1); j /= j.sum()
    pa = j.sum(1, keepdims=True); pb = j.sum(0, keepdims=True); m = j > 0
    return float((j[m] * np.log(j[m] / (pa @ pb)[m])).sum())


def dcor(a, b):
    def c(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    A, B = c(a), c(b)
    return np.sqrt(max((A * B).mean(), 0) / np.sqrt((A * A).mean() * (B * B).mean() + 1e-300))


def crit_all(C, y, z, sub, modes=("m", "c")):
    out = {}
    A = np.c_[np.ones_like(z), z]
    for mode in modes:
        yy = y if mode == "m" else y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        yrg = rankgauss(yy)
        yc = np.searchsorted(np.quantile(yy, np.linspace(0, 1, 17)[1:-1]), yy, side="right")
        out["pear_" + mode] = {k: info(v, yy) for k, v in C.items()}
        out["pearRG_" + mode] = {k: info(v, yrg) for k, v in C.items()}
        out["ewMI_" + mode] = {k: ew_mi(v, yc) for k, v in C.items()}
        out["dcor_" + mode] = {k: dcor(v[sub], yy[sub]) for k, v in C.items()}
        out["rankMI_" + mode] = {"id": ew_mi(rankgauss(C["id"]), yc)}
    return out


def judge(Ctr, Cte, ytr, yte, ztr, zte):
    from sklearn.linear_model import Ridge
    return {k: float(Ridge(alpha=1.0).fit(np.c_[ztr, Ctr[k]], ytr).score(np.c_[zte, Cte[k]], yte)) for k in Ctr}


def gam(xtr, xte, ytr, yte, ztr, zte):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer
    s = SplineTransformer(n_knots=10, degree=3, knots="quantile").fit(xtr[:, None])
    return float(Ridge(alpha=1.0).fit(np.c_[ztr, s.transform(xtr[:, None])], ytr).score(np.c_[zte, s.transform(xte[:, None])], yte))


CRIT = ["pear_m", "pearRG_m", "ewMI_m", "dcor_m", "pear_c", "pearRG_c", "ewMI_c", "dcor_c"]


def run(kind, seed, n, nperm=25):
    x, z, y = make(kind, n, seed)
    h = n // 2
    C = cands(x)
    Ctr = {k: v[:h] for k, v in C.items()}; Cte = {k: v[h:] for k, v in C.items()}
    ytr, yte, ztr, zte = y[:h], y[h:], z[:h], z[h:]
    r2 = judge(Ctr, Cte, ytr, yte, ztr, zte)
    g = gam(x[:h], x[h:], ytr, yte, ztr, zte)
    sub = np.random.default_rng(seed).choice(h, 600, replace=False)
    t = time.perf_counter(); obs = crit_all(Ctr, ytr, ztr, sub); t_obs = time.perf_counter() - t
    rng = np.random.default_rng(seed + 99)
    nulls = {c: [] for c in CRIT}
    A = np.c_[np.ones(h), ztr]; fit = A @ np.linalg.lstsq(A, ytr, rcond=None)[0]; resid = ytr - fit
    for _ in range(nperm):
        om = crit_all(Ctr, ytr[rng.permutation(h)], ztr, sub, ("m",))
        oc = crit_all(Ctr, fit + resid[rng.permutation(h)], ztr, sub, ("c",))
        for c in CRIT:
            o = om if c.endswith("_m") else oc
            nulls[c].append(max(o[c][k] - o[c]["id"] for k in Ctr if k != "id"))
    best = max(r2, key=r2.get)
    res = {"r2": r2, "gam": g, "best": best, "gain": r2[best] - r2["id"], "t_per_cand": t_obs / len(Ctr), "crit": {}}
    for c in CRIT:
        v = obs[c]; pick = max((k for k in v if k != "id"), key=v.get)
        res["crit"][c] = (pick, v[pick] - v["id"] > float(np.quantile(nulls[c], 0.95)), v[pick] - v["id"])
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=7); ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args()
    tab = {}
    for kind in KINDS:
        rs = [run(kind, s, a.rows) for s in range(a.seeds)]
        print(f"\n== {kind}: judge best={collections.Counter(r['best'] for r in rs).most_common(3)} R2 id={np.mean([r['r2']['id'] for r in rs]):.3f} "
              f"best={np.mean([r['r2'][r['best']] for r in rs]):.3f} GAM={np.mean([r['gam'] for r in rs]):.3f} t/cand={np.mean([r['t_per_cand'] for r in rs]) * 1e3:.1f}ms", flush=True)
        for c in CRIT:
            ok = acc = 0; picks = collections.Counter()
            for r in rs:
                pk, ac, _ = r["crit"][c]; acc += ac; picks[pk if ac else "-"] += 1
                if r["gain"] > 0.02:
                    ok += bool(ac and r["r2"][pk] >= r["r2"][r["best"]] - 0.02)
                else:
                    ok += (not ac)
            print(f"  {c:9s} correct {ok}/{a.seeds} accepted {acc}/{a.seeds} picks {dict(picks)}", flush=True)
            tab[(kind, c)] = ok
    print("\nSUMMARY correct decisions out of", a.seeds)
    print("crit      " + " ".join(f"{k[:9]:>9s}" for k in KINDS))
    for c in CRIT:
        print(f"{c:9s} " + " ".join(f"{tab[(k, c)]:>9d}" for k in KINDS))


if __name__ == "__main__":
    main()
