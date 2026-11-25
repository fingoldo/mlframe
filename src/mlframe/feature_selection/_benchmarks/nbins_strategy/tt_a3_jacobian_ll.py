"""Jacobian-corrected Gaussian log-likelihood as a model-free proposer / criterion for a target transform (research agent 3, 2026-11-23).

For a candidate f, with a closed-form linear fit of f(y) on x: log p(y|x) = log N(f(y); xb, s2) + log|f'(y)|. Held-out mean per-row value is comparable across f (original scale).
Decomposition: LL_f = n/2*log(1/(1-R2_f)) [Gaussian MI of f(y) with x] + marginal Gaussian LL of f(y) (with the Jacobian); the Jacobian cancels in the gain, so f enters through
R2_f and the Gaussianity of f(y). Judge: ridge on f(y), clip to the train range of f(y), Duan smearing, RMSE on the original scale (+ HistGB).

Usage: python tt_a3_jacobian_ll.py [--seeds 7] [--rows 20000] [--hgb 1]
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtri

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
EPS = 1e-9


def make(kind: str, n: int, p: int, seed: int):
    """Fixtures: the five of target_transform_mi_ext plus exp_linear, reciprocal, heavy (t, 1.5 dof noise, no right transform)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    e = rng.standard_normal(n)
    w = np.array([0.9, 0.6, 0.4, -0.3, 0.2, 0, 0, 0])[:p]
    if kind == "lognormal":
        y = np.exp(0.9 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * e)
    elif kind == "hetero_lognormal":
        y = np.exp(0.9 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * (1 + np.abs(X[:, 0])) * e)
    elif kind == "cube":
        y = (X[:, 0] + 0.7 * X[:, 1] + 0.3 * e) ** 3
    elif kind == "linear":
        y = X[:, 0] + 0.7 * X[:, 1] + 0.3 * e
    elif kind == "hetero_linear":
        y = X[:, 0] + 0.7 * X[:, 1] + 0.3 * (1 + np.abs(X[:, 0])) * e
    elif kind == "exp_linear":
        y = np.exp(0.5 * (X @ w) + 0.5 * e)
    elif kind == "reciprocal":
        y = 1.0 / (3.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.15 * e)
    elif kind == "heavy":
        y = X[:, 0] + 0.7 * X[:, 1] + 0.5 * rng.standard_t(1.5, n)
    else:
        raise ValueError(kind)
    return X, y


# ---- transforms: forward, inverse, log-Jacobian log|f'(y)|, domain mask --------------------------------------------------------------------------------------------------
def bc(y, lam):
    return np.log(y) if abs(lam) < 1e-8 else (y**lam - 1) / lam


def bc_inv(z, lam):
    return np.exp(z) if abs(lam) < 1e-8 else np.maximum(lam * z + 1, 1e-12) ** (1 / lam)


def yj(y, lam):
    out = np.empty_like(y)
    pos = y >= 0
    out[pos] = np.log1p(y[pos]) if abs(lam) < 1e-8 else ((y[pos] + 1) ** lam - 1) / lam
    neg = ~pos
    out[neg] = -np.log1p(-y[neg]) if abs(lam - 2) < 1e-8 else -(((1 - y[neg]) ** (2 - lam)) - 1) / (2 - lam)
    return out


def yj_inv(z, lam):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = np.expm1(z[pos]) if abs(lam) < 1e-8 else np.maximum(z[pos] * lam + 1, 1e-12) ** (1 / lam) - 1
    neg = ~pos
    out[neg] = -np.expm1(-z[neg]) if abs(lam - 2) < 1e-8 else 1 - np.maximum(1 - (2 - lam) * z[neg], 1e-12) ** (1 / (2 - lam))
    return out


def jac_bc(y, lam):
    return (lam - 1) * np.log(y)


def jac_yj(y, lam):
    return (lam - 1) * np.sign(y) * np.log1p(np.abs(y))


def ll_rows_homo(Xtr, ztr, Xte, zte):
    """Held-out per-row Gaussian log-density of z (no Jacobian), homoscedastic linear predictor (OLS)."""
    A = np.c_[np.ones(len(Xtr)), Xtr]
    b = np.linalg.lstsq(A, ztr, rcond=None)[0]
    s2 = max(float(np.mean((ztr - A @ b) ** 2)), EPS)
    r = zte - np.c_[np.ones(len(Xte)), Xte] @ b
    return -0.5 * np.log(2 * np.pi * s2) - 0.5 * r**2 / s2


def ll_rows_hetero(Xtr, ztr, Xte, zte):
    """Same with a log-variance model regressed on the features (one WLS pass, constant-corrected)."""
    A = np.c_[np.ones(len(Xtr)), Xtr]
    At = np.c_[np.ones(len(Xte)), Xte]
    b = np.linalg.lstsq(A, ztr, rcond=None)[0]
    for _ in range(2):
        r2 = (ztr - A @ b) ** 2
        g = np.linalg.lstsq(A, np.log(r2 + 1e-12), rcond=None)[0]
        lv = np.clip(A @ g, -30, 30)
        c = np.log(np.mean(r2 / np.exp(lv)) + EPS)
        wgt = 1 / np.sqrt(np.exp(lv + c))
        b = np.linalg.lstsq(A * wgt[:, None], ztr * wgt, rcond=None)[0]
    r2 = (ztr - A @ b) ** 2
    g = np.linalg.lstsq(A, np.log(r2 + 1e-12), rcond=None)[0]
    lv = np.clip(A @ g, -30, 30)
    c = np.log(np.mean(r2 / np.exp(lv)) + EPS)
    var = np.exp(np.clip(At @ g, -30, 30) + c)
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (zte - At @ b) ** 2 / var


def fit_lambda(fwd, jac, X, y, lo, hi):
    """Lambda maximising the in-sample homoscedastic profile log-likelihood (regression-based, not marginal normality)."""
    def neg(l):
        z = fwd(y, l)
        if not np.all(np.isfinite(z)):
            return 1e18
        return -float(np.mean(ll_rows_homo(X, z, X, z) + jac(y, l)))
    grid = np.linspace(lo, hi, 17)
    v = [neg(l) for l in grid]
    i = int(np.argmin(v))
    r = minimize_scalar(neg, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 16)]), method="bounded", options={"xatol": 1e-3})
    return float(r.x)


def candidate_set(X, y, tr):
    """name -> (fwd, inv, jac) for every candidate whose domain holds on all of y; lambdas fitted on the training rows."""
    c = {"identity": (lambda t: t, lambda z: z, lambda t: np.zeros_like(t))}
    c["cbrt"] = (np.cbrt, lambda z: z**3, lambda t: np.log(1 / 3) - (2 / 3) * np.log(np.maximum(np.abs(t), 1e-6)))
    c["asinh"] = (np.arcsinh, np.sinh, lambda t: -0.5 * np.log1p(t**2))
    if y.min() > 0:
        c["log"] = (np.log, np.exp, lambda t: -np.log(t))
        c["sqrt"] = (np.sqrt, lambda z: z**2, lambda t: -np.log(2 * np.sqrt(t)))
        c["log1p"] = (np.log1p, np.expm1, lambda t: -np.log1p(t))
        c["reciprocal"] = (lambda t: 1 / t, lambda z: 1 / np.maximum(z, 1e-12), lambda t: -2 * np.log(t))
        lam = fit_lambda(bc, jac_bc, X[tr], y[tr], -3, 3)
        c[f"boxcox"] = ((lambda t, l=lam: bc(t, l)), (lambda z, l=lam: bc_inv(z, l)), (lambda t, l=lam: jac_bc(t, l)))
        c["_lam_boxcox"] = lam
    elif y.max() < 0:
        c["reciprocal"] = (lambda t: 1 / t, lambda z: 1 / np.minimum(z, -1e-12), lambda t: -2 * np.log(-t))
    lam = fit_lambda(yj, jac_yj, X[tr], y[tr], -1, 3)
    c["yeojohnson"] = ((lambda t, l=lam: yj(t, l)), (lambda z, l=lam: yj_inv(z, l)), (lambda t, l=lam: jac_yj(t, l)))
    c["_lam_yeojohnson"] = lam
    return c


def judge_rmse(fwd, inv, X, y, tr, te, model=None, rng=None):
    """Ridge (or ``model``) on f(y); predictions clipped to the training range of f(y); Duan smearing; RMSE on the original scale."""
    from sklearn.linear_model import Ridge

    z = fwd(y)
    m = (model or Ridge(1.0)).fit(X[tr], z[tr])
    lo, hi = float(z[tr].min()), float(z[tr].max())
    res = z[tr] - m.predict(X[tr])
    pred = np.clip(m.predict(X[te]), lo, hi)
    if inv is not None and fwd(np.array([1.0]))[0] == 1.0 and fwd(np.array([2.0]))[0] == 2.0:  # identity: no smearing
        return float(np.sqrt(np.mean((pred - y[te]) ** 2)))
    e = res[rng.choice(len(res), 100, replace=False)]
    back = inv(np.clip(pred[:, None] + e[None, :], lo, hi)).mean(1)
    return float(np.sqrt(np.mean((back - y[te]) ** 2)))


def gc_mi(X, y):
    """Gaussian-copula MI (nats) of y with the joint X: -0.5*log(1-R2) of the normal-score regression."""
    def ns(v):
        r = np.argsort(np.argsort(v)) + 1.0
        return ndtri(r / (len(v) + 1))
    Z = np.c_[[ns(X[:, j]) for j in range(X.shape[1])]].T
    zy = ns(y)
    A = np.c_[np.ones(len(zy)), Z]
    r = zy - A @ np.linalg.lstsq(A, zy, rcond=None)[0]
    return float(-0.5 * np.log(np.mean(r**2) / np.var(zy)))


def signflip_p(d, rng, B=1000):
    """One-sided permutation (sign-flip) p-value of mean(d) > 0 for the paired per-row held-out LL differences."""
    obs = d.mean()
    s = rng.choice([-1.0, 1.0], size=(B, len(d)))
    return float((1 + np.sum((s * d).mean(1) >= obs)) / (B + 1))


def main():
    from mlframe.feature_selection._benchmarks.nbins_strategy.target_transform_mi_ext import equal_width_codes, mutual_information
    from sklearn.ensemble import HistGradientBoostingRegressor

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--hgb", type=int, default=1)
    a = ap.parse_args()
    kinds = ("cube", "lognormal", "hetero_lognormal", "exp_linear", "reciprocal", "linear", "hetero_linear", "heavy")
    stats = {}
    for kind in kinds:
        S = {k: [] for k in ("homo", "hetero", "gate", "mi", "hgb_homo")}
        extra = {"t_ll": [], "t_judge": [], "t_mi": [], "oracle": []}
        for seed in range(a.seeds):
            X, y = make(kind, a.rows, 8, seed)
            n = len(y)
            tr = np.arange(n) < int(0.7 * n)
            te = ~tr
            rng = np.random.default_rng(seed + 100)
            t0 = time.perf_counter()
            cand = candidate_set(X, y, tr)
            lams = {k: v for k, v in cand.items() if k.startswith("_lam")}
            cand = {k: v for k, v in cand.items() if not k.startswith("_lam")}
            llh, llt, rows_h = {}, {}, {}
            for name, (f, _inv, jac) in cand.items():
                z = f(y)
                if not np.all(np.isfinite(z)):
                    continue
                j = jac(y)
                llh[name] = float(np.mean(ll_rows_homo(X[tr], z[tr], X[te], z[te]) + j[te]))
                rh = ll_rows_hetero(X[tr], z[tr], X[te], z[te]) + j[te]
                llt[name] = float(rh.mean())
                rows_h[name] = rh
            extra["t_ll"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            rm, rg = {}, {}
            for name in llh:
                f, inv, _ = cand[name]
                rm[name] = judge_rmse(f, inv, X, y, tr, te, rng=rng)
                if a.hgb:
                    rg[name] = judge_rmse(f, inv, X, y, tr, te, model=HistGradientBoostingRegressor(max_iter=100), rng=rng)
            extra["t_judge"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            ew = {n_: sum(mutual_information(equal_width_codes(X[:, j], 10), equal_width_codes(cand[n_][0](y), 10)) for j in range(8)) for n_ in llh}
            extra["t_mi"].append(time.perf_counter() - t0)
            best = min(rm, key=rm.get)
            pick_h = max(llh, key=llh.get)
            pick_t = max(llt, key=llt.get)
            pick_m = max(ew, key=ew.get)
            # gate: best-LL candidate vs identity by a sign-flip permutation test on paired per-row held-out LL (hetero model), Bonferroni over candidates
            alt = {k: v for k, v in llt.items() if k != "identity"}
            pick_g = "identity"
            if alt:
                k = max(alt, key=alt.get)
                p = signflip_p(rows_h[k] - rows_h["identity"], rng)
                if p < 0.01 / len(alt):
                    pick_g = k
            for key, pk in (("homo", pick_h), ("hetero", pick_t), ("gate", pick_g), ("mi", pick_m)):
                S[key].append((pk, best, rm[pk] / rm["identity"], rm[pk] / rm[best]))
            if a.hgb:
                bg = min(rg, key=rg.get)
                S["hgb_homo"].append((pick_t, bg, rg[pick_t] / rg["identity"], rg[pick_t] / rg[bg]))
            extra["oracle"].append(rm[best] / rm["identity"])
            print(f"CASE {kind} s{seed} judge-best={best} (ridge ratio {rm[best]/rm['identity']:.3f}) LLhomo={pick_h} LLhet={pick_t} gate={pick_g} EWMI={pick_m} lam={ {k[5:]: round(v,2) for k,v in lams.items()} }", flush=True)
            if seed == 0:
                gm = gc_mi(X, y)
                print(f"  seed0 GC-MI(y;X)={gm:.4f} nats (rank-invariant, same for every f); per candidate: heldout LLhomo / LLhetero / ridge RMSE / EW-MI", flush=True)
                for k in llh:
                    print(f"    {k:11s} {llh[k]:9.4f} {llt[k]:9.4f} {rm[k]:10.4f} {ew[k]:.3f}", flush=True)
                # decomposition on train+test pooled for identity & best-LL: gain term vs marginal term
                for k in {pick_h, "identity"}:
                    z = cand[k][0](y)
                    A = np.c_[np.ones(n), X]
                    r2 = 1 - np.var(z - A @ np.linalg.lstsq(A, z, rcond=None)[0]) / np.var(z)
                    marg = float(np.mean(-0.5 * np.log(2 * np.pi * np.var(z)) - 0.5 + cand[k][2](y)))
                    print(f"    decomposition {k}: MIgauss(f(y);X)=-0.5log(1-R2)={-0.5*np.log(1-r2):.4f}  marginal Gaussian LL incl Jacobian={marg:.4f}  sum={-0.5*np.log(1-r2)+marg:.4f}", flush=True)
        stats[kind] = (S, extra)
    print("\nSUMMARY (per kind: picks agree with judge-best k/N; mean ridge RMSE ratio picked/identity; mean regret picked/best)")
    for kind, (S, extra) in stats.items():
        line = [f"{kind:17s}"]
        for key in ("homo", "hetero", "gate", "mi"):
            r = S[key]
            agree = sum(1 for p, b, *_ in r if p == b)
            nonid = sum(1 for p, *_ in r if p != "identity")
            line.append(f"{key}: {agree}/{len(r)} ratio {np.mean([x[2] for x in r]):.3f} regret {np.mean([x[3] for x in r]):.3f} nonid {nonid}")
        if S["hgb_homo"]:
            r = S["hgb_homo"]
            line.append(f"hgb(LLhet pick): ratio {np.mean([x[2] for x in r]):.3f}")
        line.append(f"oracle ratio {np.mean(extra['oracle']):.3f}; sec ll {np.mean(extra['t_ll']):.2f} judge(ridge+hgb) {np.mean(extra['t_judge']):.2f} ewmi {np.mean(extra['t_mi']):.2f}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
