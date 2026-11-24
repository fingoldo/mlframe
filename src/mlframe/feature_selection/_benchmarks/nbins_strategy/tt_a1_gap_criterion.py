"""Agent A1 research: information-theoretic criteria for a target transform, no model retraining (2026-11-23).

Criteria per candidate f (computed on the TRAIN rows only, half A fits, half B scores):
  R2   held-out linear R2 of f(y) on X (= I_lin = -0.5 log(1-R2); the Gaussian-linear information; invariant to affine f)
  ILIN the same in nats (monotone in R2, listed for the null-gated variants)
  LL   Box-Cox style profile log-likelihood per row on the ORIGINAL scale: -0.5 log(MSE_B(f)) + mean log|f'(y_B)|  (Jacobian handled)
  GAP  minus the linearity gap: -(R2_additive(spline) - R2_linear), both on f(y); scale free, small gap when f linearises
Null gate: sign-flip residual bootstrap from the identity linear model; threshold = 95% quantile of max over f of (score_f - score_identity).
Judge: held-out RMSE on the original scale of ridge (and HGB) fitted on f(y), inverted with a clipped inverse and generalised Duan smearing.
Usage: python tt_a1_gap_criterion.py [--seeds 7] [--rows 20000] [--hgb 1] [--boot 40]
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter

import numpy as np
from scipy import stats

warnings.simplefilter("ignore")
EPS = 1e-9


def make(kind: str, n: int, p: int, seed: int):
    """Fixtures; FIX says which candidate is the right answer."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    e = rng.standard_normal(n)
    lin = X[:, 0] + 0.7 * X[:, 1]
    het = kind.startswith("hetero_")
    b = kind.replace("hetero_", "")
    if b == "lognormal":
        y = np.exp(0.9 * X[:, 0] + 0.6 * X[:, 1] + (0.4 * (1 + np.abs(X[:, 0])) if het else 0.4) * e)
    elif b == "cube":
        y = (lin + 0.3 * e) ** 3
    elif b == "linear":
        y = lin + (0.3 * (1 + np.abs(X[:, 0])) if het else 0.3) * e
    elif b == "linear_pos":
        y = 10 + lin + 0.3 * e
    elif b == "reciprocal":
        y = 1.0 / (4 + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.05 * e)
    elif b == "heavy":
        y = lin + 0.5 * rng.standard_t(3, n)
    else:
        raise ValueError(kind)
    return X, y


FIX = {"lognormal": "log", "hetero_lognormal": "log", "cube": "cbrt", "reciprocal": "reciprocal", "linear": "identity", "hetero_linear": "identity", "linear_pos": "identity", "heavy": "identity"}


def yj_inv(z, lam):
    """Inverse Yeo-Johnson with guarded bases."""
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    with np.errstate(all="ignore"):
        out[pos] = np.log1p(z[pos]) if abs(lam) < 1e-8 else np.maximum(z[pos] * lam + 1, EPS) ** (1 / lam) - 1
        out[~pos] = -np.log1p(-z[~pos]) if abs(lam - 2) < 1e-8 else 1 - np.maximum(1 - (2 - lam) * z[~pos], EPS) ** (1 / (2 - lam))
    return out


def candidates(y: np.ndarray, rng) -> dict:
    """name -> (f, inverse, log|f'|)."""
    sg = np.sign
    c = {
        "identity": (lambda t: t, lambda z: z, lambda t: np.zeros_like(t)),
        "cbrt": (np.cbrt, lambda z: z**3, lambda t: -np.log(3) - (2 / 3) * np.log(np.maximum(np.abs(t), 1e-6))),
        "signed_sqrt": (lambda t: sg(t) * np.sqrt(np.abs(t)), lambda z: sg(z) * z**2, lambda t: -np.log(2) - 0.5 * np.log(np.maximum(np.abs(t), 1e-6))),
        "signed_square": (lambda t: sg(t) * t**2, lambda z: sg(z) * np.sqrt(np.abs(z)), lambda t: np.log(2) + np.log(np.maximum(np.abs(t), 1e-6))),
    }
    if y.min() > 0:
        c["log"] = (np.log, np.exp, lambda t: -np.log(t))
        c["reciprocal"] = (lambda t: -1.0 / t, lambda z: -1.0 / np.minimum(z, -1e-9), lambda t: -2 * np.log(t))
    sub = y if len(y) <= 4000 else rng.choice(y, 4000, replace=False)
    lam = float(stats.yeojohnson(sub)[1])
    lam = min(max(lam, -1.0), 3.0)
    c["yeo_johnson"] = (
        lambda t: stats.yeojohnson(t, lmbda=lam),
        lambda z: yj_inv(z, lam),
        lambda t: np.where(t >= 0, (lam - 1) * np.log1p(np.maximum(t, 0)), (1 - lam) * np.log1p(-np.minimum(t, 0))),
    )
    return c


class Design:
    """Linear and spline (additive) designs on halves A/B of the rows, with pseudo-inverses fitted on A."""

    def __init__(self, X):
        from sklearn.preprocessing import SplineTransformer

        n = len(X)
        self.A = np.arange(n) < n // 2
        S = SplineTransformer(n_knots=6, degree=3).fit_transform(X)
        self.D = {"lin": np.c_[np.ones(n), X], "add": np.c_[np.ones(n), S]}
        self.P = {k: np.linalg.pinv(D[self.A]) for k, D in self.D.items()}

    def heldout_pred(self, kind, Z):
        """Predictions on B of each column of Z (n x K) fitted on A."""
        return self.D[kind][~self.A] @ (self.P[kind] @ Z[self.A])


def criteria(des: Design, y: np.ndarray, cand: dict) -> dict:
    """Scores per candidate, all 'higher is better'."""
    names = list(cand)
    Z = np.column_stack([cand[k][0](y) for k in names])
    Zb = Z[~des.A]
    yb = y[~des.A]
    tot = ((Zb - Zb.mean(0)) ** 2).sum(0)
    r2 = {}
    for kind in ("lin", "add"):
        r2[kind] = 1 - ((Zb - des.heldout_pred(kind, Z)) ** 2).sum(0) / tot
    mse_lin = ((Zb - des.heldout_pred("lin", Z)) ** 2).mean(0)
    jac = np.array([np.mean(cand[k][2](yb)) for k in names])
    return {
        "R2": dict(zip(names, r2["lin"])),
        "LL": dict(zip(names, -0.5 * np.log(mse_lin) + jac)),
        "GAP": dict(zip(names, -(r2["add"] - r2["lin"]))),
        "ILIN": dict(zip(names, -0.5 * np.log(np.maximum(1 - r2["lin"], 1e-12)))),
    }


def null_threshold(des, y, key, boot, rng, q=0.95):
    """Sign-flip residual bootstrap from the identity linear model: q-quantile of max_f (score_f - score_identity)."""
    D = des.D["lin"]
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    fit = D @ beta
    r = y - fit
    ymin = y.min() if y.min() > 0 else None
    mx = []
    for _ in range(boot):
        ys = fit + rng.choice([-1.0, 1.0], len(y)) * r
        if ymin is not None:
            ys = np.maximum(ys, ymin)
        sc = criteria(des, ys, candidates(ys, rng))[key]
        mx.append(max(v - sc["identity"] for k, v in sc.items() if k != "identity"))
    return float(np.quantile(mx, q))


def judge(model_fn, X, y, tr, te, cand, rng, n_res=100):
    """RMSE on the original scale per candidate: clipped inverse, generalised smearing mean_j inv(zhat + r_j)."""
    out = {}
    for name, (f, inv, _j) in cand.items():
        z = f(y)
        m = model_fn().fit(X[tr], z[tr])
        res = z[tr] - m.predict(X[tr])
        rs = rng.choice(res, n_res, replace=False)
        lo, hi = z[tr].min(), z[tr].max()
        mg = 0.1 * (hi - lo)
        lo, hi = lo - mg, hi + mg
        if name == "reciprocal":
            hi = min(hi, z[tr].max() / 2)
        arg = np.clip(m.predict(X[te])[:, None] + rs[None, :], lo, hi)
        out[name] = float(np.sqrt(np.mean((inv(arg).mean(1) - y[te]) ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--hgb", type=int, default=1)
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--kinds", default="lognormal,hetero_lognormal,cube,reciprocal,linear,hetero_linear,linear_pos,heavy")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    keys = ["R2", "LL", "GAP"]
    rows, t_crit, t_null = [], [], []
    for kind in a.kinds.split(","):
        for seed in range(a.seeds):
            X, y = make(kind, a.rows, 8, seed)
            rng = np.random.default_rng(100 + seed)
            n = len(y)
            tr = np.arange(n) < int(0.7 * n)
            te = ~tr
            cand = candidates(y[tr], rng)
            t0 = time.perf_counter()
            des = Design(X[tr])
            sc = criteria(des, y[tr], cand)
            t_crit.append((time.perf_counter() - t0) / len(cand))
            t0 = time.perf_counter()
            thr = {k: null_threshold(des, y[tr], k, a.boot, rng) for k in keys}
            t_null.append(time.perf_counter() - t0)
            jr = judge(lambda: Ridge(1.0), X, y, tr, te, cand, rng)
            jh = judge(lambda: HistGradientBoostingRegressor(max_iter=100), X, y, tr, te, cand, rng) if a.hgb else None
            picks = {}
            for k in keys:
                s = sc[k]
                best = max(s, key=lambda q: s[q])
                picks[k] = best
                picks[k + "+null"] = best if s[best] - s["identity"] > thr[k] else "identity"
            rows.append(dict(kind=kind, seed=seed, picks=picks, jr=jr, jh=jh))
            print(f"CASE {kind} s{seed} right={FIX[kind]} ridge_best={min(jr, key=jr.get)} " + " ".join(f"{k}={v}" for k, v in picks.items()), flush=True)
    names = list(rows[0]["picks"])
    print("\nSUMMARY (right = hits vs the known answer; jbest = equals ridge-judge-best; ratio = RMSE(pick)/RMSE(identity); FP = non-identity pick on no-transform fixtures)")
    nt = [r for r in rows if FIX[r["kind"]] == "identity"]
    for k in names:
        hit = sum(r["picks"][k] == FIX[r["kind"]] for r in rows)
        jb = sum(r["picks"][k] == min(r["jr"], key=r["jr"].get) for r in rows)
        rat = np.array([r["jr"][r["picks"][k]] / r["jr"]["identity"] for r in rows])
        fp = sum(r["picks"][k] != "identity" for r in nt)
        line = f"{k:9s} right {hit}/{len(rows)} jbest {jb}/{len(rows)} ridge ratio mean {rat.mean():.3f} worst {rat.max():.3f} FP {fp}/{len(nt)}"
        if a.hgb:
            rh = np.array([r["jh"][r["picks"][k]] / r["jh"]["identity"] for r in rows])
            line += f" | hgb ratio mean {rh.mean():.3f} worst {rh.max():.3f}"
        print("SUMMARY", line)
    print("\nPER FIXTURE")
    for kind in a.kinds.split(","):
        rr = [r for r in rows if r["kind"] == kind]
        for k in names:
            cnt = Counter(r["picks"][k] for r in rr)
            rat = np.mean([r["jr"][r["picks"][k]] / r["jr"]["identity"] for r in rr])
            print(f"FIX {kind:17s} {k:9s} {dict(cnt)} ridge ratio {rat:.3f}")
        print(f"FIX {kind:17s} judge-best ridge {dict(Counter(min(r['jr'], key=r['jr'].get) for r in rr))}" + (f" hgb {dict(Counter(min(r['jh'], key=r['jh'].get) for r in rr))}" if a.hgb else ""))
    print(f"TIME criteria (R2, LL, GAP together) per candidate mean {np.mean(t_crit) * 1e3:.1f} ms; null ({a.boot} boots x 3 criteria) per case mean {np.mean(t_null):.2f} s")


if __name__ == "__main__":
    main()
