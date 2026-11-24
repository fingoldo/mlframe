"""Model-free recognition of a useful unary transform of a FEATURE for classification (research agent B1, wave 2, 2026-11-24).

Fixtures: class-conditional distribution of one feature x is a known transform of a Gaussian shift (lognormal, reciprocal, sqrt, quadratic = classes differ in variance, abs, beta/logit, count-like), two
Gaussian "other" features z carry a class shift of their own, plus controls (Gaussian shift, gamma scale shift = identity exact for logistic, Poisson, pure noise, heavy-tailed noise).
K = 2 (binary), 3 and 5 (multiclass). 10000 train rows (criteria + judge fit) and 10000 held-out rows.

Judge: held-out log loss / AUC of a multinomial logistic regression on [z, f(x)] (replace form); also LDA and Gaussian NB on [z, f(x)], a spline-GAM ceiling on [z, spline(x)] and HistGradientBoosting
on [z, x] (tree reference). Criteria (no predictive model beyond closed form / a few Newton steps): lrt (deviance gain of a one-feature multinomial logit on f(x) with the score of the other features
as offset), bd (prior weighted Bhattacharyya distance of the class-conditional Gaussians of f(x)), eta (ANOVA / point-biserial information -0.5 log(1-eta^2)), resid (information of f(x) against the
class residual y - p(z), the wave-1 feature recipe), mi_ew (plug-in MI of f(x) with the class, 16 equal-width bins: the old biased proposer), mi_rank (16 equal-frequency bins: monotone invariant).
Decision: best non-identity candidate is accepted only if its gain over identity exceeds the 95th percentile of a null of max_f(s_f - s_id). Two nulls: perm (rows of f(x) permuted against the
class, the conditional offset kept) and bs (parametric bootstrap: classes redrawn from the identity logistic fit on [z, x], the classification analogue of the sign-flip residual bootstrap of wave 1).

Usage: python tt_b1_cls_feature_transform.py --group binary|multi3|multi5 [--seeds 8] [--rows 20000] [--nperm 40] [--fixtures a,b] [--out file.json]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import collections
import json
import time
import warnings

import numpy as np
from scipy import stats

warnings.simplefilter("ignore")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

STEP = {2: 1.0, 3: 0.7, 5: 0.5}
LOGSTEP = {2: 1.5, 3: 1.0, 5: 0.7}
ZSHIFT = {2: [(-0.35, -0.21), (0.35, 0.21)]}
for _K, _r in ((3, 0.5), (5, 0.6)):
    ZSHIFT[_K] = [(_r * np.cos(2 * np.pi * k / _K), _r * np.sin(2 * np.pi * k / _K)) for k in range(_K)]

BETA = {2: [(2.0, 4.0), (4.0, 2.0)], 3: [(2.0, 5.0), (3.5, 3.5), (5.0, 2.0)]}
SQ_SCALE = {2: [0.6, 1.8], 3: [0.5, 1.0, 2.0], 5: list(np.geomspace(0.5, 2.0, 5))}
ABS_R = [0.0, 1.9, 3.4]

# kind -> (is_control, description of the right answer)
KINDS = {
    "log": (False, "log"), "log_weak": (False, "log"), "log_imb": (False, "log"), "recip": (False, "recip"), "sqrt": (False, "sqrt (weak: u ~ N(1.4+0.8c, 0.5), x = u^2, curvature small)"), "sqrt_g": (False, "sqrt (strong: sqrt(x) ~ Gamma(2.5, scale_c))"), "square": (False, "square"), "abs": (False, "abs"),
    "square_off": (False, "square about the median (sqr_raw, MRMR's uncentered square, is wrong)"), "beta_logit": (False, "logit"), "count_logn": (False, "log1p"), "heavy_signal": (False, "judge decides"),
    "ctl_linear": (True, "id"), "ctl_gamma": (True, "id"), "ctl_pois": (True, "id"), "ctl_noise": (True, "none"), "ctl_noise_heavy": (True, "none"),
}
GROUPS = {
    "binary": (2, ["log", "log_weak", "log_imb", "recip", "sqrt", "square", "abs", "beta_logit", "count_logn", "heavy_signal", "ctl_linear", "ctl_gamma", "ctl_pois", "ctl_noise", "ctl_noise_heavy"]),
    "multi3": (3, ["log", "log_weak", "recip", "square", "abs", "beta_logit", "count_logn", "ctl_linear", "ctl_gamma", "ctl_noise"]),
    "multi5": (5, ["log", "square", "ctl_linear", "ctl_noise"]),
}


def gen_x(kind, c, K, rng):
    n = len(c)
    e = rng.standard_normal(n)
    st = STEP[K]
    if kind in ("log", "log_imb"):
        return np.exp(LOGSTEP[K] * c + e)
    if kind == "log_weak":
        return np.exp(st * c + e)
    if kind == "recip":
        mu = 1.1 + {2: 0.8, 3: 0.55, 5: 0.35}[K] * c
        return 1.0 / np.maximum(mu + 0.55 * e, 0.15)
    if kind == "sqrt_g":
        return rng.gamma(2.5, 0.6 + {2: 0.5, 3: 0.35, 5: 0.2}[K] * c) ** 2
    if kind == "sqrt":
        u = 1.4 + {2: 0.8, 3: 0.55, 5: 0.35}[K] * c + 0.5 * e
        return u * u
    if kind == "square":
        return np.asarray(SQ_SCALE[K])[c] * e
    if kind == "square_off":
        return 3.0 + np.asarray(SQ_SCALE[K])[c] * e
    if kind == "abs":
        r = np.asarray(ABS_R)[c]
        sgn = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        return np.where(c == 0, 0.9 * e, sgn * (r + 0.5 * e))
    if kind == "beta_logit":
        a = np.asarray([p[0] for p in BETA[K]])[c]
        b = np.asarray([p[1] for p in BETA[K]])[c]
        return np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)
    if kind == "count_logn":
        return rng.poisson(np.exp(1.0 + {2: 1.0, 3: 0.8}[K] * c + 0.8 * e)).astype(float)
    if kind == "heavy_signal":
        return 1.2 * st * c + rng.standard_t(3, n)
    if kind == "ctl_linear":
        return st * c + e
    if kind == "ctl_gamma":
        return rng.gamma(4.0, 1.0 + {2: 0.6, 3: 0.45, 5: 0.3}[K] * c)
    if kind == "ctl_pois":
        return rng.poisson(4.0 + {2: 2.5, 3: 1.8, 5: 1.2}[K] * c).astype(float)
    if kind == "ctl_noise":
        return np.exp(e)
    if kind == "ctl_noise_heavy":
        return rng.standard_t(2, n)
    raise ValueError(kind)


def make(kind, K, n, seed):
    rng = np.random.default_rng(1000 * K + seed)
    pri = np.array([0.85, 0.15]) if kind == "log_imb" else np.full(K, 1.0 / K)
    c = rng.choice(K, size=n, p=pri)
    Z = rng.standard_normal((n, 2)) + np.asarray(ZSHIFT[K])[c]
    x = gen_x(kind, c, K, rng)
    return x, Z, c


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def cands(x, raw_sq=False):
    med = float(np.median(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25]))) + 1e-12
    mn, mx = float(x.min()), float(x.max())
    c = {"id": x}
    if mn > 0:
        c["log"] = np.log(x)
        c["recip"] = 1.0 / x
    if mn >= 0:
        c["sqrt"] = np.sqrt(x)
        c["log1p"] = np.log1p(x)
    if mn > 0 and mx < 1:
        c["logit"] = np.log(x / (1 - x))
    c["square"] = (x - med) ** 2
    c["abs"] = np.abs(x - med)
    c["tanh"] = np.tanh((x - med) / iqr)
    c["cbrt"] = np.cbrt(x)
    c["asinh"] = np.arcsinh((x - med) / iqr)
    if raw_sq:
        c["sqr_raw"] = x * x
    out = {}
    for k, v in c.items():
        v = np.asarray(v, float)
        if np.all(np.isfinite(v)) and v.std() > 0:
            out[k] = v
    return out


def stdz(F):
    m = F.mean(0)
    s = F.std(0)
    s[s == 0] = 1.0
    return (F - m) / s, m, s


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# multinomial logistic regression with offset (Newton, class 0 reference)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _ll(Phi, Y, off, W):
    L = off.copy()
    L[:, 1:] += Phi @ W
    m = L.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(L - m).sum(1))
    return float((Y * L).sum() - lse.sum()), L, lse


def fit_mnl(Phi, Y, off=None, iters=40, ridge=1e-6, tol=1e-5):
    """Newton with step halving. Phi (n,d), Y one-hot (n,K), off (n,K). Returns W (d,K-1), loglik."""
    n, d = Phi.shape
    K = Y.shape[1]
    q = K - 1
    if off is None:
        off = np.zeros((n, K))
    W = np.zeros((d, q))
    ll, L, lse = _ll(Phi, Y, off, W)
    for _ in range(iters):
        P = np.exp(L - lse[:, None])
        g = (Phi.T @ (Y[:, 1:] - P[:, 1:])).T.reshape(-1) - ridge * W.T.reshape(-1)
        H = np.zeros((q * d, q * d))
        for k in range(q):
            for l in range(k, q):
                w = P[:, k + 1] * ((1.0 if k == l else 0.0) - P[:, l + 1])
                B = (Phi * w[:, None]).T @ Phi
                H[k * d:(k + 1) * d, l * d:(l + 1) * d] = B
                if l != k:
                    H[l * d:(l + 1) * d, k * d:(k + 1) * d] = B.T
        H += ridge * np.eye(q * d)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = step.reshape(q, d).T
        t = 1.0
        ok = False
        for _h in range(12):
            Wn = W + t * step
            lln, Ln, lsen = _ll(Phi, Y, off, Wn)
            if lln >= ll - 1e-9:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        gain = lln - ll
        W, ll, L, lse = Wn, lln, Ln, lsen
        if gain < tol:
            break
    return W, ll


def predict_proba(Phi, W, K):
    L = np.zeros((Phi.shape[0], K))
    L[:, 1:] = Phi @ W
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def onehot(y, K):
    Y = np.zeros((len(y), K))
    Y[np.arange(len(y)), y] = 1.0
    return Y


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# criteria (all return arrays of length C, one entry per candidate column of the standardised candidate matrix F)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def base_offset(Zs, Y):
    Phi0 = np.c_[np.ones(len(Zs)), Zs]
    W0, ll0 = fit_mnl(Phi0, Y)
    K = Y.shape[1]
    L0 = np.zeros((len(Zs), K))
    L0[:, 1:] = Phi0 @ W0
    return L0, ll0, predict_proba(Phi0, W0, K)


def crit_lrt(F, Y, L0, ll0):
    out = np.empty(F.shape[1])
    ones = np.ones(F.shape[0])
    for j in range(F.shape[1]):
        _, ll1 = fit_mnl(np.c_[ones, F[:, j]], Y, L0, tol=1e-3)
        out[j] = 2.0 * (ll1 - ll0)
    return out


def class_moments(F, Y):
    nk = Y.sum(0)
    M = (Y.T @ F) / nk[:, None]
    V = (Y.T @ (F * F)) / nk[:, None] - M * M
    return nk, M, np.maximum(V, 1e-12)


def crit_bd_eta(F, Y):
    nk, M, V = class_moments(F, Y)
    pi = nk / nk.sum()
    K = len(nk)
    bd = np.zeros(F.shape[1])
    for k in range(K):
        for l in range(k + 1, K):
            v = V[k] + V[l]
            d = 0.25 * (M[k] - M[l]) ** 2 / v + 0.25 * np.log(0.25 * (V[k] / V[l] + V[l] / V[k] + 2.0))
            bd += pi[k] * pi[l] * d
    mu = F.mean(0)
    sst = ((F - mu) ** 2).sum(0)
    ssb = (nk[:, None] * (M - mu) ** 2).sum(0)
    eta2 = np.clip(ssb / np.maximum(sst, 1e-300), 0, 1 - 1e-12)
    return bd, -0.5 * np.log(1.0 - eta2)


def crit_resid(F, R):
    Fc = F - F.mean(0)
    Rc = R - R.mean(0)
    G = np.linalg.pinv(Rc.T @ Rc)
    B = Rc.T @ Fc
    num = np.einsum("ic,ij,jc->c", B, G, B)
    r2 = np.clip(num / np.maximum((Fc * Fc).sum(0), 1e-300), 0, 1 - 1e-12)
    return -0.5 * np.log(1.0 - r2)


def mi_codes(codes, y, nb, K):
    n = len(y)
    J = np.bincount(codes * K + y, minlength=nb * K).reshape(nb, K) / n
    pa = J.sum(1, keepdims=True)
    pb = J.sum(0, keepdims=True)
    m = J > 0
    return float((J[m] * np.log(J[m] / (pa @ pb)[m])).sum())


def crit_mi(F, y, K, nb=16):
    ew = np.empty(F.shape[1])
    rk = np.empty(F.shape[1])
    for j in range(F.shape[1]):
        f = F[:, j]
        lo, hi = f.min(), f.max()
        cw = np.clip(((f - lo) / (hi - lo + 1e-300) * nb).astype(np.int64), 0, nb - 1)
        ew[j] = mi_codes(cw, y, nb, K)
        r = stats.rankdata(f, method="average")
        cr = np.clip(((r - 0.5) / len(f) * nb).astype(np.int64), 0, nb - 1)
        rk[j] = mi_codes(cr, y, nb, K)
    return ew, rk


CRIT = ["lrt", "bd", "eta", "resid", "mi_ew", "mi_rank"]


def all_crit(F, y, Y, Zs, K, which=CRIT):
    """All criteria for candidate matrix F against classes y with the other features Zs (offset refit on y)."""
    out = {}
    if "lrt" in which or "resid" in which:
        L0, ll0, P0 = base_offset(Zs, Y)
        if "lrt" in which:
            out["lrt"] = crit_lrt(F, Y, L0, ll0)
        if "resid" in which:
            out["resid"] = crit_resid(F, Y[:, 1:] - P0[:, 1:])
    if "bd" in which or "eta" in which:
        bd, eta = crit_bd_eta(F, Y)
        out["bd"], out["eta"] = bd, eta
    if "mi_ew" in which or "mi_rank" in which:
        out["mi_ew"], out["mi_rank"] = crit_mi(F, y, K)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# judges
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def _metrics(P, y, K):
    from sklearn.metrics import roc_auc_score
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(1, keepdims=True)
    ll = float(-np.log(P[np.arange(len(y)), y]).mean())
    auc = float(roc_auc_score(y, P[:, 1])) if K == 2 else float(roc_auc_score(y, P, multi_class="ovr", average="macro"))
    return ll, auc


def judge_lr(Ftr, Fte, Ztr, Zte, ytr, yte, K, extra_id=None):
    """Held-out (logloss, auc) of the multinomial logit on [z, f(x)] for every candidate column."""
    Ytr = onehot(ytr, K)
    zm, zs = Ztr.mean(0), Ztr.std(0)
    Zs_tr, Zs_te = (Ztr - zm) / zs, (Zte - zm) / zs
    res = {}
    for name in Ftr:
        f = Ftr[name]
        m, s = f.mean(), f.std()
        a, b = (f - m) / s, (Fte[name] - m) / s
        Phi = np.c_[np.ones(len(a)), Zs_tr, a]
        W, _ = fit_mnl(Phi, Ytr)
        res[name] = _metrics(predict_proba(np.c_[np.ones(len(b)), Zs_te, b], W, K), yte, K)
    return res


def judge_append(Ftr, Fte, Ztr, Zte, ytr, yte, K):
    """Same but the candidate is appended to the raw x: [z, x, f(x)]."""
    Ytr = onehot(ytr, K)
    zm, zs = Ztr.mean(0), Ztr.std(0)
    Zs_tr, Zs_te = (Ztr - zm) / zs, (Zte - zm) / zs
    x0, x1 = Ftr["id"], Fte["id"]
    xm, xs = x0.mean(), x0.std()
    res = {}
    for name in Ftr:
        if name == "id":
            continue
        f = Ftr[name]
        m, s = f.mean(), f.std()
        Phi = np.c_[np.ones(len(f)), Zs_tr, (x0 - xm) / xs, (f - m) / s]
        Pte = np.c_[np.ones(len(x1)), Zs_te, (x1 - xm) / xs, (Fte[name] - m) / s]
        W, _ = fit_mnl(Phi, Ytr)
        res[name] = _metrics(predict_proba(Pte, W, K), yte, K)
    return res


def judge_sk(Ftr, Fte, Ztr, Zte, ytr, yte, K):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    out = {"lda": {}, "gnb": {}}
    for name in Ftr:
        A = np.c_[Ztr, Ftr[name]]
        B = np.c_[Zte, Fte[name]]
        mu, sd = A.mean(0), A.std(0)
        A, B = (A - mu) / sd, (B - mu) / sd
        out["lda"][name] = _metrics(LinearDiscriminantAnalysis().fit(A, ytr).predict_proba(B), yte, K)
        out["gnb"][name] = _metrics(GaussianNB().fit(A, ytr).predict_proba(B), yte, K)
    return out


def judge_gam(xtr, xte, Ztr, Zte, ytr, yte, K):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer
    sp = SplineTransformer(n_knots=10, degree=3, knots="quantile", extrapolation="constant").fit(xtr[:, None])
    zm, zs = Ztr.mean(0), Ztr.std(0)
    A = np.c_[(Ztr - zm) / zs, sp.transform(xtr[:, None])]
    B = np.c_[(Zte - zm) / zs, sp.transform(xte[:, None])]
    lr = LogisticRegression(C=10.0, max_iter=2000).fit(A, ytr)
    return _metrics(lr.predict_proba(B), yte, K)


def judge_hgb(Ftr, Fte, Ztr, Zte, ytr, yte, K, names):
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    out = {}
    for name in names:
        m = H(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, early_stopping=False, random_state=0).fit(np.c_[Ztr, Ftr[name]], ytr)
        out[name] = _metrics(m.predict_proba(np.c_[Zte, Fte[name]]), yte, K)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# one case
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def NONID(names):
    return [k for k in names if k != "id"]


def run_case(kind, K, seed, n, nperm, do_hgb=True, do_bs=True):
    x, Z, y = make(kind, K, n, seed)
    h = n // 2
    C = cands(x, raw_sq=(kind == "square_off"))
    names = list(C)
    Ftr = {k: v[:h] for k, v in C.items()}
    Fte = {k: v[h:] for k, v in C.items()}
    ytr, yte, Ztr, Zte = y[:h], y[h:], Z[:h], Z[h:]
    res = {"kind": kind, "K": K, "seed": seed, "names": names}
    # --- judges
    jl = judge_lr(Ftr, Fte, Ztr, Zte, ytr, yte, K)
    res["judge_lr"] = jl
    res["judge_app"] = judge_append(Ftr, Fte, Ztr, Zte, ytr, yte, K)
    res.update({"judge_" + k: v for k, v in judge_sk(Ftr, Fte, Ztr, Zte, ytr, yte, K).items()})
    res["gam"] = judge_gam(x[:h], x[h:], Ztr, Zte, ytr, yte, K)
    best = min(jl, key=lambda k: jl[k][0])
    if do_hgb:
        hn = ["id"] + ([best] if best != "id" else []) + [k for k in ("square", "abs") if k in names and k != best]
        res["judge_hgb"] = judge_hgb(Ftr, Fte, Ztr, Zte, ytr, yte, K, hn)
    # --- criteria on the train rows
    Fm = np.c_[[Ftr[k] for k in names]].T
    Fs, _, _ = stdz(Fm)
    Ytr = onehot(ytr, K)
    Zs, _, _ = stdz(Ztr)
    t = {}
    obs = {}
    for name, which in (("lrt", ["lrt"]), ("bde", ["bd", "eta"]), ("resid", ["resid"]), ("mi", ["mi_ew", "mi_rank"])):
        t0 = time.perf_counter()
        obs.update(all_crit(Fs, ytr, Ytr, Zs, K, which))
        t[name] = (time.perf_counter() - t0) / len(names)
    res["t_per_cand"] = t
    res["obs"] = {c: dict(zip(names, map(float, v))) for c, v in obs.items()}
    # --- nulls
    rng = np.random.default_rng(seed + 12345)
    iid = names.index("id")
    others = [j for j in range(len(names)) if j != iid]
    nul = {c: [] for c in CRIT}
    L0, ll0, P0 = base_offset(Zs, Ytr)
    for _ in range(nperm):
        perm = rng.permutation(h)
        o = {}
        # conditional criteria: permute the rows of the candidate matrix against (y, z, offset); marginal ones are the same as permuting y
        Fp = Fs[perm]
        o["lrt"] = crit_lrt(Fp, Ytr, L0, ll0)
        o["resid"] = crit_resid(Fp, Ytr[:, 1:] - P0[:, 1:])
        o["bd"], o["eta"] = crit_bd_eta(Fp, Ytr)
        o["mi_ew"], o["mi_rank"] = crit_mi(Fp, ytr, K)
        for c in CRIT:
            nul[c].append(float(np.max(o[c][others] - o[c][iid])))
    res["null_perm"] = nul
    if do_bs:
        # parametric bootstrap: classes redrawn from the identity logistic fit on [z, x]
        Phi = np.c_[np.ones(h), Zs, Fs[:, iid]]
        W, _ = fit_mnl(Phi, Ytr)
        Pid = predict_proba(Phi, W, K)
        cum = Pid.cumsum(1)
        nbs = {c: [] for c in CRIT}
        for _ in range(nperm):
            u = rng.random(h)[:, None]
            ys = np.minimum((u > cum).sum(1), K - 1)
            o = all_crit(Fs, ys, onehot(ys, K), Zs, K)
            for c in CRIT:
                nbs[c].append(float(np.max(o[c][others] - o[c][iid])))
        res["null_bs"] = nbs
    return res


def decisions(res):
    """Per criterion and null: (pick, accepted, gain, threshold)."""
    out = {}
    names = res["names"]
    for nk in ("null_perm", "null_bs"):
        if nk not in res:
            continue
        for c in CRIT:
            v = res["obs"][c]
            pick = max(NONID(names), key=lambda k: v[k])
            gain = v[pick] - v["id"]
            thr = float(np.quantile(res[nk][c], 0.95))
            out[(nk, c)] = (pick, bool(gain > max(thr, 0.0)), gain, thr)  # a negative null maximum (id dominates) must not accept a candidate that is worse than id
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

GAIN_NATS = 0.01
TOL_NATS = 0.005
HARM_NATS = 0.005


def classify_case(res, d):
    """Judge verdict and per criterion correctness."""
    jl = res["judge_lr"]
    best = min(jl, key=lambda k: jl[k][0])
    gain = jl["id"][0] - jl[best][0]
    exists = gain > GAIN_NATS
    out = {"best": best, "gain": gain, "exists": exists}
    for key, (pick, acc, _g, _thr) in d.items():
        harm = jl[pick][0] - jl["id"][0] > HARM_NATS
        if exists:
            ok = bool(acc and jl[pick][0] <= jl[best][0] + TOL_NATS)
        else:
            ok = not acc
        out[key] = {"ok": ok, "acc": acc, "pick": pick, "harm": bool(acc and harm)}
    return out


def summarise(rows, seeds, group):
    print(f"\n=== group {group}, seeds={seeds} ===")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    for nk in ("null_perm", "null_bs"):
        if nk not in rows[0]:
            continue
        print(f"\n--- correct decisions out of {seeds} ({nk}); gain>{GAIN_NATS} nats counts as 'a transform exists'")
        print(f"{'fixture':16s} {'judge best (count)':24s} {'gain':>6s} " + " ".join(f"{c:>8s}" for c in CRIT))
        fp = collections.defaultdict(lambda: [0, 0, 0])  # criterion -> [false pos, harm, cases]
        for kind, rs in by.items():
            cls = [classify_case(r, decisions(r)) for r in rs]
            cnt = collections.Counter(c["best"] for c in cls).most_common(2)
            gain = np.mean([c["gain"] for c in cls])
            line = f"{kind:16s} {cnt!s:24s} {gain:6.3f} "
            for c in CRIT:
                ok = sum(cl[(nk, c)]["ok"] for cl in cls)
                line += f"{ok:>8d} "
                if not any(cl["exists"] for cl in cls):
                    pass
            print(line)
            KINDS[kind][0]
            for c in CRIT:
                for cl in cls:
                    if not cl["exists"]:
                        fp[c][0] += cl[(nk, c)]["acc"]
                        fp[c][2] += 1
                    fp[c][1] += cl[(nk, c)]["harm"]
        print("false positives over all no-gain cases (accepted although judge gain <= %.3f), harmful accepts over all cases:" % GAIN_NATS)
        print("  " + "  ".join(f"{c}: FP {fp[c][0]}/{fp[c][2]} harm {fp[c][1]}" for c in CRIT))
    print("\n--- judge table: mean held-out logloss (id | judge-best | GAM ceiling | HGB id) and AUC; LDA / GNB judge-best name")
    for kind, rs in by.items():
        jl_id = np.mean([r["judge_lr"]["id"][0] for r in rs])
        jl_b = np.mean([min(v[0] for v in r["judge_lr"].values()) for r in rs])
        a_id = np.mean([r["judge_lr"]["id"][1] for r in rs])
        a_b = np.mean([max(v[1] for v in r["judge_lr"].values()) for r in rs])
        gam = np.mean([r["gam"][0] for r in rs])
        rh = [r for r in rs if "judge_hgb" in r]
        hid = np.mean([r["judge_hgb"]["id"][0] for r in rh]) if rh else float("nan")
        hbest = np.mean([min(v[0] for v in r["judge_hgb"].values()) for r in rh]) if rh else float("nan")
        lda_b = collections.Counter(min(r["judge_lda"], key=lambda k: r["judge_lda"][k][0]) for r in rs).most_common(1)
        gnb_b = collections.Counter(min(r["judge_gnb"], key=lambda k: r["judge_gnb"][k][0]) for r in rs).most_common(1)
        lrb = collections.Counter(min(r["judge_lr"], key=lambda k: r["judge_lr"][k][0]) for r in rs).most_common(1)
        print(f"{kind:16s} LR ll id {jl_id:.4f} best {jl_b:.4f} ({lrb[0][0]} {lrb[0][1]}/{len(rs)}) GAM {gam:.4f} HGB id {hid:.4f} HGB best-of-fit {hbest:.4f} | AUC id {a_id:.4f} best {a_b:.4f} | LDA {lda_b} GNB {gnb_b}")
    print("\n--- cost per candidate (ms, mean over cases): " + ", ".join(f"{k} {1e3 * np.mean([r['t_per_cand'][k] for r in rows]):.2f}" for k in rows[0]["t_per_cand"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="binary")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--nperm", type=int, default=40)
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--nohgb", action="store_true")
    ap.add_argument("--hgb_seeds", type=int, default=3)
    ap.add_argument("--nobs", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    K, kinds = GROUPS[a.group]
    if a.fixtures:
        kinds = a.fixtures.split(",")
    rows = []
    for kind in kinds:
        t0 = time.perf_counter()
        for s in range(a.seeds):
            rows.append(run_case(kind, K, s, a.rows, a.nperm, (not a.nohgb) and s < a.hgb_seeds, not a.nobs))
        print(f"[{a.group}] {kind} done in {time.perf_counter() - t0:.1f}s", flush=True)
    summarise(rows, a.seeds, a.group)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh)


if __name__ == "__main__":
    main()
