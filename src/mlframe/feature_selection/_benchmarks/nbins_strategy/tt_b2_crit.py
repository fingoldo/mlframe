"""B2 criteria: wave-1 target criterion (A1/A3 Jacobian LL + sign-flip null + held-out original-scale gate), wave-1 feature criterion (A2 Pearson information of
f(x) with the residual of y + permutation null), a multi-feature version with per-feature / max-over-features nulls, and the B2 fixes (flags in ``fixes``).
"""
from __future__ import annotations

import time

import numpy as np

from tt_b2_core import build_candidates, inv_margin

EPS = 1e-9


# ======================================================================================================================================================================
# TARGET side
# ======================================================================================================================================================================
class Des:
    """Random halves A / B of the training rows; pseudo-inverse of the design on A; everything vectorised over candidate columns."""

    def __init__(self, X: np.ndarray, rng):
        n = len(X)
        perm = rng.permutation(n)
        self.ia, self.ib = perm[: n // 2], perm[n // 2:]
        self.D = np.c_[np.ones(n), X]
        DA = self.D[self.ia]
        G = DA.T @ DA
        G[np.diag_indices_from(G)] += 1e-9 * np.trace(G) / len(G)
        self.P = np.linalg.solve(G, DA.T)
        self.DB = self.D[self.ib]
        self.p = X.shape[1]

    def pred_B(self, Z):
        return self.DB @ (self.P @ Z[self.ia])


def ll_scores(des: Des, y: np.ndarray, cands: dict, rows=None) -> dict:
    """Held-out profile log-likelihood per row, original scale: -0.5 log MSE_B(f) + mean_B log|f'(y)|. rows: optional boolean mask over B rows to score on (atom handling)."""
    names = list(cands)
    Z = np.empty((len(y), len(names)))
    for j, k in enumerate(names):
        Z[:, j] = cands[k][0](y)
    good = np.isfinite(Z).all(0)
    Z[:, ~good] = 0.0
    R = Z[des.ib] - des.pred_B(Z)
    yb = y[des.ib]
    if rows is None:
        mse = np.mean(R**2, 0)
        jac = np.array([np.mean(cands[k][2](yb)) for k in names])
    else:
        mse = np.mean(R[rows] ** 2, 0)
        jac = np.array([np.mean(cands[k][2](yb[rows])) for k in names])
    ll = -0.5 * np.log(np.maximum(mse, 1e-300)) + jac
    ll[~good | ~np.isfinite(jac)] = -np.inf
    return dict(zip(names, ll))


def _clip_bounds(y: np.ndarray):
    mn, mx = float(y.min()), float(y.max())
    lo = mn if mn >= 0 else None
    hi = mx if (mn > 0 and mx < 1) else None
    return lo, hi


def null_max(des: Des, y: np.ndarray, B: int, rng, extended: bool, floor: float, rows_fn=None, clip: bool = True) -> np.ndarray:
    """Sign-flip residual bootstrap from the identity linear fit; per draw max over non-identity candidates of LL_f - LL_identity."""
    D = des.D
    beta = np.linalg.solve(D.T @ D + 1e-9 * np.eye(D.shape[1]) * np.trace(D.T @ D) / D.shape[1], D.T @ y)
    fit = D @ beta
    r = y - fit
    lo, hi = _clip_bounds(y) if clip else (None, None)
    out = np.empty(B)
    for b in range(B):
        ys = fit + (rng.integers(0, 2, len(y)) * 2 - 1.0) * r
        if lo is not None or hi is not None:
            ys = np.clip(ys, lo, hi)
        cb = build_candidates(ys, rng, extended=extended, floor=floor)
        rows = rows_fn(ys, des) if rows_fn is not None else None
        ll = ll_scores(des, ys, cb, rows)
        out[b] = max((v - ll["identity"] for k, v in ll.items() if k != "identity"), default=-np.inf)
    return out


def heldout_orig_rmse(des: Des, y: np.ndarray, cands: dict, names: list, n_smear: int = 64, margin: float = 0.05, max_rows: int = 20000, rng=None) -> dict:
    """Original-scale held-out (B half) RMSE of the linear head fitted on A: clipped inverse, smearing from the A residuals (dof-corrected)."""
    ia, ib = des.ia, des.ib
    DA, DB = des.D[ia], des.DB
    if len(ib) > max_rows:
        sel = np.sort((rng or np.random.default_rng(0)).choice(len(ib), max_rows, replace=False))
        DB = DB[sel]
        ibs = ib[sel]
    else:
        ibs = ib
    out = {}
    dof = len(ia) / max(len(ia) - DA.shape[1], 1)
    qs = (np.arange(n_smear) + 0.5) / n_smear
    for k in names:
        f, inv = cands[k][0], cands[k][1]
        z = f(y)
        if not np.all(np.isfinite(z)):
            out[k] = np.inf
            continue
        w = des.P @ z[ia]
        za = z[ia]
        q = np.quantile(za - DA @ w, qs) * np.sqrt(dof)
        R = float(za.max() - za.min())
        zt = DB @ w
        val = np.inf
        for mg in (inv_margin(k, cands[k], margin), 0.0):
            lo, hi = float(za.min()) - mg * R, float(za.max()) + mg * R
            if k == "reciprocal":
                hi = min(hi, float(za.max()) / 2)
            sm = np.mean(inv(np.clip(zt[:, None] + q[None, :], lo, hi)), axis=1)
            if np.all(np.isfinite(sm)):
                val = float(np.sqrt(np.mean((sm - y[ibs]) ** 2)))
                break
        out[k] = val
    return out


def paired_gate(des: Des, y: np.ndarray, cands: dict, pick: str, rng, max_rows: int = 20000, n_smear: int = 64, margin: float = 0.05, B: int = 400):
    """Held-out (B half) per-row squared-error difference identity - pick on the ORIGINAL scale (smearing, clipped inverse). Returns (mean_d, p_one_sided_signflip, rmse_id, rmse_pick)."""
    ia, ib = des.ia, des.ib
    DA, DB = des.D[ia], des.DB
    sel = np.arange(len(ib))
    if len(ib) > max_rows:
        sel = np.sort(rng.choice(len(ib), max_rows, replace=False))
    DB = DB[sel]
    yb = y[ib][sel]
    dof = len(ia) / max(len(ia) - DA.shape[1], 1)
    qs = (np.arange(n_smear) + 0.5) / n_smear
    preds = {}
    for k in ("identity", pick):
        f, inv = cands[k][0], cands[k][1]
        z = f(y)
        w = des.P @ z[ia]
        za = z[ia]
        q = np.quantile(za - DA @ w, qs) * np.sqrt(dof)
        R = float(za.max() - za.min())
        zt = DB @ w
        pr = None
        for mg in (inv_margin(k, cands[k], margin), 0.0):
            lo, hi = float(za.min()) - mg * R, float(za.max()) + mg * R
            if k == "reciprocal":
                hi = min(hi, float(za.max()) / 2)
            sm = np.mean(inv(np.clip(zt[:, None] + q[None, :], lo, hi)), axis=1)
            if np.all(np.isfinite(sm)):
                pr = sm
                break
        if pr is None:
            return -np.inf, 1.0, np.inf, np.inf
        preds[k] = pr
    d = (yb - preds["identity"]) ** 2 - (yb - preds[pick]) ** 2
    obs = d.mean()
    s = rng.choice([-1.0, 1.0], size=(B, len(d)))
    p = (1 + np.sum((s * (d - 0.0)).mean(1) >= obs)) / (B + 1)
    return float(obs), float(p), float(np.sqrt(np.mean((yb - preds["identity"]) ** 2))), float(np.sqrt(np.mean((yb - preds[pick]) ** 2)))


def atom_mass(y: np.ndarray) -> tuple[float, int]:
    """(largest single-value mass, number of distinct values)."""
    v, c = np.unique(y, return_counts=True)
    return float(c.max() / len(y)), len(v)


def target_decide(X: np.ndarray, y: np.ndarray, rng, B: int = 99, q: float = 0.95, extended: bool = False, floor: float = 1e-6, fixes: tuple = (), clip_null: bool = True) -> dict:
    """Wave-1 target criterion and its stages.

    picks: ``LL`` (argmax of held-out Jacobian LL, YJ with |lambda-1|<0.1 collapsed to identity), ``LLnull`` (+ sign-flip bootstrap null, 1+count rule at level ``1-q``),
    ``W1`` (+ held-out original-scale no-worse-than-identity gate). Fix picks are keyed by name when requested via ``fixes``.
    """
    t0 = time.perf_counter()
    des = Des(X, rng)
    cands = build_candidates(y, rng, extended=extended, floor=floor)
    ll = ll_scores(des, y, cands)
    nonid = {k: v for k, v in ll.items() if k != "identity"}
    best = max(nonid, key=nonid.get) if nonid else "identity"
    gain = nonid[best] - ll["identity"] if nonid else -np.inf
    lam = cands["yeo_johnson"][3]
    collapsed = best == "yeo_johnson" and abs(lam - 1.0) < 0.1
    res = dict(ll=ll, best=best, gain=float(gain), lam=lam, n_cand=len(cands))
    res["LL"] = "identity" if (collapsed or gain <= 0) else best
    t1 = time.perf_counter()
    mx = null_max(des, y, B, rng, extended, floor, clip=clip_null)
    tau = float(np.quantile(mx, q))
    pval = float((1 + np.sum(mx >= gain)) / (B + 1))
    res.update(tau=tau, pval=pval, t_null=time.perf_counter() - t1)
    res["LLnull"] = best if (pval <= 1 - q and gain > 0 and not collapsed) else "identity"
    if res["LLnull"] != "identity":
        g = heldout_orig_rmse(des, y, cands, ["identity", res["LLnull"]], rng=rng)
        res["g2_ratio"] = g[res["LLnull"]] / g["identity"]
        res["W1"] = res["LLnull"] if g[res["LLnull"]] <= g["identity"] else "identity"
    else:
        res["W1"] = "identity"
    res["des"] = des
    res["cands"] = cands
    res["t_total"] = time.perf_counter() - t0
    return res


# ======================================================================================================================================================================
# FEATURE side
# ======================================================================================================================================================================
def feat_candidates(x: np.ndarray, extended: bool = False, ref: np.ndarray | None = None) -> dict:
    """a2 candidate set (id, log, sqrt, 1/x for x>0; (x-med)^2, |x-med|, tanh((x-med)/IQR), cbrt); parameters and domain from ``ref`` (default x itself)."""
    ref = x if ref is None else ref
    med = np.median(ref)
    sc = float(np.subtract(*np.percentile(ref, [75, 25]))) + 1e-12
    mn = float(ref.min())
    c = {"id": x}
    if mn > 0:
        c.update(log=np.log(x), sqrt=np.sqrt(x), recip=1.0 / x)
    elif extended and mn >= 0:
        c.update(sqrt=np.sqrt(np.maximum(x, 0)), log1p=np.log1p(np.maximum(x, 0)))
    c.update(square=(x - med) ** 2, abs=np.abs(x - med), tanh=np.tanh((x - med) / sc), cbrt=np.cbrt(x))
    if extended:
        c["asinh"] = np.arcsinh((x - med) / sc)
        c["sign_sqrt"] = np.sign(x - med) * np.sqrt(np.abs(x - med))
    return c


def _stdcols(F: np.ndarray) -> np.ndarray:
    F = F - F.mean(0)
    s = np.sqrt((F * F).sum(0))
    s[s == 0] = np.inf
    return F / s


def feat_scores(F: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Pearson information -0.5 log(1-rho^2) of every column of F with r."""
    rc = r - r.mean()
    rn = np.sqrt((rc * rc).sum())
    rho = (_stdcols(F).T @ rc) / (rn if rn > 0 else np.inf)
    return -0.5 * np.log(np.maximum(1 - rho**2, 1e-12))


def feat_decide(x: np.ndarray, Zo: np.ndarray | None, y: np.ndarray, rng, B: int = 100, q: float = 0.95, extended: bool = False) -> dict:
    """Wave-1 (A2) conditional Pearson-info criterion for ONE feature given the other linear columns Zo (or None)."""
    A = np.ones((len(y), 1)) if Zo is None else np.c_[np.ones(len(y)), Zo]
    r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    C = feat_candidates(x, extended)
    names = list(C)
    F = _stdcols(np.column_stack([C[k] for k in names]))
    ii = names.index("id")
    rc = r - r.mean()
    rn = np.sqrt((rc * rc).sum()) or np.inf
    rho = F.T @ rc / rn
    s = -0.5 * np.log(np.maximum(1 - rho**2, 1e-12))
    gains = {k: float(s[j] - s[ii]) for j, k in enumerate(names) if k != "id"}
    best = max(gains, key=gains.get)
    T = gains[best]
    Pm = np.stack([rng.permutation(len(y)) for _ in range(B)]).astype(np.int32)
    Rp = rc[Pm]  # B x n
    rho_b = (Rp @ F) / rn  # B x K
    sb = -0.5 * np.log(np.maximum(1 - rho_b**2, 1e-12))
    others = [j for j in range(len(names)) if j != ii]
    Tb = (sb[:, others] - sb[:, [ii]]).max(1)
    tau = float(np.quantile(Tb, q))
    pval = float((1 + np.sum(Tb >= T)) / (B + 1))
    return dict(best=best, gain=T, tau=tau, pval=pval, accept=bool(T > tau), pick=best if T > tau else "id", names=names, s=dict(zip(names, s)), Tb=Tb)


def feat_many(Xp: np.ndarray, y: np.ndarray, B: int, rng, extended: bool = False, cols=None) -> dict:
    """A2 criterion for EVERY column j of Xp at once, residual = y minus OLS on ALL other columns (Frisch-Waugh), shared permutations across features so that the
    joint null of max over features x candidates is available. Returns per feature: T_obs (max_f s_f - s_id), best name, null draws Tb (p x B)."""
    n, p = Xp.shape
    D = np.c_[np.ones(n), Xp]
    G = D.T @ D
    G[np.diag_indices_from(G)] += 1e-9 * np.trace(G) / len(G)
    Gi = np.linalg.inv(G)
    beta = Gi @ (D.T @ y)
    r_full = y - D @ beta
    Pm = np.stack([rng.permutation(n) for _ in range(B)]).astype(np.int32)
    cols = list(range(p)) if cols is None else list(cols)
    T_obs = np.zeros(len(cols))
    Tb = np.zeros((len(cols), B))
    best = []
    s_id = np.zeros(len(cols))
    for a, j in enumerate(cols):
        x = Xp[:, j]
        xt = D @ Gi[:, j + 1] / Gi[j + 1, j + 1]
        rj = r_full + beta[j + 1] * xt
        C = feat_candidates(x, extended)
        names = list(C)
        F = _stdcols(np.column_stack([C[k] for k in names]))
        ii = names.index("id")
        rc = rj - rj.mean()
        rn = np.sqrt((rc * rc).sum()) or np.inf
        rho = F.T @ rc / rn
        s = -0.5 * np.log(np.maximum(1 - rho**2, 1e-12))
        others = [t for t in range(len(names)) if t != ii]
        gains = s[others] - s[ii]
        T_obs[a] = gains.max()
        best.append(names[others[int(np.argmax(gains))]])
        sb = -0.5 * np.log(np.maximum(1 - ((rc[Pm] @ F) / rn) ** 2, 1e-12))
        Tb[a] = (sb[:, others] - sb[:, [ii]]).max(1)
        s_id[a] = s[ii]
    return dict(cols=cols, T=T_obs, Tb=Tb, best=best, s_id=s_id)


def feat_accept(m: dict, q: float = 0.95) -> dict:
    """Decision rules on the output of feat_many: boolean accept per feature for (raw per-feature null, global maxT, studentised global maxT, Bonferroni-Sidak on per-feature p)."""
    T, Tb = m["T"], m["Tb"]
    B = Tb.shape[1]
    out = {}
    out["perfeat"] = T > np.quantile(Tb, q, axis=1)
    out["maxT"] = T > np.quantile(Tb.max(0), q)
    mu, sd = Tb.mean(1, keepdims=True), Tb.std(1, keepdims=True) + 1e-12
    Zb = (Tb - mu) / sd
    Zo = (T - mu[:, 0]) / sd[:, 0]
    out["maxT_stud"] = Zo > np.quantile(Zb.max(0), q)
    pj = (1 + (Tb >= T[:, None]).sum(1)) / (B + 1)
    out["bonf"] = pj <= (1 - q) / len(T)
    out["pj"] = pj
    return out
