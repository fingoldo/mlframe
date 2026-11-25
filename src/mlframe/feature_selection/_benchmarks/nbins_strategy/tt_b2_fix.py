"""B2 fixes (target side). ``apply_fixes`` receives the wave-1 result dict (``tt_b2_crit.target_decide``) and returns extra picks keyed by fix name.

Fix names (cumulative combinations are separate keys so that each stage can be audited):
  S      support precondition: abstain (identity) when a single value carries >= 3 % of the rows (and >= 10 rows) or the target has <= 20 distinct values
  G2p    gate 2 as a paired one-sided sign-flip test on the held-out original-scale squared-error difference (identity minus pick): require mean > 0 and p <= 0.05
  SG     S and G2p together on top of LL + null
  (more are appended below as the failure modes are identified)
"""
from __future__ import annotations

import numpy as np

import tt_b2_crit as K
from tt_b2_core import inv_margin


def support_abstain(y: np.ndarray, atom_frac: float = 0.03, min_count: int = 10, max_unique: int = 20) -> bool:
    v, c = np.unique(y, return_counts=True)
    return bool((c.max() / len(y) >= atom_frac and c.max() >= min_count) or len(v) <= max_unique)


def _g2p(des, y, cands, pk, rng):
    if pk == "identity":
        return "identity", {}
    d, p, r_id, r_pk = K.paired_gate(des, y, cands, pk, rng)
    return (pk if (d > 0 and p <= 0.05) else "identity"), dict(g2_mean=d, g2_p=p, g2_rmse_id=r_id, g2_rmse_pick=r_pk)


def apply_fixes(res: dict, X, y, rng, B: int, fixes: tuple) -> dict:
    """Extra picks. Always: S, G2p, SG (cheap). With 'R': robust LL pipeline (RG, SRG). With 'E': extended candidate set pipeline (EG, SEG, E_W1)."""
    picks, diag = {}, {}
    pk = res["LLnull"]
    des, cands = res["des"], res["cands"]
    abst = support_abstain(y)
    diag["support_abstain"] = abst
    g2p, d = _g2p(des, y, cands, pk, rng)
    diag.update(d)
    picks["S"] = "identity" if abst else res["W1"]
    picks["G2p"] = g2p
    picks["SG"] = "identity" if abst else g2p
    if "R" in fixes:
        ll = ll_scores_rob(des, y, cands)
        nonid = {k: v for k, v in ll.items() if k != "identity"}
        best = max(nonid, key=nonid.get) if nonid else "identity"
        gain = nonid[best] - ll["identity"] if nonid else -np.inf
        collapsed = best == "yeo_johnson" and abs(cands["yeo_johnson"][3] - 1.0) < 0.1
        mx = null_max_fn(des, y, B, rng, False, 1e-6, ll_scores_rob)
        pval = float((1 + np.sum(mx >= gain)) / (B + 1))
        rpk = best if (pval <= 0.05 and gain > 0 and not collapsed) else "identity"
        rg, d = _g2p(des, y, cands, rpk, rng)
        picks["R"] = rpk
        picks["RG"] = rg
        picks["SRG"] = "identity" if abst else rg
        diag.update(r_best=best, r_gain=float(gain), r_pval=pval)
    if "E" in fixes:
        re_ = K.target_decide(X, y, rng, B=B, extended=True)
        picks["E_LLnull"] = re_["LLnull"]
        picks["E_W1"] = re_["W1"]
        eg, d = _g2p(re_["des"], y, re_["cands"], re_["LLnull"], rng)
        picks["EG"] = eg
        picks["SEG"] = "identity" if abst else eg
        diag.update(e_best=re_["best"], e_gain=re_["gain"], e_pval=re_["pval"])
    if "N" in fixes:  # null without clipping at the support boundary: clipping at 0 creates atoms at 0 in the draws and inflates tau for y >= 0 with zeros
        rn = K.target_decide(X, y, rng, B=B, clip_null=False)
        picks["N_LLnull"] = rn["LLnull"]
        picks["N_W1"] = rn["W1"]
        ng, d = _g2p(rn["des"], y, rn["cands"], rn["LLnull"], rng)
        picks["NG"] = ng
        diag.update(n_tau=rn["tau"], n_pval=rn["pval"], n_gain=rn["gain"])
    if "V" in fixes:
        cv = cv_decide(des, y, cands, rng)
        picks["CV"] = cv["pick"]
        # HX: union acceptance (CV accepts OR the LL pipeline accepts), the transform is the one with the lowest cross-fitted error among the two proposals
        lp = picks.get("N_W1", res["W1"])
        props = [q for q in {cv["pick"], lp} if q != "identity" and q in cv.get("mse", {})]
        picks["HX"] = min(props, key=lambda q: cv["mse"][q]) if props else "identity"
        picks["SCV"] = "identity" if abst else cv["pick"]
        cm = cv_decide(des, y, cands, rng, kind="abs")
        picks["CVm"] = cm["pick"]
        diag.update(cvm_best=cm.get("best"), cvm_p=cm["pval"], cvm_ratio=cm["ratio"])
        diag.update(cv_best=cv.get("best"), cv_T=cv["T"], cv_thr=cv["thr"], cv_p=cv["pval"], cv_ratio=cv["ratio"])
    return {"picks": picks, "diag": diag}


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# R: Huber-robust Jacobian likelihood. The Gaussian LL rewards any transform that makes the marginal tails of f(y) look normal (asinh / cbrt on heavy tails, outlier
# rows), which does not reduce the mean error. Replace -0.5 log MSE by -log s - mean huber(r/s) with s = MAD scale of the held-out residuals (the Huber normaliser is a
# constant of the loss, equal across candidates).
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def ll_scores_rob(des, y, cands, rows=None, k: float = 1.345):
    names = list(cands)
    Z = np.empty((len(y), len(names)))
    for j, nm in enumerate(names):
        Z[:, j] = cands[nm][0](y)
    good = np.isfinite(Z).all(0)
    Z[:, ~good] = 0.0
    R = Z[des.ib] - des.pred_B(Z)
    med = np.median(R, axis=0)
    s = 1.4826 * np.median(np.abs(R - med), axis=0)
    s = np.maximum(s, 1e-12)
    u = np.abs(R - med) / s
    hub = np.where(u <= k, 0.5 * u * u, k * u - 0.5 * k * k).mean(0)
    yb = y[des.ib]
    jac = np.array([np.mean(cands[nm][2](yb)) for nm in names])
    ll = -np.log(s) - hub + jac
    ll[~good | ~np.isfinite(jac)] = -np.inf
    return dict(zip(names, ll))


def null_max_fn(des, y, B, rng, extended, floor, score_fn):
    """Same sign-flip bootstrap as tt_b2_crit.null_max, with a pluggable score function."""
    D = des.D
    G = D.T @ D
    beta = np.linalg.solve(G + 1e-9 * np.eye(G.shape[0]) * np.trace(G) / G.shape[0], D.T @ y)
    fit = D @ beta
    r = y - fit
    lo, hi = K._clip_bounds(y)
    out = np.empty(B)
    for b in range(B):
        ys = fit + (rng.integers(0, 2, len(y)) * 2 - 1.0) * r
        if lo is not None or hi is not None:
            ys = np.clip(ys, lo, hi)
        cb = K.build_candidates(ys, rng, extended=extended, floor=floor)
        ll = score_fn(des, ys, cb)
        out[b] = max((v - ll["identity"] for kk, v in ll.items() if kk != "identity"), default=-np.inf)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# V: direct cross-fitted held-out comparison on the ORIGINAL scale (closed-form linear head per candidate, clipped inverse + smearing), decided by a paired sign-flip
# max-T test over candidates. It does not use the Jacobian, so atoms / bounded supports / unsafe inverses are judged by what they do to the prediction.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def cv_errors(des, y, cands, names, rng, max_rows: int = 20000, n_smear: int = 64, margin: float = 0.05, kind: str = "sq"):
    D = des.D
    qs = (np.arange(n_smear) + 0.5) / n_smear
    out = {k: [] for k in names}
    for tr, te in ((des.ia, des.ib), (des.ib, des.ia)):
        Dtr = D[tr]
        G = Dtr.T @ Dtr
        G[np.diag_indices_from(G)] += 1e-9 * np.trace(G) / len(G)
        P = np.linalg.solve(G, Dtr.T)
        if len(te) > max_rows // 2:
            te = np.sort(rng.choice(te, max_rows // 2, replace=False))
        Dte = D[te]
        dof = len(tr) / max(len(tr) - D.shape[1], 1)
        for k in names:
            f, inv = cands[k][0], cands[k][1]
            z = f(y)
            if not np.all(np.isfinite(z)):
                out[k].append(np.full(len(te), np.inf))
                continue
            w = P @ z[tr]
            za = z[tr]
            q = np.quantile(za - Dtr @ w, qs) * np.sqrt(dof)
            R = float(za.max() - za.min())
            zt = Dte @ w
            err = np.full(len(te), np.inf)
            for mg in (inv_margin(k, cands[k], margin), 0.0):
                lo, hi = float(za.min()) - mg * R, float(za.max()) + mg * R
                if k == "reciprocal":
                    hi = min(hi, float(za.max()) / 2)
                if kind == "abs":  # conditional-median prediction (what a transformed head estimates), absolute error: the MAE question
                    sm = inv(np.clip(zt, lo, hi))
                else:
                    sm = np.mean(inv(np.clip(zt[:, None] + q[None, :], lo, hi)), axis=1)
                if np.all(np.isfinite(sm)):
                    err = (y[te] - sm) ** 2 if kind == "sq" else np.abs(y[te] - sm)
                    break
            out[k].append(err)
    return {k: np.concatenate(v) for k, v in out.items()}


def cv_decide(des, y, cands, rng, B: int = 499, q: float = 0.95, max_rows: int = 20000, kind: str = "sq") -> dict:
    names = [k for k in cands if k != "identity" and not (k == "yeo_johnson" and abs(cands[k][3] - 1.0) < 0.1)]
    E = cv_errors(des, y, cands, ["identity", *names], rng, max_rows, kind=kind)
    ok = [k for k in names if np.all(np.isfinite(E[k]))]
    if not ok:
        return dict(pick="identity", T=-np.inf, thr=np.inf, pval=1.0, ratio=1.0)
    d = np.column_stack([E["identity"] - E[k] for k in ok])
    ss = np.sqrt((d**2).sum(0)) + 1e-300
    T = d.sum(0) / ss
    S = (rng.integers(0, 2, (B, len(d)), dtype=np.int8) * 2 - 1).astype(np.float32)
    Tb = (S @ d.astype(np.float32)) / ss.astype(np.float32)
    mx = Tb.max(1)
    thr = float(np.quantile(mx, q))
    sig = np.where((T > thr) & (d.sum(0) > 0))[0]
    j = int(sig[np.argmax(d.sum(0)[sig])]) if len(sig) else int(np.argmax(T))  # largest held-out gain among the significant ones
    pval = float((1 + np.sum(mx >= T[j])) / (B + 1))
    ratio = float(np.sqrt(np.mean(E[ok[j]]) / np.mean(E["identity"]))) if kind == "sq" else float(np.mean(E[ok[j]]) / np.mean(E["identity"]))
    mse = {k: float(np.mean(E[k])) for k in ok}
    mse["identity"] = float(np.mean(E["identity"]))
    return dict(pick=ok[j] if len(sig) else "identity", best=ok[j], T=float(T[j]), thr=thr, pval=pval, ratio=ratio, mse=mse)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# FEATURE side fix P: partial correlation. The wave-1 conditional criterion removes the linear part of the other columns Z from y but correlates the RAW f(x) with that
# residual; when x is correlated with Z, a candidate that is less collinear with Z scores higher for that reason alone (|x - med| beat the identity 20/20 on a linear
# lognormal x with Z ~ log x, harming RMSE by 4 %). Frisch-Waugh: residualise y AND every candidate on [1, Z], then correlate.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def feat_decide_partial(x, Zo, y, rng, B: int = 100, q: float = 0.95, extended: bool = False) -> dict:
    A = np.ones((len(y), 1)) if Zo is None else np.c_[np.ones(len(y)), Zo]
    Q, _ = np.linalg.qr(A)
    r = y - Q @ (Q.T @ y)
    C = K.feat_candidates(x, extended)
    names = list(C)
    F = np.column_stack([C[k] for k in names])
    F = F - Q @ (Q.T @ F)
    F = K._stdcols(F)
    ii = names.index("id")
    rc = r - r.mean()
    rn = np.sqrt((rc * rc).sum()) or np.inf
    s = -0.5 * np.log(np.maximum(1 - (F.T @ rc / rn) ** 2, 1e-12))
    others = [j for j in range(len(names)) if j != ii]
    gains = s[others] - s[ii]
    T = float(gains.max())
    best = names[others[int(np.argmax(gains))]]
    Pm = np.stack([rng.permutation(len(y)) for _ in range(B)]).astype(np.int32)
    sb = -0.5 * np.log(np.maximum(1 - ((rc[Pm] @ F) / rn) ** 2, 1e-12))
    Tb = (sb[:, others] - sb[:, [ii]]).max(1)
    tau = float(np.quantile(Tb, q))
    return dict(best=best, gain=T, tau=tau, pval=float((1 + np.sum(Tb >= T)) / (B + 1)), accept=bool(T > tau), pick=best if T > tau else "id")
