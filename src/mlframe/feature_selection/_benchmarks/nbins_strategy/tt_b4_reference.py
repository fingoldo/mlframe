"""Reference implementation of the model-free feature-transform recommender (research B4, 2026-11-24).

For every numeric column x_j and every valid unary candidate f in {id, log, sqrt, recip, square=(x-med)^2, abs=|x-med|, tanh=tanh((x-med)/IQR), cbrt, rankgauss}
the squared Pearson correlation s_f = rho^2(f(x_j) | Q, r) is computed, where r is the residual of y after OLS on the already selected linear columns (the design D, intercept
always included) and f(x_j) is partialled on the same design (Freedman-Lane form, so a candidate correlated with the selected columns gets credit only for what the design lacks).
All columns and candidates go through one GEMM per column block; the permutation null uses the SAME permutations of r for every column (so correlated columns share the
null structure) and gives, per permutation and column,
    T_j  = max_{f != id valid} (s_f - s_id)      the transform gain
    Rel_j = max_{f valid} s_f                     the relevance of the column
Decision rules (all computed from the same permutations):
    column : p_j = (1 + #{b: T_jb >= T_j}) / (B + 1)                    per-column null only
    maxT   : p_j = (1 + #{b: max_j' T_j'b >= T_j}) / (B + 1)            max over columns (family-wise)
    pooled : p_j = min(1, p * (1 + #{(b,j'): T_j'b >= T_j}) / (1 + B p)) null pooled over columns, Bonferroni by p
    gated  : the column must pass the relevance maxT gate (family-wise on Rel), then the per-column gain p-value is Sidak-corrected by the number of relevant columns
A candidate is ACCEPTED only if the rule passes on the discovery half A AND the gain replicates on the held-out half B (own permutation null for that one pair).

recommend_feature_transforms: one pass over all columns given `selected`.
recommend_stepwise: forward selection with transforms (relevance gate -> pick the strongest column -> transform decision -> add to design -> repeat -> backfit pass),
which is what credits the right column when features are correlated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtri

CAND_DEFAULT = ("id", "log", "sqrt", "recip", "square", "abs", "tanh", "cbrt", "rankgauss")
# MRMR's own unary table (filters/_mi_greedy_fe.py UNARY_TRANSFORMS): fixed-zero forms, no centring or scaling
CAND_MRMR = ("id", "log_abs", "sqrt_abs", "sq0", "cube", "recip0", "tanh0", "expm1c", "abs0")


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# candidate construction
# ----------------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class ColParams:
    """Per-column constants fitted on the discovery rows (replayed unchanged on any other rows)."""
    med: np.ndarray
    scale: np.ndarray
    mn: np.ndarray

    @staticmethod
    def fit(XA):
        med = np.median(XA, axis=0)
        q = np.percentile(XA, [25, 75], axis=0)
        iqr = q[1] - q[0]
        sd = XA.std(axis=0)
        scale = np.where(iqr > 1e-12, iqr, np.where(sd > 1e-12, sd, 1.0))
        return ColParams(med.astype(np.float64), scale.astype(np.float64), XA.min(axis=0).astype(np.float64))


def rg_grid(xa):
    """Rank-gauss lookup (sorted unique values, normal score of the average rank) of one column of the discovery rows."""
    u, inv, cnt = np.unique(xa, return_inverse=True, return_counts=True)
    mid = np.cumsum(cnt) - 0.5 * cnt
    sc = ndtri(mid / xa.shape[0])
    return u, sc, inv


def rg_apply(grid, x):
    u, sc, _ = grid
    return np.interp(x, u, sc)


def valid_mask(params, cands):
    """(p, K) bool: which candidate is defined for which column (decided on the discovery rows)."""
    p = params.mn.shape[0]
    v = np.ones((p, len(cands)), bool)
    for k, c in enumerate(cands):
        if c in ("log", "recip"):
            v[:, k] = params.mn > 0
        elif c == "sqrt":
            v[:, k] = params.mn >= 0
        elif c in ("log1p",):
            v[:, k] = params.mn > -1
    return v


def build_candidates(Xr, params, cands, grids=None):
    """Candidate stack (n, b, K) for the columns of Xr (n, b) with the per-column `params` (already sliced to those b columns).
    Invalid cells are filled with zeros; the caller masks them with valid_mask. rankgauss uses `grids` (list of grids, one per column) when given,
    else it is computed from the rows themselves (discovery rows)."""
    n, b = Xr.shape
    K = len(cands)
    F = np.zeros((n, b, K))
    med, scale, _mn = params.med, params.scale, params.mn
    for k, c in enumerate(cands):
        if c == "id":
            F[:, :, k] = Xr
        elif c == "log":
            F[:, :, k] = np.log(np.maximum(Xr, 1e-12))
        elif c == "sqrt":
            F[:, :, k] = np.sqrt(np.maximum(Xr, 0.0))
        elif c == "recip":
            F[:, :, k] = 1.0 / np.maximum(Xr, 1e-12)
        elif c == "square":
            F[:, :, k] = (Xr - med) ** 2
        elif c == "abs":
            F[:, :, k] = np.abs(Xr - med)
        elif c == "tanh":
            F[:, :, k] = np.tanh((Xr - med) / scale)
        elif c == "cbrt":
            F[:, :, k] = np.cbrt(Xr)
        elif c == "log1p":
            F[:, :, k] = np.log1p(np.maximum(Xr, -1 + 1e-12))
        elif c == "asinh":
            F[:, :, k] = np.arcsinh((Xr - med) / scale)
        elif c == "log_abs":
            F[:, :, k] = np.log1p(np.abs(Xr))
        elif c == "sqrt_abs":
            F[:, :, k] = np.sqrt(np.abs(Xr))
        elif c == "sq0":
            F[:, :, k] = Xr * Xr
        elif c == "cube":
            F[:, :, k] = Xr * Xr * Xr
        elif c == "recip0":
            sg = np.where(Xr >= 0, 1.0, -1.0)
            F[:, :, k] = 1.0 / (Xr + 1e-12 * sg)
        elif c == "tanh0":
            F[:, :, k] = np.tanh(Xr)
        elif c == "expm1c":
            F[:, :, k] = np.expm1(np.clip(Xr, -20.0, 20.0))
        elif c == "abs0":
            F[:, :, k] = np.abs(Xr)
        elif c == "rankgauss":
            for jj in range(b):
                if grids is not None:
                    F[:, jj, k] = rg_apply(grids[jj], Xr[:, jj])
                else:
                    u, sc, inv = rg_grid(Xr[:, jj])
                    F[:, jj, k] = sc[inv]
        else:
            raise ValueError(c)
    return F


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# screening engine
# ----------------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class Screen:
    S: np.ndarray          # (p_sub, K) observed partial s_f (-inf for invalid)
    T: np.ndarray          # (p_sub,) observed gain max_f != id (s_f - s_id)
    best: np.ndarray       # (p_sub,) index of the best non-id candidate
    Rel: np.ndarray        # (p_sub,) observed max_f s_f
    Tn: np.ndarray         # (B, p_sub) null gains
    Reln: np.ndarray       # (B, p_sub) null relevances
    cols: np.ndarray       # column ids of the rows
    seconds: float = 0.0


def _orth(D):
    Q, _ = np.linalg.qr(D)
    return Q


def screen(X, y, cols, Q, params, cands, valid, n_null, rng, block_cols=None, dtype=np.float32, grids_for=None):
    """Score all candidates of the columns `cols` of X against the residual of y on the orthonormal design Q (n, m+1, intercept included)."""
    t0 = time.perf_counter()
    n = X.shape[0]
    K = len(cands)
    cols = np.asarray(cols, dtype=np.int64)
    ps = cols.size
    # residual of y on the design, and the permuted copies projected on the complement of the design (Freedman-Lane)
    yc = np.asarray(y, dtype=np.float64)
    r = yc - Q @ (Q.T @ yc)
    R = np.empty((n, 1 + n_null), dtype=np.float64)
    R[:, 0] = r
    for bidx in range(n_null):
        R[:, 1 + bidx] = r[rng.permutation(n)]
    R -= Q @ (Q.T @ R)
    rr = np.einsum("ij,ij->j", R, R)
    R32 = R.astype(dtype)
    np.full((ps, K), -np.inf)
    Tn = np.empty((n_null, ps))
    Reln = np.empty((n_null, ps))
    kid = cands.index("id")
    nonid = np.array([k for k in range(K) if k != kid])
    if block_cols is None:
        block_cols = max(1, min(ps, int(1.5e7 // max(n * K, 1))))
    obs_all = np.full((ps, K), -np.inf)
    for j0 in range(0, ps, block_cols):
        j1 = min(ps, j0 + block_cols)
        cj = cols[j0:j1]
        Xr = X[:, cj].astype(np.float64, copy=False) if X.dtype != np.float64 else X[:, cj]
        sub = ColParams(params.med[cj], params.scale[cj], params.mn[cj])
        grids = None if grids_for is None else [grids_for(int(c)) for c in cj]
        F = build_candidates(Xr, sub, cands, grids)
        b = j1 - j0
        F = F.reshape(n, b * K)
        F -= F.mean(axis=0)
        ss = np.einsum("ij,ij->j", F, F)
        QF = Q.T @ F
        ss_res = ss - np.einsum("ij,ij->j", QF, QF)
        N = F.astype(dtype).T @ R32            # (b*K, 1+B)
        N = N.astype(np.float64)
        ok = ss_res > 1e-10 * np.maximum(ss, 1e-300)
        den = np.where(ok, ss_res, 1.0)[:, None] * rr[None, :]
        s = np.where(ok[:, None], N * N / den, 0.0)      # (b*K, 1+B)
        s = s.reshape(b, K, 1 + n_null)
        vm = valid[cj][:, :, None]
        s = np.where(vm, s, -np.inf)
        obs_all[j0:j1] = s[:, :, 0]
        sid = s[:, kid, :]                                # (b, 1+B)
        gains = s[:, nonid, :] - sid[:, None, :]          # (b, K-1, 1+B); -inf stays -inf
        gmax = gains.max(axis=1)                          # (b, 1+B)
        rel = s.max(axis=1)
        Tn[:, j0:j1] = gmax[:, 1:].T
        Reln[:, j0:j1] = rel[:, 1:].T
        if j0 == 0:
            Tobs = np.empty(ps)
            Rel = np.empty(ps)
            best = np.empty(ps, dtype=np.int64)
        Tobs[j0:j1] = gmax[:, 0]
        Rel[j0:j1] = rel[:, 0]
        best[j0:j1] = nonid[gains[:, :, 0].argmax(axis=1)]
        del F, N, s, gains
    Tobs = np.where(np.isfinite(Tobs), Tobs, 0.0)
    Tn = np.where(np.isfinite(Tn), Tn, 0.0)
    return Screen(obs_all, Tobs, best, Rel, Tn, Reln, cols, time.perf_counter() - t0)


def pvalues(T_obs, T_null, rule, studentize=False):
    """p-values of T_obs (p,) against the permutation matrix T_null (B, p) under the named family-wise rule."""
    B, p = T_null.shape
    if studentize:
        sd = T_null.std(axis=0) + 1e-12
        T_obs = T_obs / sd
        T_null = T_null / sd
    if rule == "column":
        return (1.0 + (T_null >= T_obs[None, :]).sum(axis=0)) / (B + 1.0)
    if rule == "maxT":
        M = T_null.max(axis=1)
        return (1.0 + (M[:, None] >= T_obs[None, :]).sum(axis=0)) / (B + 1.0)
    if rule == "pooled":
        pool = np.sort(T_null.ravel())
        cnt = pool.size - np.searchsorted(pool, T_obs, side="left")
        return np.minimum(1.0, p * (1.0 + cnt) / (1.0 + pool.size))
    raise ValueError(rule)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# helpers: design and replication
# ----------------------------------------------------------------------------------------------------------------------------------------------------
def design_column(x, params_j, name, grid=None):
    """One transformed column (n,) from the raw column x with the per-column constants of the discovery rows."""
    sub = ColParams(np.array([params_j[0]]), np.array([params_j[1]]), np.array([params_j[2]]))
    cands = (name,)
    F = build_candidates(x[:, None], sub, cands, grids=[grid] if name == "rankgauss" else None)
    return F[:, 0, 0]


def _col_params_j(params, j):
    return (params.med[j], params.scale[j], params.mn[j])


def replicate(XB, yB, QB, params, grid_fn, j, fname, cands, n_null, rng, stop_frac=None):
    """Gain of candidate `fname` over identity for column j on the held-out rows B (design QB), own permutation null for this single pair.
    Returns (gain, p_value). With stop_frac the permutations stop as soon as the number of exceedances makes p <= stop_frac impossible (p then reported as 1.0)."""
    n = XB.shape[0]
    x = XB[:, j].astype(np.float64)
    pj = _col_params_j(params, j)
    fv = design_column(x, pj, fname, grid_fn(j) if fname == "rankgauss" else None)
    iv = x
    yc = np.asarray(yB, dtype=np.float64)
    r = yc - QB @ (QB.T @ yc)
    fv = fv - fv.mean(); fv = fv - QB @ (QB.T @ fv)
    iv = iv - iv.mean(); iv = iv - QB @ (QB.T @ iv)
    nf, ni = float(fv @ fv), float(iv @ iv)
    def gain(rvec):
        rr_ = float(rvec @ rvec)
        if rr_ <= 0:
            return 0.0
        sf = (fv @ rvec) ** 2 / (nf * rr_) if nf > 0 else 0.0
        si = (iv @ rvec) ** 2 / (ni * rr_) if ni > 0 else 0.0
        return sf - si
    g0 = gain(r)
    cnt = 0
    cap = None if stop_frac is None else stop_frac * (n_null + 1.0)
    for _b in range(n_null):
        rp = r[rng.permutation(n)]
        rp = rp - QB @ (QB.T @ rp)
        cnt += gain(rp) >= g0
        if cap is not None and cnt + 1.0 > cap:
            return g0, 1.0
    return g0, (1.0 + cnt) / (n_null + 1.0)


@dataclass
class Result:
    table: dict
    seconds: float
    detail: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# one pass over all columns
# ----------------------------------------------------------------------------------------------------------------------------------------------------
def _resolve_selected(X, selected, names):
    if selected is None:
        return [], None
    if isinstance(selected, np.ndarray) and selected.ndim == 2:
        return [], selected
    idx = []
    for s in selected:
        idx.append(names.index(s) if isinstance(s, str) else int(s))
    return idx, None


def recommend_feature_transforms(X, y, *, selected=None, candidates=None, n_null=50, seed=0, alpha=0.05, rule="gated", replicate_half=True, n_null_rep=100,
                                 studentize=False, split=0.5, dtype=np.float32, replicate_rules=("gated", "maxT", "pooled")):
    """One pass: which unary transform of each column helps a linear / additive model beyond the identity, given the `selected` linear columns.

    X (n, p) ndarray or DataFrame, y (n,), selected: column indices / names of X that are already in the linear model (identity, leave-one-out for their own test),
    or an (n, m) design matrix of already built columns. Returns Result with table['accepted'] etc.; table also holds the p-values of every rule so a caller can
    compare rules from the same permutations.
    """
    t_start = time.perf_counter()
    names = None
    if hasattr(X, "columns"):
        names = list(X.columns)
        X = X.to_numpy()
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    cands = tuple(candidates) if candidates is not None else CAND_DEFAULT
    if "id" not in cands:
        cands = ("id", *cands)
    K = len(cands)
    rng = np.random.default_rng(seed)
    sel_idx, sel_design = _resolve_selected(X, selected, names or [])
    if replicate_half:
        perm = rng.permutation(n)
        nA = round(n * split)
        iA, iB = np.sort(perm[:nA]), np.sort(perm[nA:])
    else:
        iA = np.arange(n)
        iB = None
    XA, yA = X[iA], y[iA]
    params = ColParams.fit(XA.astype(np.float64))
    valid = valid_mask(params, cands)
    grid_cache = {}
    def grid_fn(j):
        if j not in grid_cache:
            grid_cache[j] = rg_grid(XA[:, j].astype(np.float64))
        return grid_cache[j]
    def design_for(idx_rows, sel_cols, extra, rows_X):
        cols_ = [np.ones(rows_X.shape[0])]
        for j in sel_cols:
            cols_.append(rows_X[:, j].astype(np.float64))
        if extra is not None:
            cols_.extend(list(extra[idx_rows].T))
        return np.column_stack(cols_)
    DA = design_for(iA, sel_idx, sel_design, XA)
    QA = _orth(DA)
    [j for j in range(p) if j not in sel_idx]
    grids_for = None
    def screen_cols(cols_, nn):
        """screen of columns `cols_` with nn permutations; the selected ones get their own leave-one-out design."""
        plain = [j for j in cols_ if j not in sel_idx]
        out = {}
        if plain:
            sc_ = screen(XA, yA, plain, QA, params, cands, valid, nn, rng, dtype=dtype, grids_for=grids_for)
            for q, j in enumerate(plain):
                out[j] = (sc_.T[q], sc_.Rel[q], sc_.best[q], sc_.Tn[:, q], sc_.Reln[:, q], sc_.S[q])
        for j in cols_:
            if j in sel_idx:
                keep = [q for q in sel_idx if q != j]
                Dj = design_for(iA, keep, sel_design, XA)
                sj = screen(XA, yA, [j], _orth(Dj), params, cands, valid, nn, rng, dtype=dtype, grids_for=grids_for)
                out[j] = (sj.T[0], sj.Rel[0], sj.best[0], sj.Tn[:, 0], sj.Reln[:, 0], sj.S[0])
        return out

    t_s = time.perf_counter()
    res1 = screen_cols(list(range(p)), n_null)
    sc_seconds = time.perf_counter() - t_s
    T_obs = np.zeros(p); Rel_obs = np.zeros(p); best = np.zeros(p, dtype=np.int64)
    Tn = np.zeros((n_null, p)); Reln = np.zeros((n_null, p)); S = np.full((p, K), -np.inf)
    for j, (t_, r_, b_, tn_, rn_, s_) in res1.items():
        T_obs[j] = t_; Rel_obs[j] = r_; best[j] = b_; Tn[:, j] = tn_; Reln[:, j] = rn_; S[j] = s_
    pv = {}
    for rl in ("column", "maxT", "pooled"):
        pv[rl] = pvalues(T_obs, Tn, rl, studentize)
    pv["rel_maxT"] = pvalues(Rel_obs, Reln, "maxT", studentize)
    # gated rule: relevance gate (family-wise on all columns), then a longer per-column gain null on the relevant columns only, Bonferroni by their number
    rel_cols = [j for j in range(p) if pv["rel_maxT"][j] <= alpha]
    m_rel = len(rel_cols)
    pv["gated"] = np.ones(p)
    if m_rel:
        B2 = int(max(200, np.ceil(4.0 * m_rel / alpha)))
        res2 = screen_cols(rel_cols, B2)
        for j in rel_cols:
            pj = (1.0 + (res2[j][3] >= T_obs[j]).sum()) / (B2 + 1.0)
            pv["gated"][j] = min(1.0, pj * m_rel)
    # replication on the held-out half for the discoveries of the requested rules (alpha / m_rule on a fresh null for the single pair)
    rep_p = np.full(p, np.nan); rep_gain = np.full(p, np.nan)
    disc_r = {rl: (pv[rl] <= alpha) for rl in set(replicate_rules) | {rule}}
    accepted_r = {}
    if replicate_half:
        union = np.zeros(p, bool)
        for _rl, d in disc_r.items():
            union |= d
        m_max = max([int(d.sum()) for d in disc_r.values()] + [1])
        n_rep = int(max(n_null_rep, np.ceil(2.0 * m_max / alpha)))
        XB, yB = X[iB], y[iB]
        for j in np.flatnonzero(union):
            keep = [q for q in sel_idx if q != j]
            DB = design_for(iB, keep, sel_design, XB)
            rep_gain[j], rep_p[j] = replicate(XB, yB, _orth(DB), params, grid_fn, j, cands[best[j]], cands, n_rep, rng, stop_frac=alpha)
        for rl, d in disc_r.items():
            m = max(1, int(d.sum()))
            accepted_r[rl] = d & (rep_gain > 0) & (rep_p <= alpha / m)
    else:
        accepted_r = dict(disc_r)
    primary = pv[rule]
    accepted = accepted_r[rule]
    kid = cands.index("id")
    gain_ratio = 0.5 * np.log((1 - np.maximum(S[np.arange(p), kid], 0)) / np.maximum(1 - np.maximum(S[np.arange(p), best], 0), 1e-12))
    tab = {"col": np.arange(p), "best": np.array(cands)[best], "gain": T_obs, "relevance": Rel_obs, "p": primary, "p_column": pv["column"], "p_maxT": pv["maxT"],
           "p_pooled": pv["pooled"], "p_gated": pv["gated"], "p_rel": pv["rel_maxT"], "rep_gain": rep_gain, "rep_p": rep_p, "accepted": accepted, "credit": gain_ratio,
           "accepted_by_rule": accepted_r, "disc_by_rule": disc_r}
    return Result(tab, time.perf_counter() - t_start, {"cands": cands, "S": S, "params": params, "iA": iA, "iB": iB, "screen_seconds": sc_seconds,
                                                      "T_obs": T_obs, "Tn": Tn, "Rel_obs": Rel_obs, "Reln": Reln, "best_idx": best})


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# stepwise driver
# ----------------------------------------------------------------------------------------------------------------------------------------------------
def recommend_stepwise(X, y, *, candidates=None, n_null=50, seed=0, alpha=0.05, max_steps=12, replicate_half=True, n_null_rep=100, backfit=True,
                       dtype=np.float32, gain_alpha=None, verbose=False):
    """Forward selection with transforms.

    Each round: relevance gate (max-T over the remaining columns of max_f s_f given the current design) -> strongest relevant column j* -> transform decision for j*
    (per-column gain null, since j* already passed the family-wise gate, and replication on the held-out half) -> the chosen f*(x_j*) (or identity) joins the design.
    After the loop a backfit pass re-decides every committed transform given all the others. Returns Result with table['chosen'] = {col: transform name}.
    """
    t_start = time.perf_counter()
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    cands = tuple(candidates) if candidates is not None else CAND_DEFAULT
    len(cands)
    cands.index("id")
    rng = np.random.default_rng(seed)
    if replicate_half:
        perm = rng.permutation(n)
        nA = n // 2
        iA, iB = np.sort(perm[:nA]), np.sort(perm[nA:])
    else:
        iA = np.arange(n); iB = None
    XA, yA = X[iA].astype(np.float64), y[iA]
    XB = X[iB].astype(np.float64) if iB is not None else None
    yB = y[iB] if iB is not None else None
    params = ColParams.fit(XA)
    valid = valid_mask(params, cands)
    grid_cache = {}
    def grid_fn(j):
        if j not in grid_cache:
            grid_cache[j] = rg_grid(XA[:, j])
        return grid_cache[j]
    grids_for = None
    gain_alpha = alpha if gain_alpha is None else gain_alpha
    chosen = {}      # col -> transform name
    order = []
    log = []

    def col_values(Xrows, j, fname):
        if fname == "id":
            return Xrows[:, j]
        return design_column(Xrows[:, j], _col_params_j(params, j), fname, grid_fn(j) if fname == "rankgauss" else None)

    def build_design(Xrows, skip=None):
        cols_ = [np.ones(Xrows.shape[0])]
        for j in order:
            if j != skip:
                cols_.append(col_values(Xrows, j, chosen[j]))
        return np.column_stack(cols_)

    def decide(j, Qa, Qb_rows_design_fn):
        """Transform decision for column j given the orthonormal design Qa on A. Returns (fname, info)."""
        sj = screen(XA, yA, [j], Qa, params, cands, valid, n_null, rng, dtype=dtype, grids_for=grids_for)
        pj = float((1.0 + (sj.Tn[:, 0] >= sj.T[0]).sum()) / (n_null + 1.0))
        f = cands[int(sj.best[0])]
        info = {"gain": float(sj.T[0]), "p_gain": pj, "f": f}
        if pj > gain_alpha or not np.isfinite(sj.T[0]) or sj.T[0] <= 0:
            return "id", info
        if replicate_half:
            DB = Qb_rows_design_fn()
            g, pr = replicate(XB, yB, _orth(DB), params, grid_fn, j, f, cands, n_null_rep, rng)
            info.update(rep_gain=float(g), rep_p=float(pr))
            if not (g > 0 and pr <= gain_alpha):
                return "id", info
        return f, info

    for step in range(max_steps):
        DA = build_design(XA)
        QA = _orth(DA)
        rest = [j for j in range(p) if j not in chosen]
        if not rest:
            break
        sc = screen(XA, yA, rest, QA, params, cands, valid, n_null, rng, dtype=dtype, grids_for=grids_for)
        prel = pvalues(sc.Rel, sc.Reln, "maxT")
        k = int(np.argmin(prel))
        if prel[k] > alpha:
            break
        j = int(sc.cols[np.argmax(np.where(prel <= alpha, sc.Rel, -1.0))])
        fname, info = decide(j, QA, (lambda: build_design(XB)) if replicate_half else None)
        chosen[j] = fname
        order.append(j)
        info.update(step=step, col=j, p_rel=float(prel[list(sc.cols).index(j)]))
        log.append(info)
        if verbose:
            print("step", step, info, flush=True)
    if backfit and len(order) > 1:
        for j in order:
            DA = build_design(XA, skip=j)
            DB_fn = (lambda j=j: build_design(XB, skip=j)) if replicate_half else None
            fname, info = decide(j, _orth(DA), DB_fn)
            if fname != chosen[j]:
                log.append({"backfit": True, "col": j, "from": chosen[j], "to": fname, **info})
            chosen[j] = fname
    secs = time.perf_counter() - t_start
    return Result({"chosen": dict(chosen), "order": list(order), "log": log, "n_steps": len(order)}, secs, {"cands": cands, "params": params, "iA": iA, "iB": iB})


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# applying recipes
# ----------------------------------------------------------------------------------------------------------------------------------------------------
class Recipes:
    """Fitted transforms (name per column) replayable on any rows with constants of the fitting rows."""

    def __init__(self, Xfit, chosen, cands=CAND_DEFAULT):
        Xfit = np.asarray(Xfit, dtype=np.float64)
        self.params = ColParams.fit(Xfit)
        self.chosen = {int(j): f for j, f in chosen.items() if f != "id"}
        self.grids = {j: rg_grid(Xfit[:, j]) for j, f in self.chosen.items() if f == "rankgauss"}

    def transform_cols(self, X):
        out = {}
        X = np.asarray(X, dtype=np.float64)
        for j, f in self.chosen.items():
            out[j] = design_column(X[:, j], _col_params_j(self.params, j), f, self.grids.get(j))
        return out


def all_candidate_matrix(Xfit, X, cands=CAND_DEFAULT):
    """Every valid non-identity candidate of every column: (matrix (n, M), list of (col, name)). Constants from Xfit (blind-add baseline)."""
    Xfit = np.asarray(Xfit, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    params = ColParams.fit(Xfit)
    valid = valid_mask(params, cands)
    cols, meta = [], []
    for j in range(X.shape[1]):
        for k, c in enumerate(cands):
            if c == "id" or not valid[j, k]:
                continue
            g = rg_grid(Xfit[:, j]) if c == "rankgauss" else None
            cols.append(design_column(X[:, j], _col_params_j(params, j), c, g))
            meta.append((j, c))
    return np.column_stack(cols), meta
