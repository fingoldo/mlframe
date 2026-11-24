"""B5 / question 2: one-pass, no-retraining implementations of the two information criteria, written for speed (numpy BLAS + numba), streamed in row chunks.

(a) TARGET criterion ``target_criterion``: for K candidate transforms f_k of y, the Jacobian-corrected profile log-likelihood
        LL_k = -0.5 log MSE_B(f_k) + mean_B log|f_k'(y)|
    of a ridge fit on half A evaluated on half B, plus the sign-flip residual-bootstrap null (B draws) of max_{k != id} (LL_k - LL_id).  NOTHING is refitted:
      * every ridge fit is the closed form W = M c_A with M = (G_A + reg I)^-1 fixed once; the held-out MSE is a quadratic form of sufficient statistics
            MSE_B = (z'z - 2 c_B' W + W' G_B W) / n_B,       c = D' z,  G = D'D,  D = [1, X_design],
        so a candidate costs O(m^2) after one pass over the data, independent of n;
      * the null draw y*_b = yhat + s_b * r (s_i = +-1) maps through any f as   f(y*_i) = u_i + s_bi d_i   EXACTLY, with u = (f(y) + f(2 yhat - y)) / 2 and
        d = (f(y) - f(2 yhat - y)) / 2 (two points, two signs).  So all B draws of all K candidates need ONE extra transform per row and one GEMM D'(d_k * S) of width K*B;
        the same identity holds for the Jacobian term and for z*'z* = sum u^2 + 2 sum s u d + sum d^2.
    The design can be capped to the ``cap`` columns most correlated with the candidates, so the Gram costs n cap^2, not n p^2.

(b) FEATURE criterion ``feature_criterion``: s_jk = rho^2(f_k(x_j), r) for p columns x K candidates in ONE pass (a fused numba kernel, no p x K x n temporary),
    and a permutation null that permutes r and never recomputes a transform: the transformed block Z (cb*K, rows) is built once per (row block, column block)
    and multiplied by R = [r, r_pi1 .. r_piB] (rows, B+1) with one sgemm.  Per column the threshold is the (1 - alpha) quantile over draws of max_{k != id}(s_k - s_id).
    ``null_rows`` runs the null on the first null_rows rows of every chunk (a subsample) and rescales the threshold by n_sub / n.

Providers serve ``chunk(i) -> (Xt (p, rows) float32 feature-major, y (rows,) float64)``; the first half of the chunks is the A half, the second the B half (rows are
assumed already shuffled).  Time inside ``chunk`` (data generation) accumulates in ``prov.gen_seconds`` so a benchmark can exclude it.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

LOG2 = math.log(2.0)
LOG3 = math.log(3.0)

# ------------------------------------------------------------------ providers


class ArrayProvider:
    """Serve an in-memory feature-major float32 matrix ``Xt (p, n)`` and ``y (n,)`` in row chunks."""

    def __init__(self, Xt, y, chunk_rows=5000):
        self.Xt, self.y = Xt, np.asarray(y, dtype=np.float64)
        self.p, self.n = Xt.shape
        self.chunk_rows = int(chunk_rows)
        self.nchunks = (self.n + self.chunk_rows - 1) // self.chunk_rows
        self.gen_seconds = 0.0

    def bounds(self, i):
        return i * self.chunk_rows, min(self.n, (i + 1) * self.chunk_rows)

    def chunk(self, i):
        a, b = self.bounds(i)
        return self.Xt[:, a:b], self.y[a:b]

    def target(self):
        return self.y


class GenProvider:
    """Generate chunk i deterministically and never hold X: x_j ~ N(0,1) float32, y = exp(0.8 x0 + 0.5 x1 + 0.3 x2 + 0.3 e) (log y is linear in x0..x2)."""

    def __init__(self, n, p, seed=0, chunk_rows=5000):
        self.n, self.p, self.seed, self.chunk_rows = int(n), int(p), int(seed), int(chunk_rows)
        self.nchunks = (self.n + self.chunk_rows - 1) // self.chunk_rows
        self.gen_seconds = 0.0
        self._y = None

    def bounds(self, i):
        return i * self.chunk_rows, min(self.n, (i + 1) * self.chunk_rows)

    def chunk(self, i):
        t = time.perf_counter()
        a, b = self.bounds(i)
        rows = b - a
        rng = np.random.default_rng([self.seed, i])
        Xt = rng.standard_normal((self.p, rows), dtype=np.float32)
        e = rng.standard_normal(rows)
        y = np.exp(0.8 * Xt[0] + 0.5 * Xt[1] + 0.3 * Xt[2] + 0.3 * e)
        self.gen_seconds += time.perf_counter() - t
        return Xt, y

    def target(self):
        """The whole y (n,): in a real fit y is already in memory; here one untimed sweep builds it."""
        if self._y is None:
            self._y = np.concatenate([self.chunk(i)[1] for i in range(self.nchunks)])
        return self._y


class SubProvider:
    """The first ``rows`` rows of every chunk of ``base`` (a row subsample that keeps the A / B halves)."""

    def __init__(self, base, rows):
        self.base, self.rows = base, int(rows)
        self.p, self.nchunks = base.p, base.nchunks
        self.n = sum(min(self.rows, base.bounds(i)[1] - base.bounds(i)[0]) for i in range(base.nchunks))
        self.chunk_rows = self.rows

    @property
    def gen_seconds(self):
        return self.base.gen_seconds

    def bounds(self, i):
        a, b = self.base.bounds(i)
        return i * self.rows, i * self.rows + min(self.rows, b - a)

    def chunk(self, i):
        Xt, y = self.base.chunk(i)
        return Xt[:, : self.rows], y[: self.rows]

    def target(self):
        ys = []
        for i in range(self.nchunks):
            a, b = self.base.bounds(i)
            ys.append(self.base.target()[a: a + min(self.rows, b - a)])
        return np.concatenate(ys)


def half_of(prov, i):
    return 0 if i < prov.nchunks // 2 else 1


class _Stage:
    """Context manager: wall seconds of a stage minus the data-generation seconds inside it."""

    def __init__(self, tm, name, prov):
        self.tm, self.name, self.prov = tm, name, prov

    def __enter__(self):
        self.t, self.g = time.perf_counter(), self.prov.gen_seconds
        return self

    def __exit__(self, *a):
        self.tm[self.name] = self.tm.get(self.name, 0.0) + time.perf_counter() - self.t - (self.prov.gen_seconds - self.g)


# ------------------------------------------------------------------ target transforms (numba)
#: (name, code, lambda)
TARGET_CANDS = [
    ("identity", 0, 0.0), ("cbrt", 1, 0.0), ("signed_sqrt", 2, 0.0), ("signed_square", 3, 0.0), ("log", 4, 0.0), ("neg_recip", 5, 0.0), ("asinh", 6, 0.0),
    ("yeo_johnson_0.5", 7, 0.5), ("yeo_johnson_-0.5", 7, -0.5), ("boxcox_0.25", 8, 0.25),
]


@njit(parallel=True, fastmath=True, cache=True)
def target_transforms(y, codes, lams, floor, Z, J):
    """Z[i,k] = f_k(y_i), J[i,k] = log|f_k'(y_i)|; y is clipped at ``floor`` only inside the positive-domain codes."""
    n = y.shape[0]
    K = codes.shape[0]
    for i in prange(n):
        yi = y[i]
        ay = abs(yi)
        lay = math.log(max(ay, 1e-6))
        yp = max(yi, floor)
        ly = math.log(yp)
        for k in range(K):
            c = codes[k]
            lam = lams[k]
            if c == 0:
                z = yi
                j = 0.0
            elif c == 1:
                z = np.cbrt(yi)
                j = -LOG3 - (2.0 / 3.0) * lay
            elif c == 2:
                s = math.sqrt(ay)
                z = s if yi >= 0 else -s
                j = -LOG2 - 0.5 * lay
            elif c == 3:
                z = yi * ay
                j = LOG2 + lay
            elif c == 4:
                z = ly
                j = -ly
            elif c == 5:
                z = -1.0 / yp
                j = -2.0 * ly
            elif c == 6:
                z = math.asinh(yi)
                j = -0.5 * math.log1p(yi * yi)
            elif c == 7:
                if yi >= 0:
                    l1 = math.log1p(yi)
                    z = math.expm1(lam * l1) / lam if lam != 0.0 else l1
                    j = (lam - 1.0) * l1
                else:
                    l1 = math.log1p(-yi)
                    z = -math.expm1((2.0 - lam) * l1) / (2.0 - lam)
                    j = (1.0 - lam) * l1
            else:
                z = math.expm1(lam * ly) / lam
                j = (lam - 1.0) * ly
            Z[i, k] = z
            J[i, k] = j


def cand_arrays(K, positive=True):
    """The first K candidates; the positive-domain ones (log, -1/y, Box-Cox) are dropped when y has non-positive values."""
    cs = [c for c in TARGET_CANDS if positive or c[1] not in (4, 5, 8)][:K]
    return [c[0] for c in cs], np.array([c[1] for c in cs], dtype=np.int64), np.array([c[2] for c in cs], dtype=np.float64)


def chunk_signs(seed, i, rows, B):
    """The +-1 matrix (rows, B) of chunk i; shared with the reference implementation so both see identical draws."""
    rng = np.random.default_rng([seed, 7919, i])
    return (rng.integers(0, 2, size=(rows, B), dtype=np.int8) * 2 - 1).astype(np.float32)


# ------------------------------------------------------------------ target criterion


def _inv_ridge(G, lam):
    m1 = G.shape[0]
    L = np.linalg.cholesky(G + lam * np.trace(G) / m1 * np.eye(m1))
    Li = np.linalg.solve(L, np.eye(m1))
    return Li.T @ Li


def screen_design(prov, K, cap, floor, tm, positive=True):
    """The ``cap`` columns with the largest max_k |corr(x_j, f_k(y))|: one matrix pass (X' Z for all K candidates at once)."""
    names, codes, lams = cand_arrays(K, positive)
    p = prov.p
    sxz = np.zeros((p, K)); sx = np.zeros(p); sxx = np.zeros(p); sz = np.zeros(K); szz = np.zeros(K); nn = 0
    with _Stage(tm, "screen", prov):
        for i in range(prov.nchunks):
            Xt, y = prov.chunk(i)
            Z = np.empty((len(y), K)); J = np.empty((len(y), K))
            target_transforms(y, codes, lams, floor, Z, J)
            sxz += (Xt @ Z.astype(np.float32)).astype(np.float64)
            sx += Xt.sum(1, dtype=np.float64)
            sxx += np.einsum("ij,ij->i", Xt, Xt, dtype=np.float64)
            sz += Z.sum(0); szz += np.einsum("ik,ik->k", Z, Z); nn += len(y)
        corr = (sxz - np.outer(sx, sz) / nn) / np.sqrt(np.outer(np.maximum(sxx - sx * sx / nn, 1e-30), np.maximum(szz - sz * sz / nn, 1e-30)))
        score = np.abs(corr).max(1)
    return np.sort(np.argsort(-score)[:cap])


def sufficient_stats(prov, sel, K, floor, tm, positive=True):
    """One pass: per half D'D, D'z (all K candidates), z'z, n and the held-out Jacobian sum."""
    names, codes, lams = cand_arrays(K, positive)
    m = prov.p if sel is None else len(sel)
    m1 = m + 1
    G = [np.zeros((m1, m1)), np.zeros((m1, m1))]
    C = [np.zeros((m1, K)), np.zeros((m1, K))]
    ZZ = [np.zeros(K), np.zeros(K)]
    nh = [0, 0]
    Jb = np.zeros(K)
    with _Stage(tm, "stats", prov):
        for i in range(prov.nchunks):
            h = half_of(prov, i)
            Xt, y = prov.chunk(i)
            Xs = Xt if sel is None else Xt[sel]
            rows = len(y)
            Z = np.empty((rows, K)); J = np.empty((rows, K))
            target_transforms(y, codes, lams, floor, Z, J)
            sx = Xs.sum(1, dtype=np.float64)
            G[h][0, 0] += rows
            G[h][0, 1:] += sx
            G[h][1:, 0] += sx
            G[h][1:, 1:] += (Xs @ Xs.T).astype(np.float64)
            C[h][0] += Z.sum(0)
            C[h][1:] += (Xs @ Z.astype(np.float32)).astype(np.float64)
            ZZ[h] += np.einsum("ik,ik->k", Z, Z)
            nh[h] += rows
            if h == 1:
                Jb += J.sum(0)
    return G, C, ZZ, nh, Jb


def target_null_pass(prov, sel, K, B, floor, yclip, beta, seed, tm, positive=True):
    """One pass for the sign-flip null: per half  D'u (K), D'(d_k s_b) (K*B), sum u^2 + d^2, sum s u d, held-out Jacobian sums."""
    names, codes, lams = cand_arrays(K, positive)
    m = prov.p if sel is None else len(sel)
    m1 = m + 1
    KB = K * B
    CA = np.zeros((m1, KB)); CB = np.zeros((m1, KB))
    UA = np.zeros((m1, K)); UB = np.zeros((m1, K))
    ZZ = [np.zeros((K, B)), np.zeros((K, B))]
    SJ = np.zeros((K, B))
    ybeta = beta[1:].astype(np.float32)
    with _Stage(tm, "null_pass", prov):
        for i in range(prov.nchunks):
            h = half_of(prov, i)
            Xt, y = prov.chunk(i)
            Xs = Xt if sel is None else Xt[sel]
            rows = len(y)
            yhat = beta[0] + (ybeta @ Xs).astype(np.float64)
            yref = 2.0 * yhat - y
            if yclip is not None:
                yref = np.maximum(yref, yclip)
            Zp = np.empty((rows, K)); Jp = np.empty((rows, K)); Zm = np.empty((rows, K)); Jm = np.empty((rows, K))
            target_transforms(y, codes, lams, floor, Zp, Jp)
            target_transforms(yref, codes, lams, floor, Zm, Jm)
            u = 0.5 * (Zp + Zm); d = 0.5 * (Zp - Zm)
            S = chunk_signs(seed, i, rows, B)
            U = UA if h == 0 else UB
            U[0] += u.sum(0)
            U[1:] += (Xs @ u.astype(np.float32)).astype(np.float64)
            RHS = (d.astype(np.float32)[:, :, None] * S[:, None, :]).reshape(rows, KB)
            T = CA if h == 0 else CB
            T[0] += RHS.sum(0, dtype=np.float64)
            T[1:] += (Xs @ RHS).astype(np.float64)
            ZZ[h] += (np.einsum("ik,ik->k", u, u) + np.einsum("ik,ik->k", d, d))[:, None] + 2.0 * ((u * d).T.astype(np.float32) @ S).astype(np.float64)
            if h == 1:
                SJ += (0.5 * (Jp - Jm)).T.astype(np.float32) @ S + (0.5 * (Jp + Jm)).sum(0)[:, None]
    return CA, CB, UA, UB, ZZ, SJ


def target_criterion(prov, K=9, B=40, cap=None, lam=1e-6, seed=0, alpha=0.05, timing=None, sel=None):
    """Observed LL_k, dLL_k = LL_k - LL_id and, when B > 0, the null threshold tau (the (1-alpha) quantile of max_{k != id} dLL*) and the verdict."""
    tm = {} if timing is None else timing
    y = prov.target()
    ymin = float(y.min())
    positive = ymin > 0
    K = len(cand_arrays(K, positive)[0])                            # the domain filter may drop log / reciprocal / Box-Cox
    floor = ymin if positive else 1e-12
    yclip = ymin if positive else None                              # a positive y keeps the reflected null target inside the domain of log / reciprocal
    if sel is None and cap is not None and cap < prov.p:
        sel = screen_design(prov, K, cap, floor, tm, positive)
    G, C, ZZ, nh, Jb = sufficient_stats(prov, sel, K, floor, tm, positive)
    with _Stage(tm, "solve", prov):
        M = _inv_ridge(G[0], lam)
        W = M @ C[0]
        mse = (ZZ[1] - 2 * np.einsum("mk,mk->k", C[1], W) + np.einsum("mk,mn,nk->k", W, G[1], W)) / nh[1]
        LL = -0.5 * np.log(mse) + Jb / nh[1]
        dLL = LL - LL[0]
        best = int(np.argmax(dLL[1:])) + 1
    names = cand_arrays(K, positive)[0]
    out = dict(names=names, LL=LL, dLL=dLL, best=names[best], best_gain=float(dLL[best]), m_design=len(W) - 1, sel=sel, nB=nh[1])
    if B > 0:
        with _Stage(tm, "null_beta", prov):
            Gall = G[0] + G[1]
            beta = np.linalg.solve(Gall + lam * np.trace(Gall) / Gall.shape[0] * np.eye(Gall.shape[0]), C[0][:, 0] + C[1][:, 0])
        CA, CB, UA, UB, ZZn, SJ = target_null_pass(prov, sel, K, B, floor, yclip, beta, seed, tm, positive)
        with _Stage(tm, "null_finish", prov):
            m1 = len(W)
            CAr = CA.reshape(m1, K, B) + UA[:, :, None]
            CBr = CB.reshape(m1, K, B) + UB[:, :, None]
            Wn = np.einsum("mn,nkb->mkb", M, CAr, optimize=True)
            t1 = np.einsum("mkb,mkb->kb", CBr, Wn)
            t2 = np.einsum("mkb,mkb->kb", Wn, np.einsum("mn,nkb->mkb", G[1], Wn, optimize=True))
            msen = (ZZn[1] - 2.0 * t1 + t2) / nh[1]
            LLn = -0.5 * np.log(np.maximum(msen, 1e-300)) + SJ / nh[1]
            T = (LLn[1:] - LLn[0:1]).max(0)
            tau = float(np.quantile(T, 1 - alpha))
        out.update(tau=tau, null_max=T, accepted=bool(dLL[best] > max(tau, 0.0)))
    out["timing"] = dict(tm)
    return out


def target_criterion_subnull(prov, K=9, B=40, cap=None, null_rows=2500, lam=1e-6, seed=0, alpha=0.05, timing=None, rescale=False):
    """Observed criterion on all rows; the null on the first ``null_rows`` rows of every chunk.  ``rescale=True`` multiplies the threshold by n_B(sub) / n_B(full) (the 1/n law of a
    sampling-noise null); measured in tt_b5_subnull_check.py, the target null is an EFFECT SIZE (the sign-flipped lognormal target is still skewed), so the default is no rescaling."""
    tm = {} if timing is None else timing
    obs = target_criterion(prov, K=K, B=0, cap=cap, lam=lam, seed=seed, alpha=alpha, timing=tm)
    sub = SubProvider(prov, null_rows)
    nul = target_criterion(sub, K=K, B=B, cap=cap, lam=lam, seed=seed, alpha=alpha, timing=tm, sel=obs['sel'])
    scale = nul["nB"] / obs["nB"] if rescale else 1.0
    obs.update(tau=nul["tau"] * scale, tau_sub=nul["tau"], null_max=nul["null_max"] * scale, accepted=bool(obs["best_gain"] > max(nul["tau"] * scale, 0.0)), timing=dict(tm))
    return obs


# ------------------------------------------------------------------ feature criterion (numba kernels)

FEATURE_CANDS = ["identity", "log_abs", "signed_sqrt", "signed_square", "cbrt", "abs_c", "sq_c", "tanh_z", "exp_z", "inv"]


@njit(inline="always", fastmath=True)
def _fcands(x, mu, isd, eps, K, z):
    ax = abs(x)
    d = (x - mu) * isd
    z[0] = x
    z[1] = math.log(ax + eps)
    s = math.sqrt(ax)
    z[2] = s if x >= 0 else -s
    z[3] = x * ax
    z[4] = np.cbrt(x)
    z[5] = abs(x - mu)
    z[6] = d * d
    z[7] = math.tanh(d)
    if K > 8:
        z[8] = math.exp(min(max(d, -8.0), 8.0))
    if K > 9:
        z[9] = (1.0 if x >= 0 else -1.0) / (ax + eps)


@njit(parallel=True, fastmath=True, cache=True)
def feat_obs_chunk(Xt, r, mu, isd, eps, K, acc):
    """acc[j,k] += (sum z, sum z^2, sum z r, sum z x) of the K transforms of column j over the rows of this chunk."""
    p, rows = Xt.shape
    for j in prange(p):
        z = np.empty(10)
        a = np.zeros((K, 4))
        for i in range(rows):
            xv = float(Xt[j, i])
            _fcands(xv, mu[j], isd[j], eps[j], K, z)
            rv = float(r[i])
            for k in range(K):
                zk = z[k]
                a[k, 0] += zk
                a[k, 1] += zk * zk
                a[k, 2] += zk * rv
                a[k, 3] += zk * xv
        for k in range(K):
            for q in range(4):
                acc[j, k, q] += a[k, q]


@njit(parallel=True, fastmath=True, cache=True)
def feat_fill_block(Xt, j0, nb, i0, nrows, mu, isd, eps, K, Zt, mom):
    """Zt[(jj*K + k), i] = f_k(x_{j0+jj, i0+i}) (float32) and mom[j,k] += (sum z, sum z^2, sum z x) for the rows of this block."""
    for jj in prange(nb):
        j = j0 + jj
        z = np.empty(10)
        m0 = np.zeros((K, 3))
        for i in range(nrows):
            xv = float(Xt[j, i0 + i])
            _fcands(xv, mu[j], isd[j], eps[j], K, z)
            for k in range(K):
                zk = z[k]
                Zt[jj * K + k, i] = zk
                m0[k, 0] += zk
                m0[k, 1] += zk * zk
                m0[k, 2] += zk * xv
        for k in range(K):
            for q in range(3):
                mom[j, k, q] += m0[k, q]


def perm_matrix(r, B, seed):
    """R (n, B+1) float32: column 0 is r, column b the residuals permuted by a fresh uniform permutation."""
    rng = np.random.default_rng([seed, 31337])
    R = np.empty((len(r), B + 1), dtype=np.float32)
    R[:, 0] = r
    for b in range(B):
        R[:, b + 1] = r[rng.permutation(len(r))]
    return R


def _rho2(Sz, Szz, Szx, Szr, Sr, Srr, beta, n):
    """rho^2 of f_k(x_j) with the component-plus-residual  r_b + beta_j x_j  (beta = 0 gives the plain residual).  Sz, Szz, Szx: (p, K); Szr: (p, K[, draws]); the
    identity candidate (k = 0) in Szr[:, 0] gives sum x_j r_b."""
    Sx = Sz[:, 0]
    varx = Szz[:, 0] - Sx * Sx / n
    vz = np.maximum(Szz - Sz * Sz / n, 1e-300)
    vr0 = max(Srr - Sr * Sr / n, 1e-300)
    cov_zx = Szx - Sz * Sx[:, None] / n
    if Szr.ndim == Sz.ndim + 1:
        cov = (Szr - (Sz * Sr / n)[..., None]) + (beta[:, None] * cov_zx)[..., None]
        crx = Szr[:, 0, :] - (Sx * Sr / n)[:, None]
        V = vr0 + 2.0 * beta[:, None] * crx + (beta * beta * varx)[:, None]
        return cov * cov / (vz[..., None] * V[:, None, :])
    cov = (Szr - Sz * Sr / n) + beta[:, None] * cov_zx
    crx = Szr[:, 0] - Sx * Sr / n
    V = vr0 + 2.0 * beta * crx + beta * beta * varx
    return cov * cov / (vz * V[:, None])


def feature_criterion(prov, K=9, B=40, null_rows=None, seed=0, alpha=0.05, block_k=512, row_block=2048, timing=None, beta=None):
    """Observed s_jk = rho^2(f_k(x_j), r) with r = y - mean(y), the per-column threshold from the permutation null and the verdict per column."""
    tm = {} if timing is None else timing
    p = prov.p
    beta = np.zeros(p) if beta is None else np.asarray(beta, dtype=np.float64)
    with _Stage(tm, "pilot", prov):
        Xt0, y0 = prov.chunk(0)
        mu = Xt0.mean(1, dtype=np.float64)
        sd = Xt0.std(1, dtype=np.float64) + 1e-12
        isd = 1.0 / sd
        eps = 1e-3 * sd
        yall = prov.target()
        r_all = (yall - yall[: len(y0)].mean()).astype(np.float32)
    n = prov.n
    s_obs = None
    # ---- observed statistic: fused kernel over ALL rows (skipped when the GEMM null pass covers every row anyway)
    if B <= 0 or null_rows is not None:
        with _Stage(tm, "obs_fused", prov):
            acc = np.zeros((p, K, 4))
            for i in range(prov.nchunks):
                Xt, _ = prov.chunk(i)
                a, b = prov.bounds(i)
                feat_obs_chunk(Xt, r_all[a:b], mu, isd, eps, K, acc)
            Sr = float(r_all.sum(dtype=np.float64)); Srr = float((r_all.astype(np.float64) ** 2).sum())
            s_obs = _rho2(acc[:, :, 0], acc[:, :, 1], acc[:, :, 3], acc[:, :, 2], Sr, Srr, beta, n)
    out = dict(K=K, p=p, names=FEATURE_CANDS[:K])
    if B > 0:
        sub = SubProvider(prov, null_rows) if null_rows is not None else prov
        n_u = sub.n
        with _Stage(tm, "null_setup", prov):
            if null_rows is None:
                r_u = r_all
            else:
                r_u = np.concatenate([r_all[prov.bounds(i)[0]: prov.bounds(i)[0] + (sub.bounds(i)[1] - sub.bounds(i)[0])] for i in range(prov.nchunks)])
            R = perm_matrix(r_u, B, seed)
        accn = np.zeros((p * K, B + 1))
        mom = np.zeros((p, K, 3))
        cb = max(1, block_k // K)
        Zt = np.empty((cb * K, row_block), dtype=np.float32)
        with _Stage(tm, "null_gemm", prov):
            off = 0
            for i in range(sub.nchunks):
                Xt, _ = sub.chunk(i)
                rows = Xt.shape[1]
                for i0 in range(0, rows, row_block):
                    rs = min(row_block, rows - i0)
                    Rb = R[off + i0: off + i0 + rs]
                    for j0 in range(0, p, cb):
                        nb = min(cb, p - j0)
                        feat_fill_block(Xt, j0, nb, i0, rs, mu, isd, eps, K, Zt, mom)
                        accn[j0 * K: (j0 + nb) * K] += Zt[: nb * K, :rs] @ Rb
                off += rows
        with _Stage(tm, "null_finish", prov):
            Sr = float(r_u.sum(dtype=np.float64)); Srr = float((r_u.astype(np.float64) ** 2).sum())
            sn = _rho2(mom[:, :, 0], mom[:, :, 1], mom[:, :, 2], accn.reshape(p, K, B + 1), Sr, Srr, beta, n_u)      # (p, K, B+1)
            if s_obs is None:
                s_obs = sn[:, :, 0]
            T = (sn[:, 1:, 1:] - sn[:, 0:1, 1:]).max(1)                                          # (p, B)
            if null_rows is not None:
                T = T * (n_u / n)
            tau = np.quantile(T, 1 - alpha, axis=1)
        out.update(tau=tau, null_max=T)
    gain = s_obs[:, 1:] - s_obs[:, 0:1]
    kb = gain.argmax(1) + 1
    best_gain = gain[np.arange(p), kb - 1]
    out.update(s=s_obs, best_k=kb, best_gain=best_gain)
    if B > 0:
        out["accepted"] = best_gain > np.maximum(out["tau"], 0.0)
        out["credit"] = 0.5 * np.log((1 - s_obs[:, 0]) / (1 - s_obs[np.arange(p), kb]))
    out["timing"] = dict(tm)
    return out
