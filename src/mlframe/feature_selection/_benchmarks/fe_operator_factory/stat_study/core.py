"""Numba kernels and helpers of the offset-product statistics study: rank-bin MI, shift estimators (median, zero-crossing, interaction OLS, quantile grid) evaluated over all unary x unary pairs, the preset candidate scorer and a binned rank-1 ALS."""

import warnings

import numpy as np
from numba import njit, prange

warnings.simplefilter("ignore")


@njit(cache=True)
def bins_of(f, k):
    """Equal-frequency bin codes (int8) from the merge-sort rank of ``f``."""
    n = f.size
    order = np.argsort(f, kind="mergesort")
    b = np.empty(n, np.int8)
    for r in range(n):
        b[order[r]] = (r * k) // n
    return b


@njit(cache=True)
def mi_cnt(b, yb, k, ky, mm):
    """Plug-in MI of bin codes against target codes; optional Miller-Madow bias subtraction."""
    n = b.size
    cnt = np.zeros(k * ky)
    for i in range(n):
        cnt[b[i] * ky + yb[i]] += 1.0
    px = np.zeros(k)
    py = np.zeros(ky)
    for i in range(k):
        for j in range(ky):
            c = cnt[i * ky + j]
            px[i] += c
            py[j] += c
    m = 0.0
    for i in range(k):
        for j in range(ky):
            c = cnt[i * ky + j]
            if c > 0:
                m += c / n * np.log(c * n / (px[i] * py[j]))
    if mm:
        nx = 0
        ny = 0
        for i in range(k):
            if px[i] > 0:
                nx += 1
        for j in range(ky):
            if py[j] > 0:
                ny += 1
        m -= (nx - 1) * (ny - 1) / (2.0 * n)
    return m


@njit(parallel=True, cache=True)
def mi_feats(F, yb, k, ky, mm):
    """MI of every row of ``F`` (rank-binned in parallel) against target codes."""
    m = F.shape[0]
    out = np.empty(m)
    for i in prange(m):
        out[i] = mi_cnt(bins_of(F[i], k), yb, k, ky, mm)
    return out


@njit(parallel=True, cache=True)
def mi_bins(B, yb, k, ky, mm):
    """MI of every pre-binned row of ``B`` against target codes (parallel)."""
    m = B.shape[0]
    out = np.empty(m)
    for i in prange(m):
        out[i] = mi_cnt(B[i], yb, k, ky, mm)
    return out


@njit(parallel=True, cache=True)
def bins_batch(F, k):
    """Rank-bin every row of ``F`` (parallel)."""
    m = F.shape[0]
    B = np.empty(F.shape, np.int8)
    for i in prange(m):
        B[i] = bins_of(F[i], k)
    return B


@njit(cache=True)
def zc_t(u, v, yr, nb, zthr, interp, fallback_median):
    """Zero-crossing shift: per u-bin sign of corr(v, y); the shift sits at the strongest sign flip (optionally interpolated, optionally significance-gated)."""
    n = u.size
    order = np.argsort(u)
    m = np.zeros(nb)
    su = np.zeros(nb)
    sv = np.zeros(nb)
    sy = np.zeros(nb)
    svv = np.zeros(nb)
    syy = np.zeros(nb)
    svy = np.zeros(nb)
    for r in range(n):
        i = order[r]
        b = (r * nb) // n
        m[b] += 1
        su[b] += u[i]
        sv[b] += v[i]
        sy[b] += yr[i]
        svv[b] += v[i] * v[i]
        syy[b] += yr[i] * yr[i]
        svy[b] += v[i] * yr[i]
    c = np.zeros(nb)
    mu = np.zeros(nb)
    for b in range(nb):
        if m[b] > 2:
            cv = svy[b] / m[b] - sv[b] / m[b] * sy[b] / m[b]
            vv = svv[b] / m[b] - (sv[b] / m[b]) ** 2
            vy = syy[b] / m[b] - (sy[b] / m[b]) ** 2
            c[b] = cv / np.sqrt(vv * vy + 1e-18)
            mu[b] = su[b] / m[b]
            if np.abs(c[b]) * np.sqrt(m[b]) < zthr:
                c[b] = 0.0
    best = -1.0
    t = np.nan
    last = -1
    for b in range(nb):
        if c[b] == 0.0:
            continue
        if last >= 0 and c[last] * c[b] < 0:
            s = np.abs(c[last] - c[b])
            if s > best:
                best = s
                if interp:
                    x0 = mu[last] + (mu[b] - mu[last]) * c[last] / (c[last] - c[b])
                else:
                    x0 = 0.5 * (mu[last] + mu[b])
                t = -x0
        last = b
    if np.isnan(t) and fallback_median:
        t = -np.median(u)
    return t


@njit(cache=True)
def ols_t(u, v, yr, winsor, huber):
    """One-parameter shift from the OLS fit of ``y ~ a + b u + c v + d uv``: t = c/d (optionally winsorised and Huber-reweighted)."""
    n = u.size
    umin = u.min()
    umax = u.max()
    if winsor:
        lo = np.percentile(u, 1.0)
        hi = np.percentile(u, 99.0)
        u = np.minimum(np.maximum(u, lo), hi)
        lo = np.percentile(v, 1.0)
        hi = np.percentile(v, 99.0)
        v = np.minimum(np.maximum(v, lo), hi)
    mu = u.mean()
    mv = v.mean()
    su = u.std() + 1e-12
    sv = v.std() + 1e-12
    X = np.empty((n, 4))
    for i in range(n):
        a = (u[i] - mu) / su
        b = (v[i] - mv) / sv
        X[i, 0] = 1.0
        X[i, 1] = a
        X[i, 2] = b
        X[i, 3] = a * b
    w = np.ones(n)
    beta = np.zeros(4)
    nit = 4 if huber else 1
    for _it in range(nit):
        A = np.zeros((4, 4))
        r = np.zeros(4)
        for i in range(n):
            for p in range(4):
                r[p] += w[i] * X[i, p] * yr[i]
                for q in range(4):
                    A[p, q] += w[i] * X[i, p] * X[i, q]
        for p in range(4):
            A[p, p] += 1e-9
        beta = np.linalg.solve(A, r)
        if huber:
            res = np.empty(n)
            for i in range(n):
                res[i] = yr[i] - (X[i, 0] * beta[0] + X[i, 1] * beta[1] + X[i, 2] * beta[2] + X[i, 3] * beta[3])
            s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
            for i in range(n):
                a = np.abs(res[i]) / s
                w[i] = 1.0 if a <= 1.345 else 1.345 / a
    if np.abs(beta[3]) < 1e-12:
        return np.nan
    ts = beta[2] / beta[3]
    t = ts * su - mu
    lo = -umax - 2 * su
    hi = -umin + 2 * su
    return min(max(t, lo), hi)


# modes: 0 plain, 1 median-centre, 2 ZC, 3 OLS, 4 grid(G, q in 0.1..0.9), 5 both centred
@njit(parallel=True, cache=True)
def family_eval(UX, UZ, yb, yr, mode, k, ky, mm, nbz, zthr, interp, fb, winsor, huber, G):
    """MI and shift of every (unary, unary, role) pair under shift mode 0-5 (plain, median, zero-crossing, OLS, quantile grid, both centred)."""
    nu = UX.shape[0]
    P = 2 * nu * nu
    mis = np.full(P, -1.0)
    ts = np.zeros(P)
    for p in prange(P):
        role = p // (nu * nu)
        rem = p % (nu * nu)
        iu = rem // nu
        iv = rem % nu
        if role == 0:
            u = UX[iu]
            v = UZ[iv]
        else:
            u = UZ[iu]
            v = UX[iv]
        n = u.size
        if u.std() < 1e-12 or v.std() < 1e-12:
            continue
        f = np.empty(n)
        if mode == 4:
            us = np.sort(u)
            bm = -1.0
            bt = 0.0
            for g in range(G):
                q = 0.1 + 0.8 * g / (G - 1)
                t = -us[int(q * (n - 1))]
                for i in range(n):
                    f[i] = (u[i] + t) * v[i]
                m_ = mi_cnt(bins_of(f, k), yb, k, ky, mm)
                if m_ > bm:
                    bm = m_
                    bt = t
            mis[p] = bm
            ts[p] = bt
            continue
        t = 0.0
        if mode == 1 or mode == 5:
            t = -np.median(u)
        elif mode == 2:
            t = zc_t(u, v, yr, nbz, zthr, interp, fb)
            if np.isnan(t):
                t = 0.0
        elif mode == 3:
            t = ols_t(u, v, yr, winsor, huber)
            if np.isnan(t):
                t = 0.0
        if mode == 5:
            mv_ = np.median(v)
            for i in range(n):
                f[i] = (u[i] + t) * (v[i] - mv_)
        else:
            for i in range(n):
                f[i] = (u[i] + t) * v[i]
        mis[p] = mi_cnt(bins_of(f, k), yb, k, ky, mm)
        ts[p] = t
    return mis, ts


def clean(a):
    """NaN and inf to 0."""
    return np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)


def unary_mat(UN, x):
    """Stack the cleaned unary transforms of one column into a contiguous matrix."""
    return np.ascontiguousarray(np.stack([clean(f(x)) for f in UN.values()]))


def qbin(x, k=10):
    """Equal-frequency bin codes (re-exported from ``common.binning``)."""
    n = len(x)
    r = np.empty(n, np.int64)
    r[np.argsort(x, kind="stable")] = np.arange(n)
    return np.minimum((r * k) // n, k - 1)


def rank01(y):
    """Rank of ``y`` scaled to (0, 1)."""
    return (np.argsort(np.argsort(y)) + 0.5) / len(y)


def get_presets(name):
    """Unary transforms of the named preset and the minimal binary transforms."""
    from mlframe.feature_selection.filters.feature_engineering import create_binary_transformations, create_unary_transformations

    return create_unary_transformations(name), create_binary_transformations("minimal")


def preset_cands(UX, UZ, BI):
    """list of (ids, feature matrix chunks) generator over all unary x unary x binary combos (u from x, v from z)."""
    nu = UX.shape[0]
    for iu in range(nu):
        rows = []
        ids = []
        for iv in range(nu):
            for bn, fb in BI.items():
                try:
                    val = clean(fb(UX[iu], UZ[iv]))
                except Exception:
                    continue
                if val.std() > 0:
                    rows.append(val)
                    ids.append((iu, iv, bn))
        if rows:
            yield ids, np.ascontiguousarray(np.stack(rows))


def preset_matrix_best(UX, UZ, BI, yb, k=10, mm=False):
    """Best MI over all preset unary x unary x binary combos of one pair; returns (MI, (iu, iv, binary name))."""
    best = -1.0
    bi = None
    ky = int(yb.max()) + 1
    for ids, M in preset_cands(UX, UZ, BI):
        m = mi_feats(M, yb, k, ky, mm)
        j = int(m.argmax())
        if m[j] > best:
            best = float(m[j])
            bi = ids[j]
    return best, bi


def als_rank1(x, z, y, rows_fit, B=16, iters=8):
    """Binned rank-1 fit ``f(x) g(z)`` of ``y`` by alternating least squares on ``B`` quantile bins, fitted on ``rows_fit``."""

    def ed(a):
        """Interior quantile edges of ``a`` on the fit rows."""
        return np.quantile(a[rows_fit], np.linspace(0, 1, B + 1)[1:-1])

    bx = np.searchsorted(ed(x), x)
    bz = np.searchsorted(ed(z), z)
    f = np.ones(B)
    g = np.ones(B)
    yc = y - y[rows_fit].mean()
    r = rows_fit
    for _ in range(iters):
        gz = g[bz]
        f = np.bincount(bx[r], gz[r] * yc[r], B) / (np.bincount(bx[r], gz[r] ** 2, B) + 1e-9)
        fx = f[bx]
        g = np.bincount(bz[r], fx[r] * yc[r], B) / (np.bincount(bz[r], fx[r] ** 2, B) + 1e-9)
        s = np.abs(f).max() + 1e-12
        f /= s
        g *= s
    return f[bx] * g[bz]
