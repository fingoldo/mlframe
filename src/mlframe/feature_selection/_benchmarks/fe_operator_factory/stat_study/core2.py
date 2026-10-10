"""Two-parameter extension of ``core``: closed-form ``(s, t)`` of ``(u + s)(v + t)`` from one 4x4 OLS, and a G x G grid of quantile zero-crossings."""

import numpy as np
from numba import njit, prange

from .core import bins_of, mi_cnt


@njit(cache=True)
def ols_st(u, v, yr, winsor, huber):
    """y ~ a + b*u + c*v + d*u*v  ==  d*(u + c/d)*(v + b/d) + const  ->  returns (s, t) shifts for u and v (nan if d~0)."""
    n = u.size
    umin = u.min()
    umax = u.max()
    vmin = v.min()
    vmax = v.max()
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
        return np.nan, np.nan
    s_s = beta[2] / beta[3]  # shift on standardised u
    t_s = beta[1] / beta[3]  # shift on standardised v
    s = min(max(s_s * su - mu, -umax - 2 * su), -umin + 2 * su)
    t = min(max(t_s * sv - mv, -vmax - 2 * sv), -vmin + 2 * sv)
    return s, t


# mode 6: closed-form 2-param; mode 7: GxG grid of quantile-zero-crossings (shifts for u and v)
@njit(parallel=True, cache=True)
def family_eval2(UX, UZ, yb, yr, mode, k, ky, mm, winsor, huber, G):
    """MI and (s, t) of every pair under mode 6 (closed-form 2-parameter) or 7 (G x G grid of quantile zero-crossings)."""
    nu = UX.shape[0]
    P = 2 * nu * nu
    mis = np.full(P, -1.0)
    ss = np.zeros(P)
    tt = np.zeros(P)
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
        if mode == 7:
            us = np.sort(u)
            vs = np.sort(v)
            bm = -1.0
            bs = 0.0
            bt = 0.0
            for g in range(G):
                s = -us[int((0.1 + 0.8 * g / (G - 1)) * (n - 1))]
                for h in range(G):
                    t = -vs[int((0.1 + 0.8 * h / (G - 1)) * (n - 1))]
                    for i in range(n):
                        f[i] = (u[i] + s) * (v[i] + t)
                    m_ = mi_cnt(bins_of(f, k), yb, k, ky, mm)
                    if m_ > bm:
                        bm = m_
                        bs = s
                        bt = t
            mis[p] = bm
            ss[p] = bs
            tt[p] = bt
            continue
        s, t = ols_st(u, v, yr, winsor, huber)
        if np.isnan(s):
            s = 0.0
            t = 0.0
        for i in range(n):
            f[i] = (u[i] + s) * (v[i] + t)
        mis[p] = mi_cnt(bins_of(f, k), yb, k, ky, mm)
        ss[p] = s
        tt[p] = t
    return mis, ss, tt
