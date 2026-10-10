"""CPU kernels of the offset-product operator ``(u + s) * (v + t)``.

Why the shift matters: a product ``f(c) * g(d)`` whose factor ``f`` changes sign inside the data range (``f = ln 2 + ln c`` crosses zero at ``c = 0.5``) is an interaction no
fixed unary pair can express, because the preset forms have their sign change fixed at 0. ``y ~ a + b*u + c*v + d*u*v`` is exactly ``d * (u + c/d) * (v + b/d)`` plus a constant, so one
4x4 normal-equation solve gives both shifts in closed form, with no grid.

``scan_offset_products`` is the fused scan: per (column pair, unary pair) task it fits the shifts on the even rows, then regenerates ``(u+s)*(v+t)`` on the fly (nothing is stored per
candidate) to bin it and score the plug-in MI on the even rows (the rows the shifts were fitted on) and on the odd rows (held out), next to five baselines that need no shift: ``u*v``,
``u+v``, ``u``, ``v`` and the least-squares weighted sum ``b*u + c*v`` (a product with large shifts degenerates into exactly that sum, which is no interaction). Bin edges are quantiles of a fixed-size row subsample of the even rows, so the held-out half never influences them.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["NACC", "N_EDGE_SUBSAMPLE", "N_BASELINES", "ols2_shift", "scan_offset_products", "weighted_sum_heldout_mi"]

NACC = 14  # upper triangle of the Gram matrix of [1, x, w, x*w] (10 entries) + the 4 right-hand sides
N_EDGE_SUBSAMPLE = 2048  # rows whose candidate value defines the bin edges
N_BASELINES = 5  # u*v, u+v, u, v, the least-squares weighted sum b*u + c*v
_SINGULAR_REL_TOL = 1e-12
_MIN_SPAN = 1e-300


@njit(cache=True, nogil=True)
def _solve4(acc, coef):
    """Solve the 4x4 normal equations stored in ``acc`` (see ``NACC``) by Gaussian elimination with partial pivoting; NaN coefficients when singular."""
    M = np.empty((4, 5))
    t = 0
    for j in range(4):
        for k in range(j, 4):
            M[j, k] = acc[t]
            M[k, j] = acc[t]
            t += 1
    for j in range(4):
        M[j, 4] = acc[10 + j]
    scale = 0.0
    for j in range(4):
        scale = max(scale, abs(M[j, j]))
    for col in range(4):
        piv = col
        best = abs(M[col, col])
        for r in range(col + 1, 4):
            if abs(M[r, col]) > best:
                best = abs(M[r, col])
                piv = r
        if not (best > _SINGULAR_REL_TOL * scale) or not np.isfinite(best):
            for k in range(4):
                coef[k] = np.nan
            return
        if piv != col:
            for k in range(5):
                tmp = M[col, k]
                M[col, k] = M[piv, k]
                M[piv, k] = tmp
        for r in range(col + 1, 4):
            f = M[r, col] / M[col, col]
            for k in range(col, 5):
                M[r, k] -= f * M[col, k]
    for r in range(3, -1, -1):
        s = M[r, 4]
        for k in range(r + 1, 4):
            s -= M[r, k] * coef[k]
        coef[r] = s / M[r, r]


@njit(cache=True, nogil=True)
def ols2_shift(u, v, yr, start, step, clip_u_lo, clip_u_hi, clip_v_lo, clip_v_hi, shifts):
    """Fit ``yr ~ a + b*x + c*w + d*x*w`` (``x``, ``w`` the winsorised, centred ``u``, ``v``) on rows ``start, start+step, ...`` and write the shifts ``(s, t)`` with
    ``(u + s) * (v + t) = (x + c/d) * (w + b/d)`` (up to the constant) to ``shifts[0:2]``; ``shifts[2:4]`` get the main-effect weights ``(b, c)`` of the
    interaction-free fit ``yr ~ a + b*x + c*w``. NaN shifts when the system is singular or ``d`` vanishes.

    The zero of each factor is clipped to one data span beyond the winsorisation range, so a near-zero ``d`` cannot place the crossing far outside the data."""
    n = u.shape[0]
    mu = 0.0
    mv = 0.0
    my = 0.0
    cnt = 0
    for i in range(start, n, step):
        mu += min(max(u[i], clip_u_lo), clip_u_hi)
        mv += min(max(v[i], clip_v_lo), clip_v_hi)
        my += yr[i]
        cnt += 1
    for q in range(4):
        shifts[q] = np.nan
    if cnt < 8:
        return
    mu /= cnt
    mv /= cnt
    my /= cnt
    acc = np.zeros(NACC)
    for i in range(start, n, step):
        x = min(max(u[i], clip_u_lo), clip_u_hi) - mu
        w = min(max(v[i], clip_v_lo), clip_v_hi) - mv
        yy = yr[i] - my
        f3 = x * w
        acc[0] += 1.0
        acc[1] += x
        acc[2] += w
        acc[3] += f3
        acc[4] += x * x
        acc[5] += x * w
        acc[6] += x * f3
        acc[7] += w * w
        acc[8] += w * f3
        acc[9] += f3 * f3
        acc[10] += yy
        acc[11] += x * yy
        acc[12] += w * yy
        acc[13] += f3 * yy
    coef = np.empty(4)
    _solve4(acc, coef)
    det = acc[4] * acc[7] - acc[5] * acc[5]
    if det > _SINGULAR_REL_TOL * acc[4] * acc[7]:
        shifts[2] = (acc[11] * acc[7] - acc[12] * acc[5]) / det
        shifts[3] = (acc[12] * acc[4] - acc[11] * acc[5]) / det
    d = coef[3]
    if not np.isfinite(d) or d == 0.0:
        return
    s = coef[2] / d - mu
    t = coef[1] / d - mv
    span_u = max(clip_u_hi - clip_u_lo, _MIN_SPAN)
    span_v = max(clip_v_hi - clip_v_lo, _MIN_SPAN)
    shifts[0] = min(max(s, -clip_u_hi - span_u), -clip_u_lo + span_u)
    shifts[1] = min(max(t, -clip_v_hi - span_v), -clip_v_lo + span_v)


@njit(cache=True, nogil=True)
def _edges_of(vals, edges):
    """Interior quantile edges (``edges.shape[0]`` of them) of the subsample ``vals``, sorted in place."""
    vals.sort()
    m = vals.shape[0]
    nb = edges.shape[0] + 1
    for k in range(1, nb):
        edges[k - 1] = vals[min(m - 1, (k * m) // nb)]


@njit(cache=True, nogil=True)
def _bin_of(x, edges):
    """Bin index of ``x``: the number of interior edges ``<= x`` (binary search)."""
    lo = 0
    hi = edges.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if edges[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True)
def _mi_from_counts(counts, nb, ky, n_tot):
    """Plug-in MI (nats) of an ``(nb, ky)`` joint count table holding ``n_tot`` rows."""
    if n_tot <= 0:
        return 0.0
    px = np.zeros(nb)
    py = np.zeros(ky)
    for b in range(nb):
        for c in range(ky):
            px[b] += counts[b, c]
            py[c] += counts[b, c]
    mi = 0.0
    for b in range(nb):
        for c in range(ky):
            k = counts[b, c]
            if k > 0:
                mi += (k / n_tot) * np.log(k * n_tot / (px[b] * py[c]))
    return mi


@njit(parallel=True, cache=True)
def scan_offset_products(U, tasks, yr, ycodes, ky, nb, clips, out_shift, out_mi):
    """Score every ``(column pair, unary pair)`` task.

    ``U``: ``(m, nu, n)`` float64 unary outputs of ``m`` columns (non-finite values already replaced). ``tasks``: ``(T, 4)`` int rows ``(col_u, col_v, unary_u, unary_v)``.
    ``yr``: rank-scaled target ``(n,)``; ``ycodes``: int class codes in ``[0, ky)``. ``clips``: ``(m, nu, 2)`` winsorisation bounds of each unary column.
    Writes ``out_shift[T, 2] = (s, t)`` and ``out_mi[T, 2, 1 + N_BASELINES]``: per split (0 = even rows the shifts were fitted on, 1 = odd held-out rows) the MI of the shifted
    product followed by the MI of ``u*v``, ``u+v``, ``u``, ``v``. All NaN for a task whose shifts could not be fitted."""
    T = tasks.shape[0]
    n = U.shape[2]
    nsub = N_EDGE_SUBSAMPLE
    n_even = (n + 1) // 2
    for k in prange(T):
        cu, cv, ui, vi = tasks[k, 0], tasks[k, 1], tasks[k, 2], tasks[k, 3]
        u = U[cu, ui]
        v = U[cv, vi]
        shifts = np.empty(4)
        ols2_shift(u, v, yr, 0, 2, clips[cu, ui, 0], clips[cu, ui, 1], clips[cv, vi, 0], clips[cv, vi, 1], shifts)
        s = shifts[0]
        t = shifts[1]
        wb = shifts[2]
        wc = shifts[3]
        out_shift[k, 0] = s
        out_shift[k, 1] = t
        if not (np.isfinite(s) and np.isfinite(t) and np.isfinite(wb) and np.isfinite(wc)):
            for sp in range(2):
                for q in range(1 + N_BASELINES):
                    out_mi[k, sp, q] = np.nan
            continue
        nf = 1 + N_BASELINES
        edges = np.empty((nf, nb - 1))
        sub = np.empty(nsub)
        m = min(nsub, n_even)
        for q in range(nf):
            for j in range(m):
                i = 2 * ((j * n_even) // m)
                if q == 0:
                    sub[j] = (u[i] + s) * (v[i] + t)
                elif q == 1:
                    sub[j] = u[i] * v[i]
                elif q == 2:
                    sub[j] = u[i] + v[i]
                elif q == 3:
                    sub[j] = u[i]
                elif q == 4:
                    sub[j] = v[i]
                else:
                    sub[j] = wb * u[i] + wc * v[i]
            _edges_of(sub[:m], edges[q])
        counts = np.zeros((nf, 2, nb, ky), dtype=np.int64)
        for i in range(n):
            sp = i & 1
            c = ycodes[i]
            ui_ = u[i]
            vi_ = v[i]
            counts[0, sp, _bin_of((ui_ + s) * (vi_ + t), edges[0]), c] += 1
            counts[1, sp, _bin_of(ui_ * vi_, edges[1]), c] += 1
            counts[2, sp, _bin_of(ui_ + vi_, edges[2]), c] += 1
            counts[3, sp, _bin_of(ui_, edges[3]), c] += 1
            counts[4, sp, _bin_of(vi_, edges[4]), c] += 1
            counts[5, sp, _bin_of(wb * ui_ + wc * vi_, edges[5]), c] += 1
        for sp in range(2):
            n_sp = n_even if sp == 0 else n - n_even
            for q in range(nf):
                out_mi[k, sp, q] = _mi_from_counts(counts[q, sp], nb, ky, n_sp)


@njit(parallel=True, cache=True)
def weighted_sum_heldout_mi(u, v, codes, ky, nb, n_dirs):
    """Held-out MI of the best weighted sum ``cos(t) * u / sd(u) + sin(t) * v / sd(v)`` of the two factors, the direction ``t`` chosen over ``n_dirs`` directions of the full circle by the MI on
    the even rows and scored on the odd rows (edges from a ``N_EDGE_SUBSAMPLE``-row sample of the even rows, as in the scan).

    A shifted product with large shifts degenerates into a weighted sum of its factors, and a weighted sum with better weights than the least-squares baseline wins without any interaction;
    only a gain over THIS baseline, the best additive mix of the candidate's own factors, is evidence of an interaction."""
    n = u.shape[0]
    n_even = (n + 1) // 2
    m = min(N_EDGE_SUBSAMPLE, n_even)
    su = 0.0
    sv = 0.0
    mu = 0.0
    mv = 0.0
    for i in range(0, n, 2):
        mu += u[i]
        mv += v[i]
    mu /= n_even
    mv /= n_even
    for i in range(0, n, 2):
        su += (u[i] - mu) ** 2
        sv += (v[i] - mv) ** 2
    su = np.sqrt(su / n_even) if su > 0 else 1.0
    sv = np.sqrt(sv / n_even) if sv > 0 else 1.0
    mi_even = np.empty(n_dirs)
    mi_odd = np.empty(n_dirs)
    for k in prange(n_dirs):
        th = 2.0 * np.pi * k / n_dirs
        cu = np.cos(th) / su
        cv = np.sin(th) / sv
        sub = np.empty(m)
        for j in range(m):
            i = 2 * ((j * n_even) // m)
            sub[j] = cu * u[i] + cv * v[i]
        edges = np.empty(nb - 1)
        _edges_of(sub, edges)
        counts = np.zeros((2, nb, ky), dtype=np.int64)
        for i in range(n):
            counts[i & 1, _bin_of(cu * u[i] + cv * v[i], edges), codes[i]] += 1
        mi_even[k] = _mi_from_counts(counts[0], nb, ky, n_even)
        mi_odd[k] = _mi_from_counts(counts[1], nb, ky, n - n_even)
    best = 0
    for k in range(1, n_dirs):
        if mi_even[k] > mi_even[best]:
            best = k
    return mi_odd[best]
