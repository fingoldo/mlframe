"""Fused offset-grid scorer: MI of (u_k + t_kg) * v_k for all (k, g) WITHOUT materialising the (n, K*G) matrix.

Binning semantics = repo CPU reference (_fe_edge_mi._edge_bin_codes): nbins-1 interior edges at np.quantile-style
virtual index q*(n-1) with linear interpolation of the two neighbouring order statistics, NO dedup,
searchsorted(side='right'); plug-in MI (nats), y dense int codes.  Candidate values are scrubbed (non-finite -> 0.0)
like the repo's materialise step.

Variants (all kept):
  score_offset_grid_a : per-thread n-buffer + introselect (np.partition)      -- exact by construction
  score_offset_grid_b : fine-grid histogram + exact refinement in the boundary bins, NO n-sized buffer -- exact edges
Both are independent of the thread count (candidate c is computed by exactly one thread, same arithmetic).
"""

import numpy as np
from numba import get_num_threads, njit, prange

DEFAULT_FINE_BINS = 4096  # fine histogram resolution of variant b
GATHER_CAP_FRAC = 8  # variant b: gather buffer = n // GATHER_CAP_FRAC, else fall back to variant a for that candidate
MIN_GATHER_CAP = 4096


@njit(cache=True, nogil=True, inline="always")
def _cand(u, v, t):
    """One scrubbed candidate value ``(u + t) * v``; non-finite becomes 0."""
    c = (u + t) * v
    if not np.isfinite(c):
        return 0.0
    return c


@njit(cache=True, nogil=True)
def _rank_positions(n, nbins, los, his, fracs):
    """Interior-edge anchors, same arithmetic as _fe_edge_mi._edge_bin_codes (edge k = 1..nbins-1)."""
    nq = nbins + 1
    for k in range(nq):
        pos = (k / (nq - 1)) * (n - 1)
        lo = int(np.floor(pos))
        los[k] = lo
        his[k] = lo + 1 if lo < n - 1 else lo
        fracs[k] = pos - lo


@njit(cache=True, nogil=True)
def _mi_from_codes(codes, y, ncls, nbins, hxy, hx, hy):
    """Fill the joint histograms from bin codes and return the plug-in MI."""
    n = codes.shape[0]
    hxy[:] = 0
    hx[:] = 0
    hy[:] = 0
    for i in range(n):
        b = codes[i]
        c = y[i]
        hxy[b * ncls + c] += 1
        hx[b] += 1
        hy[c] += 1
    return _mi_from_hist(hxy, hx, hy, ncls, nbins, n)


@njit(cache=True, nogil=True, fastmath=True)
def _mi_from_hist(hxy, hx, hy, ncls, nbins, n):
    """Plug-in MI (nats) from the joint and marginal histograms."""
    log_n = np.log(n)
    mi = 0.0
    for b in range(nbins):
        if hx[b] == 0:
            continue
        log_hx = np.log(hx[b])
        for c in range(ncls):
            nxy = hxy[b * ncls + c]
            if nxy == 0 or hy[c] == 0:
                continue
            mi += (nxy / n) * (np.log(nxy) + log_n - log_hx - np.log(hy[c]))
    return mi if mi > 0.0 else 0.0


@njit(cache=True, nogil=True)
def _bin_pass(u, v, t, edges, nbins, y, ncls, hxy, hx, hy, codes_out, want_codes):
    """Second pass: bin each regenerated candidate by binary search over the interior edges, histogram it with y and return the MI."""
    n = u.shape[0]
    hxy[:] = 0
    hx[:] = 0
    hy[:] = 0
    ni = nbins - 1
    for i in range(n):
        c = _cand(u[i], v[i], t)
        lo = 0
        hi = ni
        while lo < hi:
            mid = (lo + hi) // 2
            if c < edges[1 + mid]:
                hi = mid
            else:
                lo = mid + 1
        if want_codes:
            codes_out[i] = lo
        yy = y[i]
        hxy[lo * ncls + yy] += 1
        hx[lo] += 1
        hy[yy] += 1
    return _mi_from_hist(hxy, hx, hy, ncls, nbins, n)


@njit(cache=True, nogil=True)
def _edges_partition(u, v, t, buf, nbins, edges, los, his, fracs):
    """Variant a: materialise into buf, np.partition at the 2*(nbins-1) anchors (identical to the repo reference)."""
    n = u.shape[0]
    for i in range(n):
        buf[i] = _cand(u[i], v[i], t)
    nbins + 1
    kths = np.empty(2 * (nbins - 1), dtype=np.int64)
    m = 0
    for k in range(1, nbins):
        kths[m] = los[k]
        kths[m + 1] = his[k]
        m += 2
    part = np.partition(buf, kths)
    for k in range(1, nbins):
        a = part[los[k]]
        b = part[his[k]]
        edges[k] = a + (b - a) * fracs[k]


@njit(cache=True, nogil=True)
def _edges_hist_refine(u, v, t, nbins, NB, hist, mark, gbuf, edges, los, his, fracs):
    """Variant b: exact order statistics from a fine histogram + a sorted gather of only the boundary bins.
    Returns False if the gather would overflow gbuf (heavy ties / outlier-dominated range) -> caller falls back to a."""
    n = u.shape[0]
    cmin = np.inf
    cmax = -np.inf
    for i in range(n):
        c = _cand(u[i], v[i], t)
        if c < cmin:
            cmin = c
        if c > cmax:
            cmax = c
    if not (cmax > cmin):
        for k in range(1, nbins):
            edges[k] = cmin
        return True
    scale = NB / (cmax - cmin)
    hist[:] = 0
    for i in range(n):
        c = _cand(u[i], v[i], t)
        b = int((c - cmin) * scale)
        if b >= NB:
            b = NB - 1
        hist[b] += 1
    # cumulative counts before each bin, marked bins = those holding a needed rank
    mark[:] = False
    # need rank bins for los[k], his[k], k=1..nbins-1 ; ranks are ascending
    b = 0
    r_list = np.empty(2 * (nbins - 1), dtype=np.int64)
    r_bin = np.empty(2 * (nbins - 1), dtype=np.int64)
    r_cum = np.empty(2 * (nbins - 1), dtype=np.int64)
    m = 0
    for k in range(1, nbins):
        r_list[m] = los[k]
        r_list[m + 1] = his[k]
        m += 2
    cum_before = np.zeros(NB, dtype=np.int64)
    run = 0
    for b in range(NB):
        cum_before[b] = run
        run += hist[b]
    j = 0
    for b in range(NB):
        top = cum_before[b] + hist[b]
        while j < m and r_list[j] < top:
            r_bin[j] = b
            r_cum[j] = cum_before[b]
            mark[b] = True
            j += 1
        if j >= m:
            break
    tot = 0
    for b in range(NB):
        if mark[b]:
            tot += hist[b]
    if tot > gbuf.shape[0]:
        return False
    # gather
    cnt = 0
    for i in range(n):
        c = _cand(u[i], v[i], t)
        b = int((c - cmin) * scale)
        if b >= NB:
            b = NB - 1
        if mark[b]:
            gbuf[cnt] = c
            cnt += 1
    g = gbuf[:cnt]
    g.sort()
    # offset of each marked bin inside the sorted gather
    off = np.zeros(NB, dtype=np.int64)
    o = 0
    for b in range(NB):
        if mark[b]:
            off[b] = o
            o += hist[b]
    m = 0
    for k in range(1, nbins):
        a = g[off[r_bin[m]] + (r_list[m] - r_cum[m])]
        bb = g[off[r_bin[m + 1]] + (r_list[m + 1] - r_cum[m + 1])]
        edges[k] = a + (bb - a) * fracs[k]
        m += 2
    return True


@njit(parallel=True, nogil=True, cache=True)
def score_offset_grid_a(U, V, T, y, ncls, nbins):
    """U, V: (K, n) float64 C-order (column-major candidate layout); T: (K, G); y: (n,) int64 in [0, ncls). Returns (K, G) MI."""
    K, n = U.shape
    G = T.shape[1]
    C = K * G
    out = np.empty((K, G), dtype=np.float64)
    nch = min(get_num_threads(), C)
    for ch in prange(nch):
        buf = np.empty(n, dtype=np.float64)
        edges = np.empty(nbins + 1, dtype=np.float64)
        los = np.empty(nbins + 1, dtype=np.int64)
        his = np.empty(nbins + 1, dtype=np.int64)
        fracs = np.empty(nbins + 1, dtype=np.float64)
        _rank_positions(n, nbins, los, his, fracs)
        hxy = np.empty(nbins * ncls, dtype=np.int64)
        hx = np.empty(nbins, dtype=np.int64)
        hy = np.empty(ncls, dtype=np.int64)
        dummy = np.empty(1, dtype=np.int8)
        for c in range(ch, C, nch):
            k = c // G
            g = c % G
            _edges_partition(U[k], V[k], T[k, g], buf, nbins, edges, los, his, fracs)
            out[k, g] = _bin_pass(U[k], V[k], T[k, g], edges, nbins, y, ncls, hxy, hx, hy, dummy, False)
    return out


@njit(parallel=True, nogil=True, cache=True)
def score_offset_grid_b(U, V, T, y, ncls, nbins, NB):
    """Variant b: fine-histogram edges with an exact sorted gather of the boundary bins; returns (MI (K, G), per-candidate fallback flags). U, V (K, n), T (K, G), y codes (n,)."""
    K, n = U.shape
    G = T.shape[1]
    C = K * G
    out = np.empty((K, G), dtype=np.float64)
    n_fallback = np.zeros(C, dtype=np.int8)
    cap = max(MIN_GATHER_CAP, n // GATHER_CAP_FRAC)
    nch = min(get_num_threads(), C)
    for ch in prange(nch):
        hist = np.empty(NB, dtype=np.int64)
        mark = np.zeros(NB, dtype=np.bool_)
        gbuf = np.empty(cap, dtype=np.float64)
        edges = np.empty(nbins + 1, dtype=np.float64)
        los = np.empty(nbins + 1, dtype=np.int64)
        his = np.empty(nbins + 1, dtype=np.int64)
        fracs = np.empty(nbins + 1, dtype=np.float64)
        _rank_positions(n, nbins, los, his, fracs)
        hxy = np.empty(nbins * ncls, dtype=np.int64)
        hx = np.empty(nbins, dtype=np.int64)
        hy = np.empty(ncls, dtype=np.int64)
        dummy = np.empty(1, dtype=np.int8)
        buf = np.empty(1, dtype=np.float64)
        for c in range(ch, C, nch):
            k = c // G
            g = c % G
            ok = _edges_hist_refine(U[k], V[k], T[k, g], nbins, NB, hist, mark, gbuf, edges, los, his, fracs)
            if not ok:
                n_fallback[c] = 1
                if buf.shape[0] != n:
                    buf = np.empty(n, dtype=np.float64)
                _edges_partition(U[k], V[k], T[k, g], buf, nbins, edges, los, his, fracs)
            out[k, g] = _bin_pass(U[k], V[k], T[k, g], edges, nbins, y, ncls, hxy, hx, hy, dummy, False)
    return out, n_fallback


@njit(cache=True, nogil=True)
def candidate_codes_b(u, v, t, nbins, NB, codes_out):
    """Single-candidate exact bin CODES via variant b (used by the null stage, and for code-parity tests)."""
    n = u.shape[0]
    cap = max(MIN_GATHER_CAP, n // GATHER_CAP_FRAC)
    hist = np.empty(NB, dtype=np.int64)
    mark = np.zeros(NB, dtype=np.bool_)
    gbuf = np.empty(cap, dtype=np.float64)
    edges = np.empty(nbins + 1, dtype=np.float64)
    los = np.empty(nbins + 1, dtype=np.int64)
    his = np.empty(nbins + 1, dtype=np.int64)
    fracs = np.empty(nbins + 1, dtype=np.float64)
    _rank_positions(n, nbins, los, his, fracs)
    ok = _edges_hist_refine(u, v, t, nbins, NB, hist, mark, gbuf, edges, los, his, fracs)
    if not ok:
        buf = np.empty(n, dtype=np.float64)
        _edges_partition(u, v, t, buf, nbins, edges, los, his, fracs)
    ni = nbins - 1
    for i in range(n):
        c = _cand(u[i], v[i], t)
        lo = 0
        hi = ni
        while lo < hi:
            mid = (lo + hi) // 2
            if c < edges[1 + mid]:
                hi = mid
            else:
                lo = mid + 1
        codes_out[i] = lo
    return edges


@njit(parallel=True, nogil=True, cache=True)
def apply_offset_product(U, V, T, out):
    """Recipe replay / materialise: out[k, i] = scrub((U[k, i] + T[k]) * V[k, i]); U, V, out (K, n); T (K,)."""
    K, n = U.shape
    for k in prange(K):
        t = T[k]
        for i in range(n):
            out[k, i] = _cand(U[k, i], V[k, i], t)
    return out


APPLY_ROW_BLOCK = 65536


@njit(parallel=True, nogil=True, cache=True)
def apply_offset_product_1d(u, v, t, out):
    """Single-recipe replay for transform(): prange over fixed row blocks (result independent of thread count)."""
    n = u.shape[0]
    nblk = (n + APPLY_ROW_BLOCK - 1) // APPLY_ROW_BLOCK
    for b in prange(nblk):
        for i in range(b * APPLY_ROW_BLOCK, min(n, (b + 1) * APPLY_ROW_BLOCK)):
            out[i] = _cand(u[i], v[i], t)
    return out
