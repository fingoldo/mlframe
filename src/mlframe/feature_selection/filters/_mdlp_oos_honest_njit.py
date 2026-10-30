"""Numba honest supervised null for the held-out-validated MDLP strategy (``mdlp_validated``, ``fayyad_irani_validated``).

Mirrors :func:`_mdlp_validated_split.mdlp_bin_edges_oos_validated` on every permuted target: the (x, y) pairs are de-duplicated first
(the production ``_dedupe_xy`` rule, which depends on y and therefore on the permutation), a random train/holdout split is drawn, the cut
points are searched on the train part and accepted only when they hold on the holdout part, and the codes the cuts induce are scored
against the permuted target with the Miller-Madow correction. The holdout split is drawn with numba's own generator, seeded per
(permutation, column). It is a uniformly random split whose law depends only on the row count, so the null distribution is the same
procedure as production; it is NOT bit-identical to numpy's ``default_rng`` split, and is validated statistically.
"""

from __future__ import annotations


import numpy as np
from numba import njit, prange

from ._analytic_mi_null import analytic_null_min_n, _min_expected_cell  # noqa: F401  (kept for signature symmetry with the mdlp kernel)
from ._mdlp_honest_njit import _compact_labels, _codes_from_cuts, _mi_codes, _n_unique_codes, _supervised_edges
from ._numba_utils import _entropy_from_counts_njit
from .supervised_binning import _mdlp_best_split_inc_njit as _best_split


@njit(cache=True)
def oos_dedupe_njit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``_mdlp_validated_split._dedupe_xy``: returns ``(x_out, y_out)``. With no duplicate ``(x, y)`` pair the original arrays come
    back in their original order; otherwise the first representative of each pair, in ``lexsort((y, x))`` order."""
    n = x.shape[0]
    if n <= 1:
        return x, y
    o1 = np.argsort(y, kind="mergesort")
    o2 = o1[np.argsort(x[o1], kind="mergesort")]
    xs = x[o2]
    ys = y[o2]
    keep = np.empty(n, dtype=np.bool_)
    keep[0] = True
    n_keep = 1
    for i in range(1, n):
        keep[i] = (xs[i] != xs[i - 1]) or (ys[i] != ys[i - 1])
        if keep[i]:
            n_keep += 1
    if n_keep == n:
        return x, y
    xo = np.empty(n_keep, dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    yo = np.empty(n_keep, dtype=y.dtype)
    m = 0
    for i in range(n):
        if keep[i]:
            xo[m] = xs[i]
            yo[m] = ys[i]
            m += 1
    return xo, yo


@njit(cache=True)
def _oos_holdout_gain(xh: np.ndarray, yh: np.ndarray, best_split: float, nc: int, val_min: int):
    """Mirror of ``_holdout_gain``: the train-node cut ``best_split`` re-scored on the holdout rows whose label the node saw (``yh >= 0``).
    Returns ``(gain, n_l, n_r)``; ``gain`` is -1.0 when either side has fewer than ``val_min`` holdout rows."""
    cnt_l = np.zeros(nc, dtype=np.int64)
    cnt_r = np.zeros(nc, dtype=np.int64)
    n_l = 0
    n_r = 0
    for i in range(xh.shape[0]):
        c = yh[i]
        if c < 0:
            continue
        if xh[i] <= best_split:
            cnt_l[c] += 1
            n_l += 1
        else:
            cnt_r[c] += 1
            n_r += 1
    n = n_l + n_r
    if n == 0:
        return -1.0, 0, 0
    if n_l < val_min or n_r < val_min:
        return -1.0, n_l, n_r
    cnt_t = cnt_l + cnt_r
    h_full = _entropy_from_counts_njit(cnt_t, n)
    h_left = _entropy_from_counts_njit(cnt_l, n_l)
    h_right = _entropy_from_counts_njit(cnt_r, n_r)
    h_split = (n_l * h_left + n_r * h_right) / n
    return h_full - h_split, n_l, n_r


@njit(cache=True)
def oos_mdlp_cuts_njit(
    x: np.ndarray, y: np.ndarray, n_classes_total: int, min_split: int, max_depth: int, tol: float, val_min: int, holdout_frac: float
) -> np.ndarray:
    """Inner cut points of the held-out-validated MDLP on ``(x, y)`` (``x`` already de-duplicated, ``y`` raw labels 0..C-1).

    The holdout split is a random permutation drawn with numba's generator (the caller seeds it). The recursion is iterative, the same
    depth-first decisions as the production recursion: a node searches its best cut on its train rows and keeps it only if the cut holds on
    its holdout rows (gain > 0, > the Miller-Madow null mean, and >= ``tol`` times the train gain)."""
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)  # Intentional float64: bin-edge / cut vector, compared against values at full precision
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    n_hold = int(np.rint(holdout_frac * n))
    if n_hold < 1:
        n_hold = 1
    if n_hold > n:
        n_hold = n
    hold_idx = perm[:n_hold]
    train_idx = perm[n_hold:]
    xt_raw = x[train_idx]
    yt_raw = y[train_idx]
    order = np.argsort(xt_raw)
    cuts: np.ndarray = oos_cuts_from_split_njit(xt_raw[order], yt_raw[order], x[hold_idx], y[hold_idx], n_classes_total, min_split, max_depth, tol, val_min)
    return cuts


@njit(cache=True)
def oos_cuts_from_split_njit(
    xt_in: np.ndarray, yt_in: np.ndarray, xh_in: np.ndarray, yh_in: np.ndarray, n_classes_total: int, min_split: int, max_depth: int, tol: float, val_min: int
) -> np.ndarray:
    """The held-out recursion on an explicit split: ``xt_in`` sorted ascending with aligned ``yt_in``, holdout ``xh_in``/``yh_in``."""
    xt = np.ascontiguousarray(xt_in)
    yt = np.ascontiguousarray(yt_in)
    xh = np.ascontiguousarray(xh_in)
    yh = np.ascontiguousarray(yh_in)
    n = xt.shape[0] + xh.shape[0]
    nh = xh.shape[0]
    max_nodes = 2 * n + 4
    st_ts = np.empty(max_nodes, dtype=np.int64)
    st_te = np.empty(max_nodes, dtype=np.int64)
    st_hs = np.empty(max_nodes, dtype=np.int64)
    st_he = np.empty(max_nodes, dtype=np.int64)
    st_d = np.empty(max_nodes, dtype=np.int64)
    out = np.empty(n, dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    n_out = 0
    rank = np.full(n_classes_total, -1, dtype=np.int64)
    yh_c = np.empty(nh, dtype=np.int64)
    st_ts[0] = 0
    st_te[0] = xt.shape[0]
    st_hs[0] = 0
    st_he[0] = nh
    st_d[0] = 0
    top = 1
    while top > 0:
        top -= 1
        ts = st_ts[top]
        te = st_te[top]
        hs = st_hs[top]
        he = st_he[top]
        depth = st_d[top]
        n_node = te - ts
        if n_node < 2 * min_split or depth >= max_depth:
            continue
        yc, nc = _compact_labels(yt[ts:te], n_classes_total)
        if nc <= 1:
            continue
        bi, bg, _h, _nl, _nr = _best_split(xt[ts:te], yc, nc, min_split)
        if bi < 0 or bg <= 0.0:
            continue
        bs = 0.5 * (xt[ts + bi] + xt[ts + bi + 1])
        # Holdout labels in the node's class space; labels the node never saw become -1 and are dropped by the gain.
        for c in range(n_classes_total):
            rank[c] = -1
        k = 0
        for i in range(ts, te):
            c = yt[i]
            if rank[c] < 0:
                rank[c] = 0
        for c in range(n_classes_total):
            if rank[c] >= 0:
                rank[c] = k
                k += 1
        for h in range(hs, he):
            c = yh[h]
            yh_c[h] = rank[c] if (c >= 0 and c < n_classes_total) else -1
        oos_gain, oos_nl, oos_nr = _oos_holdout_gain(xh[hs:he], yh_c[hs:he], bs, nc, val_min)
        if oos_gain <= 0.0:
            continue
        n_oos = oos_nl + oos_nr
        df = nc - 1
        null_mean = df / (2.0 * n_oos) if (df > 0 and n_oos > 0) else 0.0
        if oos_gain <= null_mean or oos_gain < tol * bg:
            continue
        out[n_out] = bs
        n_out += 1
        left = bi + 1
        # Holdout rows go to the child by the same cut (x <= bs): partition the holdout segment in place (its statistics are counts only).
        a = hs
        b = he - 1
        while a <= b:
            if xh[a] <= bs:
                a += 1
            else:
                tx = xh[a]
                xh[a] = xh[b]
                xh[b] = tx
                ty = yh[a]
                yh[a] = yh[b]
                yh[b] = ty
                ty2 = yh_c[a]
                yh_c[a] = yh_c[b]
                yh_c[b] = ty2
                b -= 1
        hl = a
        st_ts[top] = ts
        st_te[top] = ts + left
        st_hs[top] = hs
        st_he[top] = hl
        st_d[top] = depth + 1
        top += 1
        st_ts[top] = ts + left
        st_te[top] = te
        st_hs[top] = hl
        st_he[top] = he
        st_d[top] = depth + 1
        top += 1
    res: np.ndarray = out[:n_out].copy()
    res.sort()
    return res


@njit(cache=True)
def _oos_column_codes(xj, yr_p, nan_rows, has_nan_col, n_classes_total, min_split, max_depth, tol, val_min, holdout_frac, q_fb, q_rf, eval_rows, xs):
    """Fill ``xs`` with the codes, at the evaluation rows, of the continuous column ``xj`` binned on the permuted raw labels ``yr_p``; returns the number of codes.

    ``xj`` holds the column with its missing values filled by the median (the cut search reads the filled column); the rows of ``nan_rows`` take one more code after the last bin."""
    xd, yd = oos_dedupe_njit(xj, yr_p)
    cuts = oos_mdlp_cuts_njit(xd, yd, n_classes_total, min_split, max_depth, tol, val_min, holdout_frac)
    final = _supervised_edges(xj, cuts, q_fb, q_rf)
    xc = _codes_from_cuts(xj, final)
    nb_reg = final.shape[0] + 1
    for t in range(eval_rows.shape[0]):
        r_t = eval_rows[t]
        xs[t] = nb_reg if (has_nan_col and nan_rows[r_t]) else xc[r_t]
    return nb_reg + 1 if has_nan_col else nb_reg


@njit(parallel=True, cache=True)
def honest_validated_null_columns(
    raw_cont: np.ndarray,
    is_cont: np.ndarray,
    fixed_codes: np.ndarray,
    fixed_nbins: np.ndarray,
    y_raw: np.ndarray,
    y_codes: np.ndarray,
    n_classes_total: int,
    perms: np.ndarray,
    min_split: int,
    max_depth: int,
    tol: float,
    val_min: int,
    holdout_frac: float,
    seed: int,
    q_fallback: np.ndarray,
    q_fallback_len: np.ndarray,
    q_refine: np.ndarray,
    q_refine_len: np.ndarray,
    eval_rows: np.ndarray,
    col_slot: np.ndarray,
    nan_cont: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-permutation, per-column null MI for the held-out-validated strategy (prange over permutations). Returns ``(plain, corrected)``
    of shape ``(perms.shape[0], k)``, the same contract as :func:`_mdlp_honest_njit.honest_null_columns`.

    ``raw_cont`` holds each continuous column with its missing values already filled by the column median (the cut search reads the filled column, as the production
    categorisation does); ``nan_cont[r, slot]`` marks the rows that were missing, which take one more code after the last bin instead of their filled value's bin."""
    n_perm_rows = perms.shape[0]
    n = perms.shape[1]
    k = is_cont.shape[0]
    slot_has_nan = np.zeros(nan_cont.shape[1], dtype=np.bool_)
    for s_ in range(nan_cont.shape[1]):
        for r_ in range(nan_cont.shape[0]):
            if nan_cont[r_, s_]:
                slot_has_nan[s_] = True
                break
    plain = np.zeros((n_perm_rows, k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    corrected = np.zeros((n_perm_rows, k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for b in prange(n_perm_rows):
        rp = perms[b]
        yc_p = np.empty(n, dtype=np.int64)
        yr_p = np.empty(n, dtype=np.int64)
        for i in range(n):
            yc_p[i] = y_codes[rp[i]]
            yr_p[i] = y_raw[rp[i]]
        # The MI is evaluated on ``eval_rows`` (the screen's row subsample, or every row): the cut search above still reads the whole
        # permuted target, exactly as the observed codes were fitted on the whole target and then scored on the subsample.
        m = eval_rows.shape[0]
        ys = np.empty(m, dtype=np.int64)
        for t in range(m):
            ys[t] = yc_p[eval_rows[t]]
        ny = 0
        for t in range(m):
            if ys[t] + 1 > ny:
                ny = ys[t] + 1
        ky_eff = 0.0
        seen_y = np.zeros(ny, dtype=np.bool_)
        for t in range(m):
            seen_y[ys[t]] = True
        for c in range(ny):
            if seen_y[c]:
                ky_eff += 1.0
        for j in range(k):
            if is_cont[j]:
                # Per-(permutation, column) seed: the holdout split is drawn by numba's generator inside the cut search.
                np.random.seed(seed + b * 1000003 + j)
                slot = col_slot[j]
                xs = np.empty(m, dtype=np.int64)
                n_x = _oos_column_codes(
                    raw_cont[:, slot], yr_p, nan_cont[:, slot], slot_has_nan[slot], n_classes_total, min_split, max_depth, tol, val_min, holdout_frac,
                    q_fallback[j, : q_fallback_len[j]], q_refine[j, : q_refine_len[j]], eval_rows, xs,
                )
            else:
                fs = col_slot[j]
                n_x = fixed_nbins[j]
                xs = np.empty(m, dtype=np.int64)
                for t in range(m):
                    xs[t] = fixed_codes[eval_rows[t], fs]
            mi = _mi_codes(xs, n_x, ys, ny)
            kx_eff = float(_n_unique_codes(xs))
            plain[b, j] = mi
            corrected[b, j] = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * m)
    return plain, corrected
