"""Table form of the TreeSHAP interaction tensor: the conditioned passes of the reference kernel (two per tree split feature, each a full path scan) replaced by per-leaf tables.

For a leaf with unique path features ``e_0..e_{ud-1}`` (zero fractions ``z_k``) and a sample whose bit mask is ``m`` (bit ``k`` = the sample satisfies every split of ``e_k``), the conditioned passes
of feature ``j = e_k`` give:

  * forced ON : the leaf survives only when bit ``k`` is set (the cold child of a split on ``j`` gets fraction 0), with fraction 1;
  * forced OFF: the leaf survives always, with the product of the cover ratios of the edges on ``j``, which is exactly ``z_k``;
  * ``j`` is not extended into the path, so the contribution to every other element ``q`` is the main-effect contribution of the path WITHOUT element ``k``: ``C[k][m without bit k][q]``.

``0.5 * (on - off)`` is therefore ``0.5 * (bit_k - z_k) * C[k][m'][q]`` added to ``Phi[j, feat(q)]`` (the reference writes ``Phi[feat(q), j]``; the symmetrisation makes the two the same). A leaf whose
path does not contain ``j`` gives equal on and off scans, so it cancels and costs nothing here. The table of a leaf holds ``ud * 2^(ud-1) * (ud-1)`` entries.

The reference (``_shap_proxy_treeshap_interactions._interaction_batch``) stays the fallback for ensembles that do not fit the budget.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange

from ._shap_proxy_treeshap import _NO_CHILD, TreeEnsemble, _extend_path, _unwound_sum
from ._shap_proxy_treeshap_table import MAX_TABLE_ENTRIES, PAYOFF_SHARE, build_tables

MAX_ITABLE_ENTRIES = MAX_TABLE_ENTRIES  # double entries of the interaction tables (192 MB)


@njit(cache=True, parallel=True)
def _build_itables(leaf_ud, leaf_feat_off, leaf_zero, leaf_itab_off, leaf_value, itable):
    """``itable[off + (k * 2^(ud-1) + m') * (ud-1) + q]`` = contribution to the q-th element other than ``k`` of the path without element ``k`` at the compressed mask ``m'``."""
    for leaf in prange(leaf_ud.shape[0]):
        ud = leaf_ud[leaf]
        if ud < 2:
            continue
        off = leaf_feat_off[leaf]
        nd = ud - 1
        pf_feat = np.zeros(nd + 1, dtype=np.int64)
        pf_zero = np.empty(nd + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        pf_one = np.empty(nd + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        pweight = np.empty(nd + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        half = 1 << nd
        for k in range(ud):
            base = leaf_itab_off[leaf] + k * half * nd
            for m in range(half):
                _extend_path(pf_feat, pf_zero, pf_one, pweight, 0, 0, 1.0, 1.0, -1)
                for q in range(nd):
                    e = q if q < k else q + 1
                    _extend_path(pf_feat, pf_zero, pf_one, pweight, 0, q + 1, leaf_zero[off + e], 1.0 if (m >> q) & 1 else 0.0, 0)
                for q in range(1, nd + 1):
                    w = _unwound_sum(pf_zero, pf_one, pweight, 0, nd, q)
                    itable[base + m * nd + q - 1] = w * (pf_one[q] - pf_zero[q]) * leaf_value[leaf]


@njit(cache=True, parallel=True)
def _interaction_table_batch(
    X, Phi, phi_main, children_left, children_right, children_default, features, thresholds, tree_roots, node_slot, leaf_of_node,
    leaf_ud, leaf_feat_off, leaf_feat, leaf_zero, leaf_table_off, table, leaf_itab_off, itable, max_depth,
):
    """Fill ``Phi`` (n, P, P) and ``phi_main`` (n, P), parallel over rows: one node walk per tree with the bit mask, at each leaf ``ud`` main-table adds and ``ud (ud-1)`` interaction adds."""
    n = X.shape[0]
    stack_size = 2 * (max_depth + 2)
    for i in prange(n):
        st_node = np.empty(stack_size, dtype=np.int64)
        st_mask = np.empty(stack_size, dtype=np.int64)
        xi = X[i]
        phi_i = phi_main[i]
        Phi_i = Phi[i]
        for t in range(tree_roots.shape[0]):
            sp = 0
            st_node[sp] = tree_roots[t]
            st_mask[sp] = -1
            sp += 1
            while sp > 0:
                sp -= 1
                node = st_node[sp]
                mask = st_mask[sp]
                if children_left[node] == _NO_CHILD:
                    leaf = leaf_of_node[node]
                    ud = leaf_ud[leaf]
                    if ud > 0:
                        _leaf_adds1(phi_i, Phi_i, mask & ((1 << ud) - 1), ud, leaf_feat_off[leaf], leaf_table_off[leaf], leaf_itab_off[leaf], leaf_feat, leaf_zero, table, itable)
                    continue
                xv = xi[features[node]]
                if np.isnan(xv):
                    hot = children_default[node]
                elif xv < thresholds[node]:
                    hot = children_left[node]
                else:
                    hot = children_right[node]
                cold = children_right[node] if hot == children_left[node] else children_left[node]
                st_node[sp] = cold
                st_mask[sp] = mask & ~(1 << node_slot[node])
                sp += 1
                st_node[sp] = hot
                st_mask[sp] = mask
                sp += 1
        _symmetrise_and_diagonal(Phi_i, phi_i)


@njit(cache=True)
def _route_row(mask, xi, c0, c1, root, inner, children_left, children_right, features, thresholds, left_is_default, node_bit):
    """Branch-free route of one row through one tree: the bit mask of every node of the tree (``mask[node - root]``)."""
    mask[0] = -1
    for c in range(c0, c1):
        node = inner[c]
        xv = xi[features[node]]
        go_left = left_is_default[node] if xv != xv else xv < thresholds[node]
        m = mask[node - root]
        mc = m & ~node_bit[node]
        mask[children_left[node] - root] = m if go_left else mc
        mask[children_right[node] - root] = mc if go_left else m


@njit(cache=True)
def _route_rows4(mk, x0, x1, x2, x3, c0, c1, root, inner, children_left, children_right, features, thresholds, left_is_default, node_bit):
    """``_route_row`` for four rows at once (``mk[q]`` = the masks of row ``q``): the node record is loaded once and the four routes are independent."""
    for q in range(4):
        mk[q, 0] = -1
    for c in range(c0, c1):
        node = inner[c]
        f = features[node]
        th = thresholds[node]
        dl = left_is_default[node]
        nb = ~node_bit[node]
        li = children_left[node] - root
        ri = children_right[node] - root
        me = node - root
        v0 = x0[f]
        v1 = x1[f]
        v2 = x2[f]
        v3 = x3[f]
        g0 = dl if v0 != v0 else v0 < th
        g1 = dl if v1 != v1 else v1 < th
        g2 = dl if v2 != v2 else v2 < th
        g3 = dl if v3 != v3 else v3 < th
        m = mk[0, me]
        mc = m & nb
        mk[0, li] = m if g0 else mc
        mk[0, ri] = mc if g0 else m
        m = mk[1, me]
        mc = m & nb
        mk[1, li] = m if g1 else mc
        mk[1, ri] = mc if g1 else m
        m = mk[2, me]
        mc = m & nb
        mk[2, li] = m if g2 else mc
        mk[2, ri] = mc if g2 else m
        m = mk[3, me]
        mc = m & nb
        mk[3, li] = m if g3 else mc
        mk[3, ri] = mc if g3 else m


@njit(cache=True)
def _leaf_adds1(phi_i, Phi_i, mk, ud, foff, lo, ibase, leaf_feat, leaf_zero, table, itable):
    """Main-table and interaction-table adds of one leaf for one row (``mk`` = the mask of the row restricted to the ``ud`` elements of the leaf)."""
    base = lo + mk * ud
    for j in range(ud):
        phi_i[leaf_feat[foff + j]] += table[base + j]
    if ud > 1:
        nd = ud - 1
        half = 1 << nd
        for k in range(ud):
            factor = 0.5 * (((mk >> k) & 1) - leaf_zero[foff + k])
            mc = (mk & ((1 << k) - 1)) | ((mk >> (k + 1)) << k)
            row = Phi_i[leaf_feat[foff + k]]
            b = ibase + (k * half + mc) * nd
            for q in range(nd):
                row[leaf_feat[foff + (q if q < k else q + 1)]] += factor * itable[b + q]


@njit(cache=True)
def _leaf_adds4(phi_main, Phi, r, m0, m1, m2, m3, ud, foff, lo, ibase, leaf_feat, leaf_zero, table, itable):
    """``_leaf_adds1`` for rows ``r .. r + 3``: the four chains of read-modify-write adds are independent and overlap."""
    for j in range(ud):
        ft = leaf_feat[foff + j]
        phi_main[r, ft] += table[lo + m0 * ud + j]
        phi_main[r + 1, ft] += table[lo + m1 * ud + j]
        phi_main[r + 2, ft] += table[lo + m2 * ud + j]
        phi_main[r + 3, ft] += table[lo + m3 * ud + j]
    if ud > 1:
        nd = ud - 1
        half = 1 << nd
        for k in range(ud):
            z = leaf_zero[foff + k]
            f0 = 0.5 * (((m0 >> k) & 1) - z)
            f1 = 0.5 * (((m1 >> k) & 1) - z)
            f2 = 0.5 * (((m2 >> k) & 1) - z)
            f3 = 0.5 * (((m3 >> k) & 1) - z)
            low = (1 << k) - 1
            kb = ibase + k * half * nd
            b0 = kb + ((m0 & low) | ((m0 >> (k + 1)) << k)) * nd
            b1 = kb + ((m1 & low) | ((m1 >> (k + 1)) << k)) * nd
            b2 = kb + ((m2 & low) | ((m2 >> (k + 1)) << k)) * nd
            b3 = kb + ((m3 & low) | ((m3 >> (k + 1)) << k)) * nd
            fk = leaf_feat[foff + k]
            row0 = Phi[r, fk]
            row1 = Phi[r + 1, fk]
            row2 = Phi[r + 2, fk]
            row3 = Phi[r + 3, fk]
            for q in range(nd):
                ft = leaf_feat[foff + (q if q < k else q + 1)]
                row0[ft] += f0 * itable[b0 + q]
                row1[ft] += f1 * itable[b1 + q]
                row2[ft] += f2 * itable[b2 + q]
                row3[ft] += f3 * itable[b3 + q]


@njit(cache=True)
def _symmetrise_and_diagonal(Phi_i, phi_i):
    """Symmetrise the off-diagonal in place (the (i,j) and (j,i) passes are numerically independent), THEN fill the diagonal so the row-sum identity holds against the symmetrised matrix."""
    P = phi_i.shape[0]
    for a in range(P):
        for b in range(a + 1, P):
            avg = 0.5 * (Phi_i[a, b] + Phi_i[b, a])
            Phi_i[a, b] = avg
            Phi_i[b, a] = avg
    for j in range(P):
        row_sum = 0.0
        for p in range(P):
            if p != j:
                row_sum += Phi_i[j, p]
        Phi_i[j, j] = phi_i[j] - row_sum


@njit(cache=True, parallel=True)
def _interaction_table_batch_blocked(
    X, Phi, phi_main, children_left, children_right, features, thresholds, left_is_default, node_bit, tree_roots, max_tree_nodes,
    inner, inner_ptr, leaves, leaf_ptr, leaf_of_node, leaf_ud, leaf_feat_off, leaf_feat, leaf_zero, leaf_table_off, table, leaf_itab_off, itable, block,
):
    """``_interaction_table_batch`` with the loop order of the main table kernel: preorder route and leaf passes (branch-free route), trees outer over blocks of ``block`` rows so one tree's interaction-table
    slice is read once per block and stays in L2 instead of streaming the whole table once per row, and four rows abreast so the read-modify-write chains of the leaf adds overlap. The rows left over
    (fewer than four) run the single-row path; the symmetrisation and the diagonal follow per row."""
    n = X.shape[0]
    n_blocks = (n + block - 1) // block
    for bk in prange(n_blocks):
        r0 = bk * block
        r1 = min(n, r0 + block)
        mask = np.empty(max_tree_nodes, dtype=np.int64)
        mk4 = np.empty((4, max_tree_nodes), dtype=np.int64)
        for t in range(tree_roots.shape[0]):
            root = tree_roots[t]
            c0 = inner_ptr[t]
            c1 = inner_ptr[t + 1]
            r = r0
            while r + 4 <= r1:
                _route_rows4(mk4, X[r], X[r + 1], X[r + 2], X[r + 3], c0, c1, root, inner, children_left, children_right, features, thresholds, left_is_default, node_bit)
                for c in range(leaf_ptr[t], leaf_ptr[t + 1]):
                    node = leaves[c]
                    leaf = leaf_of_node[node]
                    ud = leaf_ud[leaf]
                    if ud > 0:
                        me = node - root
                        full = (1 << ud) - 1
                        _leaf_adds4(
                            phi_main, Phi, r, mk4[0, me] & full, mk4[1, me] & full, mk4[2, me] & full, mk4[3, me] & full, ud, leaf_feat_off[leaf], leaf_table_off[leaf],
                            leaf_itab_off[leaf], leaf_feat, leaf_zero, table, itable,
                        )
                r += 4
            while r < r1:
                _route_row(mask, X[r], c0, c1, root, inner, children_left, children_right, features, thresholds, left_is_default, node_bit)
                for c in range(leaf_ptr[t], leaf_ptr[t + 1]):
                    node = leaves[c]
                    leaf = leaf_of_node[node]
                    ud = leaf_ud[leaf]
                    if ud > 0:
                        _leaf_adds1(
                            phi_main[r], Phi[r], mask[node - root] & ((1 << ud) - 1), ud, leaf_feat_off[leaf], leaf_table_off[leaf], leaf_itab_off[leaf], leaf_feat, leaf_zero, table, itable
                        )
                r += 1
        for r in range(r0, r1):
            _symmetrise_and_diagonal(Phi[r], phi_main[r])


def _split_features_per_tree(ensemble: TreeEnsemble) -> float:
    """Mean number of distinct split features of a tree (the reference runs two conditioned scans per one)."""
    roots = np.asarray(ensemble.tree_roots, dtype=np.int64)
    feats = np.asarray(ensemble.features)
    bounds = np.append(roots, feats.shape[0])
    counts = []
    for t in range(roots.shape[0]):
        f = feats[bounds[t] : bounds[t + 1]]
        counts.append(np.unique(f[f >= 0]).shape[0])
    return float(np.mean(counts)) if counts else 0.0


def build_interaction_tables(ensemble: TreeEnsemble, n_rows: Optional[int] = None) -> Optional[dict]:
    """The main tables plus the interaction tables of ``ensemble``, or ``None`` when the main tables do not fit, the interaction tables exceed ``MAX_ITABLE_ENTRIES``, or (with ``n_rows``) their build work
    ``sum(ud * 2^(ud-1) * ud^2)`` is not below ``PAYOFF_SHARE`` of the reference's per-row work (two full path scans per tree split feature)."""
    tab = build_tables(ensemble)
    if tab is None:
        return None
    ud = tab["leaf_ud"]
    pw = np.int64(1) << np.maximum(ud - 1, 0)
    isizes = np.where(ud > 1, ud * pw * (ud - 1), 0).astype(np.int64)
    if int(isizes.sum()) > MAX_ITABLE_ENTRIES:
        return None
    if n_rows is not None:
        u2 = ud.astype(np.float64) ** 2  # Intentional float64: work estimate
        build = float((np.where(ud > 1, ud * pw, 0) * u2).sum())
        ref_row = 2.0 * float(u2.sum()) * _split_features_per_tree(ensemble)
        if build >= PAYOFF_SHARE * float(n_rows) * ref_row:
            return None
    leaf_itab_off = np.concatenate(([0], np.cumsum(isizes)[:-1])).astype(np.int64)
    itable = np.zeros(int(isizes.sum()), dtype=np.float64)  # Intentional float64: the table is the exact reference
    _build_itables(ud, tab["leaf_feat_off"], tab["leaf_zero"], leaf_itab_off, tab["leaf_value"], itable)
    return {**tab, "leaf_itab_off": leaf_itab_off, "itable": itable}


def interaction_tensor_table(ensemble: TreeEnsemble, X: np.ndarray, tables: Optional[dict] = None, block: int = -1) -> Optional[tuple]:
    """``(Phi (n, P, P), phi (n, P), base)`` from the interaction tables, or ``None`` when the ensemble does not fit them (the caller keeps the conditioned reference kernel)."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    Xf = ensemble.as_float32(X)
    tab = tables if tables is not None else build_interaction_tables(ensemble, int(Xf.shape[0]))
    if tab is None:
        return None
    n, P = Xf.shape[0], ensemble.n_features
    Phi = np.zeros((n, P, P), dtype=np.float64)  # Intentional float64: the caller's tensor contract
    phi_main = np.zeros((n, P), dtype=np.float64)  # Intentional float64: the caller's tensor contract
    lin = tab.get("linear")
    if lin is not None and block != 0:
        with parallel_kernel_entry():
            _interaction_table_batch_blocked(
                Xf, Phi, phi_main, ensemble.children_left, ensemble.children_right, ensemble.features, ensemble.thresholds, lin["left_is_default"], lin["node_bit"], ensemble.tree_roots,
                lin["max_tree_nodes"], lin["inner"], lin["inner_ptr"], lin["leaves"], lin["leaf_ptr"], tab["leaf_of_node"], tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_feat"],
                tab["leaf_zero"], tab["leaf_table_off"], tab["table"], tab["leaf_itab_off"], tab["itable"], block if block > 0 else 16,
            )
        return Phi, phi_main, float(ensemble.base_offset)
    with parallel_kernel_entry():
        _interaction_table_batch(
            Xf, Phi, phi_main, ensemble.children_left, ensemble.children_right, ensemble.children_default, ensemble.features, ensemble.thresholds, ensemble.tree_roots,
            tab["node_slot"], tab["leaf_of_node"], tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_feat"], tab["leaf_zero"], tab["leaf_table_off"], tab["table"],
            tab["leaf_itab_off"], tab["itable"], ensemble.max_depth,
        )
    return Phi, phi_main, float(ensemble.base_offset)
