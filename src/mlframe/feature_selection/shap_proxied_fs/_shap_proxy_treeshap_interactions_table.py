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

MAX_ITABLE_ENTRIES = MAX_TABLE_ENTRIES  # float64 entries of the interaction tables (192 MB)


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
    P = phi_main.shape[1]
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
                        mask = mask & ((1 << ud) - 1)
                        foff = leaf_feat_off[leaf]
                        base = leaf_table_off[leaf] + mask * ud
                        for j in range(ud):
                            phi_i[leaf_feat[foff + j]] += table[base + j]
                        if ud > 1:
                            nd = ud - 1
                            half = 1 << nd
                            ibase = leaf_itab_off[leaf]
                            for k in range(ud):
                                factor = 0.5 * (((mask >> k) & 1) - leaf_zero[foff + k])
                                mc = (mask & ((1 << k) - 1)) | ((mask >> (k + 1)) << k)
                                row = Phi_i[leaf_feat[foff + k]]
                                b = ibase + (k * half + mc) * nd
                                for q in range(nd):
                                    e = q if q < k else q + 1
                                    row[leaf_feat[foff + e]] += factor * itable[b + q]
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


def interaction_tensor_table(ensemble: TreeEnsemble, X: np.ndarray, tables: Optional[dict] = None) -> Optional[tuple]:
    """``(Phi (n, P, P), phi (n, P), base)`` from the interaction tables, or ``None`` when the ensemble does not fit them (the caller keeps the conditioned reference kernel)."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    tab = tables if tables is not None else build_interaction_tables(ensemble, int(Xf.shape[0]))
    if tab is None:
        return None
    n, P = Xf.shape[0], ensemble.n_features
    Phi = np.zeros((n, P, P), dtype=np.float64)  # Intentional float64: the caller's tensor contract
    phi_main = np.zeros((n, P), dtype=np.float64)  # Intentional float64: the caller's tensor contract
    with parallel_kernel_entry():
        _interaction_table_batch(
            Xf, Phi, phi_main, ensemble.children_left, ensemble.children_right, ensemble.children_default, ensemble.features, ensemble.thresholds, ensemble.tree_roots,
            tab["node_slot"], tab["leaf_of_node"], tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_feat"], tab["leaf_zero"], tab["leaf_table_off"], tab["table"],
            tab["leaf_itab_off"], tab["itable"], ensemble.max_depth,
        )
    return Phi, phi_main, float(ensemble.base_offset)
