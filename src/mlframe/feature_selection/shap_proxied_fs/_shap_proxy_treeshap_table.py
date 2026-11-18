"""Table form of the path-dependent TreeSHAP (the idea of FastTreeSHAP v2): the per-leaf weight work is paid once per ensemble, not once per sample and leaf.

In the path algorithm every sample walks every leaf and spends ``O(depth^2)`` there: the unwound sums of the leaf's path. But the path of a leaf is fixed by the tree (its unique split features and their
zero fractions, the product of the cover ratios of the edges taken), and the only thing a sample contributes is the vector of *one fractions*: for each unique feature of the path, whether the sample
satisfies every split of that feature on the path (0 or 1). A leaf with ``ud`` unique features therefore has at most ``2^ud`` distinct contributions per element, and the table is built once with the
same EXTEND / unwound-sum arithmetic as the path kernel. A sample then only walks the nodes keeping a bit mask (a cold step clears the bit of its node's feature slot) and, at a leaf, adds ``ud`` table entries.

The element order does not matter (the Shapley weights are symmetric in the elements), so the elements of a leaf are its unique features in order of first appearance on the path. The table is used only
when its size fits ``MAX_TABLE_ENTRIES`` and every leaf has at most ``MAX_UNIQUE`` unique features; otherwise the caller keeps the path kernel.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange

from ._shap_proxy_treeshap import _NO_CHILD, TreeEnsemble, _extend_path, _unwound_sum

MAX_UNIQUE = 12  # a mask has 2^12 = 4096 entries per leaf at most
MAX_TABLE_ENTRIES = 24_000_000  # float64 entries of the contribution table (192 MB); float32 tables take half


@njit(cache=True)
def _walk_leaves(children_left, children_right, features, node_sample_weight, tree_roots, max_depth, node_slot, leaf_ud, leaf_feat_off, leaf_feat, leaf_zero, leaf_of_node, count_only):
    """Structural walk of every tree (no sample): fills each node's feature slot and, per leaf, its unique-feature count, features and zero fractions. Returns ``(n_leaves, n_feat_entries)``.

    The state of level ``l`` (unique features and zero fractions of the path above and including the edge into the node) is rebuilt from level ``l - 1`` when the node is popped, so siblings share
    the intact parent state."""
    width = max_depth + 2
    st_feat = np.empty((width + 1, width), dtype=np.int64)
    st_zero = np.empty((width + 1, width), dtype=np.float64)  # Intentional float64: zero fractions are products of cover ratios
    cnt = np.zeros(width + 1, dtype=np.int64)
    stack_node = np.empty(2 * width + 4, dtype=np.int64)
    stack_level = np.empty(2 * width + 4, dtype=np.int64)
    stack_parent = np.empty(2 * width + 4, dtype=np.int64)
    n_leaves = 0
    n_feat = 0
    for t in range(tree_roots.shape[0]):
        sp = 0
        stack_node[sp] = tree_roots[t]
        stack_level[sp] = 0
        stack_parent[sp] = -1
        sp += 1
        cnt[0] = 0
        while sp > 0:
            sp -= 1
            node = stack_node[sp]
            level = stack_level[sp]
            parent = stack_parent[sp]
            if level > 0:
                # state of this level = parent's state with the edge (parent -> node) folded into the parent's feature slot
                k = node_slot[parent]
                c = cnt[level - 1]
                for j in range(c):
                    st_feat[level, j] = st_feat[level - 1, j]
                    st_zero[level, j] = st_zero[level - 1, j]
                if k == c:
                    st_feat[level, k] = features[parent]
                    st_zero[level, k] = 1.0
                    cnt[level] = c + 1
                else:
                    cnt[level] = c
                w_parent = node_sample_weight[parent]
                st_zero[level, k] *= node_sample_weight[node] / w_parent if w_parent > 0.0 else 0.0
            if children_left[node] == _NO_CHILD:
                ud = cnt[level]
                if not count_only:
                    leaf_ud[n_leaves] = ud
                    leaf_feat_off[n_leaves] = n_feat
                    for j in range(ud):
                        leaf_feat[n_feat + j] = st_feat[level, j]
                        leaf_zero[n_feat + j] = st_zero[level, j]
                    leaf_of_node[node] = n_leaves
                n_leaves += 1
                n_feat += ud
                continue
            f = features[node]
            k = cnt[level]
            for j in range(cnt[level]):
                if st_feat[level, j] == f:
                    k = j
                    break
            node_slot[node] = k
            stack_node[sp] = children_right[node]
            stack_level[sp] = level + 1
            stack_parent[sp] = node
            sp += 1
            stack_node[sp] = children_left[node]
            stack_level[sp] = level + 1
            stack_parent[sp] = node
            sp += 1
    return n_leaves, n_feat


@njit(cache=True, parallel=True)
def _build_tables(leaf_ud, leaf_feat_off, leaf_zero, leaf_table_off, leaf_value, table):
    """Contribution table: for leaf ``L`` and mask ``m`` (bit ``i`` = the sample satisfies every split of element ``i``), entry ``(m, i)`` is the contribution to element ``i``'s feature, computed with the
    path kernel's own EXTEND / unwound-sum routines."""
    for leaf in prange(leaf_ud.shape[0]):
        ud = leaf_ud[leaf]
        off = leaf_feat_off[leaf]
        pf_feat = np.zeros(ud + 1, dtype=np.int64)
        pf_zero = np.empty(ud + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        pf_one = np.empty(ud + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        pweight = np.empty(ud + 1, dtype=np.float64)  # Intentional float64: the path kernel's weights are double
        base = leaf_table_off[leaf]
        for m in range(1 << ud):
            _extend_path(pf_feat, pf_zero, pf_one, pweight, 0, 0, 1.0, 1.0, -1)
            for i in range(ud):
                _extend_path(pf_feat, pf_zero, pf_one, pweight, 0, i + 1, leaf_zero[off + i], 1.0 if (m >> i) & 1 else 0.0, 0)
            for i in range(1, ud + 1):
                w = _unwound_sum(pf_zero, pf_one, pweight, 0, ud, i)
                table[base + m * ud + i - 1] = w * (pf_one[i] - pf_zero[i]) * leaf_value[leaf]


@njit(cache=True, parallel=True)
def _table_batch(X, phi, children_left, children_right, children_default, features, thresholds, tree_roots, node_slot, leaf_of_node, leaf_ud, leaf_feat_off, leaf_feat, leaf_table_off, table, max_depth):
    """Fill ``phi`` (n, f) from the tables, parallel over rows: walk each tree keeping the mask, add ``ud`` table entries at each leaf."""
    n = X.shape[0]
    stack_size = 2 * (max_depth + 2)
    for i in prange(n):
        st_node = np.empty(stack_size, dtype=np.int64)
        st_mask = np.empty(stack_size, dtype=np.int64)
        xi = X[i]
        phi_i = phi[i]
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
                        base = leaf_table_off[leaf] + (mask & ((1 << ud) - 1)) * ud
                        foff = leaf_feat_off[leaf]
                        for j in range(ud):
                            phi_i[leaf_feat[foff + j]] += table[base + j]
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


@njit(cache=True)
def _tree_orders(children_left, children_right, tree_roots, n_nodes, inner, inner_ptr, leaves, leaf_ptr):
    """Preorder lists of each tree's inner nodes and leaves (global node ids), tree after tree, with their CSR pointers. Returns False when a tree's nodes do not all lie in
    ``[root_t, root_{t+1})`` (the linear kernel indexes its mask scratch by ``node - root``)."""
    stack = np.empty(n_nodes + 1, dtype=np.int64)
    ni = 0
    nl = 0
    n_trees = tree_roots.shape[0]
    for t in range(n_trees):
        inner_ptr[t] = ni
        leaf_ptr[t] = nl
        lo = tree_roots[t]
        hi = tree_roots[t + 1] if t + 1 < n_trees else n_nodes
        sp = 0
        stack[sp] = tree_roots[t]
        sp += 1
        while sp > 0:
            sp -= 1
            node = stack[sp]
            if node < lo or node >= hi:
                return False
            if children_left[node] == _NO_CHILD:
                leaves[nl] = node
                nl += 1
            else:
                inner[ni] = node
                ni += 1
                stack[sp] = children_right[node]
                sp += 1
                stack[sp] = children_left[node]
                sp += 1
    inner_ptr[n_trees] = ni
    leaf_ptr[n_trees] = nl
    return True


@njit(cache=True, parallel=True)
def _table_batch_linear(
    X, phi, children_left, children_right, features, thresholds, left_is_default, node_bit, tree_roots, max_tree_nodes,
    inner, inner_ptr, leaves, leaf_ptr, leaf_of_node, leaf_ud, leaf_feat_off, leaf_feat, leaf_table_off, table,
):
    """The table kernel without a DFS stack: per tree one pass over the inner nodes in preorder computes both children's masks branch-free (the route is a select, not a jump), then one pass over the
    leaves adds the table entries. The mask scratch is indexed by ``node - root`` and stays in L1."""
    n = X.shape[0]
    for i in prange(n):
        mask = np.empty(max_tree_nodes, dtype=np.int64)
        xi = X[i]
        phi_i = phi[i]
        for t in range(tree_roots.shape[0]):
            root = tree_roots[t]
            mask[0] = -1
            for c in range(inner_ptr[t], inner_ptr[t + 1]):
                node = inner[c]
                xv = xi[features[node]]
                go_left = left_is_default[node] if xv != xv else xv < thresholds[node]
                m = mask[node - root]
                mc = m & ~node_bit[node]
                mask[children_left[node] - root] = m if go_left else mc
                mask[children_right[node] - root] = mc if go_left else m
            for c in range(leaf_ptr[t], leaf_ptr[t + 1]):
                node = leaves[c]
                leaf = leaf_of_node[node]
                ud = leaf_ud[leaf]
                if ud > 0:
                    base = leaf_table_off[leaf] + (mask[node - root] & ((1 << ud) - 1)) * ud
                    foff = leaf_feat_off[leaf]
                    for j in range(ud):
                        phi_i[leaf_feat[foff + j]] += table[base + j]


@njit(cache=True, parallel=True)
def _table_batch_blocked(
    X, phi, children_left, children_right, features, thresholds, left_is_default, node_bit, tree_roots, max_tree_nodes,
    inner, inner_ptr, leaves, leaf_ptr, leaf_of_node, leaf_ud, leaf_feat_off, leaf_feat, leaf_table_off, table, block,
):
    """``_table_batch_linear`` with two changes of loop order. (1) Trees outer over blocks of ``block`` rows: one tree's node records, leaf records and table slice are read once per block instead of once
    per row, so they stay in L1 while the ensemble (hundreds of KB of metadata plus the table) streams through L2/L3 once per block. (2) Four rows advance together through each tree: the leaf adds of one row
    chain through memory (every leaf of a tree adds into the features near its root, a load-add-store-forward chain of about 12 cycles per add), the four rows are independent chains the core overlaps, and the
    node and leaf records are loaded once for four rows. The tail (fewer than four rows) runs the single-row loop."""
    n = X.shape[0]
    n_blocks = (n + block - 1) // block
    for b in prange(n_blocks):
        r0 = b * block
        r1 = min(n, r0 + block)
        mk = np.empty((4, max_tree_nodes), dtype=np.int64)
        for t in range(tree_roots.shape[0]):
            root = tree_roots[t]
            c0 = inner_ptr[t]
            c1 = inner_ptr[t + 1]
            l0 = leaf_ptr[t]
            l1 = leaf_ptr[t + 1]
            r = r0
            while r + 4 <= r1:
                x0 = X[r]
                x1 = X[r + 1]
                x2 = X[r + 2]
                x3 = X[r + 3]
                p0 = phi[r]
                p1 = phi[r + 1]
                p2 = phi[r + 2]
                p3 = phi[r + 3]
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
                for c in range(l0, l1):
                    node = leaves[c]
                    leaf = leaf_of_node[node]
                    ud = leaf_ud[leaf]
                    if ud > 0:
                        me = node - root
                        lo = leaf_table_off[leaf]
                        foff = leaf_feat_off[leaf]
                        full = (1 << ud) - 1
                        b0 = lo + (mk[0, me] & full) * ud
                        b1 = lo + (mk[1, me] & full) * ud
                        b2 = lo + (mk[2, me] & full) * ud
                        b3 = lo + (mk[3, me] & full) * ud
                        for j in range(ud):
                            ft = leaf_feat[foff + j]
                            p0[ft] += table[b0 + j]
                            p1[ft] += table[b1 + j]
                            p2[ft] += table[b2 + j]
                            p3[ft] += table[b3 + j]
                r += 4
            while r < r1:
                xi = X[r]
                phi_i = phi[r]
                mk[0, 0] = -1
                for c in range(c0, c1):
                    node = inner[c]
                    xv = xi[features[node]]
                    go_left = left_is_default[node] if xv != xv else xv < thresholds[node]
                    m = mk[0, node - root]
                    mc = m & ~node_bit[node]
                    mk[0, children_left[node] - root] = m if go_left else mc
                    mk[0, children_right[node] - root] = mc if go_left else m
                for c in range(l0, l1):
                    node = leaves[c]
                    leaf = leaf_of_node[node]
                    ud = leaf_ud[leaf]
                    if ud > 0:
                        base = leaf_table_off[leaf] + (mk[0, node - root] & ((1 << ud) - 1)) * ud
                        foff = leaf_feat_off[leaf]
                        for j in range(ud):
                            phi_i[leaf_feat[foff + j]] += table[base + j]
                r += 1


PAYOFF_SHARE = 0.6  # the table is built when its build work is below this share of the path kernel's work over the rows


def build_tables(ensemble: TreeEnsemble, n_rows: Optional[int] = None) -> Optional[dict]:
    """The structural arrays and the contribution table of ``ensemble``, or ``None`` when a leaf has more than ``MAX_UNIQUE`` unique features, the table would exceed ``MAX_TABLE_ENTRIES``, or (with
    ``n_rows``) the build work ``sum(2^ud * ud^2)`` is not below ``PAYOFF_SHARE`` of the path kernel's work ``n_rows * sum(ud^2)`` (few rows: the path kernel is cheaper than building the table)."""
    n_nodes = int(ensemble.children_left.shape[0])
    node_slot = np.zeros(n_nodes, dtype=np.int64)
    leaf_of_node = np.full(n_nodes, -1, dtype=np.int64)
    dummy_i = np.empty(1, dtype=np.int64)
    dummy_f = np.empty(1, dtype=np.float64)  # Intentional float64: unused placeholder of the count pass
    n_leaves, n_feat = _walk_leaves(
        ensemble.children_left, ensemble.children_right, ensemble.features, ensemble.node_sample_weight, ensemble.tree_roots, ensemble.max_depth, node_slot, dummy_i, dummy_i, dummy_i, dummy_f, leaf_of_node, True
    )
    leaf_ud = np.zeros(n_leaves, dtype=np.int64)
    leaf_feat_off = np.zeros(n_leaves, dtype=np.int64)
    leaf_feat = np.zeros(n_feat, dtype=np.int64)
    leaf_zero = np.zeros(n_feat, dtype=np.float64)  # Intentional float64: zero fractions
    _walk_leaves(
        ensemble.children_left, ensemble.children_right, ensemble.features, ensemble.node_sample_weight, ensemble.tree_roots, ensemble.max_depth, node_slot, leaf_ud, leaf_feat_off, leaf_feat, leaf_zero, leaf_of_node, False
    )
    if leaf_ud.size == 0 or int(leaf_ud.max()) > MAX_UNIQUE:
        return None
    sizes = (np.int64(1) << leaf_ud) * leaf_ud
    if int(sizes.sum()) > MAX_TABLE_ENTRIES:
        return None
    if n_rows is not None:
        ud2 = leaf_ud.astype(np.float64) ** 2  # Intentional float64: work estimate
        if float(((np.int64(1) << leaf_ud) * ud2).sum()) >= PAYOFF_SHARE * float(n_rows) * float(ud2.sum()):
            return None
    leaf_table_off = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    leaf_value = np.zeros(n_leaves, dtype=np.float64)  # Intentional float64: leaf outputs
    is_leaf = leaf_of_node >= 0
    leaf_value[leaf_of_node[is_leaf]] = ensemble.values[is_leaf]
    table = np.zeros(int(sizes.sum()), dtype=np.float64)  # Intentional float64: the table is the exact reference
    _build_tables(leaf_ud, leaf_feat_off, leaf_zero, leaf_table_off, leaf_value, table)
    return {
        "node_slot": node_slot, "leaf_of_node": leaf_of_node, "leaf_ud": leaf_ud, "leaf_feat_off": leaf_feat_off, "leaf_feat": leaf_feat, "leaf_table_off": leaf_table_off, "table": table,
        "leaf_zero": leaf_zero, "leaf_value": leaf_value, "linear": _linear_arrays(ensemble, node_slot),
    }


def _linear_arrays(ensemble: TreeEnsemble, node_slot: np.ndarray) -> Optional[dict]:
    """Preorder node lists and per-node route data of the linear kernel, or ``None`` when a tree's node ids are not contiguous (the stack kernel then answers)."""
    n_nodes = int(ensemble.children_left.shape[0])
    n_trees = int(ensemble.tree_roots.shape[0])
    inner = np.empty(n_nodes, dtype=np.int64)
    leaves = np.empty(n_nodes, dtype=np.int64)
    inner_ptr = np.zeros(n_trees + 1, dtype=np.int64)
    leaf_ptr = np.zeros(n_trees + 1, dtype=np.int64)
    if not _tree_orders(ensemble.children_left, ensemble.children_right, ensemble.tree_roots, n_nodes, inner, inner_ptr, leaves, leaf_ptr):
        return None
    bounds = np.append(np.asarray(ensemble.tree_roots, dtype=np.int64), n_nodes)
    return {
        "inner": inner[: inner_ptr[-1]].copy(), "inner_ptr": inner_ptr, "leaves": leaves[: leaf_ptr[-1]].copy(), "leaf_ptr": leaf_ptr,
        "left_is_default": np.asarray(ensemble.children_default == ensemble.children_left),
        "node_bit": (np.int64(1) << node_slot.astype(np.int64)),
        "max_tree_nodes": int(np.diff(bounds).max()),
    }


def _auto_block(n_rows: int) -> int:
    """Rows per block of the blocked kernel: 64 (measured best on 1 and 8 threads), fewer for small inputs so that every thread still gets about two blocks. Always at least 4 (one row group)."""
    import numba

    per_thread = n_rows // (2 * max(int(numba.get_num_threads()), 1))
    return int(max(4, min(64, per_thread - per_thread % 4)))


def treeshap_phi_base_table(ensemble: TreeEnsemble, X: np.ndarray, tables: Optional[dict] = None, kernel: str = "blocked", block: int = 0) -> Optional[tuple]:
    """``(phi (n, f) float64, base)`` from the contribution tables, or ``None`` when the ensemble does not fit them (the caller keeps the path kernel). ``tables`` reuses a ``build_tables`` result.

    ``kernel`` picks the sample loop: ``"blocked"`` (default: preorder passes, row blocks, four rows abreast; ``block`` 0 = automatic), ``"linear"`` (preorder passes, one row at a time) or ``"stack"`` (the
    DFS-stack kernel); the first two fall back to ``"stack"`` when the tree node ids are not contiguous. All three agree to ~1e-14 (the order of the leaf adds differs)."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    tab = tables if tables is not None else build_tables(ensemble, int(np.asarray(X).shape[0]))
    if tab is None:
        return None
    Xf = ensemble.as_float32(X)
    phi = np.zeros((Xf.shape[0], ensemble.n_features), dtype=np.float64)  # Intentional float64: the caller's phi contract
    lin = tab.get("linear") if kernel != "stack" else None
    if lin is not None:
        args = (
            Xf, phi, ensemble.children_left, ensemble.children_right, ensemble.features, ensemble.thresholds, lin["left_is_default"], lin["node_bit"], ensemble.tree_roots,
            lin["max_tree_nodes"], lin["inner"], lin["inner_ptr"], lin["leaves"], lin["leaf_ptr"], tab["leaf_of_node"], tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_feat"],
            tab["leaf_table_off"], tab["table"],
        )
        with parallel_kernel_entry():
            if kernel == "linear":
                _table_batch_linear(*args)
            else:
                _table_batch_blocked(*args, block if block > 0 else _auto_block(Xf.shape[0]))
        return phi, float(ensemble.base_offset)
    with parallel_kernel_entry():
        _table_batch(
            Xf, phi, ensemble.children_left, ensemble.children_right, ensemble.children_default, ensemble.features, ensemble.thresholds, ensemble.tree_roots,
            tab["node_slot"], tab["leaf_of_node"], tab["leaf_ud"], tab["leaf_feat_off"], tab["leaf_feat"], tab["leaf_table_off"], tab["table"], ensemble.max_depth,
        )
    return phi, float(ensemble.base_offset)
