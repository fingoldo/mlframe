"""Fast custom path-dependent TreeSHAP *interaction* values (sample-parallel numba).

Companion to ``_shap_proxy_treeshap`` (main-effect TreeSHAP). This module computes the (n, P, P)
SHAP interaction tensor that ``shap.TreeExplainer.shap_interaction_values`` produces, but with the
self-contained numba kernel that already powers the main-effect path -- avoiding the painfully slow
``shap`` Python/Cython interaction call on wide proxy widths.

Algorithm (Lundberg's interaction extension -- the same recurrence the reference shap C++
``tree_shap_recursive`` uses with its ``condition`` / ``condition_feature`` / ``condition_fraction``
parameters, Algorithm 2 of "Consistent Individualized Feature Attribution for Tree Ensembles"):

  * The OFF-DIAGONAL ``Phi_ij`` (i != j) is HALF the change in feature ``i``'s attribution caused by
    conditioning on feature ``j``:  ``Phi_ij = (phi_i|j_on - phi_i|j_off) / 2``.
  * A conditioned pass forces feature ``j`` onto a fixed branch. At a split on ``j`` the path is NOT
    extended with ``j`` (so ``j`` never receives attribution in that pass) and the ``condition_fraction``
    carried to the children encodes the forcing:  for ``condition > 0`` (forced "on") the cold child
    gets ``condition_fraction = 0`` (pruned) and the hot child keeps the full fraction; for
    ``condition < 0`` (forced "off") both children's fractions are multiplied by their cover ratio.
    The leaf attribution is scaled by the surviving ``condition_fraction``.
  * The DIAGONAL ``Phi_ii`` is the residual main effect:  ``Phi_ii = phi_i - sum_{j != i} Phi_ij``,
    which makes the row-sum identity ``sum_k Phi_ik == phi_i`` hold by construction.

Per tree we run ONE unconditioned scan (the main effect ``phi``) plus, for each feature ``j`` that
actually splits in the ensemble, TWO conditioned scans (on/off). The kernel is parallel over samples;
the per-sample work loops trees x conditioning-features, so the cost is
``O(n * n_trees * depth * n_split_feats)`` rather than ``O(n * P^2)`` blindly.

The output ``Phi`` (n, P, P) is in margin / log-odds space and satisfies, numerically:
  * EXACT symmetry  ``Phi_ij == Phi_ji``  (the off-diagonal is symmetrised before the diagonal fill),
  * row-sum identity ``sum_k Phi_ik == phi_i`` (the main-effect SHAP value from the existing kernel),
  * parity with ``shap.TreeExplainer.shap_interaction_values`` to ~1e-4 (see the parity test).

Scope mirrors the main-effect kernel: xgboost ``XGBRegressor`` / binary ``XGBClassifier`` via the
shared ``extract_ensemble``; unsupported models fall back to the ``shap`` library in the caller.
ALL kernel versions are kept (this is a NEW kernel; the main-effect one in ``_shap_proxy_treeshap`` is
untouched) so we can re-bench per HW.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import (
    _NO_CHILD,
    _extend_path,
    _unwind_path,
    _unwound_sum,
)


@njit(cache=True)
def _treeshap_one_tree_conditioned(
    x, phi_row,
    children_left, children_right, children_default,
    features, thresholds, values, node_sample_weight,
    root, width,
    condition, condition_feature,
    pf_feat, pf_zero, pf_one, pweight,
    st_node, st_level, st_ud, st_zero, st_one, st_pfeat, st_cfrac,
):
    """Iterative path-dependent TreeSHAP scan for one tree, optionally CONDITIONED on a feature.

    Mirrors the reference shap C++ ``tree_shap_recursive`` frame for frame (explicit DFS stack because
    numba miscompiles recursion under parallel=True+cache=True, as in the main-effect kernel). Each
    frame carries the incoming (node, level, unique_depth, zero_frac, one_frac, parent_feature_index,
    condition_fraction); ``level`` selects the path window so siblings start from the same prefix.

    ``condition == 0``: ordinary TreeSHAP -> accumulates the main effect ``phi`` (matches the kernel in
    ``_shap_proxy_treeshap``). ``condition == +1`` forces ``condition_feature`` "on"; ``-1`` forces it
    "off". Conditioning skips EXTEND of the conditioned feature, splits the ``condition_fraction`` per
    the reference, and scales the leaf attribution by the surviving fraction."""
    sp = 0
    st_node[sp] = root
    st_level[sp] = 0
    st_ud[sp] = 0
    st_zero[sp] = 1.0
    st_one[sp] = 1.0
    st_pfeat[sp] = -1
    st_cfrac[sp] = 1.0
    sp += 1

    while sp > 0:
        sp -= 1
        node = st_node[sp]
        level = st_level[sp]
        unique_depth = st_ud[sp]
        parent_zero = st_zero[sp]
        parent_one = st_one[sp]
        parent_feat = st_pfeat[sp]
        cond_frac = st_cfrac[sp]
        off = level * width

        if cond_frac == 0.0:
            continue  # no weight reaches us (the pruned conditioned branch)

        # Copy the parent's path prefix into this level's window, then (maybe) EXTEND. We copy
        # ``unique_depth + 1`` entries to mirror the reference ``std::copy(.., parent + unique_depth + 1)``
        # so that when EXTEND is SKIPPED (the conditioned feature) the slot at ``unique_depth`` keeps the
        # parent's value instead of stale window data from an earlier sibling frame.
        if level > 0:
            poff = (level - 1) * width
            for k in range(unique_depth + 1):
                pf_feat[off + k] = pf_feat[poff + k]
                pf_zero[off + k] = pf_zero[poff + k]
                pf_one[off + k] = pf_one[poff + k]
                pweight[off + k] = pweight[poff + k]
        # Skip extending the conditioned feature so it never receives attribution.
        if condition == 0 or condition_feature != parent_feat:
            _extend_path(pf_feat, pf_zero, pf_one, pweight, off, unique_depth, parent_zero, parent_one, parent_feat)

        if children_left[node] == _NO_CHILD:
            leaf_val = values[node]
            i = 1
            while i <= unique_depth:
                w = _unwound_sum(pf_zero, pf_one, pweight, off, unique_depth, i)
                phi_row[pf_feat[off + i]] += w * (pf_one[off + i] - pf_zero[off + i]) * leaf_val * cond_frac
                i += 1
            continue

        f = features[node]
        xv = x[f]
        if np.isnan(xv):
            hot = children_default[node]
        elif xv < thresholds[node]:
            hot = children_left[node]
        else:
            hot = children_right[node]
        cold = children_right[node] if hot == children_left[node] else children_left[node]

        w_node = node_sample_weight[node]
        hot_w = node_sample_weight[hot] / w_node if w_node > 0.0 else 0.0
        cold_w = node_sample_weight[cold] / w_node if w_node > 0.0 else 0.0

        # If feature ``f`` already appears on the path, UNWIND it and carry its prior fractions forward.
        incoming_zero = 1.0
        incoming_one = 1.0
        path_index = 0
        k = 1
        while k <= unique_depth:
            if pf_feat[off + k] == f:
                path_index = k
                break
            k += 1
        child_depth = unique_depth + 1  # depth passed to children before the conditioned adjustment
        if path_index != 0:
            incoming_zero = pf_zero[off + path_index]
            incoming_one = pf_one[off + path_index]
            _unwind_path(pf_feat, pf_zero, pf_one, pweight, off, unique_depth, path_index)
            child_depth = unique_depth  # we removed one entry; children re-extend back to this depth

        # Split / divide the condition_fraction among the two recursive calls (reference logic).
        hot_cfrac = cond_frac
        cold_cfrac = cond_frac
        if condition > 0 and f == condition_feature:
            cold_cfrac = 0.0
            child_depth -= 1  # conditioned feature occupies no path slot at the child
        elif condition < 0 and f == condition_feature:
            hot_cfrac = cond_frac * hot_w
            cold_cfrac = cond_frac * cold_w
            child_depth -= 1

        # Push cold then hot so hot (popped last) is processed first. The CHILD level is child_depth+1;
        # the child's frame carries unique_depth=child_depth and parent_feature_index=f.
        st_node[sp] = cold
        st_level[sp] = level + 1
        st_ud[sp] = child_depth
        st_zero[sp] = incoming_zero * cold_w
        st_one[sp] = 0.0
        st_pfeat[sp] = f
        st_cfrac[sp] = cold_cfrac
        sp += 1
        st_node[sp] = hot
        st_level[sp] = level + 1
        st_ud[sp] = child_depth
        st_zero[sp] = incoming_zero * hot_w
        st_one[sp] = incoming_one
        st_pfeat[sp] = f
        st_cfrac[sp] = hot_cfrac
        sp += 1


@njit(cache=True, parallel=True)
def _interaction_batch(
    X, Phi, phi_main,
    children_left, children_right, children_default,
    features, thresholds, values, node_sample_weight,
    tree_roots, max_path, cond_feats,
):
    """Fill ``Phi`` (n, P, P) and ``phi_main`` (n, P), parallel over rows. Excludes the base offset.

    ``cond_feats`` is the sorted array of distinct split-feature indices in the ensemble: the only
    features that need conditioning passes (conditioning on a feature the ensemble never splits on is a
    no-op). For each such feature ``j`` we run an on-pass and an off-pass; ``Phi[:, i, j]`` for i != j
    is half their difference. The diagonal is filled afterwards from the row-sum identity."""
    n = X.shape[0]
    n_trees = tree_roots.shape[0]
    P = phi_main.shape[1]
    n_cond = cond_feats.shape[0]
    width = max_path + 2
    n_levels = max_path + 3       # conditioned passes can reach one extra level via child_depth+1
    stack_size = 2 * (max_path + 3)
    scratch = width * n_levels

    for i in prange(n):
        pf_feat = np.empty(scratch, dtype=np.int64)
        pf_zero = np.empty(scratch, dtype=np.float64)
        pf_one = np.empty(scratch, dtype=np.float64)
        pweight = np.empty(scratch, dtype=np.float64)
        st_node = np.empty(stack_size, dtype=np.int64)
        st_level = np.empty(stack_size, dtype=np.int64)
        st_ud = np.empty(stack_size, dtype=np.int64)
        st_zero = np.empty(stack_size, dtype=np.float64)
        st_one = np.empty(stack_size, dtype=np.float64)
        st_pfeat = np.empty(stack_size, dtype=np.int64)
        st_cfrac = np.empty(stack_size, dtype=np.float64)
        xi = X[i]

        phi_unc = np.zeros(P, dtype=np.float64)
        phi_on = np.zeros(P, dtype=np.float64)
        phi_off = np.zeros(P, dtype=np.float64)

        # Unconditioned pass -> main effect phi.
        for t in range(n_trees):
            _treeshap_one_tree_conditioned(
                xi, phi_unc, children_left, children_right, children_default,
                features, thresholds, values, node_sample_weight,
                tree_roots[t], width, 0, -1,
                pf_feat, pf_zero, pf_one, pweight,
                st_node, st_level, st_ud, st_zero, st_one, st_pfeat, st_cfrac,
            )
        for p in range(P):
            phi_main[i, p] = phi_unc[p]

        # Conditioning passes: one (on, off) pair per distinct split feature.
        for c in range(n_cond):
            j = cond_feats[c]
            for p in range(P):
                phi_on[p] = 0.0
                phi_off[p] = 0.0
            for t in range(n_trees):
                _treeshap_one_tree_conditioned(
                    xi, phi_on, children_left, children_right, children_default,
                    features, thresholds, values, node_sample_weight,
                    tree_roots[t], width, 1, j,
                    pf_feat, pf_zero, pf_one, pweight,
                    st_node, st_level, st_ud, st_zero, st_one, st_pfeat, st_cfrac,
                )
            for t in range(n_trees):
                _treeshap_one_tree_conditioned(
                    xi, phi_off, children_left, children_right, children_default,
                    features, thresholds, values, node_sample_weight,
                    tree_roots[t], width, -1, j,
                    pf_feat, pf_zero, pf_one, pweight,
                    st_node, st_level, st_ud, st_zero, st_one, st_pfeat, st_cfrac,
                )
            # Off-diagonal Phi[i, p, j] = (phi_on[p] - phi_off[p]) / 2 for p != j.
            for p in range(P):
                if p != j:
                    Phi[i, p, j] = 0.5 * (phi_on[p] - phi_off[p])

        # Symmetrise the off-diagonal in place (the (i,j) vs (j,i) passes are numerically independent),
        # THEN fill the diagonal so the row-sum identity holds against the symmetrised matrix.
        for a in range(P):
            for b in range(a + 1, P):
                avg = 0.5 * (Phi[i, a, b] + Phi[i, b, a])
                Phi[i, a, b] = avg
                Phi[i, b, a] = avg
        for j in range(P):
            row_sum = 0.0
            for p in range(P):
                if p != j:
                    row_sum += Phi[i, j, p]
            Phi[i, j, j] = phi_main[i, j] - row_sum


def interaction_tensor_numba(ensemble, X: np.ndarray):
    """Run the numba path-dependent TreeSHAP interaction kernel.

    Returns ``(Phi, phi, base)`` where ``Phi`` is (n, P, P) float64 interaction values in margin /
    log-odds space, ``phi`` is the (n, P) main-effect SHAP (== ``Phi.sum(axis=2)``, the row-sum
    identity), and ``base`` is the ensemble intercept so ``base + Phi.sum((1,2)) == model margin``.
    The off-diagonal is symmetrised inside the kernel so the exact symmetry invariant holds.
    """
    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n = Xf.shape[0]
    P = ensemble.n_features
    Phi = np.zeros((n, P, P), dtype=np.float64)
    phi_main = np.zeros((n, P), dtype=np.float64)

    # Only features that actually split need conditioning passes.
    split_feats = ensemble.features[ensemble.features >= 0]
    cond_feats = np.unique(split_feats).astype(np.int64) if split_feats.size else np.empty(0, dtype=np.int64)

    _interaction_batch(
        Xf, Phi, phi_main,
        ensemble.children_left, ensemble.children_right, ensemble.children_default,
        ensemble.features, ensemble.thresholds, ensemble.values, ensemble.node_sample_weight,
        ensemble.tree_roots, ensemble.max_depth, cond_feats,
    )
    return Phi, phi_main, float(ensemble.base_offset)
