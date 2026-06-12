"""Fast custom path-dependent TreeSHAP for the SHAP-proxied feature selector.

Profiling showed OOF-SHAP is the wide-data hotspot (~96 s for 2000 features): the ``shap`` library's
``TreeExplainer`` re-walks Python/Cython per call and its ``GPUTreeExplainer`` is broken on our boxes.
This module is a faithful, self-contained re-implementation of Lundberg's *path-dependent* TreeSHAP
(Algorithm 2 of "Consistent Individualized Feature Attribution for Tree Ensembles", 2019) that:

  * extracts an xgboost booster's trees once into flat numba-friendly tensors (no per-call Python),
  * runs the per-sample polynomial EXTEND/UNWIND scan in numba, **parallel over samples** (``prange``),
  * reuses the precomputed tensors across every row and every fold,
  * returns ``(phi, base)`` with EXACT additivity ``base + phi.sum(1) == model margin`` (tree_path_dependent).

It matches ``shap.TreeExplainer(..., feature_perturbation="tree_path_dependent")`` to ~1e-4 (see the
parity test). The numba kernel is the always-available fallback; an optional cupy variant lives in
``_shap_proxy_treeshap_gpu`` and is selected by the dispatcher in ``_shap_proxy_explain`` purely by
data size + HW. ALL kernel versions are kept (never in-place replaced) so we can re-bench per HW.

Critical correctness invariants discovered while validating against xgboost 3.x:
  * Feature values are compared as **float32** (xgboost casts internally); using float64 misroutes a
    handful of rows near a split threshold and breaks additivity. We cast X to float32 for routing.
  * Routing is ``x < split_condition -> "yes"/left``; NaN follows the node's ``missing`` child.
  * The path-dependent conditional-expectation weight is the node ``cover`` (hessian-weighted sample
    count) ratio child/parent -- exactly what ``tree_path_dependent`` uses.

Scope: xgboost ``XGBRegressor`` / ``XGBClassifier`` (binary) AND lightgbm ``LGBMRegressor`` /
``LGBMClassifier`` (binary), single output, margin space. Both families map onto the same flat node
tensors so the kernels are shared unchanged (lightgbm's ``<=`` routing and sample-count cover are
normalised in its extractor). Other model types fall back to the ``shap`` library via the dispatcher.
SHAP *interaction* values have their own shared-tensor kernel in ``_shap_proxy_treeshap_interactions``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange

# Sentinel for "no child" (leaf) in the flat children arrays.
_NO_CHILD = -1


@dataclass
class TreeEnsemble:
    """Flat, contiguous per-node tensors for an entire ensemble, concatenated tree-after-tree.

    All arrays are indexed by a GLOBAL node id (running offset across trees). ``tree_offsets`` gives
    the global id of each tree's root so the kernel can iterate tree-by-tree without ragged arrays.
    """

    children_left: np.ndarray   # (N,) int32  global id of "yes"/left child, or _NO_CHILD for leaf
    children_right: np.ndarray  # (N,) int32  global id of "no"/right child
    children_default: np.ndarray  # (N,) int32 global id of "missing" child
    features: np.ndarray        # (N,) int32  split feature index, or -1 for leaf
    thresholds: np.ndarray      # (N,) float32 split_condition (float32 to match xgboost routing)
    values: np.ndarray          # (N,) float64 leaf output (0.0 for internal nodes)
    node_sample_weight: np.ndarray  # (N,) float64 node cover (hessian-weighted count)
    tree_roots: np.ndarray      # (T,) int32  global id of each tree's root node
    max_depth: int              # max tree depth across the ensemble (sizes the per-thread scratch)
    base_offset: float          # ensemble intercept (base_score) added once to the margin
    n_features: int


def _extract_xgboost_ensemble(booster, n_features: int) -> TreeEnsemble:
    """Parse an xgboost booster's JSON dump into flat per-node tensors (done once per fit)."""
    cfg = json.loads(booster.save_config())
    lp = cfg["learner"]["learner_model_param"]
    # base_score is serialised like "[3E-1]"; strip the brackets.
    base_raw = str(lp.get("base_score", "0")).strip("[]")
    base_offset = float(base_raw)
    # For logistic objectives xgboost stores base_score in PROBABILITY space but leaf values and the
    # output margin live in LOG-ODDS space, so the margin intercept is logit(base_score). Detect the
    # objective from the config and transform; regression objectives keep base_score as-is.
    objective = str(cfg["learner"]["objective"].get("name", "")).lower()
    if "logistic" in objective or "logitraw" in objective or objective.startswith("binary:"):
        p = min(max(base_offset, 1e-7), 1.0 - 1e-7)
        base_offset = float(np.log(p / (1.0 - p)))

    fmap = {name: i for i, name in enumerate(booster.feature_names)} if booster.feature_names else None
    dumps = booster.get_dump(dump_format="json", with_stats=True)

    cl, cr, cd, feat, thr, val, cover = [], [], [], [], [], [], []
    tree_roots = []
    max_depth = 0

    for raw in dumps:
        root = json.loads(raw)
        base = len(cl)
        tree_roots.append(base)
        # Flatten this tree depth-first; map local nodeid -> global id via a per-tree dict.
        # First pass: collect nodes by nodeid so children can be resolved to global ids.
        nodes = {}

        def _collect(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            nodes[node["nodeid"]] = node
            for ch in node.get("children", []):
                _collect(ch, depth + 1)

        _collect(root, 0)
        local_ids = sorted(nodes.keys())
        local_to_global = {nid: base + k for k, nid in enumerate(local_ids)}
        for nid in local_ids:
            node = nodes[nid]
            if "leaf" in node:
                cl.append(_NO_CHILD)
                cr.append(_NO_CHILD)
                cd.append(_NO_CHILD)
                feat.append(-1)
                thr.append(0.0)
                val.append(float(node["leaf"]))
                cover.append(float(node.get("cover", 1.0)))
            else:
                cl.append(local_to_global[node["yes"]])
                cr.append(local_to_global[node["no"]])
                cd.append(local_to_global[node["missing"]])
                sp = node["split"]
                fi = fmap[sp] if (fmap is not None and sp in fmap) else int(str(sp).lstrip("f"))
                feat.append(int(fi))
                thr.append(float(node["split_condition"]))
                val.append(0.0)
                cover.append(float(node.get("cover", 1.0)))

    children_left = np.asarray(cl, dtype=np.int32)
    children_right = np.asarray(cr, dtype=np.int32)
    values = np.asarray(val, dtype=np.float64)
    node_sample_weight = np.asarray(cover, dtype=np.float64)
    tree_roots_arr = np.asarray(tree_roots, dtype=np.int32)

    # SHAP's tree_path_dependent base value is NOT base_score: it is base_score plus, per tree, the
    # cover-weighted mean leaf value (the tree's unconditional expected output). phi measures the
    # deviation from THAT conditional expectation, so additivity requires this base. Compute each
    # tree's root expectation E[root] = (cover_L*E[L] + cover_R*E[R]) / cover_node bottom-up.
    expected = base_offset + _ensemble_expected_value(
        children_left, children_right, values, node_sample_weight, tree_roots_arr)

    return TreeEnsemble(
        children_left=children_left,
        children_right=children_right,
        children_default=np.asarray(cd, dtype=np.int32),
        features=np.asarray(feat, dtype=np.int32),
        thresholds=np.asarray(thr, dtype=np.float32),
        values=values,
        node_sample_weight=node_sample_weight,
        tree_roots=tree_roots_arr,
        max_depth=int(max_depth),
        base_offset=float(expected),
        n_features=int(n_features),
    )


def _ensemble_expected_value(children_left, children_right, values, cover, tree_roots) -> float:
    """Sum over trees of the cover-weighted mean leaf value (each tree's unconditional expectation).

    Iterative post-order via the fact that within a tree global ids are emitted root-first in sorted
    nodeid order, so a child always has a higher global id than its parent -> a single reverse pass
    propagates leaf expectations upward. Returns the summed root expectation across all trees.
    """
    n_nodes = children_left.shape[0]
    exp = values.astype(np.float64).copy()  # leaves already hold their value; internals overwritten
    # Reverse pass: when we reach an internal node, its children are already finalised.
    for nid in range(n_nodes - 1, -1, -1):
        lc = children_left[nid]
        if lc == _NO_CHILD:
            continue
        rc = children_right[nid]
        c = cover[nid]
        if c > 0.0:
            exp[nid] = (cover[lc] * exp[lc] + cover[rc] * exp[rc]) / c
        else:
            exp[nid] = 0.0
    total = 0.0
    for t in range(tree_roots.shape[0]):
        total += exp[tree_roots[t]]
    return float(total)


# --------------------------------------------------------------------------------------------------
# numba kernel: Lundberg path-dependent TreeSHAP (Algorithm 2), sample-parallel via prange.
#
# This mirrors the reference shap C++ ``tree_shap_recursive``: per (sample, tree) we recurse root->leaf
# carrying a "path" of unique features seen, each entry holding the fraction of "ones" (feature in the
# conditioning set) and "zeros" (not) paths plus a polynomial weight ``pweight``. EXTEND grows the
# path on descent; UNWIND removes a feature that recurs deeper; at a leaf we attribute, for each path
# feature, the include-minus-exclude weighted leaf value. Summed over trees this gives the exact
# tree_path_dependent attribution with base + phi.sum == margin.
#
# The path is stored in a single flat scratch buffer; recursion level ``L`` owns a fresh window so the
# parent's path is preserved across the two child calls. Each level gets a fixed (max_path+2) slot so
# offset arithmetic stays O(1): off = level * (max_path + 2).
# --------------------------------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _extend_path(pf_feat, pf_zero, pf_one, pweight, off, unique_depth, zero_frac, one_frac, feat_idx):
    """EXTEND: append a split (``feat_idx``) at position ``unique_depth`` and update the polynomial."""
    pf_feat[off + unique_depth] = feat_idx
    pf_zero[off + unique_depth] = zero_frac
    pf_one[off + unique_depth] = one_frac
    pweight[off + unique_depth] = 1.0 if unique_depth == 0 else 0.0
    i = unique_depth - 1
    while i >= 0:
        pweight[off + i + 1] += one_frac * pweight[off + i] * (i + 1.0) / (unique_depth + 1.0)
        pweight[off + i] = zero_frac * pweight[off + i] * (unique_depth - i) / (unique_depth + 1.0)
        i -= 1


@njit(cache=True, inline="always")
def _unwind_path(pf_feat, pf_zero, pf_one, pweight, off, unique_depth, path_index):
    """UNWIND: remove the split at ``path_index`` in place (inverse of EXTEND)."""
    one_frac = pf_one[off + path_index]
    zero_frac = pf_zero[off + path_index]
    next_one = pweight[off + unique_depth]
    i = unique_depth - 1
    while i >= 0:
        if one_frac != 0.0:
            tmp = pweight[off + i]
            pweight[off + i] = next_one * (unique_depth + 1.0) / ((i + 1.0) * one_frac)
            next_one = tmp - pweight[off + i] * zero_frac * (unique_depth - i) / (unique_depth + 1.0)
        else:
            pweight[off + i] = pweight[off + i] * (unique_depth + 1.0) / (zero_frac * (unique_depth - i))
        i -= 1
    # Shift the path entries above ``path_index`` down by one.
    i = path_index
    while i < unique_depth:
        pf_feat[off + i] = pf_feat[off + i + 1]
        pf_zero[off + i] = pf_zero[off + i + 1]
        pf_one[off + i] = pf_one[off + i + 1]
        i += 1


@njit(cache=True, inline="always")
def _unwound_sum(pf_zero, pf_one, pweight, off, unique_depth, path_index):
    """Sum of path weights as if the split at ``path_index`` were UNWOUND (non-destructive)."""
    one_frac = pf_one[off + path_index]
    zero_frac = pf_zero[off + path_index]
    next_one = pweight[off + unique_depth]
    total = 0.0
    if one_frac != 0.0:
        i = unique_depth - 1
        while i >= 0:
            tmp = next_one / ((i + 1.0) * one_frac)
            total += tmp
            next_one = pweight[off + i] - tmp * zero_frac * (unique_depth - i)
            i -= 1
        total *= (unique_depth + 1.0)
    else:
        i = unique_depth - 1
        while i >= 0:
            total += pweight[off + i] / (zero_frac * (unique_depth - i))
            i -= 1
        total *= (unique_depth + 1.0)
    return total


@njit(cache=True)
def _treeshap_one_tree(
    x, phi_row,
    children_left, children_right, children_default,
    features, thresholds, values, node_sample_weight,
    root, width,
    pf_feat, pf_zero, pf_one, pweight,
    st_node, st_level, st_ud, st_zero, st_one, st_feat,
):
    """Iterative (explicit-stack) descent for one tree -- numba recursion miscompiles under
    parallel=True+cache=True, so we manage the DFS frames ourselves. ``level`` selects each frame's
    path window; the parent window (``level-1``) is copied in so siblings start from the same prefix.
    Hot child (the one x routes to) is pushed last so it is processed first (matches the reference)."""
    sp = 0
    st_node[sp] = root
    st_level[sp] = 0
    st_ud[sp] = 0
    st_zero[sp] = 1.0
    st_one[sp] = 1.0
    st_feat[sp] = -1
    sp += 1

    while sp > 0:
        sp -= 1
        node = st_node[sp]
        level = st_level[sp]
        unique_depth = st_ud[sp]
        zero_frac = st_zero[sp]
        one_frac = st_one[sp]
        feat_idx = st_feat[sp]
        off = level * width

        # Copy the parent's path prefix into this level's window, then EXTEND with the incoming split.
        if level > 0:
            poff = (level - 1) * width
            for k in range(unique_depth):
                pf_feat[off + k] = pf_feat[poff + k]
                pf_zero[off + k] = pf_zero[poff + k]
                pf_one[off + k] = pf_one[poff + k]
                pweight[off + k] = pweight[poff + k]
        _extend_path(pf_feat, pf_zero, pf_one, pweight, off, unique_depth, zero_frac, one_frac, feat_idx)

        if children_left[node] == _NO_CHILD:
            leaf_val = values[node]
            i = 1
            while i <= unique_depth:
                w = _unwound_sum(pf_zero, pf_one, pweight, off, unique_depth, i)
                phi_row[pf_feat[off + i]] += w * (pf_one[off + i] - pf_zero[off + i]) * leaf_val
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
        next_unique = unique_depth + 1
        if path_index != 0:
            incoming_zero = pf_zero[off + path_index]
            incoming_one = pf_one[off + path_index]
            _unwind_path(pf_feat, pf_zero, pf_one, pweight, off, unique_depth, path_index)
            next_unique = unique_depth

        # Push cold then hot so hot (popped last) is processed first.
        st_node[sp] = cold
        st_level[sp] = level + 1
        st_ud[sp] = next_unique
        st_zero[sp] = incoming_zero * cold_w
        st_one[sp] = 0.0
        st_feat[sp] = f
        sp += 1
        st_node[sp] = hot
        st_level[sp] = level + 1
        st_ud[sp] = next_unique
        st_zero[sp] = incoming_zero * hot_w
        st_one[sp] = incoming_one
        st_feat[sp] = f
        sp += 1


@njit(cache=True, parallel=True)
def _treeshap_batch(
    X, phi,
    children_left, children_right, children_default,
    features, thresholds, values, node_sample_weight,
    tree_roots, max_path,
):
    """Fill ``phi`` (n, f) for all samples, parallel over rows. Excludes the base offset."""
    n = X.shape[0]
    n_trees = tree_roots.shape[0]
    width = max_path + 2          # entries available to each path window
    n_levels = max_path + 2       # root..deepest leaf
    stack_size = 2 * (max_path + 2)  # DFS stack depth bound (two children per pushed level)
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
        st_feat = np.empty(stack_size, dtype=np.int64)
        xi = X[i]
        phi_i = phi[i]
        for t in range(n_trees):
            _treeshap_one_tree(
                xi, phi_i, children_left, children_right, children_default,
                features, thresholds, values, node_sample_weight,
                tree_roots[t], width,
                pf_feat, pf_zero, pf_one, pweight,
                st_node, st_level, st_ud, st_zero, st_one, st_feat,
            )


def treeshap_phi_base_numba(ensemble: TreeEnsemble, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Run the numba path-dependent TreeSHAP. Returns ``(phi (n,f) float64, base float)``.

    ``X`` is cast to float32 for routing (xgboost's internal comparison dtype). ``phi`` is in margin /
    log-odds space; ``base`` is the ensemble intercept so ``base + phi.sum(1)`` is the model margin.
    """
    Xf = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    n = Xf.shape[0]
    phi = np.zeros((n, ensemble.n_features), dtype=np.float64)
    _treeshap_batch(
        Xf, phi,
        ensemble.children_left, ensemble.children_right, ensemble.children_default,
        ensemble.features, ensemble.thresholds, ensemble.values, ensemble.node_sample_weight,
        ensemble.tree_roots, ensemble.max_depth,
    )
    return phi, float(ensemble.base_offset)


def _extract_lightgbm_ensemble(booster, n_features: int) -> TreeEnsemble:
    """Parse a lightgbm booster's ``dump_model`` JSON into the SAME flat per-node tensors as xgboost.

    Two structural differences from xgboost are normalised here so the UNMODIFIED numba/cupy kernels
    apply unchanged:
      * LightGBM routes ``x <= threshold -> left``; the kernel routes ``x < threshold -> left``. We
        convert each threshold via ``nextafter(t, +inf)`` so ``x < nextafter(t)`` iff ``x <= t`` in the
        float32 routing dtype the kernel uses (verified exact: see the parity test).
      * The path-dependent cover is the node sample count (``internal_count`` / ``leaf_count``), the
        lightgbm analogue of xgboost's hessian-weighted ``cover``.
    LightGBM folds its ``boost_from_average`` initial score into the first tree's leaves, so the base
    offset is just the cover-weighted ensemble expectation (no separate intercept term) -- matching
    ``shap``'s ``expected_value`` to machine precision (see the additivity test)."""
    md = booster.dump_model()
    cl, cr, cd, feat, thr, val, cover = [], [], [], [], [], [], []
    tree_roots = []
    max_depth = 0

    for tinfo in md["tree_info"]:
        root = tinfo["tree_structure"]
        base = len(cl)
        tree_roots.append(base)
        # DFS pre-order: parent emitted before children so global ids increase down the tree (the
        # invariant ``_ensemble_expected_value`` relies on). Collect nodes, assigning each a global id.
        flat = []

        def _collect(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            node["_gid"] = base + len(flat)
            flat.append(node)
            if "left_child" in node:
                _collect(node["left_child"], depth + 1)
                _collect(node["right_child"], depth + 1)

        _collect(root, 0)
        for node in flat:
            if "leaf_value" in node:
                cl.append(_NO_CHILD)
                cr.append(_NO_CHILD)
                cd.append(_NO_CHILD)
                feat.append(-1)
                thr.append(0.0)
                val.append(float(node["leaf_value"]))
                cover.append(float(node.get("leaf_count", 1.0)))
            else:
                lc = node["left_child"]["_gid"]
                rc = node["right_child"]["_gid"]
                cl.append(lc)
                cr.append(rc)
                feat.append(int(node["split_feature"]))
                # Convert ``x <= t`` to the kernel's ``x < t'`` via nextafter (float32 routing dtype).
                t = np.float32(float(node["threshold"]))
                thr_le = float(np.nextafter(t, np.float32(np.inf)))
                thr.append(thr_le)
                # NaN routing depends on lightgbm's ``missing_type``: only "NaN" honours ``default_left``;
                # "None"/"Zero" treat a missing value AS 0.0 (and route it by the threshold) -- so we bake
                # that by pointing children_default at whichever child 0.0 routes to (the kernel always
                # sends NaN to children_default, so this reproduces lightgbm's "NaN as 0" semantics).
                missing_type = str(node.get("missing_type", "None"))
                if missing_type == "NaN":
                    cd.append(lc if node.get("default_left", True) else rc)
                else:
                    cd.append(lc if (0.0 < thr_le) else rc)
                val.append(0.0)
                cover.append(float(node.get("internal_count", 1.0)))

    children_left = np.asarray(cl, dtype=np.int32)
    children_right = np.asarray(cr, dtype=np.int32)
    values = np.asarray(val, dtype=np.float64)
    node_sample_weight = np.asarray(cover, dtype=np.float64)
    tree_roots_arr = np.asarray(tree_roots, dtype=np.int32)
    base_offset = _ensemble_expected_value(
        children_left, children_right, values, node_sample_weight, tree_roots_arr)

    return TreeEnsemble(
        children_left=children_left,
        children_right=children_right,
        children_default=np.asarray(cd, dtype=np.int32),
        features=np.asarray(feat, dtype=np.int32),
        thresholds=np.asarray(thr, dtype=np.float32),
        values=values,
        node_sample_weight=node_sample_weight,
        tree_roots=tree_roots_arr,
        max_depth=int(max_depth),
        base_offset=float(base_offset),
        n_features=int(n_features),
    )


def is_supported_xgboost(estimator) -> bool:
    """True for a fitted xgboost regressor / binary classifier we can extract a booster from."""
    name = type(estimator).__name__
    if name not in ("XGBRegressor", "XGBClassifier"):
        return False
    try:
        booster = estimator.get_booster()
    except Exception:
        return False
    if booster is None:
        return False
    # Multiclass (num_class > 2) is out of scope for v1 (single-output coalition margin).
    try:
        cfg = json.loads(booster.save_config())
        n_class = int(cfg["learner"]["learner_model_param"].get("num_class", "0"))
        if n_class > 2:
            return False
    except Exception:
        pass
    return True


def is_supported_lightgbm(estimator) -> bool:
    """True for a fitted lightgbm regressor / binary classifier we can extract a booster from.

    Multiclass (``num_tree_per_iteration`` > 1 / ``num_class`` > 1) is out of scope (single-output
    coalition margin), mirroring the xgboost gate."""
    name = type(estimator).__name__
    if name not in ("LGBMRegressor", "LGBMClassifier", "Booster"):
        return False
    try:
        booster = estimator if name == "Booster" else estimator.booster_
    except Exception:
        return False
    if booster is None:
        return False
    try:
        md = booster.dump_model()
        if int(md.get("num_tree_per_iteration", 1)) > 1 or int(md.get("num_class", 1)) > 1:
            return False
    except Exception:
        return False
    return True


def _lightgbm_booster_and_nfeatures(estimator):
    booster = estimator if type(estimator).__name__ == "Booster" else estimator.booster_
    try:
        n_features = int(estimator.n_features_in_)
    except Exception:
        try:
            n_features = int(booster.num_feature())
        except Exception:
            n_features = int(booster.dump_model().get("max_feature_idx", 0)) + 1
    return booster, n_features


def extract_ensemble(estimator) -> Optional[TreeEnsemble]:
    """Extract the flat ensemble tensors from a supported xgboost OR lightgbm estimator, else ``None``.

    Both model families map onto the SAME flat node tensors so the numba/cupy kernels are shared
    unchanged; lightgbm's ``<=`` routing and sample-count cover are normalised inside its extractor."""
    if is_supported_xgboost(estimator):
        booster = estimator.get_booster()
        try:
            n_features = int(estimator.n_features_in_)
        except Exception:
            n_features = len(booster.feature_names) if booster.feature_names else 0
        return _extract_xgboost_ensemble(booster, n_features)
    if is_supported_lightgbm(estimator):
        booster, n_features = _lightgbm_booster_and_nfeatures(estimator)
        return _extract_lightgbm_ensemble(booster, n_features)
    return None
