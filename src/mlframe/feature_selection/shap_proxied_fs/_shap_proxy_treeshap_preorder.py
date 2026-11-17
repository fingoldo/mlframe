"""Static preorder records of a tree ensemble for the device table TreeSHAP kernels.

The table form visits every node of every tree for every sample, so the visiting order is the same for all samples (left child first): the nodes can be laid out in that order once and a kernel can walk the
ensemble as a flat loop over the records instead of a per-thread stack of node ids. The addresses of the loads then do not depend on the previous step, the loads of a warp are broadcasts of one 16-byte record,
and the stack of the walk shrinks to one mask per depth for each child side.

Record of one node (``width`` int32 columns, the first four are the 16-byte record the kernels read):
``x = depth | is_left << 8 | is_leaf << 9 | default_left << 10``; split: ``y`` feature, ``z`` float32 threshold bits, ``w`` feature slot; leaf: ``y`` unique-feature count, ``z`` offset of its features, ``w`` offset of its
table. With ``width`` 5 the fifth column is the offset of the leaf's pair table (the interaction kernel).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

from ._shap_proxy_treeshap import TreeEnsemble


@njit(cache=True)
def _preorder_nodes(children_left, children_right, tree_roots, max_depth):
    """Node ids of every tree in left-first preorder with their depths and left-child flags, trees one after another."""
    n_nodes = children_left.shape[0]
    order = np.empty(n_nodes, dtype=np.int64)
    depth = np.empty(n_nodes, dtype=np.int64)
    is_left = np.empty(n_nodes, dtype=np.int64)
    stack_node = np.empty(2 * max_depth + 8, dtype=np.int64)
    stack_depth = np.empty(2 * max_depth + 8, dtype=np.int64)
    stack_left = np.empty(2 * max_depth + 8, dtype=np.int64)
    k = 0
    for t in range(tree_roots.shape[0]):
        sp = 0
        stack_node[sp] = tree_roots[t]
        stack_depth[sp] = 0
        stack_left[sp] = 0
        sp += 1
        while sp > 0:
            sp -= 1
            node = stack_node[sp]
            order[k] = node
            depth[k] = stack_depth[sp]
            is_left[k] = stack_left[sp]
            k += 1
            if children_left[node] >= 0:
                stack_node[sp] = children_right[node]
                stack_depth[sp] = depth[k - 1] + 1
                stack_left[sp] = 0
                sp += 1
                stack_node[sp] = children_left[node]
                stack_depth[sp] = depth[k - 1] + 1
                stack_left[sp] = 1
                sp += 1
    return order[:k], depth[:k], is_left[:k]


def preorder_records(ensemble: TreeEnsemble, tab: dict, width: int = 4, pair_off: Optional[np.ndarray] = None) -> np.ndarray:
    """``(n_nodes, width)`` int32 records of ``ensemble`` in static preorder (see the module docstring); ``tab`` is a ``build_tables`` result, ``pair_off`` the per-leaf pair-table offsets for ``width`` 5."""
    order, depth, is_left = _preorder_nodes(ensemble.children_left, ensemble.children_right, ensemble.tree_roots, int(ensemble.max_depth))
    cl = ensemble.children_left[order]
    leaf = cl < 0
    rec = np.zeros((order.shape[0], width), dtype=np.int32)
    rec[:, 0] = depth | (is_left << 8) | (leaf.astype(np.int64) << 9) | ((ensemble.children_default[order] == cl).astype(np.int64) << 10)
    inner = ~leaf
    rec[inner, 1] = ensemble.features[order[inner]]
    rec[inner, 2] = np.ascontiguousarray(ensemble.thresholds, dtype=np.float32).view(np.int32)[order[inner]]
    rec[inner, 3] = tab["node_slot"][order[inner]]
    lid = tab["leaf_of_node"][order[leaf]]
    rec[leaf, 1] = tab["leaf_ud"][lid]
    rec[leaf, 2] = tab["leaf_feat_off"][lid]
    rec[leaf, 3] = tab["leaf_table_off"][lid]
    if width > 4:
        if pair_off is None:
            raise ValueError("width > 4 needs the per-leaf pair-table offsets")
        rec[leaf, 4] = pair_off[lid]
    return rec
