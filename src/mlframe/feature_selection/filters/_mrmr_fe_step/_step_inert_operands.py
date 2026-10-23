"""Engineered pair candidates with an inert raw leaf do not enter the selection pool.

A candidate such as ``max(rint(c),neg(e))`` can pass the pair gates while one of its raw leaves does nothing: on a clipped ``c`` the ``max`` returns ``rint(c)`` on almost every row, and permuting ``e``
leaves its information about the target unchanged. Admitted, it is selected, fed forward as an operand of a composite that is chosen for its other operand, and the finished selection references a noise
column.

A leaf ``r`` of a candidate ``f`` is inert when ``f`` carries no information about ``r`` beyond what the other raw leaves of ``f`` already give: ``CMI(r; f | other leaves)`` does not clear its
within-stratum permutation floor (the test the raw-redundancy sweep applies to a raw against its children). Only candidates with two or more raw leaves are tested, and only the candidates a pair
already admitted, so the cost is a handful of conditional nulls per pair.
"""

from __future__ import annotations

import logging
from mlframe.utils.log_throttle import log_throttle
import re
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[^A-Za-z0-9_.]+")


def _raw_leaves(name: str, raw_name_set: "set[str]") -> "list[str]":
    """The distinct raw column names that appear as tokens of the engineered column name, in order of appearance."""
    out: "list[str]" = []
    for token in _TOKEN.split(str(name)):
        if token in raw_name_set and token not in out:
            out.append(token)
    return out


def _has_inert_leaf(leaves: "list[str]", name_to_idx: "dict[str, int]", data: Any, codes: np.ndarray, seed: int) -> "tuple[str, float, float] | None":
    """``(leaf, cmi, floor)`` of the first raw leaf of the candidate (given by its binned ``codes``) that is conditionally independent of it given the other leaves, else ``None``."""
    from .._fe_raw_redundancy_helpers import _excess_and_floor
    from .._mi_greedy_cmi_fe import _renumber_joint

    n_cols = int(data.shape[1])
    for leaf in leaves:
        li = name_to_idx.get(leaf)
        others = [name_to_idx[o] for o in leaves if o != leaf and o in name_to_idx]
        if li is None or li >= n_cols or not others or any(o >= n_cols for o in others):
            continue
        z, _card = _renumber_joint(*[np.ascontiguousarray(data[:, o]) for o in others])
        cmi, floor, _excess = _excess_and_floor(np.ascontiguousarray(data[:, li]), np.ascontiguousarray(codes), z, seed=seed)
        if not cmi > floor:
            logger.debug("FE admission: leaf %r is inert for the candidate (CMI %.5f does not clear its floor %.5f).", leaf, cmi, floor)
            return leaf, float(cmi), float(floor)
    return None


def filter_inert_leaf_block(
    self: Any, this_pair_features: list, transformed_vals: np.ndarray, new_cols: list, new_nbins: Any, cols: list, data: Any, discretise: Callable[[np.ndarray], np.ndarray], verbose: int
) -> tuple:
    """Drop the candidates of one pair block that have an inert raw leaf; same return shape and re-indexing as ``_filter_unreplayable_pair_block``.

    ``data`` is the binned matrix that holds the raw columns (named by ``cols``); ``discretise(values)`` bins a ``(n, k)`` block of candidate values into the codes the pool uses. With no ``data``
    (or on any failure of the test) the block is returned unchanged.
    """
    if data is None or not this_pair_features:
        return this_pair_features, transformed_vals, new_cols, new_nbins, this_pair_features
    raw_names = set(getattr(self, "feature_names_in_", []))
    name_to_idx = {str(c): i for i, c in enumerate(cols)}
    seed = int(getattr(self, "random_seed", 0) or 0)
    keep_positions = []
    for pos, (_config, j) in enumerate(this_pair_features):
        leaves = _raw_leaves(new_cols[j], raw_names)
        if len(leaves) >= 2:
            try:
                codes = np.asarray(discretise(np.asarray(transformed_vals)[:, [j]]))[:, 0]
                inert = _has_inert_leaf(leaves, name_to_idx, data, codes, seed)
                if inert is not None:
                    from .._fe_rejection_ledger import record_fe_rejection

                    record_fe_rejection(
                        self, gate="inert_leaf", candidate=new_cols[j], operands=tuple(leaves), operator="pair", observed=inert[1], threshold=inert[2],
                        reason=f"leaf {inert[0]} carries no information about the candidate beyond the other leaves",
                    )
                    continue
            except Exception as exc:
                log_throttle(logger, "fe_inert_leaf_test_failed", logging.WARNING, "FE admission: the inert-leaf test of %r failed (%s: %s); the candidate stays.", new_cols[j], type(exc).__name__, exc)
        keep_positions.append(pos)
    if len(keep_positions) == len(this_pair_features):
        return this_pair_features, transformed_vals, new_cols, new_nbins, this_pair_features
    if verbose:
        logger.info("MRMR FE pair stage: excluded %d candidate(s) with an inert raw leaf before selection.", len(this_pair_features) - len(keep_positions))
    kept_original = [this_pair_features[pos] for pos in keep_positions]
    column_idx = [this_pair_features[pos][1] for pos in keep_positions]
    filtered_features = [(this_pair_features[pos][0], new_j) for new_j, pos in enumerate(keep_positions)]
    filtered_vals = transformed_vals[:, column_idx] if column_idx else transformed_vals[:, :0]
    filtered_cols = [new_cols[j] for j in column_idx]
    filtered_nbins = np.asarray(new_nbins)[column_idx]
    return filtered_features, filtered_vals, filtered_cols, filtered_nbins, kept_original
