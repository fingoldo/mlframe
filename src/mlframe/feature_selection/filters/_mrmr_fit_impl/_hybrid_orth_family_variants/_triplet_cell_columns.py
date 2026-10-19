"""Joint-cell columns for the triplets the cross-basis triplet stage finds over few-level raw columns.

The stage engineers a polynomial product of three standardised legs. For three binary legs that product is the parity of the three, so a conjunction
of them (A AND B AND C) is invisible to it and the stage's winner carries no usable signal. The joint cell table keeps every combination apart and
so holds whatever any function of the three legs can carry; it is appended next to the product for the triples the stage already chose, and only when it carries information the product
does not: for a three-way XOR the parity IS the target, the cell table adds nothing, and the simpler product stands alone.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..._fe_frame_ops import fe_extract_columns, fe_is_numeric_col

logger = logging.getLogger(__name__)


def _legs(name: str) -> "list[str]":
    """Operand names of a ``a*b*c__...`` triplet column name."""
    return name.split("__", 1)[0].split("*")


def triplet_cell_columns(X: Any, triplet_names: "list[str]", y_codes: Any, seed: int = 0) -> "tuple[dict[str, np.ndarray], dict[str, Any]]":
    """``({cell3 name: codes}, {cell3 name: recipe})`` for the triplets in ``triplet_names`` whose three legs are raw numeric columns of ``X`` with few levels.

    A triplet with a leg that is missing, non-numeric or has too many levels yields nothing, and so does one whose cell table adds no information about ``y_codes`` beyond the triplet's own
    product (conditional information above the within-stratum permutation null); a cell column already in ``X`` is not rebuilt.
    """
    from .._selection_compaction import _synthesized_cell_triple
    from ..._fe_raw_redundancy_helpers import _excess_and_floor
    from ..._fe_usability_signal import _crit_np_dtype
    from ..._orthogonal_triplet_fe import _quantile_codes

    y = np.asarray(y_codes).ravel()
    columns: "dict[str, np.ndarray]" = {}
    recipes: "dict[str, Any]" = {}
    present = set(X.columns)
    for nm in triplet_names:
        legs = _legs(nm)
        if len(legs) != 3 or len(set(legs)) != 3 or not all(c in present and fe_is_numeric_col(X, c) for c in legs):
            continue
        cell_name = "cell3(" + ",".join(sorted(legs)) + ")"
        if cell_name in present or cell_name in columns:
            continue
        frame = pd.DataFrame(fe_extract_columns(X, legs))
        made = _synthesized_cell_triple(frame, tuple(legs))
        if made is None:
            continue
        name, rec, codes = made
        product = _quantile_codes(np.nan_to_num(np.asarray(fe_extract_columns(X, [nm])[nm], dtype=_crit_np_dtype())), 10)
        cmi, floor, excess = _excess_and_floor(np.asarray(codes, dtype=np.int64), y, product, seed=seed)
        if not (cmi > floor and excess >= floor):
            continue
        columns[name] = np.asarray(codes, dtype=_crit_np_dtype())
        recipes[name] = rec
    return columns, recipes
