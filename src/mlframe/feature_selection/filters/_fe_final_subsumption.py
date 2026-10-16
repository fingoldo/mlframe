"""End-of-fit subsumption pass: drop a selected column that carries no information about the target beyond the real compounds.

The earlier stages each judge a candidate against the survivors they can see at that moment. A column admitted by an early stage can
end up redundant once a later stage has added the compound that carries its signal, and a later stage can re-admit a column an earlier
one dropped. This pass runs after every stage has finished, so the verdict is taken against the final set of multi-source compounds.

A column is dropped when the conditional information it carries about the target, given all the compounds, collapses to the
permutation floor (the same information-theoretic verdict the retention pass uses). Compounds themselves are never dropped, and with no
compound present nothing is dropped.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from ._fe_retention_subsumption import retention_form_is_subsumed


def final_subsumption_prune(
    candidates: Mapping[str, np.ndarray],
    anchors: Sequence[str],
    y_binned: np.ndarray,
    y_continuous: Optional[np.ndarray] = None,
    seed: int = 0,
) -> list[str]:
    """Names of the non-anchor ``candidates`` whose information about ``y`` is subsumed by the ``anchors`` compounds.

    ``candidates`` maps every selected column name to its full-n continuous values. ``anchors`` are the multi-source compounds that
    define the subsumption reference. Returns names in ``candidates`` order.
    """
    anchor_names = [a for a in anchors if a in candidates]
    if not anchor_names:
        return []
    anchor_values = [np.asarray(candidates[a], dtype=_crit_np_dtype()).ravel() for a in anchor_names]
    anchor_set = set(anchor_names)
    dropped: list[str] = []
    for name, values in candidates.items():
        if name in anchor_set:
            continue
        if retention_form_is_subsumed(
            cand_continuous=np.asarray(values, dtype=_crit_np_dtype()).ravel(),
            incumbent_continuous=anchor_values,
            y_binned=np.asarray(y_binned).ravel(),
            y_continuous=y_continuous,
            seed=seed,
        ):
            dropped.append(name)
    return dropped
