"""Does a raw operand add information about y beyond the selected engineered feature that contains it?

Two post-selection passes put a raw operand back next to its engineered feature: the emit-both re-attach and the never-empty raw stand-in. Both used a
MARGINAL test (an uncorrected 32-permutation p-value, or no test at all: the largest marginal MI), so an operand the engineered feature already explains was
re-attached by chance. A 3-way XOR is the clean case: the operands carry no marginal signal, the composite carries all of it, and one of the three operands
was kept next to the composite. The question that decides whether a raw column is worth emitting is conditional: given the engineered feature, does the
operand still tell something about y.

The statistic is the plug-in I(operand; y | engineered feature) over the bins of the engineered feature, against a within-stratum permutation of the operand
(the same null the triplet scorer uses). An operand is kept only when it clears the null against EVERY engineered feature that contains it.
"""

from __future__ import annotations

import zlib

import numpy as np

_PERMUTATIONS = 199  # the smallest p-value is 1 / 200, below any alpha shared over a handful of operands
_MIN_STRATUM_ROWS = 30  # strata smaller than this carry no usable conditional estimate


def operand_adds_information(data, target_col: int, operand_col: int, owner_cols, alpha: float, seed: int, key: str) -> bool:
    """True when the operand's conditional MI given each owning engineered feature is significant at ``alpha``.

    ``data`` is the discretised factors matrix; ``operand_col``, ``owner_cols`` and ``target_col`` are column indices into it. ``key`` names the operand so
    its permutation stream does not depend on the order the operands are tested in.
    """
    from mlframe.feature_selection.filters._cmi_strata_njit import permutation_sources, stratified_cmi, stratified_cmi_null
    from mlframe.feature_selection.filters._orthogonal_triplet_fe import _dense_codes

    y = data[:, int(target_col)]
    for owner in owner_cols:
        x_d, y_d, s_d = _dense_codes(data[:, int(operand_col)], y, data[:, int(owner)])
        observed = float(stratified_cmi(x_d, y_d, s_d, _MIN_STRATUM_ROWS))
        rng = np.random.default_rng([int(seed), zlib.crc32(f"{key}|{owner}".encode())])
        null = stratified_cmi_null(x_d, y_d, s_d, permutation_sources(s_d, _PERMUTATIONS, rng), _MIN_STRATUM_ROWS)
        if (1.0 + float(np.count_nonzero(null >= observed))) / (_PERMUTATIONS + 1.0) >= alpha:
            return False  # this owner explains the operand
    return True


__all__ = ["operand_adds_information"]
