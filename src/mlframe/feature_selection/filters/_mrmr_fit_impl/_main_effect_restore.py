"""Main effects of a selected joint-cell composite, kept beside it (the hierarchy principle).

A joint cell table of three few-level operands holds every function of them, so the drop passes rightly find each operand "subsumed" by it. What they cannot see is that an operand
which carries most of the composite's information is the main effect the composite only modulates: the composite is then that operand plus an interaction on top of it, and a downstream
model reads the main effect directly from the operand but not from a nominal cell code. An operand that carries a small part of the information, whose association with the target
exists only because the conjunction contains it, is not a main effect and stays out.

The test is a break-even comparison on the debiased excesses the other passes use, with no threshold of its own: an operand ``X`` is restored when its marginal excess exceeds the excess
that the composite adds to ``X`` (``excess(X; y) > excess(cell; y | X)``), that is, when the composite is mostly ``X``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CELL_PREFIX = "cell3("


def run_main_effect_restore(self, X, selected_vars, cols, data, classes_y, verbose) -> "tuple[bool, list[int]]":
    """Re-attach the raw operands of a selected ``cell3`` composite that carry more of its information than it adds on top of them; returns ``(changed, selected_vars)``."""
    try:
        cells = [r for r in (self._engineered_recipes_ or []) if str(getattr(r, "name", "")).startswith(_CELL_PREFIX)]
        if not cells:
            return False, selected_vars
        from .._fe_raw_redundancy_helpers import _excess_and_floor
        from ._selection_compaction import _replayed_codes

        raw_names = [str(c) for c in self.feature_names_in_]
        index = {str(c): i for i, c in enumerate(cols)}
        selected = {raw_names[int(v)] for v in selected_vars if 0 <= int(v) < len(raw_names)}
        y = np.asarray(classes_y).ravel()
        seed = int(getattr(self, "random_seed", 0) or 0)
        restored: "list[str]" = []
        for rec in cells:
            cell = None
            for src in (str(s) for s in (getattr(rec, "src_names", None) or ())):
                if src in selected or src not in index or src not in raw_names:
                    continue
                if cell is None:
                    cell = _replayed_codes(rec, X)
                x = np.asarray(data[:, index[src]]).astype(np.int64)
                _, own_floor, own = _excess_and_floor(x, y, None, seed=seed)
                _, _, added = _excess_and_floor(cell, y, x, seed=seed)
                if own > own_floor and own > added:
                    restored.append(src)
                    selected.add(src)
        if not restored:
            return False, selected_vars
        selected_vars = [*selected_vars, *(raw_names.index(nm) for nm in restored)]
        self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) - set(restored)
        self.support_ = np.array(selected_vars, dtype=np.int64)
        self._main_effects_restored_ = list(restored)
        if verbose:
            logger.info("MRMR main-effect restore: %s carry more of the joint-cell composite's information than it adds on top of them; kept beside it.", restored)
        return True, selected_vars
    except Exception:
        logger.warning("MRMR main-effect restore failed; the selection is left as it was.", exc_info=True)
        return False, selected_vars
