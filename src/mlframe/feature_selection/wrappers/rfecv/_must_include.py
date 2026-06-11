"""``must_include`` partition for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s pre-while setup. Validates that every
``self.must_include`` entry is present in ``X`` (pandas: by name;
ndarray: by integer index in range), splits the feature universe into
the pinned set + the optimiser's search-complement, and warns when
``must_include`` exhausts every feature.

Re-imported at the parent's module bottom so historical
``from ._fit import _resolve_must_include`` keeps resolving
transparently.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _resolve_must_include(self, *, X, original_features, verbose):
    """Return ``(original_features_minus_pinned, must_include_resolved)``.

    Also assigns ``self._must_include_resolved`` for downstream readers.
    """
    must_include_resolved: list = []
    if self.must_include:
        if isinstance(X, pd.DataFrame):
            missing = [m for m in self.must_include if m not in original_features]
        else:
            # ndarray: must_include must be integer indices in [0, n_cols)
            p = X.shape[1]
            missing = [m for m in self.must_include if not (isinstance(m, (int, np.integer)) and 0 <= int(m) < p)]
        if missing:
            raise ValueError(
                f"must_include contains entries not in X: {missing}. "
                f"Available: {list(original_features)[:20]}..."
            )
        must_include_resolved = list(self.must_include)
        # Remove from search universe; the optimiser explores only the
        # COMPLEMENT. Final support_ = must_include + optimiser's pick.
        original_features = [c for c in original_features if c not in must_include_resolved]
        if verbose:
            logger.info(
                "must_include: %d feature(s) pinned, %d searched.",
                len(must_include_resolved), len(original_features),
            )
        if len(original_features) == 0:
            logger.warning(
                "must_include exhausts every feature in X; nothing for "
                "the optimiser to pick. Fitting on the pinned set only."
            )
    self._must_include_resolved = must_include_resolved
    return original_features, must_include_resolved
