"""One-line INFO summary of what a finished ``MRMR.fit`` selected (raw count of N inputs, engineered extras, first 30 names)."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.feature_selection._selection_log import log_selection

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def log_mrmr_fit_summary(mrmr: Any, elapsed_s: float) -> None:
    """Emit ``MRMR: selected K of N features (+E engineered) in T s: [names]`` at INFO; silent when the fit left no single-output ``support_`` (multi-output)."""
    try:
        support = getattr(mrmr, "support_", None)
        n_in = getattr(mrmr, "n_features_in_", None)
        if support is None or n_in is None or not hasattr(mrmr, "feature_names_in_"):
            return
        names = [str(n) for n in mrmr.get_feature_names_out()]
        support = np.asarray(support)
        n_raw = int(support.sum()) if support.dtype == bool else int(support.size)  # bool mask or integer indices
        log_selection(logger, "MRMR", n_raw, int(n_in), names, elapsed=elapsed_s, extra=f"+{len(names) - n_raw} engineered", respect_quiet=True)
    except Exception as exc:
        logger.debug("MRMR fit summary failed: %r", exc)
