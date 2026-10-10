"""Step 1 of ``screen_predictors`` (subsample-index application), carved out of ``_screen_predictors`` to keep it under 1000 lines."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _screen_predictors_step1_array_returned_encodings(subsample_idx, factors_data, targets_data, _screen_full_factors):
    """Step 1 of screen_predictors: lines starting at ``if subsample_idx is not None:``."""
    if subsample_idx is not None:
        try:
            _sidx = np.asarray(subsample_idx)
            if _sidx.ndim == 1 and 0 < _sidx.shape[0] < len(factors_data) and int(_sidx.max()) < len(factors_data):
                _sidx = _sidx.astype(np.int64, copy=False)
                _same_t = targets_data is factors_data
                _screen_full_factors = factors_data
                factors_data = factors_data[_sidx]
                if _same_t:
                    targets_data = factors_data
                elif targets_data is not None and len(targets_data) == len(_screen_full_factors):
                    targets_data = targets_data[_sidx]
        except Exception as e:
            log_throttle(logger, "screen_predictors.subsample_index", logging.WARNING, "screening: applying the subsample index failed (%s), the full factors are used", e)
            _screen_full_factors = None
    return _screen_full_factors, factors_data
