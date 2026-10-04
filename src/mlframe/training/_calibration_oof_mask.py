"""Calibration-source selection and NaN masking of temporal OOF probabilities before they feed a calibrator."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def mask_nonfinite_oof_rows(
    oof_x: np.ndarray, oof_y: np.ndarray, *, metrics: Optional[dict] = None, where: str = "post_calibrate_model"
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows whose OOF probability is non-finite (expanding-window warm-up rows carry NaN) jointly with their labels.

    The number of dropped rows is logged and stamped as ``metrics["oof_nonfinite_rows_masked"]`` so a temporal suite shows
    explicitly how much of the train block never received an out-of-fold prediction. Raises when nothing finite remains.
    """
    oof_x = np.asarray(oof_x)
    oof_y = np.asarray(oof_y)
    finite = np.isfinite(oof_x)
    if finite.ndim > 1:
        finite = finite.all(axis=tuple(range(1, finite.ndim)))
    n_bad = int(finite.size - int(finite.sum()))
    if n_bad == 0:
        return oof_x, oof_y
    if n_bad == finite.size:
        raise ValueError(f"{where}: every OOF probability is non-finite, nothing to calibrate on")
    logger.info("%s: masked %d of %d non-finite OOF rows (temporal warm-up) before fitting", where, n_bad, finite.size)
    _stamp(metrics, n_bad)
    return oof_x[finite], oof_y[finite]


def _stamp(metrics: Any, n_bad: int) -> None:
    """Record the masked-row count on the metrics dict when one is available."""
    if isinstance(metrics, dict):
        metrics["oof_nonfinite_rows_masked"] = n_bad


def multi_output_calibration_source(model: Any, calib_probs: Any, calib_target: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(probs, labels)`` the per-class isotonic calibrator of ``post_calibrate_model`` is fit on.

    Caller-provided ``calib_probs`` / ``calib_target`` win, then the OOF probabilities stamped on the model; there is no test-slice fallback.
    """
    # Fit per-class isotonic on the calibration source. Prefer caller-provided (calib_probs, calib_target);
    # fall back to OOF-train probs stamped on the model; only as last resort -- and only with an explicit ``calib_idx``
    # confirmed disjoint from test_idx above -- do we draw from train_idx via target_series. Pure test-slice
    # calibration (the historical default ``test_probs[:calib_set_size]``) is no longer supported here: it leaks.
    if calib_probs is not None:
        _calib_p = np.asarray(calib_probs)
        _calib_y = np.asarray(calib_target)
    else:
        _oof_probs_mo = getattr(model, "oof_probs", None)
        if _oof_probs_mo is not None:
            _calib_p = np.asarray(_oof_probs_mo)
            # oof_probs are in train-row order (cross_val_predict); pair each
            # with its OWN row's label via the train-aligned oof_target. The
            # old ``target_series.iloc[:len(oof)]`` positional slice is only
            # correct when train is the leading contiguous block, so under a
            # shuffled / group-aware split it fit the calibrator on
            # mismatched (prob, label) pairs.
            _oof_y_mo = getattr(model, "oof_target", None)
            if _oof_y_mo is None:
                raise ValueError(
                    "post_calibrate_model (multi-output): model.oof_probs is present but "
                    "model.oof_target is missing, so OOF probs cannot be aligned to their "
                    "labels. Retrain so oof_target is stamped, or pass calib_probs+calib_target."
                )
            _calib_y = np.asarray(_oof_y_mo)[: _calib_p.shape[0]]
        else:
            raise ValueError(
                "post_calibrate_model (multi-output): no calibration source available. Pass calib_probs+calib_target "
                "(OOF-train probs preferred) or train the model with oof_n_splits>=2 so model.oof_probs is stamped."
            )

    return _calib_p, _calib_y
