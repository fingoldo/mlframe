"""Leaky-val pre-screen: drop hopeless ensemble components BEFORE the expensive honest-OOF refits.

The dummy-floor gate at the end of the cross-target ensemble phase discards most components (14 of 21 in prod) after
each has been K-fold OOF-refit. Predicting with the already-trained components on a frame they were not fully trained
on costs no refit at all, and a component whose leaky RMSE cannot clear the dummy floor even with a generous safety
margin will not clear it honestly either. The honest gate still runs afterwards, so this stage only decides what is
worth refitting.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .._prediction_memo import memo_predict

logger = logging.getLogger(__name__)

PRESCREEN_SAFETY = 1.5
"""Leaky RMSE is divided by this before it is compared to the floor, so only hopeless components are dropped."""

_MIN_FINITE_ROWS = 10
"""Below this many jointly finite rows the leaky RMSE is noise, so the component is kept."""


def prescreen_frame(ext_X: Any, ext_y: Any, val_df: Any, y_full: Any, val_idx: Any) -> tuple[Any, Any]:
    """The frame and target the pre-screen predicts on, and their absence as ``(None, None)``.

    The OOF source decides where the WEIGHTING surface comes from; the pre-screen only needs a frame the trained
    components can predict on, so it falls back to the validation split whenever the OOF path supplies none. Tying it
    to ``oof_holdout_source='external_val'`` meant it never ran under the default K-fold source.
    """
    if ext_X is not None and ext_y is not None:
        return ext_X, ext_y
    if val_df is None or val_idx is None:
        return None, None
    try:
        val_y = np.asarray(y_full)[val_idx]
    except (TypeError, IndexError):
        return None, None
    if val_y is None or len(val_y) == 0:
        return None, None
    return val_df, val_y


def leaky_rmse_keep_mask(components: Sequence[Any], component_names: Sequence[str], X: Any, y: Any, dummy_floor: float) -> tuple[list[bool], list[str]]:
    """Per-component keep flags and the dropped components' labels, judged on leaky RMSE against ``dummy_floor``.

    A component whose ``predict`` raises, or that has too few jointly finite rows to score, is kept: the screen exists
    to save refits, never to decide a borderline case.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    keep_mask: list[bool] = []
    dropped: list[str] = []
    for comp, name in zip(components, component_names):
        try:
            preds = memo_predict(comp, X)
            finite = np.isfinite(preds) & np.isfinite(y_arr)
            if finite.sum() < _MIN_FINITE_ROWS:
                keep_mask.append(True)
                continue
            resid = preds[finite] - y_arr[finite]
            leaky_rmse = float(np.sqrt(np.mean(resid * resid)))
            if leaky_rmse / PRESCREEN_SAFETY > dummy_floor:
                keep_mask.append(False)
                dropped.append(f"{name}(leakyRMSE={leaky_rmse:.4g})")
            else:
                keep_mask.append(True)
        except Exception as e:
            logger.debug("suppressed: leaky-RMSE pre-check failed for this candidate, keeping it: %s", e)
            keep_mask.append(True)
    return keep_mask, dropped


def dummy_floor_from_metadata(metadata: dict, target_type: Any, target_name: Any) -> float | None:
    """The strongest dummy baseline's primary-metric value for this target, or ``None`` when it is unusable.

    Assumes an RMSE-family regression primary metric: the value is compared directly against component RMSEs.
    """
    raw = (metadata.get("dummy_baselines", {}) or {}).get(str(target_type), {}).get(str(target_name), {})
    if not isinstance(raw, dict):
        return None
    data = raw.get("data", {}) or {}
    strongest = raw.get("strongest")
    primary_metric = raw.get("primary_metric")
    if not (strongest and primary_metric and strongest in data):
        return None
    value = data[strongest].get(primary_metric)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)
