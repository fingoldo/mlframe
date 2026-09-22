"""Leaky-val pre-screen: drop hopeless ensemble components BEFORE the expensive honest-OOF refits.

The dummy-floor gate at the end of the cross-target ensemble phase discards most components (14 of 21 in prod) after
each has been K-fold OOF-refit. Predicting with the already-trained components on a frame they were not fully trained
on costs no refit at all, and a component whose leaky RMSE cannot clear the dummy floor even with a generous safety
margin will not clear it honestly either. The honest gate still runs afterwards, so this stage only decides what is
worth refitting.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

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


def same_split_dummy_rmse(metadata: dict, target_type: Any, target_name: Any, oof_names: Sequence[str], oof_rmses: Any, oof_y: Any) -> float | None:
    """The strongest dummy's RMSE on the same OOF rows the components are scored on, or ``None`` when it cannot be had.

    The dummy baseline in metadata is a VAL-split number, and comparing it with the components' train K-fold OOF RMSEs let
    weak components through on a hard val period and dropped good ones on an easy one. On the OOF rows: the in-pool
    ``lag_predict`` column when present, else the strongest constant strategy (mean / median / quantile) evaluated on the
    OOF targets. Only when neither applies is the val-split value used, with a WARNING naming the mismatch.
    """
    names = list(oof_names)
    if "lag_predict" in names:
        v = float(np.asarray(oof_rmses, dtype=np.float64)[names.index("lag_predict")])
        if np.isfinite(v):
            return v
    raw = (metadata.get("dummy_baselines", {}) or {}).get(str(target_type), {}).get(str(target_name), {})
    strongest = raw.get("strongest") if isinstance(raw, dict) else None
    y = np.asarray(oof_y, dtype=np.float64).reshape(-1) if oof_y is not None else np.empty(0)
    y = y[np.isfinite(y)]
    if strongest and y.size:
        const = None
        if strongest == "mean":
            const = float(np.mean(y))
        elif strongest == "median":
            const = float(np.median(y))
        elif str(strongest).startswith("quantile_p"):
            try:
                const = float(np.quantile(y, float(str(strongest)[len("quantile_p") :]) / 100.0))
            except ValueError:
                const = None
        if const is not None:
            return float(np.sqrt(np.mean((y - const) ** 2)))
    val_value = dummy_floor_from_metadata(metadata, target_type, target_name)
    if val_value is not None:
        logger.warning(
            "[CompositeCrossTargetEnsemble] target=%r: the strongest dummy (%s) has no same-split OOF estimate; the dummy floor "
            "falls back to its val-split RMSE, which is not measured on the rows the components are scored on.", target_name, strongest,
        )
    return val_value


def apply_dummy_floor_gate(
    cfg: Any, metadata: Mapping[str, Any], target_type: Any, target_name: str,
    components: list, names: list, rmses: np.ndarray, pred_matrix: np.ndarray | None, y_holdout: np.ndarray,
) -> tuple[list, list, np.ndarray, np.ndarray | None]:
    """Drop the components whose honest-OOF RMSE exceeds the strongest dummy's same-split RMSE (times 1 + tolerance).

    A trained model that loses to a parameter-free dummy on the honest holdout cannot improve the ensemble; keeping it
    dilutes the NNLS weights. When every component would be dropped all are kept (the honest gate then falls back to the
    best single). Returns ``(components, names, rmses, pred_matrix)``, filtered or unchanged; ``ct_ensemble_dummy_floor_enabled``
    (default True) turns it off.
    """
    # Dummy-floor gate: drop any component whose honest-OOF RMSE exceeds the raw target's strongest-dummy RMSE by more than the configured tolerance. A trained model that loses to a parameter-free dummy on the honest holdout cannot improve the ensemble; keeping it dilutes NNLS weights and harms test performance.
    # The dummy's primary_metric value is compared directly against component OOF RMSEs, so the floor is unit-consistent only while the regression primary is RMSE (currently the only option).
    _dummy_floor_enabled = bool(getattr(
        cfg,
        "ct_ensemble_dummy_floor_enabled", True,
    ))
    _dummy_floor_tol = float(getattr(
        cfg,
        "ct_ensemble_dummy_floor_tolerance", 0.0,
    ))
    if (_dummy_floor_enabled
            and pred_matrix is not None
            and pred_matrix.shape[1] > 0
            and len(rmses) > 0):
        # The floor is measured on the same OOF rows as the components it gates (see same_split_dummy_rmse).
        _dummy_floor_rmse = same_split_dummy_rmse(metadata, target_type, target_name, names, rmses, y_holdout)
        if _dummy_floor_rmse is not None:
            _dummy_floor_rmse *= 1.0 + _dummy_floor_tol
        if _dummy_floor_rmse is not None:
            _keep_idx = [_i for _i in range(len(rmses)) if np.isfinite(rmses[_i]) and rmses[_i] <= _dummy_floor_rmse]
            _dropped_idx = [_i for _i in range(len(rmses)) if _i not in set(_keep_idx)]
            if _dropped_idx and len(_keep_idx) >= 1:
                _dropped_names = [f"{names[_i]}(OOF={rmses[_i]:.4g})" for _i in _dropped_idx]
                _floor_base = _dummy_floor_rmse / (1.0 + _dummy_floor_tol)
                logger.warning(
                    "[CompositeCrossTargetEnsemble] target='%s' "
                    "dummy-floor gate fired: dropping %d/%d "
                    "component(s) whose OOF RMSE > the strongest "
                    "dummy's same-split OOF RMSE %.4g x (1+%.2f) = %.4g. "
                    "Dropped: %s",
                    target_name, len(_dropped_idx),
                    len(rmses),
                    _floor_base, _dummy_floor_tol,
                    _dummy_floor_rmse, _dropped_names,
                )
                components = [components[_i] for _i in _keep_idx]
                names = [names[_i] for _i in _keep_idx]
                rmses = rmses[_keep_idx]
                pred_matrix = pred_matrix[:, _keep_idx]
            elif not _keep_idx:
                logger.warning(
                    "[CompositeCrossTargetEnsemble] target='%s' "
                    "dummy-floor gate would drop ALL %d "
                    "component(s) (every OOF RMSE > %.4g); "
                    "keeping all to avoid empty pool. The "
                    "honest-OOF gate below will fall back to "
                    "best single.",
                    target_name, len(rmses),
                    _dummy_floor_rmse,
                )

    return components, names, rmses, pred_matrix
