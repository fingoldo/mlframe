"""Wrap-pass watchdog: checks a wrapped composite against an oracle built from the raw split frame.

Two checks, each independent of the path the wrapper itself takes:

* **base read**: the base the wrapper reads at predict (``_extract_base_for_transform``) must equal the base read straight
  from the split frame, the columns the spec's params were fitted on. A different base at predict time is one of the
  failures the old check named but, comparing the wrapper with its own machinery, could not see.
* **additive error**: for a transform whose inverse is ``T + g(base)`` (``Transform.additive_in_t``), the y-error equals
  the T-error row by row, with the true ``T`` computed from the split's real y and base. ``T_hat`` comes from the wrapper's
  own inner input (its pre-pipeline applied, the group column dropped), and the check skips rows the train-envelope clip
  pinned to a bound and rows outside the fitted base range, so neither the clip nor the soft base shrink trips it.

Grouped transforms get their groups from the wrapper's ``group_column``. A check that cannot run logs at WARNING: a silent
DEBUG line had left whole transform families without coverage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle

from ..composite import get_transform
from mlframe.training.composite.estimator.shared import extract_base_matrix as _extract_base_matrix
from mlframe.training.composite.estimator.shared import extract_groups as _extract_groups
from mlframe.training.composite.transforms.shared import call_transform
from ._prediction_memo import memo_predict

# The parent module's logger name: these lines predate the split, and log filters select them by that name.
logger = logging.getLogger("mlframe.training.core._phase_composite_wrapping")

_WATCHDOG_RELATIVE_THRESHOLD = 0.01
"""The additive check fires when y-MAE and T-MAE differ by more than this fraction of the T-MAE."""

WATCHDOG_SAMPLE_ROWS = 2_000
"""Rows the watchdog checks per composite when the wrap pass skips its metric block."""


def _watchdog_base_columns(spec: dict) -> tuple[str, ...]:
    """Full ordered base-column tuple for a spec, mirroring the wrapper's ``_resolve_base_columns``.

    Multi-base specs (``linear_residual_multi`` and any future multi-base transform) carry the secondary bases in ``extra_base_columns``; the
    watchdog must feed the transform's ``forward``/``inverse`` the same ``(n, K)`` base matrix the wrapper uses, otherwise those calls raise
    "base has 1 columns but fitted alphas has K entries" and the watchdog silently swallows the error -- leaving the multi-base family (exactly
    the one most prone to wrapper-math bugs) with zero coverage. Single-base specs return a 1-tuple, so the 1-D fast path below is unchanged.
    """
    _bc = spec.get("base_column") if isinstance(spec, dict) else None
    if not _bc:
        return ()
    _extra = tuple(spec.get("extra_base_columns") or ()) if isinstance(spec, dict) else ()
    return (_bc, *_extra)


def _watchdog_extract_base(split_df: Any, base_columns: tuple[str, ...]) -> np.ndarray:
    """Extract the watchdog base array for ``base_columns`` from ``split_df``.

    For a single base column returns a 1-D float64 array (bit-identical to the prior ``np.asarray(split_df[col]).astype(np.float64)`` pull);
    for K>=2 columns returns the canonical ``(n, K)`` matrix via the same ``_extract_base_matrix`` helper the wrapper uses, so the transform's
    ``forward``/``inverse`` see all K bases. Format-native (no whole-frame copy): single-col path stays a narrow column pull, multi-col path
    routes through the polars ``.select`` / pandas ``.loc`` single-buffer extractor.
    """
    if len(base_columns) == 1:
        return np.asarray(split_df[base_columns[0]]).astype(np.float64)
    return _extract_base_matrix(split_df, base_columns)


def _inside_fit_range(params: dict, base: np.ndarray) -> np.ndarray:
    """Rows whose every base column lies inside the range captured at fit: the soft base shrink leaves these untouched."""
    from mlframe.training.composite.estimator.shared import BASE_FIT_RANGE_KEY

    rng = params.get(BASE_FIT_RANGE_KEY) if isinstance(params, dict) else None
    b2 = base.reshape(-1, 1) if base.ndim == 1 else base
    if not isinstance(rng, dict) or "lo" not in rng:
        return np.ones(b2.shape[0], dtype=bool)
    return np.asarray(np.all((b2 >= np.asarray(rng["lo"])) & (b2 <= np.asarray(rng["hi"])), axis=1), dtype=bool)


def _warn(key: str, msg: str, *args: Any) -> None:
    """Throttled WARNING under the watchdog's logger."""
    log_throttle(logger, key, logging.WARNING, msg, *args)


def _check_base_read(wrapper: Any, split_df: Any, base: np.ndarray, composite_name: str, split_name: str) -> None:
    """The base the wrapper reads at predict must equal the base read from the split frame."""
    wrapper_base = np.asarray(wrapper._extract_base_for_transform(split_df, wrapper._resolve_base_columns()), dtype=np.float64)
    if wrapper_base.shape != base.shape:
        _warn("composite_wrap_watchdog_base_shape", "[CompositeTargetEstimator.watchdog] composite='%s' split='%s': the wrapper reads a base of shape "
              "%s at predict, the spec's base columns give %s.", composite_name, split_name, wrapper_base.shape, base.shape)
        return
    both = np.isfinite(wrapper_base) & np.isfinite(base)
    if both.any() and not np.allclose(wrapper_base[both], base[both], rtol=1e-9, atol=0.0):
        _warn("composite_wrap_watchdog_base_mismatch", "[CompositeTargetEstimator.watchdog] composite='%s' split='%s': the base the wrapper reads at "
              "predict differs from the spec's base columns in the split frame (max abs diff %.4g).", composite_name, split_name,
              float(np.max(np.abs(wrapper_base[both] - base[both]))))


def _check_additive(wrapper: Any, transform: Any, params: dict, split_df: Any, y_split: np.ndarray, base: Any, groups: Any,
                    composite_name: str, split_name: str) -> None:
    """y-MAE must equal T-MAE for an additive-in-T transform, with the true T from the split's real y and base."""
    from mlframe.training.composite.estimator.shared import apply_t_clip as _apply_t_clip
    from mlframe.training.composite.estimator.shared import inner_input

    t_true = np.asarray(call_transform(transform, "forward", y_split, base, params, groups=groups), dtype=np.float64)
    t_hat = np.asarray(wrapper.estimator_.predict(inner_input(wrapper, split_df, transform)), dtype=np.float64).reshape(-1)
    wrapper_params = getattr(wrapper, "fitted_params_", None) or params  # carries the T clip and the base fit range
    t_hat = _apply_t_clip(wrapper, t_hat, wrapper_params)[0]
    y_hat = np.asarray(memo_predict(wrapper, split_df), dtype=np.float64).reshape(-1)
    rows = np.isfinite(t_true) & np.isfinite(t_hat) & np.isfinite(y_hat) & np.isfinite(y_split)
    # The train-envelope clip moves only rows it pins to a bound; those are left out, every other row must match.
    rows &= (y_hat > float(wrapper_params.get("y_clip_low", -np.inf))) & (y_hat < float(wrapper_params.get("y_clip_high", np.inf)))
    if base is not None:
        rows &= _inside_fit_range(wrapper_params, np.asarray(base, dtype=np.float64))
    if int(rows.sum()) < 10:
        return
    mae_t = float(np.mean(np.abs(t_hat[rows] - t_true[rows])))
    mae_y = float(np.mean(np.abs(y_hat[rows] - y_split[rows])))
    rel = abs(mae_t - mae_y) / max(mae_t, 1e-9)
    if rel > _WATCHDOG_RELATIVE_THRESHOLD:
        _warn("composite_wrap_watchdog_additive_divergence", "[CompositeTargetEstimator.watchdog] composite='%s' split='%s' transform=%s: y-MAE=%.4g "
              "differs from T-MAE=%.4g by %.1f%% on %d in-range rows; an additive inverse gives identical errors. Probable causes: the inner "
              "does not return T (transformer state lost through clone/pickle), the inverse is applied twice, or the inner sees a different "
              "feature frame than it was fitted on.", composite_name, split_name, transform.name, mae_y, mae_t, rel * 100.0, int(rows.sum()))


def _is_additive_in_t(transform_name: str) -> bool:
    """True when the ``MAE_T == MAE_y`` watchdog invariant holds for this transform.

    Read off ``Transform.additive_in_t``: the hand-kept list included ``quantile_residual`` (``y = T * IQR + median``, so the
    invariant never holds and the watchdog false-fired) and missed most additive transforms. An out-of-fold train forward
    (target encoding) answers the fit rows differently from the inverse, so it is left out too.
    """
    try:
        t = get_transform(transform_name)
    except (KeyError, ValueError):
        return False
    return bool(getattr(t, "additive_in_t", False)) and not getattr(t, "oof_train_forward", False)


def run_wrap_watchdog(wrapper: Any, spec: Any, split_df: Any, y_split: Any, *, composite_name: str, split_name: str) -> None:
    """Run both checks for one wrapped composite on one split; any failure to run them is logged at WARNING."""
    t_name = spec.get("transform_name") if isinstance(spec, dict) else None
    if not t_name or not hasattr(wrapper, "estimator_") or not hasattr(wrapper, "_extract_base_for_transform"):
        return
    try:
        transform = get_transform(t_name)
        params = spec.get("fitted_params", {}) or {}
        y_arr = np.asarray(y_split, dtype=np.float64).reshape(-1)
        groups = None
        if transform.requires_groups:
            if not getattr(wrapper, "group_column", None):
                return
            groups = _extract_groups(split_df, wrapper.group_column)
        base = None
        if transform.requires_base:
            cols = _watchdog_base_columns(spec)
            if not cols or not all(c in split_df for c in cols):
                return
            base = _watchdog_extract_base(split_df, cols)
            _check_base_read(wrapper, split_df, base, composite_name, split_name)
        if _is_additive_in_t(t_name):
            _check_additive(wrapper, transform, params, split_df, y_arr, base, groups, composite_name, split_name)
    except Exception as exc:
        _warn("composite_wrap_watchdog_failed", "[CompositeTargetEstimator.watchdog] check could not run for composite='%s' split='%s' (%s: %s).",
              composite_name, split_name, type(exc).__name__, exc)


def watchdog_sample(split_df: Any, y_split: np.ndarray, n: int = WATCHDOG_SAMPLE_ROWS) -> tuple[Any, np.ndarray]:
    """The first ``n`` rows of a split, as frame and target: the cheap watchdog pass when the metric block is skipped."""
    if len(y_split) <= n:
        return split_df, y_split
    head = split_df.head(n) if hasattr(split_df, "head") else split_df[:n]
    return head, y_split[:n]


__all__ = ["run_wrap_watchdog", "watchdog_sample"]
