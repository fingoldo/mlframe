"""``CompositeTargetEstimator.from_fitted_inner``: wrap an already-fitted inner model into a y-scale predictor.

The classmethod on the estimator keeps its full signature and docstring and delegates here, like the predict /
update families (``_predict`` / ``_update``), keeping ``_estimator.py`` under the module size limit.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ._estimator_helpers import _y_quantile_grid

from . import _y_train_clip_bounds
from ._routing import resolve_transform as get_transform

# Same logger name as before the carve, so anything filtering these records by logger name keeps seeing them.
logger = logging.getLogger("mlframe.training.composite.estimator._estimator")


def from_fitted_inner(
    cls: type,
    fitted_inner: Any,
    transform_name: str,
    base_column: str,
    transform_fitted_params: dict[str, Any],
    y_train: np.ndarray,
    fallback_predict: str = "y_train_median",
    base_columns: Sequence[str] | None = None,
    inner_pre_pipeline: Any = None,
    base_train: np.ndarray | None = None,
    group_column: str | None = None,
    recurrence_continuation: bool = False,
    target_name: str | None = None,
) -> Any:
    """Body of ``CompositeTargetEstimator.from_fitted_inner`` (see its docstring)."""
    instance = cls(
        base_estimator=fitted_inner,
        transform_name=transform_name,
        base_column=base_column,
        base_columns=base_columns,
        fallback_predict=fallback_predict,
        drop_invalid_rows=True,
        group_column=group_column,
        recurrence_continuation=recurrence_continuation,
    )
    # Validate we can lookup the transform up-front so a typo
    # surfaces here, not on first predict.
    get_transform(transform_name)

    y_train = np.asarray(y_train).reshape(-1).astype(np.float64)
    finite = np.isfinite(y_train)
    if finite.size == 0 or not finite.any():
        y_train_median = float("nan")
        y_clip_low, y_clip_high = float("-inf"), float("inf")
    else:
        y_train_median = float(np.median(y_train[finite]))
        y_clip_low, y_clip_high = _y_train_clip_bounds(y_train[finite])

    # T-scale clip bounds. This route has no direct T_train (caller passed
    # y_train + params). For UNARY transforms (``requires_base=False``: log_y,
    # cbrt_y, yeo_johnson_y, quantile_normal_y, y_quantile_clip) the T-scale
    # is a fitted function of y ALONE, so T_train is OFFSET from 0; a symmetric
    # ``+/-10*std(y)`` band centered at 0 mis-centers it and clips the
    # in-distribution T_hat flat. The unary forward needs no base, so we
    # reconstruct the EXACT T_train via ``transform.forward(y, zeros, params)``
    # and apply the same MAD envelope the .fit() path uses (median(T)+/-10*MAD,
    # widened to the observed [T_min, T_max]). For BASE-dependent transforms
    # base is unavailable here, so we keep the conservative ``+/-10*y_std``
    # proxy -- correct because the additive-residual cores (diff /
    # linear_residual) have T centered near 0 by OLS construction.
    # The gate counts FINITE values (``finite.sum()``), mirroring .fit(); using
    # ``finite.size`` let a mostly-NaN y_train estimate the band from ~2 points.
    t_clip_low, t_clip_high = float("-inf"), float("inf")
    # Discovery stamps the exact T-train envelope into the spec (it has the base column there). Prefer it: the y_std proxy
    # below assumes T lives on y's scale, false for SCALED residuals (quantile_residual divides by a per-bin IQR), where it
    # clipped every test row to one bound and turned the prediction into a constant.
    _env_lo = transform_fitted_params.get("t_train_envelope_low") if hasattr(transform_fitted_params, "get") else None
    _env_hi = transform_fitted_params.get("t_train_envelope_high") if hasattr(transform_fitted_params, "get") else None
    if _env_lo is not None and _env_hi is not None and np.isfinite(_env_lo) and np.isfinite(_env_hi) and _env_hi >= _env_lo:
        t_clip_low, t_clip_high = float(_env_lo), float(_env_hi)
    elif int(finite.sum()) >= 10:
        _transform = get_transform(transform_name)
        _t_train_recon: np.ndarray | None = None
        if not _transform.requires_base:
            # Unary: reconstruct exact T from y alone (base ignored by the
            # unary registry adapter, so a zeros placeholder is sound).
            try:
                _y_fin = y_train[finite]
                _t_train_recon = np.asarray(
                    _transform.forward(
                        _y_fin, np.zeros_like(_y_fin), dict(transform_fitted_params),
                    ),
                    dtype=np.float64,
                ).reshape(-1)
            except Exception as _recon_err:  # pragma: no cover - defensive
                logger.warning(
                    "[CompositeTargetEstimator.from_fitted_inner] unary T "
                    "reconstruction failed for transform '%s' (%r); falling "
                    "back to the y_std envelope proxy.",
                    transform_name, _recon_err,
                )
                _t_train_recon = None
        if _t_train_recon is not None:
            t_finite = _t_train_recon[np.isfinite(_t_train_recon)]
            if t_finite.size >= 10:
                t_med = float(np.median(t_finite))
                t_mad = float(np.median(np.abs(t_finite - t_med)))
                if t_mad > 0:
                    t_clip_low = t_med - 10.0 * t_mad
                    t_clip_high = t_med + 10.0 * t_mad
                    t_clip_low = min(t_clip_low, float(t_finite.min()))
                    t_clip_high = max(t_clip_high, float(t_finite.max()))
        else:
            y_std = float(np.std(y_train[finite]))
            if y_std > 0:
                t_envelope = 10.0 * y_std
                t_clip_low = -t_envelope
                t_clip_high = +t_envelope

    instance.estimator_ = fitted_inner
    # The entry's fitted pre_pipeline: the inner was trained on its output, so predict applies it to the suite-stage frame for
    # the inner only, while the base keeps the raw stage the transform params were fit on.
    instance.inner_pre_pipeline_ = inner_pre_pipeline
    if target_name:
        instance.target_name_ = str(target_name)
    instance.fitted_params_ = {
        **dict(transform_fitted_params),
        "y_clip_low": y_clip_low,
        "y_clip_high": y_clip_high,
        "y_train_median": y_train_median, "y_train_quantile_grid": _y_quantile_grid(y_train),
        "t_clip_low": t_clip_low,
        "t_clip_high": t_clip_high,
    }
    if recurrence_continuation:
        instance.fitted_params_["recurrence_continuation"] = True
    # Same base calibration range fit() captures, so the default-ON soft base-shrink and its deep-OOD fallback are live on the
    # suite path too. A spec that already carries a range (stamped by discovery) keeps it; train bases refresh it otherwise.
    if base_train is not None:
        from . import _soft_shrink as _soft_shrink

        _soft_shrink.capture_base_fit_range(instance, get_transform(transform_name), np.asarray(base_train, dtype=np.float64))
    # Inherit feature_names_in_ from the already-fitted inner so the
    # predict-side column-subset fallback can resolve the wrapper's expected
    # columns; without it the wrapper is fed the post-extensions pca/svd-only
    # frame while its inner was trained on the raw-plus-extension frame, and
    # CatBoost raises a feature-name mismatch.
    _inner_names = getattr(fitted_inner, "feature_names_in_", None)
    if _inner_names is None:
        _inner_names = getattr(fitted_inner, "feature_names_", None)
    if _inner_names is not None:
        try:
            instance.feature_names_in_ = list(_inner_names)
        except TypeError as _names_err:
            # Slotted / read-only inner instance rejected the assignment. Surface
            # so the operator sees this rather than waiting for the CatBoost
            # ``At position 0 should be feature with name x0 (found pca0)`` crash
            # at predict time which doesn't trace back to this propagation step.
            logger.warning(
                "CompositeEstimator: failed to propagate inner.feature_names_in_ "
                "onto wrapper (%s); predict-time may raise feature-name mismatch "
                "on CB/LGB/XGB inner model. Inner type: %s, inner names count: %d.",
                _names_err, type(fitted_inner).__name__, len(list(_inner_names)),
            )
    # Stamp the wrapper-level feature count so ``n_features_in_`` is
    # consistent with ``feature_names_in_``. ``from_fitted_inner`` does not
    # support grouped transforms (no group_column arg), so the inner's
    # feature count already equals what the wrapper exposes; prefer the
    # inherited name list when present, else the inner's scalar.
    _ffi_names = getattr(instance, "feature_names_in_", None)
    if _ffi_names is not None:
        instance._n_features_in_wrapper = len(_ffi_names)
    else:
        _inner_n = getattr(fitted_inner, "n_features_in_", None)
        if _inner_n is not None:
            instance._n_features_in_wrapper = int(_inner_n)
    instance.runtime_stats_ = {
        "predict_calls": 0,
        "predict_rows_total": 0,
        "domain_violation_rows": 0,
        "y_clip_low_hits": 0,
        "y_clip_high_hits": 0,
        "t_clip_low_hits": 0,
        "t_clip_high_hits": 0,
    }
    # Stamp the construction-source flag so __sklearn_clone__ can refuse cloning a wrapper whose fitted state lives outside the __init__ signature. sklearn.base.clone() would otherwise return a silent unfitted shell and the first predict() call on the clone would raise NotFittedError mid-pipeline. The legitimate clone-on-unfitted-spec flow (sklearn.Pipeline, GridSearchCV) goes through __init__ and never trips this flag.
    instance._built_via_from_fitted_inner = True
    return instance
