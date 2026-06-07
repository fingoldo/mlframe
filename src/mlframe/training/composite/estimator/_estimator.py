"""``CompositeTargetEstimator`` -- the main composite-target estimator class.

Wave 102 (2026-05-21): split out from ``composite_estimator.py`` to keep that
file below the 1k-line monolith threshold. Behaviour preserved bit-for-bit;
the class is re-exported from ``composite_estimator`` so existing imports
continue to work.

Top-level helpers (_y_train_clip_bounds, _extract_base, _extract_groups,
_extract_base_matrix, _is_polars_df) stay in the parent module; this
sibling imports them back from the partial-module state at load time.
The parent's bottom re-export triggers the sibling's load AFTER those
helpers are defined, so the partial-module lookup succeeds.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from collections import deque
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    pl = None
    _HAS_POLARS = False

# Wave 102: parent helpers needed by CompositeTargetEstimator's methods.
# The parent's bottom-of-file re-export triggers our load AFTER these
# helpers have been bound at the parent's top, so the partial-module
# lookup succeeds.
from . import (
    _y_train_clip_bounds,
    _extract_base,
    _extract_groups,
    _extract_base_matrix,
    _is_polars_df,
    _to_1d_numpy,
)
# Wave 102 split missed re-importing get_transform + DomainViolationError
# alongside the parent helpers above; the fitted-from-spec / fit / predict /
# predict_invert paths all use them, so leaving them unimported turned every
# CompositeTargetEstimator instantiation into a NameError at the very first
# call site. 2026-05-21 fix.
from ..transforms import get_transform, DomainViolationError

logger = logging.getLogger(__name__)


class CompositeTargetEstimator(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper that fits an inner regressor on a
    transformed target and inverts at predict time.

    Parameters
    ----------
    base_estimator
        Any sklearn-compatible regressor with ``fit(X, y)`` /
        ``predict(X)``. The wrapper clones it at ``fit`` time so the
        unfitted prototype passed in stays clean. ``feature_importances_``,
        ``columns``, ``get_booster()``, and other common attributes are
        delegated through the wrapper transparently.
    transform_name
        One of :func:`list_transforms`. Determines the forward / inverse
        applied to the target.
    base_column
        Name of the column in ``X`` carrying the base feature. Required:
        the wrapper extracts ``base`` from this column at both ``fit``
        and ``predict`` time. If the inner feature-selection step drops
        the column, ``predict`` will raise ``KeyError``.
    fallback_predict
        Strategy for rows where the inverse transform cannot be applied
        (e.g. ``base[row] = inf`` for ``ratio``, or ``base[row] <= 0``
        for ``logratio``). Default ``"y_train_median"`` substitutes the
        median of the training target; ``"nan"`` returns NaN so callers
        can decide.
    drop_invalid_rows
        If True (default), rows that fail ``transform.domain_check`` at
        fit time are dropped before fitting the inner estimator. If
        False, ``fit`` raises :exc:`DomainViolationError` instead.
    online_refit_enabled
        Rolling-buffer streaming-refit knob. When True, the wrapper carries a last-N rolling buffer of (y, base) observations across :meth:`update` calls and re-estimates alpha / beta in-place when the streaming-mean z-score exceeds ``online_refit_z_threshold``. Default OFF because stateful estimators break ``sklearn.clone()`` (the clone starts with an empty buffer); enable only when calling :meth:`update` explicitly from a streaming harness.
    online_refit_buffer_n
        Max rows retained in the rolling buffer. Older entries are evicted FIFO.
    online_refit_z_threshold
        Absolute z-score on the running residual mean above which alpha / beta are refit.
    online_refit_min_buffer_n
        Minimum buffer size before a refit can fire; prevents single-row noise from triggering an alpha flip.

    Attributes set at fit
    ---------------------
    fitted_params_
        Dict carrying transform-specific fitted parameters (e.g.
        ``alpha``, ``beta`` for linear_residual) plus the post-inverse
        y-clip bounds (``y_clip_low``, ``y_clip_high``) and
        ``y_train_median`` for fallback predictions.
    estimator_
        The fitted clone of ``base_estimator``.
    feature_names_in_
        Column names seen at fit (best effort: pandas / polars).
    runtime_stats_
        Live counters: ``domain_violation_rate`` (fraction of rows at
        predict where ``base`` was non-finite or out of domain),
        ``y_clip_hit_rate`` (fraction of predictions clipped to the
        y-bounds). Populated lazily on first ``predict`` and updated
        cumulatively across calls.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        transform_name: str = "diff",
        base_column: str = "",
        fallback_predict: str = "y_train_median",
        drop_invalid_rows: bool = True,
        runtime_stats_callback: Callable[[dict[str, Any]], None] | None = None,
        auto_variance_stabilise: bool = False,
        base_columns: Sequence[str] | None = None,
        group_column: str | None = None,
        online_refit_enabled: bool = False,
        online_refit_buffer_n: int = 10_000,
        online_refit_z_threshold: float = 3.0,
        online_refit_min_buffer_n: int = 200,
    ) -> None:
        self.base_estimator = base_estimator
        self.transform_name = transform_name
        self.base_column = base_column
        # Rolling-buffer streaming alpha refit. When ``online_refit_enabled=True``, the wrapper carries a rolling buffer of last-N (y, base) observations across ``update()`` calls; each update runs ``streaming_alpha_check_and_refit`` and, when |z| > threshold, updates ``self.fitted_params_["alpha"]`` / ``["beta"]`` in-place so subsequent predict() calls use the drift-corrected coefficients. Default OFF: stateful estimators break sklearn.clone() (cloned instance starts with empty buffer) so the flag is explicit opt-in. The buffer fields use trailing underscore (``self._buffer_y_``) to mark runtime-only state; sklearn.clone() ignores those.
        self.online_refit_enabled = online_refit_enabled
        self.online_refit_buffer_n = online_refit_buffer_n
        self.online_refit_z_threshold = online_refit_z_threshold
        self.online_refit_min_buffer_n = online_refit_min_buffer_n
        # Grouped-transform support. When the configured ``transform_name``
        # resolves to a transform with ``requires_groups=True`` (currently
        # only ``linear_residual_grouped``), the wrapper extracts a 1-D
        # groups ndarray from this column and passes it as a kwarg to
        # fit / forward / inverse. None for ungrouped transforms; the
        # wrapper validates the pair at fit-time and raises a clear
        # error if a grouped transform is configured without group_column.
        self.group_column = group_column
        # Multi-base support. ``base_columns`` is the canonical multi-column path used by
        # ``linear_residual_multi`` and any future multi-base transform.
        # When None and ``base_column`` is non-empty, falls back to a
        # single-element tuple so legacy callers continue to work
        # unchanged. When both are passed, ``base_columns`` wins (the
        # single-column ``base_column`` is treated as a legacy alias
        # for back-compat).
        self.base_columns = base_columns
        self.fallback_predict = fallback_predict
        self.drop_invalid_rows = drop_invalid_rows
        # When transform is ratio/logratio and
        # caller doesn't provide explicit sample_weight, auto-compute
        # variance-stabilising weights ~ 1/|base| (capped) to flatten
        # heteroscedasticity that ratio/logratio targets exhibit
        # (residuals on T-scale scale with |base| under multiplicative
        # DGP; LightGBM minimising MSE on T over-fits to the high-
        # variance regime). Default off -- opt-in because it changes
        # the loss; recommended on heavy-tail targets where logratio
        # was already chosen.
        self.auto_variance_stabilise = auto_variance_stabilise
        # Optional callback fired at the end of every ``predict`` call
        # with a snapshot of the per-batch counters. Use to hook into
        # Prometheus / StatsD / DataDog without coupling the wrapper
        # to a metrics library. The callback receives a dict with
        # keys ``batch_n``, ``batch_domain_violation_rows``,
        # ``batch_y_clip_low_hits``, ``batch_y_clip_high_hits``,
        # plus the cumulative-since-fit counters under
        # ``cumulative_*``. Errors raised by the callback are logged
        # at DEBUG and swallowed -- monitoring failures must never
        # poison the predict path.
        self.runtime_stats_callback = runtime_stats_callback

    # ------------------------------------------------------------------
    # Predict family -- thin in-body delegating stubs.
    #
    # The implementations live in ``_composite_target_estimator_predict`` and
    # were historically bound only via runtime class-attribute assignment at
    # this module's bottom, which made them invisible to mypy / IDEs / help().
    # These stubs make the public predict surface discoverable while the heavy
    # bodies stay carved out.
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> "np.ndarray":
        """y-scale point prediction (inner predict on T-scale, then invert). See ``_composite_target_estimator_predict.predict``."""
        from . import _predict as _pred
        return _pred.predict(self, X)

    def predict_quantile(self, X: Any, alpha: "float | Sequence[float]" = 0.5) -> "np.ndarray":
        """y-scale quantile prediction by inverting the inner's T-scale quantile. See ``_composite_target_estimator_predict.predict_quantile``."""
        from . import _predict as _pred
        return _pred.predict_quantile(self, X, alpha)

    # ------------------------------------------------------------------
    # Alternate constructor: post-hoc wrapping
    # ------------------------------------------------------------------

    @classmethod
    def from_fitted_inner(
        cls,
        fitted_inner: Any,
        transform_name: str,
        base_column: str,
        transform_fitted_params: dict[str, Any],
        y_train: np.ndarray,
        fallback_predict: str = "y_train_median",
        base_columns: Sequence[str] | None = None,
    ) -> CompositeTargetEstimator:
        """Build a wrapper around an ALREADY-FITTED inner model.

        This is the path used by the suite-level integration: the
        per-target training loop trains models on the T-scale composite
        target unaware of the wrapper, then post-hoc wrapping converts
        each fitted model to a y-scale predictor.

        Bypasses the wrapper's own ``fit`` because the inner is
        already fitted and re-fitting would either re-train (waste)
        or fail (no train data at this point in the suite). We
        populate the same ``estimator_`` / ``fitted_params_`` /
        ``runtime_stats_`` state that ``fit`` would have set, so
        ``predict`` is contract-identical to fit-then-predict.

        Parameters
        ----------
        fitted_inner
            The already-fitted base estimator. Caller is responsible
            for ensuring it was trained on the T-scale composite target
            (i.e. ``transform.forward(y, base, transform_fitted_params)``).
        transform_name, base_column
            See :meth:`__init__`.
        transform_fitted_params
            Transform-specific fitted parameters (``alpha`` / ``beta``
            for linear_residual, ``mad_eff`` / ``median_t`` for
            logratio, ``eps`` for ratio). Comes from
            :class:`CompositeSpec.fitted_params`.
        y_train
            Training-row ``y`` values used by the discovery pass. The
            wrapper computes ``y_train_median``, ``y_clip_low``, and
            ``y_clip_high`` from this so the predict-time fallbacks
            and post-inverse y-clip work without needing the train
            data later.
        fallback_predict
            See :meth:`__init__`.
        """
        instance = cls(
            base_estimator=fitted_inner,
            transform_name=transform_name,
            base_column=base_column,
            base_columns=base_columns,
            fallback_predict=fallback_predict,
            drop_invalid_rows=True,
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

        # T-scale clip bounds (mirror the .fit() path). On the
        # from_fitted_inner route we don't have direct T_train access -
        # caller passed y_train + transform_fitted_params. Reconstruct
        # a conservative T envelope from y_train statistics when possible
        # (transforms y - alpha*base have T-range bounded by y-range +
        # alpha*base-range, but base isn't available here, so we widen
        # to +/- 10 * y_std as a conservative T-envelope proxy that still
        # catches order-of-magnitude blow-ups while leaving in-distribution
        # T predictions untouched).
        # ``finite`` is a boolean mask, so ``finite.size`` is len(y_train) -- the
        # gate must count FINITE values (mirror the .fit() path, which sizes the
        # already-filtered finite array). Using .size let a mostly-NaN y_train
        # estimate the T-clip envelope from as few as 2 unrepresentative points.
        if int(finite.sum()) >= 10:
            y_std = float(np.std(y_train[finite]))
            if y_std > 0:
                t_envelope = 10.0 * y_std
                t_clip_low = -t_envelope
                t_clip_high = +t_envelope
            else:
                t_clip_low, t_clip_high = float("-inf"), float("inf")
        else:
            t_clip_low, t_clip_high = float("-inf"), float("inf")

        instance.estimator_ = fitted_inner
        instance.fitted_params_ = {
            **dict(transform_fitted_params),
            "y_clip_low": y_clip_low,
            "y_clip_high": y_clip_high,
            "y_train_median": y_train_median,
            "t_clip_low": t_clip_low,
            "t_clip_high": t_clip_high,
        }
        # Inherit feature_names_in_ from the already-fitted inner. The
        # __init__ path captures it via .fit() (line 618); the
        # from_fitted_inner path must mirror that so predict-side
        # ``predict_from_models`` can resolve the wrapper's expected
        # column list via getattr(model, "feature_names_in_"). Without it
        # the predict-side column-subset / df_pre_pipeline fallback path
        # (predict.py:1168-1194) skips, and the wrapper is fed the
        # post-extensions pca-only / svd-only frame while its inner
        # CatBoost/LGB/XGB was trained on the raw-plus-extension frame.
        # CatBoost then raises ``At position 0 should be feature with
        # name x0 (found pca0)``. Surfaced by fuzz iter-340 (composite +
        # PCA) / iter-79 family (composite + TruncatedSVD).
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
        instance.runtime_stats_ = {
            "predict_calls": 0,
            "predict_rows_total": 0,
            "domain_violation_rows": 0,
            "y_clip_low_hits": 0,
            "y_clip_high_hits": 0,
        }
        # Stamp the construction-source flag so __sklearn_clone__ can refuse cloning a wrapper whose fitted state lives outside the __init__ signature. sklearn.base.clone() would otherwise return a silent unfitted shell and the first predict() call on the clone would raise NotFittedError mid-pipeline. The legitimate clone-on-unfitted-spec flow (sklearn.Pipeline, GridSearchCV) goes through __init__ and never trips this flag.
        instance._built_via_from_fitted_inner = True
        return instance

    def __sklearn_clone__(self) -> "CompositeTargetEstimator":
        """Refuse cloning a wrapper built via :meth:`from_fitted_inner`.

        ``from_fitted_inner`` assigns ``estimator_`` / ``fitted_params_`` /
        ``runtime_stats_`` / ``feature_names_in_`` directly on the
        instance, bypassing ``__init__``. ``sklearn.base.clone()`` copies
        only ``get_params()`` output, so a clone of a from_fitted_inner
        instance silently loses every fitted attribute and the first
        downstream ``predict`` call raises ``NotFittedError`` from deep
        inside whichever pipeline cloned it.

        Standard ``fit()``-built instances clone normally via the
        default sklearn path (the flag is only stamped by
        from_fitted_inner).
        """
        if getattr(self, "_built_via_from_fitted_inner", False):
            raise NotImplementedError(
                "CompositeTargetEstimator: refusing to clone an instance built via "
                "from_fitted_inner. The fitted state (estimator_, fitted_params_, "
                "runtime_stats_, feature_names_in_) lives outside the __init__ "
                "signature, so sklearn.base.clone() would silently produce an "
                "unfitted shell and the first predict() call on the clone would "
                "raise NotFittedError mid-pipeline. If you need a fresh wrapper, "
                "construct it via the standard CompositeTargetEstimator(...) "
                "constructor + fit(). If you need to deepcopy the fitted state, "
                "use copy.deepcopy(instance) (not sklearn.base.clone)."
            )
        # Default sklearn clone semantics: reconstruct via class + cloned init params.
        # Mirrors what BaseEstimator's default __sklearn_clone__ does (sklearn>=1.3).
        klass = self.__class__
        new_params = {name: clone(val, safe=False) for name, val in self.get_params(deep=False).items()}
        return klass(**new_params)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def _resolve_base_columns(self) -> tuple[str, ...]:
        """Canonical multi-column form of ``base_columns`` / ``base_column``.

        Priority order:
        1. ``self.base_columns`` if non-None and non-empty (multi-base path).
        2. ``self.base_column`` if non-empty (single-base legacy path,
           wrapped as a one-tuple).
        3. Empty tuple (caller config error -- callers must validate).

        Returns a tuple so it is hashable + immutable + safe to ship
        into ``fitted_params_`` / spec serialization.
        """
        if self.base_columns:
            return tuple(self.base_columns)
        if self.base_column:
            return (self.base_column,)
        return ()

    def _extract_base_for_transform(
        self, X: Any, base_columns: tuple[str, ...],
    ) -> np.ndarray:
        """Pull base values; return 1-D when K=1 (so legacy transforms
        with 1-D fit/forward/inverse keep working), 2-D when K>=2."""
        if len(base_columns) == 1:
            return _extract_base(X, base_columns[0])
        return _extract_base_matrix(X, base_columns)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> CompositeTargetEstimator:
        if self.base_estimator is None:
            raise ValueError("CompositeTargetEstimator: base_estimator must not be None.")
        transform = get_transform(self.transform_name)
        base_columns = self._resolve_base_columns()
        # Pack J: unary y-transforms (cbrt_y, log_y, ...) have ``requires_base=False`` and must not require a base column. Feed a zeros placeholder so downstream calls keep their (y, base, params) signatures without branching.
        if not transform.requires_base:
            y_arr = _to_1d_numpy(y).astype(np.float64)
            base_arr = np.zeros_like(y_arr)
        else:
            if not base_columns:
                raise ValueError(
                    "CompositeTargetEstimator: either base_column (str) or "
                    "base_columns (sequence) must be supplied."
                )
            y_arr = _to_1d_numpy(y).astype(np.float64)
            base_arr = self._extract_base_for_transform(X, base_columns)
            if len(y_arr) != len(base_arr):
                raise ValueError(
                    f"CompositeTargetEstimator.fit: y has {len(y_arr)} rows but X "
                    f"has {len(base_arr)} -- caller passed misaligned inputs."
                )

        # Apply domain check before fitting transform params: invalid
        # rows must NOT bias the OLS / MAD computation.
        valid = transform.domain_check(y_arr, base_arr)
        valid = np.asarray(valid)
        if valid.ndim != 1:
            raise ValueError(
                f"CompositeTargetEstimator.fit: transform '{self.transform_name}' "
                f"domain_check returned ndim={valid.ndim}; expected 1-D boolean mask."
            )
        n_invalid = int((~valid).sum())
        if n_invalid > 0:
            if not self.drop_invalid_rows:
                raise DomainViolationError(
                    f"CompositeTargetEstimator.fit: transform '{self.transform_name}' "
                    f"requires domain conditions; {n_invalid} of {len(y_arr)} rows violate. "
                    "Set drop_invalid_rows=True to drop them automatically."
                )
            logger.info(
                "[CompositeTargetEstimator] dropping %d/%d (%.2f%%) rows "
                "violating domain of transform '%s'.",
                n_invalid, len(y_arr), 100.0 * n_invalid / len(y_arr), self.transform_name,
            )

        y_train = y_arr[valid]
        base_train = base_arr[valid]

        if y_train.size == 0:
            raise DomainViolationError(
                f"CompositeTargetEstimator.fit: 0 rows pass domain_check for "
                f"transform '{self.transform_name}'. Check inputs."
            )

        # Fit transform-specific params on the valid train rows.
        # Pass sample_weight through to weight-aware transforms
        # (currently only ``linear_residual``); other transforms
        # ignore the kwarg.
        sample_weight_train = (
            np.asarray(sample_weight)[valid] if sample_weight is not None else None
        )
        # Grouped-transform support. When the transform requires_groups=True,
        # extract group labels from the configured group_column and pass them
        # as kwargs through fit / forward. The transform's own fit signature
        # enforces presence.
        groups_train: np.ndarray | None = None
        if transform.requires_groups:
            if not self.group_column:
                raise ValueError(
                    f"CompositeTargetEstimator: transform '{self.transform_name}' "
                    f"requires groups; configure ``group_column`` on the wrapper."
                )
            groups_train = _extract_groups(X, self.group_column)[valid]
        # transform_fit_kwargs is separate from fit_kwargs (which is
        # the caller's pass-through to the inner estimator's .fit() at
        # the bottom of this method). Only ``groups`` flows into the
        # transform; sample_weight is also threaded through but lives
        # in its own kwarg name.
        transform_fit_kwargs: dict[str, Any] = {}
        if groups_train is not None:
            transform_fit_kwargs["groups"] = groups_train
        try:
            transform_params = transform.fit(
                y_train, base_train,
                sample_weight=sample_weight_train,
                **transform_fit_kwargs,
            )
        except TypeError:
            # Transform.fit doesn't accept sample_weight (most don't).
            transform_params = transform.fit(
                y_train, base_train, **transform_fit_kwargs,
            )

        # Compute T on the valid rows. Grouped transforms need the
        # groups kwarg for forward as well.
        transform_forward_kwargs: dict[str, Any] = (
            {"groups": groups_train} if groups_train is not None else {}
        )
        t_train = transform.forward(
            y_train, base_train, transform_params, **transform_forward_kwargs,
        )

        # Sanity: T must be finite or the inner estimator will choke.
        if not np.all(np.isfinite(t_train)):
            n_bad = int((~np.isfinite(t_train)).sum())
            raise DomainViolationError(
                f"CompositeTargetEstimator.fit: transform '{self.transform_name}' produced "
                f"{n_bad} non-finite T values on train. Likely numerical issue (extreme y / base)."
            )

        # Subset X to valid rows. Branch on type so we don't pull
        # pandas APIs on polars frames. Wrapper passes through whatever
        # frame the caller provided to the inner estimator; if the
        # inner doesn't accept that frame type (e.g. LightGBM 4.5 +
        # sklearn 1.6 on polars), the caller is responsible for the
        # conversion BEFORE reaching the wrapper. mlframe strategies
        # already do this at the suite level. We must not silently
        # materialise large polars frames -- that defeats the whole
        # zero-copy Arrow path on multi-GB datasets.
        X_valid = self._subset_rows(X, valid)
        # For grouped transforms, the group_column is metadata for the
        # transform (per-row alpha lookup) and is commonly non-numeric
        # (e.g. group_id strings). Tree models like LightGBM reject object
        # dtypes outright; the user-friendly behaviour is to drop the column
        # from X_valid before passing to the inner so the wrapper hides this
        # plumbing entirely.
        if transform.requires_groups and self.group_column:
            X_valid = self._drop_columns(X_valid, [self.group_column])
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[valid]
        elif (getattr(self, "auto_variance_stabilise", False)
                and self.transform_name in ("ratio", "logratio")):
            # Auto-compute variance-stabilising weights for ratio/logratio.
            # For multiplicative DGP, residual variance scales with
            # |base|. Weight ~ 1/|base| (capped at the 5th percentile
            # to avoid blow-up on near-zero base) flattens it.
            # ``base_train`` is already filtered to valid rows (line 471);
            # re-applying the full-length ``valid`` mask here would either
            # IndexError (when some rows were dropped) or be a misleading
            # no-op (when none were).
            base_valid = base_train.astype(np.float64)
            abs_base = np.abs(base_valid)
            floor_q = float(np.quantile(
                abs_base[abs_base > 0], 0.05,
            )) if (abs_base > 0).any() else 1.0
            abs_base_clipped = np.maximum(abs_base, floor_q)
            w = 1.0 / abs_base_clipped
            # Normalise to mean 1 so loss scale matches unweighted.
            w *= w.size / w.sum()
            sample_weight = w

        # Fit the inner estimator. Clone first so the unfitted
        # prototype passed in stays untouched and sklearn.clone() of
        # the wrapper produces a fresh inner.
        estimator = clone(self.base_estimator)
        if sample_weight is not None:
            try:
                estimator.fit(X_valid, t_train, sample_weight=sample_weight, **fit_kwargs)
            except TypeError:
                # Inner doesn't accept sample_weight; fall back.
                logger.info(
                    "[CompositeTargetEstimator] inner estimator '%s' does not accept "
                    "sample_weight; ignoring.",
                    type(self.base_estimator).__name__,
                )
                estimator.fit(X_valid, t_train, **fit_kwargs)
        else:
            estimator.fit(X_valid, t_train, **fit_kwargs)

        # Stash post-inverse y-clip bounds + train median for fallback.
        # Wave 21 P1: use np.nanmedian so a NaN-bearing y_train doesn't
        # silently make y_train_median == NaN -- the fallback constant is
        # used at predict time when transform is unsafe; pre-fix the
        # fallback predicted NaN for ALL rows that triggered it.
        y_clip_low, y_clip_high = _y_train_clip_bounds(y_train)
        y_train_median = float(np.nanmedian(y_train))
        if not np.isfinite(y_train_median):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "composite_estimator: y_train has no finite values; "
                "y_train_median fallback set to 0.0. Predict-time "
                "fallback will return 0 instead of NaN.",
            )
            y_train_median = 0.0

        # T-scale clip bounds. Inner predict() can blow out far past the
        # T-train envelope on heavy-tail residual targets (observed in prod:
        # XGB reg:pseudohubererror with un-calibrated
        # huber_slope=1.0 produced T_hat in [-50, +340] for T_train in
        # [-50, +50]; the additive inverse then pushed y_hat 340 above
        # the train envelope and the y-clip only catches the part outside
        # [y_min, y_max], not the wildly-extrapolated middle).
        # T-clip BEFORE inverse uses MAD-scaled bounds (median(T) +/- 10*MAD)
        # so the in-distribution mass is unaffected while gross blow-up
        # is bounded. MAD-based to be robust to a few outlying T values
        # in the train fold itself.
        t_finite = t_train[np.isfinite(t_train)]
        if t_finite.size >= 10:
            t_med = float(np.median(t_finite))
            t_mad = float(np.median(np.abs(t_finite - t_med)))
            if t_mad > 0:
                t_clip_low = t_med - 10.0 * t_mad
                t_clip_high = t_med + 10.0 * t_mad
                t_observed_min = float(t_finite.min())
                t_observed_max = float(t_finite.max())
                t_clip_low = min(t_clip_low, t_observed_min)
                t_clip_high = max(t_clip_high, t_observed_max)
            else:
                t_clip_low, t_clip_high = float("-inf"), float("inf")
        else:
            t_clip_low, t_clip_high = float("-inf"), float("inf")

        self.estimator_ = estimator
        self.fitted_params_ = {
            **transform_params,
            "y_clip_low": y_clip_low,
            "y_clip_high": y_clip_high,
            "y_train_median": y_train_median,
            "t_clip_low": t_clip_low,
            "t_clip_high": t_clip_high,
            "n_train_valid": int(y_train.size),
            "n_train_invalid": n_invalid,
        }
        # Best-effort feature names (pandas / polars). Narrowed except: an ndarray X legitimately lacks ``.columns`` (AttributeError); anything else (e.g. a polars frame whose column iteration raises) is anomalous and should surface, not silently swallow under bare Exception.
        try:
            self.feature_names_in_ = list(X.columns)
        except AttributeError:
            pass
        # Live counters initialised lazily by predict.
        self.runtime_stats_ = {
            "predict_calls": 0,
            "predict_rows_total": 0,
            "domain_violation_rows": 0,
            "y_clip_low_hits": 0,
            "y_clip_high_hits": 0,
        }
        return self

    @staticmethod
    def _subset_rows(X: Any, mask: np.ndarray) -> Any:
        """Row-subset X, preserving the dataframe flavour. Polars / pandas
        / ndarray supported. Raises TypeError otherwise."""
        if _is_polars_df(X):
            # ``_is_polars_df`` only returns True when the module-level
            # ``pl`` reference is the real polars module, so we can use
            # it directly without an extra import.
            return X.filter(pl.Series(mask))
        if isinstance(X, pd.DataFrame):
            return X.loc[mask].reset_index(drop=True)
        if isinstance(X, np.ndarray):
            return X[mask]
        raise TypeError(
            f"CompositeTargetEstimator: unsupported X type {type(X).__name__} for row subsetting."
        )

    @staticmethod
    def _drop_columns(X: Any, columns: Sequence[str]) -> Any:
        """Return ``X`` without ``columns``, preserving frame flavour.

        Used to strip the wrapper's plumbing columns (group_column for
        grouped transforms) before passing ``X`` to the inner estimator -
        tree models like LightGBM reject object/string dtypes that the
        wrapper needs for per-row group lookups.

        Silently no-op for columns not present (the caller may pass
        columns that were already dropped upstream by feature selection).
        """
        # Polars
        if _is_polars_df(X):
            present = [c for c in columns if c in X.columns]
            return X.drop(present) if present else X
        if isinstance(X, pd.DataFrame):
            present = [c for c in columns if c in X.columns]
            return X.drop(columns=present) if present else X
        # ndarray has no columns -> nothing to drop.
        return X



# Method rebinding from sibling carves. Done at module bottom so the parent class is fully constructed; identity preserved (parent.X is sibling.X) which keeps isinstance / hasattr / sklearn introspection unchanged. Mirror of the RFECV.fit carve pattern.
from . import _utils as _utils  # noqa: E402
from . import _predict as _pred  # noqa: E402
from . import _update as _upd  # noqa: E402

CompositeTargetEstimator._require_fitted = _utils._require_fitted
CompositeTargetEstimator.get_booster = _utils.get_booster
CompositeTargetEstimator.feature_importances_ = property(_utils.feature_importances_)
CompositeTargetEstimator.coef_ = property(_utils.coef_)
CompositeTargetEstimator.intercept_ = property(_utils.intercept_)
CompositeTargetEstimator.booster_ = property(_utils.booster_)
CompositeTargetEstimator.n_features_in_ = property(_utils.n_features_in_)

CompositeTargetEstimator._predict_unclipped = _pred._predict_unclipped
CompositeTargetEstimator.predict_pre_clip = _pred.predict_pre_clip
# ``predict`` / ``predict_quantile`` are now defined as in-body delegating stubs
# on the class (discoverable to mypy / IDE / help()); the heavy implementations
# stay in ``_pred`` and are reached via those stubs.

CompositeTargetEstimator.update = _upd.update
CompositeTargetEstimator.get_buffer_state = _upd.get_buffer_state
