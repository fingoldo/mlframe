"""CompositeTargetEstimator + helpers: sklearn-compatible wrapper that hides the transform-and-invert loop from downstream callers. Split out of composite.py to keep wrapper surface independent of CompositeTargetDiscovery; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import logging
import warnings
from collections import deque
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover - polars optional dep
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """ENS-P2-6: prefer explicit isinstance check over duck-typing on
    ``hasattr(x, "to_pandas")`` (which mis-detects any object exposing that
    method - mocks, custom wrappers, sklearn pipeline stubs)."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from .composite_transforms import (
    DomainViolationError,
    Transform,
    UnknownTransformError,
    _TRANSFORMS_REGISTRY,
    get_transform,
)

logger = logging.getLogger(__name__)


# Bounds for the post-inverse y-clip, expressed as multipliers on the
# ``[Q001(y_train), Q999(y_train)]`` envelope. Values outside this
# extended envelope are unphysical for the training distribution and
# almost certainly the result of ``exp`` / division blow-up.
_Y_CLIP_LOW_FRAC: float = 0.1
_Y_CLIP_HIGH_FRAC: float = 10.0


# ----------------------------------------------------------------------

def _y_train_clip_bounds(y_train: np.ndarray) -> tuple[float, float]:
    """Compute post-inverse y-clip bounds from train target.

    Bounds are extended Q001 / Q999 envelope multiplied by safety
    factors to keep predictions inside a physically plausible range
    even when the inverse transform produces extreme values for
    out-of-distribution rows. The asymmetric multipliers (0.1 lower,
    10x upper) reflect the typical heavy-tail asymmetry of regression
    targets in finance / volume / rate domains; symmetric clip would
    bite legitimate upper-tail predictions on log-normally distributed
    targets.
    """
    finite = y_train[np.isfinite(y_train)]
    if finite.size == 0:
        return float("-inf"), float("inf")
    q_low = float(np.quantile(finite, 0.001))
    q_high = float(np.quantile(finite, 0.999))
    # Edge case: q_low or q_high is 0 -> multiplier collapses bound;
    # fall back to absolute envelope around 0.
    span = q_high - q_low
    if span <= 0:
        # Constant target on train; allow +/- 10% wiggle.
        med = float(np.median(finite))
        return med - 0.1 * abs(med) - 1e-6, med + 0.1 * abs(med) + 1e-6
    low = q_low - (1.0 - _Y_CLIP_LOW_FRAC) * span
    high = q_high + (_Y_CLIP_HIGH_FRAC - 1.0) * span
    return low, high


from .utils import coerce_to_1d_numpy as _to_1d_numpy  # noqa: E402,F401


def _extract_base(X: Any, base_column: str) -> np.ndarray:
    """Pull base values from X (pandas / polars / structured ndarray).

    Raises ``KeyError`` with a helpful message if the column is missing
    -- this most commonly bites callers who configured MRMR / RFECV
    that dropped the base column before reaching the wrapper. The
    message points at the fix (``forced_keep_columns`` in the feature
    selection config).
    """
    # Polars
    if _is_polars_df(X):
        if base_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: base column '{base_column}' missing from X. "
                "If feature selection (MRMR/RFECV) is dropping it, add base_column "
                "to forced_keep_columns in the feature selection config."
            )
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view. astype(copy=False)
        # avoids a second copy when the dtype already matches.
        return X.get_column(base_column).to_numpy().astype(np.float64, copy=False)
    if isinstance(X, pd.DataFrame):
        if base_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: base column '{base_column}' missing from X. "
                "Columns: " + ", ".join(map(str, X.columns[:8])) + ("..." if len(X.columns) > 8 else "")
            )
        return X[base_column].to_numpy(dtype=np.float64)
    raise TypeError(
        f"CompositeTargetEstimator: unsupported X type {type(X).__name__}; "
        "pass pandas / polars DataFrame or a structured ndarray with named columns."
    )


def _extract_groups(X: Any, group_column: str) -> np.ndarray:
    """Pull group labels from X without numeric casting.

    Unlike :func:`_extract_base`, this preserves the dtype (string /
    integer / categorical) so per-row group lookups work in the
    grouped transform. Returns a 1-D ndarray.
    """
    if _is_polars_df(X):
        if group_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: group column '{group_column}' "
                f"missing from X."
            )
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view.
        return X.get_column(group_column).to_numpy()
    if isinstance(X, pd.DataFrame):
        if group_column not in X.columns:
            raise KeyError(
                f"CompositeTargetEstimator: group column '{group_column}' "
                f"missing from X."
            )
        return X[group_column].to_numpy()
    raise TypeError(
        f"CompositeTargetEstimator: unsupported X type {type(X).__name__}; "
        "pass pandas / polars DataFrame."
    )


def _extract_base_matrix(X: Any, base_columns: Sequence[str]) -> np.ndarray:
    """Multi-column variant of :func:`_extract_base`.

    Returns a 2-D ``(n, K)`` ndarray where K = ``len(base_columns)``.
    For K=1 this is identical to ``_extract_base(...).reshape(-1, 1)``.
    The K-column case is used by ``linear_residual_multi`` and any
    future multi-base transform.

    Same missing-column / unsupported-type semantics as
    :func:`_extract_base`. Errors include the offending column name so
    feature-selection drops are easy to diagnose.
    """
    if len(base_columns) == 0:
        raise ValueError(
            "CompositeTargetEstimator: base_columns is empty; multi-base "
            "transforms require at least one base column."
        )
    # Single-select fast path: a polars ``.select(cols).to_numpy()`` materialises all K cols in one Arrow
    # buffer (~3-5x faster than the per-column-then-column_stack loop on K>=4 cols, validated on a
    # 1M-row x 8-col synthetic in bench_extract_base_matrix.py). Same for pandas: ``loc[:, cols].to_numpy()``
    # avoids per-column dispatch.
    cols_list = list(base_columns)
    if _is_polars_df(X):
        missing = [c for c in cols_list if c not in X.columns]
        if missing:
            raise KeyError(
                f"CompositeTargetEstimator: base columns {missing!r} missing from X. "
                "If feature selection (MRMR/RFECV) is dropping them, add the base_columns "
                "to forced_keep_columns in the feature selection config."
            )
        arr = X.select(cols_list).to_numpy()
        return arr.astype(np.float64, copy=False) if arr.dtype != np.float64 else arr
    if isinstance(X, pd.DataFrame):
        missing = [c for c in cols_list if c not in X.columns]
        if missing:
            raise KeyError(
                f"CompositeTargetEstimator: base columns {missing!r} missing from X. "
                "Columns: " + ", ".join(map(str, X.columns[:8])) + ("..." if len(X.columns) > 8 else "")
            )
        return X.loc[:, cols_list].to_numpy(dtype=np.float64, copy=False)
    # Fallback: route per-column for unknown X type (preserves prior behaviour for ndarray-with-names etc.).
    cols = [_extract_base(X, c) for c in cols_list]
    return np.column_stack(cols)


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

        instance.estimator_ = fitted_inner
        instance.fitted_params_ = {
            **dict(transform_fitted_params),
            "y_clip_low": y_clip_low,
            "y_clip_high": y_clip_high,
            "y_train_median": y_train_median,
        }
        instance.runtime_stats_ = {
            "predict_calls": 0,
            "predict_rows_total": 0,
            "domain_violation_rows": 0,
            "y_clip_low_hits": 0,
            "y_clip_high_hits": 0,
        }
        return instance

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
        # (e.g. well_id strings). Tree models like LightGBM reject object
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
        y_clip_low, y_clip_high = _y_train_clip_bounds(y_train)
        y_train_median = float(np.median(y_train))

        self.estimator_ = estimator
        self.fitted_params_ = {
            **transform_params,
            "y_clip_low": y_clip_low,
            "y_clip_high": y_clip_high,
            "y_train_median": y_train_median,
            "n_train_valid": int(y_train.size),
            "n_train_invalid": n_invalid,
        }
        # Best-effort feature names (pandas / polars).
        try:
            self.feature_names_in_ = list(X.columns)
        except Exception:
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

    def _predict_unclipped(self, X: Any) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Internal: compute the pre-clip y-scale prediction plus the row count and the params dict.

        Pulled out of ``predict`` so callers that need the raw (un-clipped) y-hat for diagnostics (e.g. honest pre-clip train
        RMSE - which the clip cannot improve on rows that are in-envelope by construction) can get it without re-running
        the inverse twice. ``predict`` is implemented as a thin clip-applying wrapper around this.
        """
        if not hasattr(self, "estimator_"):
            raise RuntimeError(
                "CompositeTargetEstimator.predict called before fit."
            )
        transform = get_transform(self.transform_name)
        params = self.fitted_params_

        # Determine the row count for the prediction. For unary
        # transforms (``requires_base=False``) we cannot rely on a base
        # column for the size; ask the inner estimator's prediction
        # length implicitly by deferring extraction.
        if transform.requires_base:
            base_columns = self._resolve_base_columns()
            base_arr = self._extract_base_for_transform(X, base_columns)
        else:
            # Placeholder zeros sized to predict-time inputs. We let the
            # inner predict (below) drive the row count, and re-size the
            # placeholder after t_hat is known. Cheap (zeros allocate
            # lazily on virtually-mmapped memory).
            base_arr = None
        # Domain check at predict: y is unknown, so we ask the transform
        # to gate on base-side conditions only (e.g. base > 0 for
        # logratio, |base| > 0 for ratio). The ``y=None`` sentinel is
        # part of the domain_check contract -- transforms whose forward
        # is purely y-side (none here) would still need a y; the four
        # core transforms all degrade cleanly. For unary transforms
        # (no base) we delegate to the transform's own domain_check on
        # ``y=None, base=None`` which the registry adapter handles by
        # returning all-True for the t_hat row count below.
        if transform.requires_base:
            domain_ok = transform.domain_check(None, base_arr)
        else:
            domain_ok = None  # sized after t_hat is computed

        # Grouped-transform support at predict.
        # Extract per-row group labels and thread them as kwargs to the
        # inverse call. Unseen groups (not present at fit) fall back to
        # global alpha/beta inside the transform's inverse impl, so no
        # caller-side handling is needed here.
        inverse_kwargs: dict[str, Any] = {}
        if transform.requires_groups:
            if not self.group_column:
                raise ValueError(
                    f"CompositeTargetEstimator.predict: transform "
                    f"'{self.transform_name}' requires groups but "
                    f"``group_column`` is not configured."
                )
            inverse_kwargs["groups"] = _extract_groups(X, self.group_column)

        # Pass X through to the inner unchanged. NEVER materialise the
        # frame here -- on a 100 GB polars frame a silent conversion
        # blows the host out of memory. Caller is responsible for
        # ensuring the frame type is acceptable to the inner estimator
        # (mlframe strategies handle this at the suite level).
        # For grouped transforms: strip group_column from X before
        # predict so the inner doesn't see the (typically string)
        # plumbing column -- same logic as fit().
        X_for_inner = (
            self._drop_columns(X, [self.group_column])
            if transform.requires_groups and self.group_column
            else X
        )
        t_hat = np.asarray(
            self.estimator_.predict(X_for_inner), dtype=np.float64,
        ).reshape(-1)

        # Unary transforms have no base column at predict-time; size the
        # placeholder + domain mask to match t_hat. Inverse ignores the
        # base arg (registry adapter; see composite_transforms.py).
        if not transform.requires_base:
            base_arr = np.zeros_like(t_hat)
            domain_ok = np.ones_like(t_hat, dtype=bool)

        # Apply inverse only on valid rows; fill the rest with fallback.
        if domain_ok.all():
            y_hat = transform.inverse(t_hat, base_arr, params, **inverse_kwargs)
        else:
            y_hat = np.full_like(t_hat, fill_value=np.nan, dtype=np.float64)
            # Inverse on valid rows only; placeholder base for invalid
            # rows is irrelevant since we overwrite immediately.
            base_safe = np.where(domain_ok, base_arr, 1.0)
            y_hat_valid = transform.inverse(
                t_hat, base_safe, params, **inverse_kwargs,
            )
            y_hat[domain_ok] = y_hat_valid[domain_ok]
            # Fallback for invalid rows.
            if self.fallback_predict == "y_train_median":
                y_hat[~domain_ok] = params["y_train_median"]
            elif self.fallback_predict == "nan":
                pass  # already NaN
            else:
                raise ValueError(
                    f"CompositeTargetEstimator: unknown fallback_predict "
                    f"'{self.fallback_predict}'; choose 'y_train_median' or 'nan'."
                )

        n_violation = int((~domain_ok).sum())
        n_rows = int(t_hat.size)
        meta = {"params": params, "n_violation": n_violation, "n_rows": n_rows}
        return y_hat, n_rows, meta

    def predict_pre_clip(self, X: Any) -> np.ndarray:
        """Return the inverse-of-transform y-prediction WITHOUT the train-envelope clip applied.

        The post-hoc clip ``[y_clip_low, y_clip_high]`` is a no-op on train rows by construction (they ARE the envelope) so any
        train metric computed on the clipped prediction is identical to the un-clipped one. The clip only does work on val /
        test rows that drift outside the train range. Computing pre- AND post-clip RMSE separately exposes the clip's actual
        contribution per split instead of folding the no-op train case into a falsely "improved" headline number.
        """
        y_hat_unclipped, _, _ = self._predict_unclipped(X)
        return y_hat_unclipped

    def predict(self, X: Any) -> np.ndarray:
        y_hat, n, meta = self._predict_unclipped(X)
        params = meta["params"]
        n_violation = meta["n_violation"]
        t_hat_size = meta["n_rows"]

        # Post-inverse y-clip. Prediction outside the train envelope is
        # almost always exp() / division blow-up; clip and count for
        # observability.
        low = params["y_clip_low"]
        high = params["y_clip_high"]
        low_hits = int(np.sum(y_hat < low))
        high_hits = int(np.sum(y_hat > high))
        if low_hits or high_hits:
            y_hat = np.clip(y_hat, low, high)

        # Update runtime_stats_ cumulatively.
        n = t_hat_size
        rs = self.runtime_stats_
        rs["predict_calls"] += 1
        rs["predict_rows_total"] += n
        rs["domain_violation_rows"] += n_violation
        rs["y_clip_low_hits"] += low_hits
        rs["y_clip_high_hits"] += high_hits

        # Optional metrics callback (Prometheus / StatsD / DataDog hook).
        # Failures are logged at DEBUG and swallowed -- monitoring must
        # never break inference.
        cb = getattr(self, "runtime_stats_callback", None)
        if cb is not None:
            try:
                cb({
                    "transform_name": self.transform_name,
                    "base_column": self.base_column,
                    "batch_n": n,
                    "batch_domain_violation_rows": n_violation,
                    "batch_y_clip_low_hits": low_hits,
                    "batch_y_clip_high_hits": high_hits,
                    "cumulative_predict_calls": rs["predict_calls"],
                    "cumulative_predict_rows_total": rs["predict_rows_total"],
                    "cumulative_domain_violation_rows": rs["domain_violation_rows"],
                    "cumulative_y_clip_low_hits": rs["y_clip_low_hits"],
                    "cumulative_y_clip_high_hits": rs["y_clip_high_hits"],
                })
            except Exception as cb_err:
                logger.debug(
                    "[CompositeTargetEstimator] runtime_stats_callback failed: %s",
                    cb_err,
                )
        return y_hat

    # ------------------------------------------------------------------
    # Quantile predictions (ConfidenceAnalysisConfig integration)
    # ------------------------------------------------------------------

    def predict_quantile(
        self, X: Any, alpha: float | Sequence[float] = 0.5,
    ) -> np.ndarray:
        """y-scale quantile prediction by inverting the inner's
        T-scale quantile.

        Requires the inner estimator to expose ``predict_quantile(X,
        alpha)`` -- e.g. CatBoost ``MultiQuantile``, LightGBM
        ``quantile_alpha``, sklearn ``QuantileRegressor``. The wrapper
        calls inner -> ``T_q`` then applies the transform's inverse.

        Accepts either a scalar ``alpha`` (returns ``(n_samples,)``) or
        an array-like of K quantile levels (returns ``(n_samples, K)``);
        the 2-D path preserves the quantile dimension so per-quantile
        ensemble blending (``predict_quantile_ensemble``) can operate
        column-wise instead of collapsing to a single point estimate.

        **Quantile preservation under inverse**:

        | transform        | inverse                  | preserves quantiles?               |
        |------------------|--------------------------|------------------------------------|
        | ``diff``           | ``T + base``               | always (monotonic in T)            |
        | ``linear_residual``| ``T + alpha*base + beta``  | always                             |
        | ``logratio``       | ``base * exp(softcap(T))`` | yes (base > 0 already required)    |
        | ``ratio``          | ``T * base``               | flips when ``base < 0``; raises    |

        For ``ratio`` with mixed-sign base raises ``NotImplementedError``
        rather than silently swap the high / low quantiles.
        """
        if not hasattr(self, "estimator_"):
            raise RuntimeError(
                "CompositeTargetEstimator.predict_quantile called before fit."
            )
        inner = self.estimator_
        if not hasattr(inner, "predict_quantile"):
            raise NotImplementedError(
                f"inner estimator {type(inner).__name__!r} does not expose "
                "predict_quantile(X, alpha); wrap a quantile regressor "
                "(CatBoost MultiQuantile / LightGBM quantile_alpha / "
                "sklearn QuantileRegressor) instead."
            )

        transform = get_transform(self.transform_name)
        params = self.fitted_params_
        base_columns = self._resolve_base_columns()
        base_arr = self._extract_base_for_transform(X, base_columns)

        # Sign-flip guard for ratio: T < 0 and base < 0 produces
        # positive y; high T-quantile would swap to low y-quantile.
        if self.transform_name == "ratio":
            if np.any(base_arr < 0):
                raise NotImplementedError(
                    "predict_quantile is not supported for transform 'ratio' "
                    "when base contains negative values: y = T * base flips "
                    "the quantile ordering on negative-base rows. Use "
                    "predict() for point predictions or switch transform."
                )

        # logratio domain check: base must be > 0 (already required at
        # fit time, but predict-time inputs may differ).
        if self.transform_name == "logratio":
            if np.any(base_arr <= 0):
                raise NotImplementedError(
                    "predict_quantile for transform 'logratio' requires "
                    "base > 0 on every row. Filter the input or use "
                    "predict() with the wrapper's domain fallback."
                )

        alpha_is_scalar = np.isscalar(alpha)
        try:
            t_raw = np.asarray(
                inner.predict_quantile(X, alpha), dtype=np.float64,
            )
        except TypeError:
            # Some libs name the kwarg differently (e.g. quantile=, q=).
            t_raw = np.asarray(
                inner.predict_quantile(X, alpha=alpha), dtype=np.float64,
            )

        # Preserve quantile dimensionality: scalar alpha -> 1-D (n_samples,);
        # array alpha -> 2-D (n_samples, K). Flattening unconditionally would
        # collapse a (n_samples, K) multi-quantile head into one mean-like
        # vector and destroy the predictive interval downstream blenders need.
        low = params.get("y_clip_low", float("-inf"))
        high = params.get("y_clip_high", float("inf"))
        if alpha_is_scalar:
            t_q = t_raw.reshape(-1)
            y_q = transform.inverse(t_q, base_arr, params)
            return np.clip(y_q, low, high)

        if t_raw.ndim == 1:
            # Inner emits per-alpha 1-D and was called with a vector -- broadcast
            # against the single base column we already extracted (which is 1-D).
            y_q = transform.inverse(t_raw, base_arr, params)
            return np.clip(y_q, low, high).reshape(-1, 1)
        if t_raw.ndim != 2:
            raise ValueError(
                f"CompositeTargetEstimator.predict_quantile: inner returned ndim={t_raw.ndim}; expected 1 or 2."
            )
        # Per-column inverse: base_arr is (n_samples,), so reshape to broadcast.
        cols: list[np.ndarray] = []
        for k in range(t_raw.shape[1]):
            y_col = transform.inverse(t_raw[:, k], base_arr, params)
            cols.append(np.clip(y_col, low, high))
        return np.column_stack(cols)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def _require_fitted(self, attr: str) -> Any:
        """sklearn convention: pre-fit access to ``feature_importances_``/``coef_``/
        ``intercept_`` raises ``NotFittedError`` (was: silent ``None`` return).
        """
        est = getattr(self, "estimator_", None)
        if est is None:
            raise NotFittedError(
                f"CompositeTargetEstimator has no fitted ``estimator_``; "
                f"call .fit(...) before accessing .{attr}."
            )
        return est

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self._require_fitted("feature_importances_"), "feature_importances_", None)

    @property
    def coef_(self) -> np.ndarray | None:
        return getattr(self._require_fitted("coef_"), "coef_", None)

    @property
    def intercept_(self) -> float | None:
        return getattr(self._require_fitted("intercept_"), "intercept_", None)

    def get_booster(self):
        """XGBoost shim. Uses the same ``_require_fitted`` path as the
        sklearn-convention properties so the error class is
        ``NotFittedError`` rather than the bespoke ``RuntimeError`` it
        used to raise - callers can catch both shapes uniformly."""
        return self._require_fitted("get_booster").get_booster()

    @property
    def booster_(self):
        """LightGBM shim. Raises ``NotFittedError`` when called before
        ``fit`` so callers fail loudly instead of receiving a silent
        ``None`` and crashing downstream (sklearn convention)."""
        return getattr(self._require_fitted("booster_"), "booster_", None)

    @property
    def n_features_in_(self) -> int | None:
        """sklearn convention: raise ``NotFittedError`` before fit so
        callers can't probe an unfit wrapper for feature count and
        silently receive ``None``."""
        return getattr(self._require_fitted("n_features_in_"), "n_features_in_", None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Rolling-buffer streaming alpha refit
    # ------------------------------------------------------------------

    def update(self, y_recent: Any, base_recent: Any) -> dict[str, Any]:
        """Streaming-update interface: append new (y, base) observations to a rolling buffer and run a drift check.

        Caller invokes this method on incoming production data; when the buffer fills past ``online_refit_min_buffer_n`` AND the Chow-style z-score crosses ``online_refit_z_threshold``, the wrapper's ``fitted_params_["alpha"]`` / ``["beta"]`` get updated in-place so subsequent predict() calls use the drift-corrected coefficients.

        Only supported for the ``linear_residual`` transform (the only one with closed-form alpha/beta in the fitted params; other transforms have transform-specific params that aren't suitable for streaming refit). For other transforms, raises ``NotImplementedError``.

        Parameters
        ----------
        y_recent, base_recent
            New observation arrays (1-D, equal length). Appended to the rolling buffer; oldest rows evicted (FIFO) once the buffer reaches ``online_refit_buffer_n``.

        Returns
        -------
        info: dict carrying the same fields as ``streaming_alpha_check_and_refit`` (refit / z_score / alpha_buffer / beta_buffer / reason) PLUS ``buffer_n_total`` (current buffer size after the update).
        """
        if not getattr(self, "online_refit_enabled", False):
            raise RuntimeError(
                "CompositeTargetEstimator.update: online_refit_enabled is False. Set it to True in __init__ to enable streaming refit."
            )
        if self.transform_name not in ("linear_residual",):
            raise NotImplementedError(
                f"streaming alpha refit only supported for 'linear_residual'; got transform_name={self.transform_name!r}. Other transforms have transform-specific params (eps for ratio, mad_eff for logratio, per-bin medians for quantile_residual, etc.) that don't fit the closed-form alpha/beta refit pattern."
            )
        if not hasattr(self, "fitted_params_"):
            raise RuntimeError(
                "CompositeTargetEstimator.update called before fit (no fitted_params_ to refit)."
            )
        # Lazy-init the rolling buffers on first update.
        if not hasattr(self, "_buffer_y_"):
            self._buffer_y_ = deque(maxlen=int(self.online_refit_buffer_n))
            self._buffer_base_ = deque(maxlen=int(self.online_refit_buffer_n))
        y_arr = np.asarray(y_recent, dtype=np.float64).reshape(-1)
        base_arr = np.asarray(base_recent, dtype=np.float64).reshape(-1)
        if y_arr.size != base_arr.size:
            raise ValueError(
                f"CompositeTargetEstimator.update: y_recent ({y_arr.size} rows) and base_recent ({base_arr.size} rows) must have equal length."
            )
        self._buffer_y_.extend(y_arr.tolist())
        self._buffer_base_.extend(base_arr.tolist())
        buffer_n = len(self._buffer_y_)
        # Lazy import to break the composite_estimator <-> composite_streaming
        # cycle (composite_streaming lazy-imports _linear_residual_fit from
        # composite, which re-exports CompositeTargetEstimator from us).
        from .composite_streaming import streaming_alpha_check_and_refit
        # Run drift check; the helper handles the buffer-too-small case.
        new_alpha, new_beta, info = streaming_alpha_check_and_refit(
            np.asarray(self._buffer_y_, dtype=np.float64),
            np.asarray(self._buffer_base_, dtype=np.float64),
            current_alpha=float(self.fitted_params_.get("alpha", 0.0)),
            current_beta=float(self.fitted_params_.get("beta", 0.0)),
            z_threshold=float(self.online_refit_z_threshold),
            min_buffer_n=int(self.online_refit_min_buffer_n),
        )
        info["buffer_n_total"] = buffer_n
        if info.get("refit"):
            # Update params in-place. The wrapper's predict() reads these on every call so the next predict will use the drifted alpha / beta.
            self.fitted_params_["alpha"] = new_alpha
            self.fitted_params_["beta"] = new_beta
            logger.info(
                "[CompositeTargetEstimator.update] streaming refit fired (z=%.2f > %.2f). alpha %.4f -> %.4f, beta %.4f -> %.4f. buffer_n=%d",
                info["z_score"], self.online_refit_z_threshold,
                info["alpha_buffer"] if info["alpha_buffer"] is not None else float("nan"),
                new_alpha,
                info["beta_buffer"] if info["beta_buffer"] is not None else float("nan"),
                new_beta, buffer_n,
            )
        return info

    def get_buffer_state(self) -> dict[str, Any]:
        """Diagnostic: returns the current rolling-buffer state without exposing the deque internals to callers.

        Useful for monitoring / unit tests. Returns ``{"buffer_n": int, "buffer_full": bool, "alpha_current": float, "beta_current": float}``.
        """
        buf_n = len(getattr(self, "_buffer_y_", []))
        return {
            "buffer_n": buf_n,
            "buffer_full": buf_n >= int(self.online_refit_buffer_n),
            "alpha_current": float(getattr(self, "fitted_params_", {}).get("alpha", float("nan"))),
            "beta_current": float(getattr(self, "fitted_params_", {}).get("beta", float("nan"))),
        }


# ----------------------------------------------------------------------
# Per-quantile ensemble blending
# ----------------------------------------------------------------------

def predict_quantile_ensemble(
    members: Sequence[Any],
    X: Any,
    quantiles: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Blend multi-quantile predictions from ``members`` PER-QUANTILE COLUMN.

    Each member ``m`` must expose ``predict_quantile(X, alpha)`` returning a scalar-alpha
    1-D vector OR a multi-alpha 2-D ``(n_samples, n_quantiles)`` matrix. The blend is
    elementwise (per row x per quantile column) -- it preserves the quantile dimension
    rather than collapsing to a single ``.predict()``-style scalar per row.

    Why this matters: arithmetic-mean of the POINT predictions of K quantile heads
    discards the predictive interval entirely (each member would have answered a
    different question if asked for q10 vs q90). The correct blend stacks the q10
    columns from every member, the q50 columns from every member, the q90 columns
    from every member, and averages each STACK independently.

    Parameters
    ----------
    members : sequence of fitted estimators
        Each must expose ``predict_quantile(X, alpha=float)``. When the member's
        ``predict_quantile`` natively supports a multi-quantile array (CatBoost
        MultiQuantile, XGBoost ``quantile_alpha=[...]``), it is called once per
        ``alpha`` for back-compat with the scalar-alpha contract on
        ``CompositeTargetEstimator.predict_quantile``.
    X : feature frame
        Passed through to each member unchanged.
    quantiles : sequence of float
        Quantile levels to blend at. Must be sorted ascending; values in (0, 1).
    weights : sequence of float, optional
        Per-member non-negative weights. Renormalised to sum to 1. ``None`` means
        equal weights (arithmetic mean across members per column).

    Returns
    -------
    np.ndarray, shape (n_samples, n_quantiles)
        Blended per-quantile predictions in original y-scale.

    Raises
    ------
    ValueError
        On empty member list, mismatched per-member output shapes, mismatched
        quantile sets across members, weights length mismatch, non-sorted /
        out-of-range quantiles, or weights summing to zero.
    """
    if not members:
        raise ValueError("predict_quantile_ensemble: members is empty")

    quantiles_arr = np.asarray(quantiles, dtype=np.float64)
    if quantiles_arr.ndim != 1 or quantiles_arr.size == 0:
        raise ValueError(
            f"predict_quantile_ensemble: quantiles must be a non-empty 1-D sequence, got shape {quantiles_arr.shape!r}"
        )
    if np.any((quantiles_arr <= 0.0) | (quantiles_arr >= 1.0)):
        raise ValueError(
            f"predict_quantile_ensemble: quantiles must be strictly between 0 and 1; got {quantiles_arr.tolist()!r}"
        )
    if not np.all(np.diff(quantiles_arr) > 0):
        raise ValueError(
            f"predict_quantile_ensemble: quantiles must be sorted ascending and unique; got {quantiles_arr.tolist()!r}"
        )

    if weights is None:
        w = np.full(len(members), 1.0 / len(members), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(members),):
            raise ValueError(
                f"predict_quantile_ensemble: weights length {w.shape!r} != n_members {len(members)}"
            )
        if np.any(w < 0):
            raise ValueError(
                f"predict_quantile_ensemble: weights must be non-negative; got {w.tolist()!r}"
            )
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("predict_quantile_ensemble: weights sum to zero")
        w = w / w_sum

    # Materialise every member's (n_samples, n_quantiles) matrix. Different quantile sets
    # MUST raise -- silently truncating the union would hand back a confidence interval
    # at the wrong nominal coverage.
    per_member: list[np.ndarray] = []
    expected_shape: Optional[Tuple[int, int]] = None
    quantiles_list = [float(v) for v in quantiles_arr]
    for idx, m in enumerate(members):
        if not hasattr(m, "predict_quantile"):
            raise ValueError(
                f"predict_quantile_ensemble: member {idx} ({type(m).__name__!r}) lacks predict_quantile(X, alpha); "
                "wrap each member in CompositeTargetEstimator or another quantile-aware estimator."
            )
        # Single batched call when the member's ``predict_quantile`` accepts a sequence
        # (e.g. CatBoost MultiQuantile, CompositeTargetEstimator). We attempt one batched
        # call and fall back to per-alpha scalar calls only on TypeError / ValueError /
        # mismatched-shape -- LightGBM and sklearn QuantileRegressor are scalar-only.
        member_mat: Optional[np.ndarray] = None
        try:
            batched = m.predict_quantile(X, quantiles_list)
            batched_arr = np.asarray(batched, dtype=np.float64)
            if batched_arr.ndim == 2 and batched_arr.shape[1] == len(quantiles_list):
                member_mat = batched_arr
        except (TypeError, ValueError):
            member_mat = None
        if member_mat is None:
            cols: list[np.ndarray] = []
            for alpha_v in quantiles_arr:
                try:
                    pred = m.predict_quantile(X, float(alpha_v))
                except TypeError:
                    pred = m.predict_quantile(X, alpha=float(alpha_v))
                pred_arr = np.asarray(pred, dtype=np.float64)
                if pred_arr.ndim == 2:
                    if pred_arr.shape[1] != 1:
                        raise ValueError(
                            f"predict_quantile_ensemble: member {idx} returned shape {pred_arr.shape!r} for scalar alpha={alpha_v}; "
                            "expected (n_samples,) or (n_samples, 1)."
                        )
                    pred_arr = pred_arr[:, 0]
                elif pred_arr.ndim != 1:
                    raise ValueError(
                        f"predict_quantile_ensemble: member {idx} predict_quantile returned ndim={pred_arr.ndim} (expected 1 or 2)"
                    )
                cols.append(pred_arr)
            member_mat = np.column_stack(cols)
        if expected_shape is None:
            expected_shape = member_mat.shape
        elif member_mat.shape != expected_shape:
            raise ValueError(
                f"predict_quantile_ensemble: member {idx} produced shape {member_mat.shape!r}, "
                f"member 0 produced {expected_shape!r}. Per-quantile blend requires identical "
                "(n_samples, n_quantiles) across members -- different quantile sets are ambiguous "
                "(union vs intersection vs interpolated re-evaluation)."
            )
        per_member.append(member_mat)

    # Stack along a new member axis -> (M, N, K), then weighted-mean over M -> (N, K).
    stacked = np.stack(per_member, axis=0)
    # Broadcast weights along (N, K) without materialising a full (M, N, K) weight tensor.
    blended = np.tensordot(w, stacked, axes=(0, 0))
    return blended
