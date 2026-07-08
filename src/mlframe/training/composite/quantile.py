"""Native quantile (pinball) composite estimator.

``CompositeQuantileEstimator`` fits ONE inner regressor per requested quantile
on the transform ``T = f(y, base)`` (reusing the existing transform registry --
``linear_residual`` / ``diff`` / ``logratio`` / the unary y-transforms / ...),
predicts each quantile on the T-scale, inverts to the y-scale via the wrapper's
transform inverse, and ENFORCES non-crossing (sorts the predicted quantiles
per row so ``q_low <= ... <= q_high``).

Why a dedicated estimator (vs ``CompositeTargetEstimator.predict_quantile``).
The single-inner ``predict_quantile`` requires the inner to natively expose a
multi-quantile head (CatBoost ``MultiQuantile`` / sklearn ``QuantileRegressor``).
The far more common production case is a plain GBDT (LightGBM / XGBoost) trained
with the pinball objective at a FIXED alpha -- one model per quantile. This
estimator orchestrates that fan-out: each inner is a clone of ``base_estimator``
with its ``alpha`` (LightGBM) / ``alpha`` of the pinball objective set to the
requested quantile, fit on the SAME ``T`` target. Because each inner minimises
the pinball loss at its alpha, its point ``.predict()`` IS the alpha-quantile of
``T``; the wrapper's inverse then maps that to the alpha-quantile of ``y`` for
the monotone-in-T transforms.

Reuse. The per-quantile fit/predict path is delegated to a private
``CompositeTargetEstimator`` per quantile, so ALL of the transform fit, the
domain filtering, the T-clip, the y-clip, the fallback routing, the grouped /
multi-base plumbing, and the NaN-safety guards are shared verbatim -- this class
adds only (a) the per-quantile alpha wiring onto the inner and (b) the
non-crossing sort. No transform is added, so no provenance / formula entry is
needed.

Non-crossing. Pinball-trained heads are fit INDEPENDENTLY, so nothing forces
``y_hat(q=0.1) <= y_hat(q=0.9)`` row-wise; they can cross, especially on small
samples / extrapolation. We restore monotonicity by sorting each row of the
``(n, n_q)`` prediction matrix ascending (the standard rearrangement; Chernozhukov
et al. 2010 show sorting quantile estimates never increases estimation error and
restores monotonicity). For transforms whose inverse is DECREASING in T
(``reciprocal_residual``, ``y = 1/(T + 1/base)``) the inverse FLIPS quantile order:
the head trained for the tau-quantile of T produces the (1-tau)-quantile of y. The
estimator detects this at fit (``_inverse_decreasing_``) and serves each column from
the COMPLEMENTARY head, so the column labels are correct independent of the
non-crossing sort -- ``predict_quantile(X)[:, j]`` is always the requested level's
y-quantile, not its mirror. For ``ratio`` with mixed-sign base the inverse flips
quantile order PER ROW (sign-dependent), which a fit-time scalar flag cannot
disambiguate -- that case raises in the inner head's ``predict_quantile``.

cProfile (fit + predict_quantile, n=50k x 2 cols, 5 quantiles, LightGBM
n_estimators=100): 0.98 s total, of which ~0.74 s is inside LightGBM ``train`` /
``basic.py:update`` (the inner boosting of the 5 heads) and ~0.23 s the inner
``predict`` calls. The wrapper-side work -- the single shared transform fit, the
per-head inverse, and the per-row non-crossing ``np.sort`` -- is <2 ms combined,
so there is NO actionable wrapper-side speedup: the cost is the K inner fits,
each of which already uses LightGBM's own ``n_jobs`` threading. (The heads are
fit serially; a process-pool fan-out is a future option but would not be
bit-identical and is dominated by the already-parallel inner anyway.)
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .estimator import CompositeTargetEstimator
from .transforms import get_transform

logger = logging.getLogger(__name__)

# Default quantile grid: a symmetric 5-level set giving an 80% central band
# (0.1, 0.9) plus the 50% median and the 0.25 / 0.75 quartiles. Covers the most
# common interval-reporting needs out of the box.
_DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _set_inner_quantile_alpha(estimator: Any, q: float) -> Any:
    """Return a clone of ``estimator`` configured to learn the ``q``-quantile.

    Sets the pinball objective + alpha on the clone via ``set_params`` for the
    three GBDT families this codebase already pulls in, falling back to a bare
    ``alpha`` / ``quantile`` param for sklearn-style quantile regressors:

    - LightGBM (``LGBMRegressor``): ``objective='quantile'`` + ``alpha=q``.
    - XGBoost (``XGBRegressor``, >=1.7): ``objective='reg:quantileerror'`` +
      ``quantile_alpha=q``.
    - sklearn ``QuantileRegressor`` / ``GradientBoostingRegressor(loss='quantile')``:
      ``quantile=q`` / ``alpha=q``.

    The configuration is best-effort and introspection-gated on the clone's
    declared ``get_params`` keys so an unknown estimator is not handed a param
    it would reject; if NO recognised quantile knob is present we raise, because
    silently fitting K identical mean-regressors would return K identical
    "quantiles" (a silent correctness bug, not a degraded-but-valid result).
    """
    inner = clone(estimator)
    try:
        valid_keys = set(inner.get_params(deep=False).keys())
    except Exception:  # pragma: no cover - non-sklearn estimator
        valid_keys = set()

    type_name = type(inner).__name__
    set_any = False
    # LightGBM: objective='quantile', alpha=q. ``alpha`` is a LightGBM
    # **kwargs param so it is NOT in ``get_params`` until set, but
    # ``set_params(alpha=...)`` stores it; gate on the type + ``objective``.
    if "objective" in valid_keys and "LGBM" in type_name:
        inner.set_params(objective="quantile", alpha=float(q))
        set_any = True
    # XGBoost >=1.7 pinball: reg:quantileerror + quantile_alpha.
    elif "quantile_alpha" in valid_keys:
        params: dict[str, Any] = {"quantile_alpha": float(q)}
        if "objective" in valid_keys:
            params["objective"] = "reg:quantileerror"
        inner.set_params(**params)
        set_any = True
    # sklearn QuantileRegressor: quantile=q (linear pinball).
    elif "quantile" in valid_keys:
        inner.set_params(quantile=float(q))
        set_any = True
    # sklearn GradientBoostingRegressor(loss='quantile'): alpha=q.
    elif "alpha" in valid_keys and "loss" in valid_keys:
        inner.set_params(loss="quantile", alpha=float(q))
        set_any = True
    # Generic fallback: a bare ``alpha`` knob (e.g. CatBoost single-quantile
    # via loss_function, or a custom pinball wrapper exposing alpha).
    elif "alpha" in valid_keys:
        inner.set_params(alpha=float(q))
        set_any = True

    if not set_any:
        raise ValueError(
            f"CompositeQuantileEstimator: inner estimator {type_name!r} exposes "
            "no recognised quantile knob (objective/alpha for LightGBM, "
            "quantile_alpha for XGBoost, quantile for sklearn QuantileRegressor, "
            "loss/alpha for GradientBoostingRegressor). Pass a pinball-capable "
            "regressor, or set the objective on the prototype yourself."
        )
    return inner


def _transform_inverse_decreasing(transform_name: str) -> bool:
    """True when the transform inverse ``y = inverse(T, base)`` DECREASES in ``T``.

    For such transforms the head trained for the tau-quantile of ``T`` produces the
    (1-tau)-quantile of ``y`` (the inverse flips the order), so the per-head
    columns must be re-labelled by their complementary level. Detected by probing
    the registered inverse with an increasing ``T`` ramp at a representative
    positive base; base-free / sign-ambiguous transforms (``ratio`` with mixed
    base) are NOT handled here and are caught by ``predict_quantile`` of the inner.
    """
    try:
        transform = get_transform(transform_name)
    except Exception:  # pragma: no cover - unknown name surfaces earlier at fit
        return False
    if not getattr(transform, "requires_base", True):
        # Unary y-transforms (log_y / cbrt_y / yeo_johnson_y) are all monotone
        # INCREASING in T by construction; no flip.
        return False
    # Probe a SMALL increasing T ramp around 0 at a representative positive base.
    # Reciprocal-type inverses have a pole at ``T = -1/base``; a wide ramp would
    # straddle it and read as non-monotone. A tight ramp (+/- 1e-3) stays on one
    # branch so the local sign of dy/dT is read cleanly.
    t_probe = np.array([-1e-3, 0.0, 1e-3], dtype=np.float64)
    base_probe = np.full(3, 2.0, dtype=np.float64)
    try:
        params = transform.fit(np.array([1.0, 2.0, 3.0]), base_probe)
        y_probe = np.asarray(transform.inverse(t_probe, base_probe, params), dtype=np.float64).reshape(-1)
    except Exception:  # pragma: no cover - probe failure -> assume increasing
        return False
    if y_probe.shape[0] != 3 or not np.all(np.isfinite(y_probe)):
        return False
    # Strictly decreasing across the ramp -> the inverse flips quantile order.
    return bool(y_probe[0] > y_probe[1] > y_probe[2])


class CompositeQuantileEstimator(BaseEstimator, RegressorMixin):
    """Native pinball composite: one inner per quantile on the transform ``T``.

    Parameters
    ----------
    base_estimator
        Unfitted pinball-capable inner prototype. Cloned once PER requested
        quantile; each clone gets its ``alpha`` set to that quantile via
        :func:`_set_inner_quantile_alpha` (LightGBM ``objective='quantile'`` /
        XGBoost ``reg:quantileerror`` / sklearn ``QuantileRegressor``). The
        prototype passed in stays untouched.
    transform_name
        One of :func:`mlframe.training.composite.list_transforms`. Determines the
        forward / inverse applied to the target. Same registry the point
        estimator uses; ``linear_residual`` is the canonical choice (additive,
        always quantile-preserving under inverse).
    base_column / base_columns / group_column
        Same semantics as :class:`CompositeTargetEstimator` -- the base feature
        the transform residualises against (single or multi), and the group
        label column for grouped transforms.
    quantiles
        Ascending sequence of quantile levels in (0, 1). Default
        ``(0.1, 0.25, 0.5, 0.75, 0.9)``. Validated at fit; ``predict_quantile``
        may request a SUBSET of these (re-using the already-fitted heads) but
        not a level that was not fitted.
    fallback_predict / drop_invalid_rows
        Forwarded to each per-quantile :class:`CompositeTargetEstimator`.
    enforce_non_crossing
        When True (default), sort each row of the predicted ``(n, n_q)`` matrix
        ascending so ``q_low <= ... <= q_high``. Set False only for diagnosing
        the raw per-head crossing rate.

    Attributes set at fit
    ---------------------
    estimators_
        Dict ``{quantile: fitted CompositeTargetEstimator}``.
    quantiles_
        The fitted quantile grid (sorted ndarray).
    feature_names_in_
        Column names seen at fit (best effort), inherited from the median head.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        transform_name: str = "linear_residual",
        base_column: str = "",
        quantiles: Sequence[float] | None = None,
        fallback_predict: str = "y_train_median",
        drop_invalid_rows: bool = True,
        base_columns: Sequence[str] | None = None,
        group_column: str | None = None,
        enforce_non_crossing: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.transform_name = transform_name
        self.base_column = base_column
        self.quantiles = quantiles
        self.fallback_predict = fallback_predict
        self.drop_invalid_rows = drop_invalid_rows
        self.base_columns = base_columns
        self.group_column = group_column
        self.enforce_non_crossing = enforce_non_crossing

    def _resolve_quantiles(self) -> np.ndarray:
        """Validate + sort the configured quantile grid into a 1-D ndarray."""
        q = self.quantiles if self.quantiles is not None else _DEFAULT_QUANTILES
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_arr.size == 0:
            raise ValueError(f"CompositeQuantileEstimator: quantiles is empty; got {self.quantiles!r}.")
        if np.any((q_arr <= 0.0) | (q_arr >= 1.0)):
            raise ValueError("CompositeQuantileEstimator: quantiles must be strictly in (0, 1); " f"got {q_arr.tolist()!r}.")
        q_sorted = np.unique(q_arr)
        if q_sorted.size != q_arr.size:
            raise ValueError("CompositeQuantileEstimator: quantiles must be unique; got " f"{q_arr.tolist()!r}.")
        return q_sorted

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> "CompositeQuantileEstimator":
        """Fit one inner per quantile on the transform ``T``.

        Validates the transform up front, then for each quantile ``q`` builds a
        :class:`CompositeTargetEstimator` whose inner is a clone of
        ``base_estimator`` configured for the ``q``-quantile pinball loss, and
        fits it on ``(X, y)``. The transform fit / domain / clip state is
        identical across heads (same y, base, params) -- only the inner's pinball
        alpha differs. ``sample_weight`` / ``**fit_kwargs`` flow to every head.
        Returns ``self``.
        """
        if self.base_estimator is None:
            raise ValueError("CompositeQuantileEstimator: base_estimator must not be None.")
        # Surface a typo'd transform name at fit, not on the first predict.
        get_transform(self.transform_name)
        quantiles = self._resolve_quantiles()

        estimators: dict[float, CompositeTargetEstimator] = {}
        for q in quantiles:
            inner_q = _set_inner_quantile_alpha(self.base_estimator, float(q))
            head = CompositeTargetEstimator(
                base_estimator=inner_q,
                transform_name=self.transform_name,
                base_column=self.base_column,
                base_columns=self.base_columns,
                group_column=self.group_column,
                fallback_predict=self.fallback_predict,
                drop_invalid_rows=self.drop_invalid_rows,
            )
            head.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
            estimators[float(q)] = head

        self.estimators_ = estimators
        self.quantiles_ = quantiles
        # Detect a monotone-DECREASING transform inverse (e.g. reciprocal_residual,
        # y = 1/(T + 1/base)): a head trained for the tau-quantile of T then yields
        # the (1-tau)-quantile of y, so the column labelled tau must take the head
        # trained at (1-tau). Without this the raw (enforce_non_crossing=False)
        # columns are silently swapped: column "0.1" carries the 0.9 y-quantile.
        self._inverse_decreasing_ = _transform_inverse_decreasing(self.transform_name)
        # Inherit feature names from the median (or first) head for sklearn
        # introspection parity; all heads saw the same X so any head works.
        ref_head = estimators[float(quantiles[np.argmin(np.abs(quantiles - 0.5))])]
        ref_names = getattr(ref_head, "feature_names_in_", None)
        if ref_names is not None:
            self.feature_names_in_ = list(ref_names)
        cols = getattr(X, "columns", None)
        if cols is not None:
            self.n_features_in_ = len(cols)
        elif getattr(X, "shape", None) is not None and len(X.shape) >= 2:
            self.n_features_in_ = int(X.shape[1])
        return self

    def predict_quantile(
        self, X: Any, quantiles: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict the y-scale quantiles; return an ``(n, n_q)`` matrix.

        Each requested ``q`` must have been fitted (a subset of the fit-time
        grid is allowed; an unseen level raises). Every head's point
        ``.predict()`` is the alpha-quantile of ``T`` (pinball-trained), inverted
        to y-scale by that head's transform inverse. The resulting columns are
        stacked in ASCENDING-quantile order and, when ``enforce_non_crossing``,
        each row is sorted ascending so ``q_low <= ... <= q_high``.

        Returns ``(n_samples, len(quantiles))``. NaN-safe: a row whose base is
        out of domain routes through each head's ``fallback_predict`` (never a
        silent NaN), and the non-crossing sort is applied with ``np.sort`` which
        is order-preserving on any finite values.
        """
        if not hasattr(self, "estimators_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("CompositeQuantileEstimator.predict_quantile called before fit.")
        n_in = getattr(self, "n_features_in_", None)
        cols = getattr(X, "columns", None)
        n_x = len(cols) if cols is not None else (int(X.shape[1]) if getattr(X, "shape", None) is not None and len(X.shape) >= 2 else None)
        if n_in is not None and n_x is not None and n_x != n_in:
            raise ValueError(f"CompositeQuantileEstimator.predict_quantile: X has {n_x} features, but the estimator was fit with {n_in}.")
        if quantiles is None:
            req = self.quantiles_
        else:
            req = np.asarray(quantiles, dtype=np.float64).reshape(-1)
            if req.size == 0:
                raise ValueError("CompositeQuantileEstimator.predict_quantile: quantiles is empty.")
            req = np.sort(req)

        decreasing = bool(getattr(self, "_inverse_decreasing_", False))
        q_cols: list[np.ndarray] = []
        for q in req:
            # Decreasing inverse: the y-quantile at level q is produced by the head
            # trained for the (1-q)-quantile of T. Use the complementary head so
            # the returned column is correctly labelled regardless of the sort.
            head_q = 1.0 - float(q) if decreasing else float(q)
            head = self._lookup_head(head_q)
            y_q = np.asarray(head.predict(X), dtype=np.float64).reshape(-1)
            q_cols.append(y_q)
        out = np.column_stack(q_cols)

        if self.enforce_non_crossing and out.shape[1] > 1:
            # Sort each row ascending -> q_low <= ... <= q_high. np.sort pushes
            # NaN to the end per row, but the fallback routing already replaced
            # every out-of-domain row with a finite constant in each column, so
            # under the default fallback no NaN reaches here; under
            # fallback_predict='nan' the NaN columns sort to the high end, which
            # is the documented uninformative-but-valid behaviour.
            out = np.sort(out, axis=1)
        return out

    def predict(self, X: Any) -> np.ndarray:
        """Median (0.5-quantile) point prediction.

        When 0.5 was fitted, returns that head's prediction directly; otherwise
        returns the quantile nearest 0.5 from the fitted grid (so ``predict`` is
        always defined for a fitted estimator).
        """
        q_arr = self.quantiles_
        nearest = float(q_arr[np.argmin(np.abs(q_arr - 0.5))])
        return np.asarray(
            self._lookup_head(nearest).predict(X), dtype=np.float64,
        ).reshape(-1)

    def _lookup_head(self, q: float) -> CompositeTargetEstimator:
        """Resolve the fitted head for quantile ``q`` (exact, with float tol)."""
        if q in self.estimators_:
            return self.estimators_[q]
        # Float-key tolerance: a caller passing 0.1 vs np.float64(0.1) etc.
        for fitted_q, head in self.estimators_.items():
            if abs(fitted_q - q) <= 1e-9:
                return head
        raise ValueError(
            f"CompositeQuantileEstimator: quantile {q!r} was not fitted; "
            f"fitted quantiles are {sorted(self.estimators_)!r}. predict_quantile "
            "may only request a subset of the fit-time grid."
        )
