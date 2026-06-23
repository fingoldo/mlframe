"""Full predictive-distribution composite + proper scoring (CRPS).

``CompositeDistributionEstimator`` turns the per-quantile composite into a full
predictive distribution. It fits a :class:`CompositeQuantileEstimator` at a DENSE
grid of quantile levels (default ``0.05 .. 0.95`` in 0.05 steps) with the same
non-crossing guarantee, then exposes the standard UQ surface built ON TOP of that
quantile representation:

- :meth:`predict_quantile` -- thin pass-through to the fitted quantile heads.
- :meth:`predict_cdf` -- the step-CDF implied by the predicted quantiles: at a
  query value ``t`` the predicted ``F(t) = level`` of the largest fitted quantile
  whose value is ``<= t`` (a right-continuous step function on the quantile grid).
- :meth:`sample` -- inverse-CDF (quantile-function) sampling: draw ``u ~ U(0,1)``
  and read off the predicted quantile at level ``u`` via linear interpolation in
  the quantile-level domain (the standard probability-integral-transform sampler).
- :meth:`crps` -- the Continuous Ranked Probability Score computed from the
  quantile representation via the pinball-integral / quantile-decomposition
  identity ``CRPS(F, y) = (2/K) * sum_k pinball_{q_k}(y, Q(q_k))`` (Gneiting &
  Raftery 2007; the quantile decomposition of the CRPS). Lower is better; it is a
  STRICTLY PROPER score, so a sharper-yet-calibrated heteroscedastic distribution
  beats a wide homoscedastic one.

Reuse. ALL quantile logic (per-head pinball alpha wiring, the transform
fit/inverse, the domain filtering, the non-crossing sort) lives in
:class:`CompositeQuantileEstimator` and is NOT duplicated here -- this class holds
one as ``quantile_estimator_`` and only adds the distribution-level surface
(CDF / sampling / CRPS) on top of its ``predict_quantile`` output.

Memory. No frame copy: ``predict_quantile`` already pulls the narrow base column
via the inner heads; this class operates only on the resulting ``(n, K)`` quantile
matrix (small -- K levels, never the feature frame). The CRPS pinball integral is
a vectorised numpy reduction over that matrix.

cProfile (fit + crps, n=20k x 2 cols, K=19 dense levels, stub inner): the wrapper
work outside the K inner fits is the single ``predict_quantile`` call plus a
vectorised ``(n, K)`` pinball reduction -- <1 ms combined; cost is the K inner
boosting fits, already parallel inside the inner. No actionable wrapper-side
speedup (mirrors the quantile estimator's profile note).
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .quantile import CompositeQuantileEstimator

logger = logging.getLogger(__name__)

# Dense default grid: 0.05 .. 0.95 step 0.05 (19 levels). Symmetric, includes the
# 0.5 median, gives a fine enough quantile mesh that the pinball-integral CRPS is
# a close approximation of the continuous CRPS while keeping K inner fits modest.
_DEFAULT_DENSE_QUANTILES: tuple[float, ...] = tuple(
    round(0.05 * k, 2) for k in range(1, 20)
)


class CompositeDistributionEstimator(BaseEstimator, RegressorMixin):
    """Full predictive distribution via a dense quantile composite + CRPS.

    Parameters
    ----------
    base_estimator
        Unfitted pinball-capable inner prototype (LightGBM ``objective='quantile'``
        / XGBoost ``reg:quantileerror`` / sklearn ``QuantileRegressor``). Cloned
        once per dense quantile level by the inner
        :class:`CompositeQuantileEstimator`.
    transform_name, base_column, base_columns, group_column,
    fallback_predict, drop_invalid_rows, enforce_non_crossing
        Forwarded verbatim to :class:`CompositeQuantileEstimator`.
    quantiles
        Ascending dense grid of levels in (0, 1). Default ``0.05 .. 0.95`` step
        0.05 (19 levels). A denser grid sharpens the CRPS integral at the cost of
        more inner fits.

    Attributes set at fit
    ---------------------
    quantile_estimator_
        The fitted :class:`CompositeQuantileEstimator` (holds the per-level heads).
    quantiles_
        The fitted dense level grid (sorted ndarray).
    feature_names_in_
        Inherited from the inner quantile estimator (best effort).
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

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> "CompositeDistributionEstimator":
        """Fit the dense quantile composite that backs the distribution.

        Delegates entirely to :class:`CompositeQuantileEstimator` (one inner per
        dense level on the transform ``T``, non-crossing enforced). Returns
        ``self``.
        """
        if self.base_estimator is None:
            raise ValueError(
                "CompositeDistributionEstimator: base_estimator must not be None."
            )
        quantiles = (
            self.quantiles if self.quantiles is not None
            else _DEFAULT_DENSE_QUANTILES
        )
        qest = CompositeQuantileEstimator(
            base_estimator=self.base_estimator,
            transform_name=self.transform_name,
            base_column=self.base_column,
            quantiles=quantiles,
            fallback_predict=self.fallback_predict,
            drop_invalid_rows=self.drop_invalid_rows,
            base_columns=self.base_columns,
            group_column=self.group_column,
            enforce_non_crossing=self.enforce_non_crossing,
        )
        qest.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        self.quantile_estimator_ = qest
        self.quantiles_ = np.asarray(qest.quantiles_, dtype=np.float64)
        ref_names = getattr(qest, "feature_names_in_", None)
        if ref_names is not None:
            self.feature_names_in_ = list(ref_names)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "quantile_estimator_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                "CompositeDistributionEstimator: call fit before this method."
            )

    # ------------------------------------------------------------------
    # Quantile / point pass-through
    # ------------------------------------------------------------------
    def predict_quantile(
        self, X: Any, quantiles: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Predict y-scale quantiles ``(n, n_q)`` (pass-through to the heads)."""
        self._check_fitted()
        return self.quantile_estimator_.predict_quantile(X, quantiles=quantiles)

    def predict(self, X: Any) -> np.ndarray:
        """Median (0.5-quantile) point prediction."""
        self._check_fitted()
        return self.quantile_estimator_.predict(X)

    def _quantile_matrix(self, X: Any) -> np.ndarray:
        """The full dense ``(n, K)`` non-crossing quantile matrix on the y-scale."""
        return np.asarray(
            self.quantile_estimator_.predict_quantile(X, quantiles=self.quantiles_),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # CDF
    # ------------------------------------------------------------------
    def predict_cdf(self, X: Any, y_grid: Sequence[float]) -> np.ndarray:
        """Step-CDF implied by the predicted quantiles, evaluated at ``y_grid``.

        Returns an ``(n_samples, len(y_grid))`` matrix where entry ``[i, j]`` is
        the predicted ``F_i(y_grid[j])`` -- the probability the i-th sample's
        target is ``<= y_grid[j]``. The CDF is the right-continuous step function
        defined by the predicted quantiles: ``F(t)`` equals the LARGEST fitted
        quantile LEVEL whose predicted VALUE ``Q(level) <= t`` (and 0 below the
        lowest predicted quantile). This is monotone non-decreasing in ``t`` by
        construction (the quantile values are non-crossing / sorted ascending per
        row), so the returned rows are valid CDFs.
        """
        self._check_fitted()
        qmat = self._quantile_matrix(X)  # (n, K), ascending per row
        levels = self.quantiles_  # (K,), ascending
        t = np.asarray(y_grid, dtype=np.float64).reshape(-1)  # (G,)
        # For each (row i, query j): count fitted quantiles with value <= t_j, take
        # the level of the highest such. Vectorised via broadcasting: compare the
        # (n, K, 1) quantile values against (1, 1, G) queries.
        leq = qmat[:, :, None] <= t[None, None, :]  # (n, K, G) bool
        # Level assigned to each satisfied quantile (broadcast), else 0; the max
        # over K gives the highest satisfied level (= step-CDF value). Below the
        # lowest quantile no level is satisfied -> max of an all-zero slice = 0.
        level_if = np.where(leq, levels[None, :, None], 0.0)  # (n, K, G)
        cdf = level_if.max(axis=1)  # (n, G)
        return cdf

    # ------------------------------------------------------------------
    # Sampling (inverse-CDF / probability integral transform)
    # ------------------------------------------------------------------
    def sample(
        self, X: Any, n: int, random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw ``n`` samples per row via inverse-CDF (quantile-function) sampling.

        Returns an ``(n_samples, n)`` matrix of draws. For each draw we sample
        ``u ~ U(0, 1)`` and read off the predicted quantile at level ``u`` by
        LINEAR INTERPOLATION of that row's quantile function in the level domain
        (``np.interp`` over the fitted ``(level, value)`` knots, with the endpoints
        held flat below/above the lowest/highest fitted level). This is the
        standard probability-integral-transform sampler: ``Q(U) ~ F`` when ``U`` is
        uniform, so the draws follow the predicted distribution.
        """
        self._check_fitted()
        if n <= 0:
            raise ValueError(f"CompositeDistributionEstimator.sample: n must be > 0; got {n}.")
        rng = (
            random_state if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        qmat = self._quantile_matrix(X)  # (n_rows, K)
        levels = self.quantiles_  # (K,)
        n_rows = qmat.shape[0]
        u = rng.random((n_rows, n))  # (n_rows, n) in [0, 1)
        out = np.empty((n_rows, n), dtype=np.float64)
        # np.interp is 1-D; loop per row over the (small) quantile knots. n_rows is
        # the sample count, K is tiny -- this is a light pass, not a hot kernel.
        for i in range(n_rows):
            out[i, :] = np.interp(u[i, :], levels, qmat[i, :])
        return out

    # ------------------------------------------------------------------
    # CRPS (proper score, quantile decomposition)
    # ------------------------------------------------------------------
    def crps(self, X: Any, y_true: Any, reduce: str = "mean") -> Any:
        """Continuous Ranked Probability Score from the quantile representation.

        Uses the quantile decomposition of the CRPS (Gneiting & Raftery 2007):

        ``CRPS(F_i, y_i) = (2 / K) * sum_k rho_{q_k}(y_i - Q_i(q_k))``

        where ``rho_q(u) = u * (q - 1{u < 0})`` is the pinball (quantile) loss and
        ``{q_k}`` is the fitted dense level grid. As the grid densifies this
        converges to the integral form ``2 * integral_0^1 pinball_q dq`` = the CRPS.
        Lower is better; the score is strictly proper, so a calibrated SHARP
        distribution scores below a wider one on the same data.

        Parameters
        ----------
        reduce : {"mean", "none"}
            ``"mean"`` (default) returns the scalar mean CRPS over rows; ``"none"``
            returns the per-row ``(n_samples,)`` vector.
        """
        self._check_fitted()
        qmat = self._quantile_matrix(X)  # (n, K)
        levels = self.quantiles_  # (K,)
        y = np.asarray(y_true, dtype=np.float64).reshape(-1)  # (n,)
        if y.shape[0] != qmat.shape[0]:
            raise ValueError(
                "CompositeDistributionEstimator.crps: y_true length "
                f"{y.shape[0]} != n_samples {qmat.shape[0]}."
            )
        # Pinball loss per (row, level): u = y - Q ; rho = u*(q - 1{u<0}).
        u = y[:, None] - qmat  # (n, K)
        rho = u * (levels[None, :] - (u < 0.0).astype(np.float64))  # (n, K)
        per_row = (2.0 / levels.shape[0]) * rho.sum(axis=1)  # (n,)
        if reduce == "none":
            return per_row
        if reduce == "mean":
            return float(np.mean(per_row))
        raise ValueError(
            f"CompositeDistributionEstimator.crps: reduce must be 'mean' or 'none', "
            f"got {reduce!r}."
        )
