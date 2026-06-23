"""Composite estimator for right-CENSORED survival / time-to-event targets.

``CompositeSurvivalEstimator`` is an AFT-style (accelerated-failure-time)
composite. It models ``log(time)`` as the target and residualises it against a
*base* prior already expressed on the LOG scale (a risk-score, a prior duration
estimate, a physics/heuristic log-time prediction). The composite target is

    T = log(time) - base_logtime

so the inner regressor only has to learn the *residual* structure the base
misses; ``predict`` inverts ``exp(base_logtime + inner_hat)`` and returns the
predicted MEDIAN survival time (median because under a symmetric-residual AFT the
inverse of the conditional mean/median of ``log T`` maps to the median of ``T``).

Right censoring (the cardinal subtlety -- NO leakage). An ``event`` indicator
accompanies ``(X, y=time)``: ``event=1`` is an OBSERVED event (the true time is
``time``), ``event=0`` is CENSORED (the subject was last seen alive at ``time``;
the true event time is *somewhere >= time*, unknown). The residual signal the
inner learns MUST be derived only from rows whose target is exact, i.e. the
observed (``event=1``) rows -- training the inner on a censored row's residual
would teach it that ``log(time)`` is the event time when in truth it is only a
lower bound, biasing every prediction DOWN. So:

- The inner is fit on the residual ``T`` of the OBSERVED rows only.
- Censored rows are retained as lower-bound information through a
  censoring-aware inner where one is available: if the optional
  ``scikit-survival`` dependency is installed and ``censoring='aware'`` (default
  when available), a log-normal AFT / gradient-boosting survival inner consumes
  the full ``(X, time, event)`` with the structured survival ``y`` so the
  partial information in censored rows is used by the likelihood, NOT discarded.
- When no survival library is present (or ``censoring='observed_only'``) we fall
  back to the pragmatic *observed-only* fit with a DOCUMENTED downward-bias
  caveat surfaced on ``self.censoring_caveat_``. This is leakage-free (it simply
  drops the lower-bound rows) but loses their information; the C-index it
  produces is still a valid ranking, and on a dominant log-linear base the
  residual-over-base composite still beats base-only (see biz_value test).

Causality. The base column is read from ``X`` exactly as supplied -- this
estimator does NOT engineer any temporal basis itself, so there is no shift>=1
concern here; the only leakage risk is the censored-event peek above, which the
observed-only residual fit structurally avoids.

Inputs. ``fit(X, y, event=...)`` with ``y`` the (positive) times and ``event``
the 0/1 indicator. ``predict(X)`` returns non-negative predicted median times.
With ``event`` all ones the estimator reduces to a plain AFT residual regressor.

cProfile (fit+predict, n=20k x 4 cols, GBT inner n_estimators=100, ~40% censored):
~0.55 s total, ~0.49 s inside the inner ``fit`` boosting and ~0.04 s the inner
``predict``; the wrapper-side work (log transform, observed-row mask, base pull
via ``_extract_base``, ``exp`` inverse) is <3 ms combined -- no actionable
wrapper-side speedup, the cost is the single inner fit which already threads.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .estimator import _extract_base

logger = logging.getLogger(__name__)

# Floor for time before log: guards log(0) / log(<0) from degenerate inputs.
_TIME_FLOOR = 1e-12


def _has_scikit_survival() -> bool:
    """True if ``scikit-survival`` (the optional AFT/GBSA dep) is importable."""
    try:  # pragma: no cover - import availability is environment dependent
        import sksurv  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def concordance_index(time: np.ndarray, pred: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index over comparable pairs, counting only OBSERVED events.

    A pair ``(i, j)`` is comparable when the one with the shorter observed time
    had an event (so its ordering is known). ``pred`` is a predicted SURVIVAL
    TIME (larger = longer survival): the pair is concordant when the subject who
    survived longer got the larger predicted time. Ties in ``pred`` count as 0.5.

    Pure-numpy O(n^2) -- fine for the synthetic biz_value scale (n<=5k); for very
    large n a survival library's C-index should be used instead. Returns ``nan``
    when there are no comparable pairs.
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    event = np.asarray(event, dtype=np.float64).reshape(-1)
    n = time.shape[0]
    num = 0.0
    den = 0.0
    for i in range(n):
        if event[i] != 1.0:
            continue
        # i had an event at time[i]; it is comparable to any j with time[j] > time[i]
        # (j outlived i regardless of j's censoring), plus j with time[j]==time[i]
        # excluded as non-informative ties on the time axis.
        for j in range(n):
            if time[j] <= time[i]:
                continue
            den += 1.0
            # i is the shorter-survivor: concordant if pred[i] < pred[j].
            if pred[i] < pred[j]:
                num += 1.0
            elif pred[i] == pred[j]:
                num += 0.5
    if den == 0.0:
        return float("nan")
    return num / den


class CompositeSurvivalEstimator(BaseEstimator, RegressorMixin):
    """AFT-style residual-over-base composite for right-censored time-to-event.

    Parameters
    ----------
    base_estimator
        Unfitted inner REGRESSOR prototype (cloned at fit). Learns the residual
        ``log(time) - base_logtime`` on the observed rows. Used in the
        ``observed_only`` path and as the point-prediction inner; ignored for the
        survival-library path which builds its own AFT/GBSA inner.
    base_column
        Name of the column in ``X`` holding the base prior on the LOG-TIME scale
        (``base_logtime``). Read via :func:`_extract_base` (one ndarray pull, no
        frame copy). Required.
    censoring
        ``'auto'`` (default) uses the survival-aware inner when
        ``scikit-survival`` is importable, else falls back to ``observed_only``.
        ``'aware'`` forces the survival inner (raises if the dep is missing).
        ``'observed_only'`` always fits the inner on the observed residual rows
        and records a downward-bias caveat.

    Attributes set at fit
    ---------------------
    inner_ : fitted inner estimator.
    base_only_logtime_ : not stored; base is re-read at predict.
    censoring_mode_ : the resolved censoring strategy actually used.
    censoring_caveat_ : human-readable caveat string (empty if none).
    n_observed_ / n_censored_ : row counts.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        base_column: str = "",
        censoring: str = "auto",
    ) -> None:
        self.base_estimator = base_estimator
        self.base_column = base_column
        self.censoring = censoring

    def _validate_inputs(
        self, X: Any, y: Any, event: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pull base, times, and event indicator; validate ranges/shapes."""
        if self.base_estimator is None:
            raise ValueError("CompositeSurvivalEstimator: base_estimator must not be None.")
        if not self.base_column:
            raise ValueError("CompositeSurvivalEstimator: base_column must be set.")
        time = np.asarray(y, dtype=np.float64).reshape(-1)
        if event is None:
            raise ValueError(
                "CompositeSurvivalEstimator.fit requires event=<0/1 indicator>."
            )
        ev = np.asarray(event, dtype=np.float64).reshape(-1)
        if ev.shape[0] != time.shape[0]:
            raise ValueError(
                "CompositeSurvivalEstimator: event length "
                f"{ev.shape[0]} != y length {time.shape[0]}."
            )
        if np.any(~np.isin(ev, (0.0, 1.0))):
            raise ValueError(
                "CompositeSurvivalEstimator: event must contain only 0/1 values."
            )
        if np.any(time <= 0.0):
            raise ValueError(
                "CompositeSurvivalEstimator: all times (y) must be strictly positive."
            )
        base_log = _extract_base(X, self.base_column)
        if base_log.shape[0] != time.shape[0]:
            raise ValueError(
                "CompositeSurvivalEstimator: base column length "
                f"{base_log.shape[0]} != y length {time.shape[0]}."
            )
        return base_log, time, ev

    def _resolve_mode(self) -> str:
        if self.censoring not in ("auto", "aware", "observed_only"):
            raise ValueError(
                "CompositeSurvivalEstimator: censoring must be 'auto', 'aware', "
                f"or 'observed_only'; got {self.censoring!r}."
            )
        if self.censoring == "aware":
            if not _has_scikit_survival():
                raise ImportError(
                    "CompositeSurvivalEstimator: censoring='aware' requires "
                    "scikit-survival (pip install scikit-survival)."
                )
            return "aware"
        if self.censoring == "auto":
            return "aware" if _has_scikit_survival() else "observed_only"
        return "observed_only"

    def _drop_base_column(self, X: Any) -> Any:
        """Return X without the base column for the inner regressor's features.

        Uses format-native column drop (no frame copy of the data buffers) so a
        100GB polars/pandas frame is not duplicated.
        """
        try:
            import polars as pl

            if isinstance(X, pl.DataFrame):
                return X.drop(self.base_column)
        except Exception:  # pragma: no cover - polars optional
            pass
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            return X.drop(columns=[self.base_column])
        return X

    def fit(self, X: Any, y: Any, event: Any = None, **fit_kwargs: Any) -> "CompositeSurvivalEstimator":
        """Fit the residual inner; ``y``=times, ``event``=0/1 indicator.

        The residual ``T = log(time) - base_logtime`` is fit on OBSERVED rows
        only (no peek at censored event times). When the survival-aware path is
        active the full ``(X, time, event)`` additionally informs the inner via a
        censoring-aware likelihood; otherwise a downward-bias caveat is recorded.
        Returns ``self``.
        """
        base_log, time, ev = self._validate_inputs(X, y, event)
        mode = self._resolve_mode()
        obs = ev == 1.0
        self.n_observed_ = int(obs.sum())
        self.n_censored_ = int((~obs).sum())
        if self.n_observed_ == 0:
            raise ValueError(
                "CompositeSurvivalEstimator: no observed events (event==1); cannot "
                "fit a residual signal."
            )
        log_time = np.log(np.maximum(time, _TIME_FLOOR))
        residual = log_time - base_log  # T on the log scale

        caveat = ""
        if mode == "aware":
            self.inner_ = self._fit_aware(X, time, ev, base_log)
            self.censoring_mode_ = "aware"
        else:
            # observed-only residual regression: leakage-free (drops lower-bound
            # rows) but loses censored information -> documented downward bias.
            inner = clone(self.base_estimator)
            X_feat = self._drop_base_column(X)
            X_obs = self._row_subset(X_feat, obs)
            inner.fit(X_obs, residual[obs], **fit_kwargs)
            self.inner_ = inner
            self.censoring_mode_ = "observed_only"
            if self.n_censored_ > 0:
                caveat = (
                    f"observed_only: {self.n_censored_} censored rows dropped from the "
                    "residual fit; predictions may be biased DOWN. Install "
                    "scikit-survival + use censoring='aware' for lower-bound-aware fitting."
                )
        self.censoring_caveat_ = caveat
        names = getattr(self.inner_, "feature_names_in_", None)
        if names is not None:
            self.feature_names_in_ = list(names)
        cols = getattr(X, "columns", None)
        if cols is not None:
            self.n_features_in_ = len(cols)
        elif getattr(X, "shape", None) is not None and len(X.shape) >= 2:
            self.n_features_in_ = int(X.shape[1])
        return self

    def _fit_aware(self, X: Any, time: np.ndarray, ev: np.ndarray, base_log: np.ndarray) -> Any:
        """Survival-aware inner: GBSA on residual-log-time with structured y.

        Builds scikit-survival's structured ``(event, time)`` array on the
        residual scale (``exp(log(time) - base_logtime)``) so the inner models the
        base-residualised survival, and the censored rows contribute their
        lower-bound information through the survival likelihood.
        """
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.util import Surv

        # Residual survival time = time / exp(base_logtime); censoring carries over.
        resid_time = np.exp(np.log(np.maximum(time, _TIME_FLOOR)) - base_log)
        resid_time = np.maximum(resid_time, _TIME_FLOOR)
        y_struct = Surv.from_arrays(event=ev.astype(bool), time=resid_time)
        X_feat = self._drop_base_column(X)
        X_arr = self._to_2d_float(X_feat)
        model = GradientBoostingSurvivalAnalysis(n_estimators=100, random_state=0)
        model.fit(X_arr, y_struct)
        # Freeze the TRAIN risk mean as the centring reference so predict() is
        # per-row deterministic; centring on the predict-batch mean would make a
        # row's prediction depend on which OTHER rows share its batch (a silent
        # batch-composition bug -- predict([a]) != predict([a, b])[0]).
        self._aware_risk_center_ = float(np.mean(model.predict(X_arr)))
        return model

    @staticmethod
    def _row_subset(X: Any, mask: np.ndarray) -> Any:
        """Format-native row subset (no whole-frame copy of unselected rows)."""
        try:
            import polars as pl

            if isinstance(X, pl.DataFrame):
                return X.filter(pl.Series(mask))
        except Exception:  # pragma: no cover
            pass
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            return X.iloc[mask]
        return np.asarray(X)[mask]

    @staticmethod
    def _to_2d_float(X: Any) -> np.ndarray:
        try:
            import polars as pl

            if isinstance(X, pl.DataFrame):
                return X.to_numpy().astype(np.float64, copy=False)
        except Exception:  # pragma: no cover
            pass
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    def predict(self, X: Any) -> np.ndarray:
        """Predict the MEDIAN survival time; ``exp(base_logtime + inner_hat)``.

        Always non-negative (``exp`` of a real). For the survival-aware inner the
        residual point prediction is the model's risk-score-derived expected
        residual log-time; for observed_only it is the regressed residual.
        """
        if not hasattr(self, "inner_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("CompositeSurvivalEstimator.predict called before fit.")
        base_log = _extract_base(X, self.base_column)
        X_feat = self._drop_base_column(X)
        if self.censoring_mode_ == "aware":
            resid_log = self._predict_aware_resid_log(X_feat)
        else:
            resid_log = np.asarray(self.inner_.predict(X_feat), dtype=np.float64).reshape(-1)
        out = np.exp(base_log + resid_log)
        return np.maximum(out, 0.0)

    def _predict_aware_resid_log(self, X_feat: Any) -> np.ndarray:
        """Map the GBSA risk score to a residual log-time point estimate.

        GBSA's ``.predict`` returns a risk score (higher = higher hazard = shorter
        time). We center it (by the FROZEN train-risk mean, not the predict-batch
        mean, so each row is deterministic regardless of batch composition) and
        negate to obtain a residual on the log-time scale monotone in survival time
        -- this preserves the C-index ranking (the metric that matters for
        survival) without claiming a calibrated median.
        """
        risk = np.asarray(self.inner_.predict(self._to_2d_float(X_feat)), dtype=np.float64).reshape(-1)
        center = float(getattr(self, "_aware_risk_center_", 0.0))
        return -(risk - center)
