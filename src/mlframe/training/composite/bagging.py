"""``BaggedCompositeEstimator`` -- variance reduction + epistemic UQ for composites.

Fits ``n_estimators`` clones of a :class:`CompositeTargetEstimator` (or any
sklearn-compatible regressor) on bootstrap resamples of the training rows --
each member also gets a distinct inner ``random_state`` when its estimator
exposes one -- and averages their per-row predictions for a lower-variance
point estimate.

The across-member spread is an EPISTEMIC-uncertainty signal: where the
training data is dense, the members agree (small spread); in extrapolation
regions (few / no nearby train rows) the bootstrap resamples disagree and the
spread widens. :meth:`predict_std` exposes that spread and
:meth:`predict_interval_epistemic` turns it into a Gaussian band.

Design
------
- sklearn-compatible: ``fit`` / ``predict`` / ``get_params`` / ``set_params``
  / :func:`sklearn.clone` all behave per convention; the base estimator is
  cloned per member so the unfitted prototype stays clean.
- Deterministic given ``random_state``: a seeded ``np.random.RandomState``
  draws each member's bootstrap indices AND each member's inner seed, so two
  fits with the same seed produce bit-identical predictions.
- No frame copies on the hot path. Bootstrap row selection uses the
  flavour-native subsetter (polars ``.filter`` via take / pandas ``.iloc`` /
  ndarray fancy-index) -- never ``df.copy()`` on a possibly-100GB frame. Each
  member holds a reference into the SAME backing frame (the resample is an
  index gather, materialised once per member by the inner estimator's fit).
- Parallelisable: members are independent, so ``n_jobs`` could fan the fit
  across joblib workers. Kept SEQUENTIAL here for simplicity + determinism;
  the (documented) parallel option is a drop-in ``joblib.Parallel`` over the
  member loop -- noted but not enabled so a 100GB frame is not pickled to N
  workers by accident.

Biz value
---------
On a noisy target the bagged composite has LOWER out-of-sample RMSE than a
single composite (bootstrap averaging cancels the per-member variance), and
``predict_std`` is LARGER in extrapolation regions (few train rows) than in
dense regions -- the hallmark of an epistemic signal. Both are pinned by
``tests/training/test_biz_val_training_composite_bagging.py``.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)

try:
    import polars as pl

    _HAS_POLARS = True
except Exception:  # pragma: no cover - polars optional
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


def _n_rows(X: Any) -> int:
    """Row count across pandas / polars / ndarray / list-of-rows."""
    shape = getattr(X, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(X)


def _take_rows(X: Any, idx: np.ndarray) -> Any:
    """Gather rows ``idx`` (possibly with repeats) WITHOUT copying the whole frame.

    Flavour-native: polars ``DataFrame[idx]`` / pandas ``.iloc`` / ndarray fancy
    index. Each path materialises only the selected (n_idx) rows -- on a bootstrap
    resample that is the same row-count as the input, but it is a single gather,
    not a frame ``.copy()`` followed by a mask.
    """
    if _is_polars_df(X):
        return X[idx.tolist()]
    iloc = getattr(X, "iloc", None)
    if iloc is not None:
        return iloc[idx]
    return np.asarray(X)[idx]


def _take_1d(y: Any, idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(y)
    return np.asarray(arr[idx])


class BaggedCompositeEstimator(BaseEstimator, RegressorMixin):
    """Bootstrap-bagged ensemble of composite estimators for variance + UQ.

    Parameters
    ----------
    base_estimator
        The prototype regressor (typically a
        :class:`CompositeTargetEstimator`). Cloned once per member; the
        prototype passed in is never fitted. Required.
    n_estimators
        Number of bootstrap members. More members -> lower point-estimate
        variance + a smoother ``predict_std``, at linear fit cost. Default 10.
    bootstrap
        When True (default) each member trains on a with-replacement resample
        of the rows (classic bagging). When False members differ ONLY by their
        inner ``random_state`` (useful when the inner estimator is itself
        stochastic and you want seed-ensembling without resampling).
    max_samples
        Fraction (0,1] of ``n`` rows drawn per bootstrap member. Default 1.0
        (draw ``n`` rows with replacement -- the standard bootstrap). Smaller
        values increase member diversity / spread.
    vary_inner_random_state
        When True (default) each member also receives a distinct seed pushed
        into the cloned estimator's ``random_state`` param (and the nested
        ``base_estimator__random_state`` for a composite wrapper) when that
        param exists. Decorrelates members even when ``bootstrap=False``.
    random_state
        Seed for the bootstrap-index + member-seed draws. With a fixed seed the
        whole fit (and therefore every prediction) is deterministic.
    n_jobs
        Reserved. Members are independent and could be fit in parallel via
        ``joblib.Parallel``; the current implementation is SEQUENTIAL for
        determinism + to avoid pickling a large frame to N workers. Accepting
        the param keeps the public signature stable for a future parallel
        backend. Default 1 (sequential).
    aggregation
        How ``predict`` collapses the per-member prediction matrix to a point estimate. ``"trimmed_mean"`` (default) drops the
        symmetric ``trim_fraction`` extremes per row then averages -- robust to a wild bootstrap member on heavy-tailed / outlier
        targets, near-mean-efficient on clean Gaussian data. ``"mean"`` is the legacy Gaussian-MLE aggregator; ``"median"`` is the
        maximally-robust option. ``predict_std`` / ``predict_interval_epistemic`` always use the mean+std (the epistemic spread is
        a Gaussian-band notion independent of the point-estimate aggregator).
    trim_fraction
        Symmetric fraction (each tail) dropped when ``aggregation="trimmed_mean"``. Default 0.2 (the robust knee across clean +
        heavy-tail + 5-30% outlier contamination, see qual-17 bench); must be in [0, 0.5).

    Attributes set at fit
    ---------------------
    estimators_
        List of the ``n_estimators`` fitted member estimators.
    n_features_in_ / feature_names_in_
        Mirrored from the first member when available (sklearn convention).
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 10,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        vary_inner_random_state: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        aggregation: str = "trimmed_mean",
        trim_fraction: float = 0.2,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.vary_inner_random_state = vary_inner_random_state
        self.random_state = random_state
        self.n_jobs = n_jobs
        # sklearn contract: __init__ only stores params verbatim (no validation / no transformation), so set_params round-trips and
        # clone reconstructs faithfully. aggregation / trim_fraction are validated in fit instead.
        # ``trimmed_mean`` (symmetric 20% trim) is the default point-estimate aggregator: on heavy-tailed / outlier-contaminated
        # targets it lowers honest-holdout RMSE+MAE materially while costing only a fraction of a percent under clean Gaussian
        # noise. The 0.2 trim (vs the earlier 0.1) is the robust knee from the trim-fraction sweep (bench bench_bagging_trim_fraction_qual17):
        # across clean / heavy-tail t(2) / 5-30% outlier contamination, 0.2 wins honest-holdout RMSE on 28/35 seed-scenario cells
        # vs the 0.1 incumbent (7/7 on EVERY contaminated scenario, ~5-9% RMSE drop) for an immaterial ~0.3% clean cost; it captures
        # essentially all the contamination benefit of larger trims without sitting on the grid edge (near-median, contamination-overfit risk).
        # ``mean`` is the legacy aggregator (Gaussian-MLE optimal, but a single wild member skews it); ``median`` is the maximally-robust
        # fallback. Pickle-replay: old saved models lack these attrs -> predict() reads them via getattr with the LEGACY ``mean``
        # default so a v1 model replays byte-identically.
        self.aggregation = aggregation
        self.trim_fraction = trim_fraction

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _set_member_seed(self, estimator: Any, seed: int) -> None:
        """Push ``seed`` into whichever ``random_state`` params the member exposes.

        Tries the top-level ``random_state`` and the nested
        ``base_estimator__random_state`` (the composite-wrapper path) so a
        stochastic inner GBDT actually decorrelates. Silently no-ops on a param
        the estimator does not declare -- ``set_params`` raises on unknown keys,
        so we filter to the declared param set first.
        """
        try:
            valid = set(estimator.get_params(deep=True).keys())
        except Exception:  # pragma: no cover - non-sklearn estimator
            return
        to_set = {}
        if "random_state" in valid:
            to_set["random_state"] = seed
        if "base_estimator__random_state" in valid:
            to_set["base_estimator__random_state"] = seed
        if to_set:
            try:
                estimator.set_params(**to_set)
            except (ValueError, TypeError) as err:  # pragma: no cover - defensive
                logger.debug(
                    "BaggedCompositeEstimator: could not set member seed (%r); " "member stays at its default random_state.",
                    err,
                )

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
        **fit_kwargs: Any,
    ) -> "BaggedCompositeEstimator":
        """Fit ``n_estimators`` members on bootstrap resamples of (X, y).

        ``sample_weight`` (when given) is resampled with the same bootstrap
        indices and passed through to each member's ``fit`` (members that do
        not accept it ignore it). ``**fit_kwargs`` forward to every member.
        Returns ``self``.
        """
        if self.aggregation not in ("trimmed_mean", "mean", "median"):
            raise ValueError(f"BaggedCompositeEstimator: aggregation must be one of " f"'trimmed_mean' / 'mean' / 'median'; got {self.aggregation!r}.")
        if not (0.0 <= self.trim_fraction < 0.5):
            raise ValueError(f"BaggedCompositeEstimator: trim_fraction must be in [0, 0.5); got {self.trim_fraction!r}.")
        if self.base_estimator is None:
            raise ValueError("BaggedCompositeEstimator: base_estimator must not be None.")
        if self.n_estimators < 1:
            raise ValueError(f"BaggedCompositeEstimator: n_estimators must be >=1, got " f"{self.n_estimators}.")
        if not (0.0 < self.max_samples <= 1.0):
            raise ValueError(f"BaggedCompositeEstimator: max_samples must be in (0, 1], got " f"{self.max_samples}.")

        n = _n_rows(X)
        if n == 0:
            raise ValueError("BaggedCompositeEstimator.fit: X has 0 rows.")
        n_draw = max(1, int(round(self.max_samples * n)))
        rng = np.random.RandomState(self.random_state)
        sw_arr = None if sample_weight is None else np.asarray(sample_weight)

        estimators: List[Any] = []
        for _m in range(int(self.n_estimators)):
            if self.bootstrap:
                idx = rng.randint(0, n, size=n_draw)
            else:
                # No resampling: every member sees all rows; diversity comes
                # solely from the per-member inner seed. Still draw from rng so
                # the member-seed stream below stays seed-deterministic.
                idx = np.arange(n)
            member_seed = int(rng.randint(0, np.iinfo(np.int32).max))

            est = clone(self.base_estimator)
            if self.vary_inner_random_state:
                self._set_member_seed(est, member_seed)

            X_m = _take_rows(X, idx) if self.bootstrap else X
            y_m = _take_1d(y, idx) if self.bootstrap else np.asarray(y)
            if sw_arr is not None:
                sw_m = sw_arr[idx] if self.bootstrap else sw_arr
                est.fit(X_m, y_m, sample_weight=sw_m, **fit_kwargs)
            else:
                est.fit(X_m, y_m, **fit_kwargs)
            estimators.append(est)

        self.estimators_ = estimators
        # sklearn-convention mirrors from the first member (best effort).
        first = estimators[0]
        n_feat = getattr(first, "n_features_in_", None)
        if n_feat is not None:
            self.n_features_in_ = n_feat
        names = getattr(first, "feature_names_in_", None)
        if names is not None:
            self.feature_names_in_ = list(names)
        return self

    def _member_predictions(self, X: Any) -> np.ndarray:
        """Stack member predictions -> (n_estimators, n_rows) float64 matrix."""
        if not getattr(self, "estimators_", None):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("BaggedCompositeEstimator: call fit before predict.")
        preds = [np.asarray(est.predict(X), dtype=np.float64).reshape(-1) for est in self.estimators_]
        return np.vstack(preds)

    def predict(self, X: Any) -> np.ndarray:
        """Lower-variance, outlier-robust point estimate across bootstrap members.

        Aggregation is selected by ``self.aggregation`` (default ``trimmed_mean``). Pickle-replay: models saved before this
        attribute existed are read via getattr with the legacy ``mean`` default, so an old saved model replays byte-identically.
        """
        members = self._member_predictions(X)
        aggregation = getattr(self, "aggregation", "mean")
        if aggregation == "mean":
            return np.asarray(members.mean(axis=0))
        if aggregation == "median":
            return np.asarray(np.median(members, axis=0))
        m = members.shape[0]
        k = int(np.floor(getattr(self, "trim_fraction", 0.1) * m))
        if k == 0:
            # Too few members to trim symmetrically -- the trimmed mean degenerates to the plain mean.
            return np.asarray(members.mean(axis=0))
        ordered = np.sort(members, axis=0)
        return np.asarray(ordered[k : m - k].mean(axis=0))

    def predict_std(self, X: Any) -> np.ndarray:
        """Across-member prediction spread -- an epistemic-uncertainty signal.

        Population std (ddof=0) of the member predictions per row, so it is
        EXACTLY 0 when every member predicts identically (e.g. a deterministic
        base estimator with ``bootstrap=False`` + ``vary_inner_random_state=False``)
        and grows where the bootstrap members disagree (extrapolation regions).
        Always >= 0.
        """
        return np.asarray(self._member_predictions(X).std(axis=0, ddof=0))

    def predict_interval_epistemic(
        self, X: Any, z: float = 1.96,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gaussian epistemic band ``(mean - z*std, mean + z*std)``.

        ``z`` is the standard-normal multiplier (1.96 ~ 95%). This is an
        EPISTEMIC interval (model-disagreement only); it does NOT include
        aleatoric / observation noise, so for calibrated marginal coverage pair
        it with split-conformal on a member's residuals. Returns ``(lower,
        upper)`` y-scale arrays.
        """
        preds = self._member_predictions(X)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0, ddof=0)
        return mean - z * std, mean + z * std

    # Delegate a couple of common introspection attributes to the first member
    # so a bagged composite is a near drop-in for a single one in diagnostics.
    @property
    def feature_importances_(self) -> np.ndarray:
        """Member-averaged ``feature_importances_`` (NotFittedError pre-fit)."""
        if not getattr(self, "estimators_", None):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("BaggedCompositeEstimator: feature_importances_ needs fit first.")
        mats = [np.asarray(e.feature_importances_, dtype=np.float64) for e in self.estimators_]
        return np.asarray(np.mean(np.vstack(mats), axis=0))
