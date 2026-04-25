"""Selection-bias-aware learning for "positives are heavily over-observed" data.

Use case (the one that prompted this module):
- You have ground-truth positives from a source that mostly only surfaces
  positives (e.g. a marketplace that only lists hired jobs).
- For most periods the dataset is positive-only — non-positives are
  systematically MISSING because you never observed them.
- For a SUBSET of periods you have full labels (you scraped *all* rows
  including unhired ones during a 1-month window).
- A naive classifier on the combined data sees P(y=1) ≈ 95% in train,
  predicts ~95% on everything, and gets blown out on TEST when the
  true prior is ~40%.

This module ships three strategies, picked via the ``strategy`` kwarg
on :class:`PULearningWrapper`:

1. ``"unbiased_only"`` — train the base classifier on the unbiased
   subset only. Simplest and best calibration when the unbiased subset
   is reasonably sized (>1k positive samples). Ignores biased data
   entirely. Works because the unbiased subset is a clean i.i.d.
   sample of the true population.

2. ``"prior_shift_correction"`` — train on the full data normally,
   then apply Saerens-Latinne-Decaestecker (2002) prior-shift
   correction at inference time:
   ``P_target(y=1|x) ∝ P_train(y=1|x) * (P_target(y)/P_train(y))``.
   Uses the full dataset for discrimination (so AUC tends to be
   highest) and corrects for the marginal-prior shift between train
   and target distributions. Assumes P(x|y) is the same in biased and
   unbiased periods (no covariate drift), which is reasonable when
   the bias is a label-selection mechanism and not a feature-distribution
   shift.

3. ``"elkan_noto"`` — Elkan & Noto (KDD 2008) PU classifier. Trains a
   proxy ``g(x) = P(s=1|x)`` with balanced sample weights (mandatory
   when s is severely skewed, which is the typical case here),
   estimates ``c = P(s=1|y=1)`` from the unbiased positives, and
   recovers ``f(x) = clip(g(x)/c, 0, 1)``. Theoretically elegant; in
   practice often beaten by the simpler strategies when the unbiased
   subset is small.

Default is ``"auto"``: chooses ``unbiased_only`` if the unbiased subset
has ≥ 2 × ``min_unbiased_positives`` rows on each class, otherwise
falls back to ``prior_shift_correction``.

Naive importance-weighting (downweight biased positives by
``true_prior / observed_pos_rate``) is intentionally NOT shipped: when
biased data is positive-only (the common case), the effective weighted
prior remains near ~0.9 even with the "correct" weight, because the
denominator (sum of weights × P(y=1)) is dominated by biased y=1 rows.
Saerens prior-shift correction is the inference-time-equivalent fix
that actually achieves target prior recovery.

References:
    Elkan, C. and Noto, K. (2008). "Learning Classifiers from Only
    Positive and Unlabeled Data." KDD 2008.

    Saerens, M., Latinne, P., Decaestecker, C. (2002). "Adjusting the
    outputs of a classifier to new a priori probabilities: a simple
    procedure." Neural Computation 14(1).

Public surface:
- PULearningWrapper — sklearn-style wrapper with the three strategies.
- estimate_c_from_unbiased_positives — standalone c estimator (Elkan-Noto).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

logger = logging.getLogger(__name__)


CEstimationMethod = Literal["mean_unbiased_pos", "max_unbiased_pos", "median_unbiased_pos"]
PUStrategy = Literal["auto", "unbiased_only", "prior_shift_correction", "elkan_noto"]


def estimate_c_from_unbiased_positives(
    proxy_probs: np.ndarray,
    method: CEstimationMethod = "mean_unbiased_pos",
) -> float:
    """Estimate ``c = P(s=1|y=1)`` from the proxy classifier's output on
    truly-positive unbiased rows.

    Parameters
    ----------
    proxy_probs : ndarray
        ``g(x) = P(s=1|x)`` evaluated on rows where the TRUE label is 1
        (typically the unbiased-period positives). For Elkan-Noto's
        constant-c assumption to hold these should be a clean random
        sample of the positive class.
    method : {"mean_unbiased_pos", "max_unbiased_pos", "median_unbiased_pos"}
        Aggregation rule. Mean (default) is the lowest-variance estimator
        but biased toward 0 if the proxy is noisy. Max is the original
        Elkan-Noto Method 2 — robust to label noise on negatives but
        very high variance. Median is a compromise.

    Returns
    -------
    float
        Estimated c in (0, 1]. Caller is responsible for handling
        c <= 0 / c > 1.
    """
    if proxy_probs.size == 0:
        raise ValueError("proxy_probs is empty — cannot estimate c.")
    if method == "mean_unbiased_pos":
        return float(np.mean(proxy_probs))
    if method == "max_unbiased_pos":
        return float(np.max(proxy_probs))
    if method == "median_unbiased_pos":
        return float(np.median(proxy_probs))
    raise ValueError(f"Unknown c-estimation method: {method!r}")


class PULearningWrapper(BaseEstimator, ClassifierMixin):
    """Selection-bias-aware classifier for positive-mostly training data.

    Wraps a sklearn-compatible binary classifier and trains it under one
    of three strategies that correct for "positives are heavily
    over-observed" selection bias. See the module docstring for the
    full explanation of when each strategy is the right pick.

    Parameters
    ----------
    base_estimator : sklearn-like classifier
        Anything with ``fit(X, y, [sample_weight=...])``,
        ``predict_proba(X) -> (N, 2)``, and ``classes_``. Tested with
        sklearn LR, HGB, XGB, LGB, CB.
    strategy : {"auto", "unbiased_only", "prior_shift_correction", "elkan_noto"}
        See module docstring. Default ``"auto"`` picks based on the
        size of the unbiased subset.
    true_prior : float, optional
        True population P(y=1). Used by ``prior_shift_correction`` as
        the target marginal at inference time. If ``None``, estimated
        from the unbiased subset's positive rate (assumes unbiased
        subset is representative of the target population).
    c_estimation_method : {"mean_unbiased_pos", "max_unbiased_pos", "median_unbiased_pos"}
        Used only for ``elkan_noto``. Default
        ``"mean_unbiased_pos"`` is lowest-variance.
    min_c_warn : float, default 0.05
        For ``elkan_noto``: warn when estimated c falls below this.
    min_unbiased_positives : int, default 50
        Minimum truly-positive unbiased samples required to fit. Below
        this any of the three strategies is unreliable; ``fit`` raises.
    balance_proxy : bool, default True
        For ``elkan_noto``: balance s=0 vs s=1 via sample_weight when
        training the proxy. Mandatory in the typical regime where s is
        severely skewed (s=1 ≫ s=0). Disable only if the base
        estimator handles imbalance natively (e.g. CB
        ``auto_class_weights='Balanced'``).
    auto_strategy_unbiased_count_threshold : int, default 1000
        For ``strategy="auto"``: use ``unbiased_only`` if the unbiased
        subset has at least this many positive AND this many negative
        samples; else fall back to ``importance_weighted``.

    Attributes
    ----------
    strategy_ : str
        The strategy actually used (resolved from ``"auto"``).
    base_estimator_ : fitted classifier
        The underlying classifier. For ``elkan_noto`` this predicts s;
        for the other strategies it predicts y directly.
    c_ : float, optional
        Estimated ``P(s=1|y=1)``. Only set for ``elkan_noto``.
    train_prior_ : float, optional
        Observed positive rate in the full training set. Only set for
        ``prior_shift_correction`` (used by Saerens correction at
        inference time).
    estimated_prior_ : float
        Implied / target population P(y=1). Useful for sanity-checking.
    classes_ : ndarray
        ``[0, 1]`` — binary by construction.
    """

    def __init__(
        self,
        base_estimator: Any,
        strategy: PUStrategy = "auto",
        true_prior: Optional[float] = None,
        c_estimation_method: CEstimationMethod = "mean_unbiased_pos",
        min_c_warn: float = 0.05,
        min_unbiased_positives: int = 50,
        balance_proxy: bool = True,
        auto_strategy_unbiased_count_threshold: int = 1000,
    ):
        self.base_estimator = base_estimator
        self.strategy = strategy
        self.true_prior = true_prior
        self.c_estimation_method = c_estimation_method
        self.min_c_warn = min_c_warn
        self.min_unbiased_positives = min_unbiased_positives
        self.balance_proxy = balance_proxy
        self.auto_strategy_unbiased_count_threshold = auto_strategy_unbiased_count_threshold

    def fit(
        self,
        X: Any,
        y: np.ndarray,
        *,
        is_unbiased: np.ndarray,
        **fit_params: Any,
    ) -> "PULearningWrapper":
        """Fit the PU classifier.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix. Forwarded to the base estimator as-is.
        y : ndarray of shape (n_samples,)
            Observed labels. In biased periods (``is_unbiased=False``)
            every observed row is positive, so y must be 1. In unbiased
            periods, y is the TRUE label (0 or 1).
        is_unbiased : ndarray of bool, shape (n_samples,)
            True for rows from a period where both classes were observed
            (so y is reliable). False for biased / positive-only rows.
        **fit_params
            Forwarded to ``base_estimator.fit``.
        """
        y = np.asarray(y).astype(np.int8, copy=False)
        is_unbiased = np.asarray(is_unbiased).astype(bool, copy=False)

        if y.shape[0] != is_unbiased.shape[0]:
            raise ValueError(
                f"y and is_unbiased length mismatch: y={y.shape[0]}, "
                f"is_unbiased={is_unbiased.shape[0]}"
            )

        unique_y = set(np.unique(y).tolist())
        if not unique_y.issubset({0, 1}):
            raise ValueError(
                f"PULearningWrapper is binary-only. Got y values: {unique_y}"
            )

        # Sanity check: biased rows should all be observed-positive.
        biased = ~is_unbiased
        if biased.sum() > 0 and y[biased].min() < 1:
            logger.warning(
                "%d biased rows have y=0 — selection-bias theory assumes "
                "biased rows are observed-positive only (y=1). The "
                "strategy will treat these as positives for proxy / "
                "weighting purposes; your `is_unbiased` flag may be "
                "inverted or the biased filter is leaking negatives.",
                int((y[biased] == 0).sum()),
            )

        # Count unbiased positives and negatives
        n_ub_pos = int((is_unbiased & (y == 1)).sum())
        n_ub_neg = int((is_unbiased & (y == 0)).sum())
        if n_ub_pos < self.min_unbiased_positives:
            raise ValueError(
                f"Need >= {self.min_unbiased_positives} unbiased positive "
                f"samples; got {n_ub_pos}. Either lower min_unbiased_positives "
                "at your own risk, or collect more fully-labeled data — the "
                "model can't learn calibration without it."
            )

        # Resolve "auto" strategy.
        strategy = self.strategy
        if strategy == "auto":
            if (n_ub_pos >= self.auto_strategy_unbiased_count_threshold
                    and n_ub_neg >= self.auto_strategy_unbiased_count_threshold):
                strategy = "unbiased_only"
            else:
                strategy = "prior_shift_correction"
            logger.info("PULearningWrapper auto-strategy: %s "
                        "(unbiased pos=%d, neg=%d, threshold=%d)",
                        strategy, n_ub_pos, n_ub_neg,
                        self.auto_strategy_unbiased_count_threshold)
        self.strategy_ = strategy

        if strategy == "unbiased_only":
            self._fit_unbiased_only(X, y, is_unbiased, **fit_params)
        elif strategy == "prior_shift_correction":
            self._fit_prior_shift_correction(X, y, is_unbiased, **fit_params)
        elif strategy == "elkan_noto":
            self._fit_elkan_noto(X, y, is_unbiased, **fit_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = getattr(
            self.base_estimator_, "n_features_in_",
            X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else None,
        )
        return self

    # ------------------------------------------------------------------
    # Strategy: unbiased_only
    # ------------------------------------------------------------------
    def _fit_unbiased_only(self, X, y, is_unbiased, **fit_params):
        X_ub = self._subset(X, is_unbiased)
        y_ub = y[is_unbiased]
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X_ub, y_ub, **fit_params)
        self.estimated_prior_ = float(y_ub.mean())

    # ------------------------------------------------------------------
    # Strategy: prior_shift_correction (Saerens et al. 2002)
    # ------------------------------------------------------------------
    def _fit_prior_shift_correction(self, X, y, is_unbiased, **fit_params):
        """Train normally; correct for prior shift at predict time."""
        if self.true_prior is not None:
            target_prior = float(self.true_prior)
        else:
            ub_y = y[is_unbiased]
            target_prior = float(ub_y.mean())
            logger.info(
                "PULearningWrapper(prior_shift_correction): true_prior not "
                "provided; estimated from unbiased subset = %.3f.",
                target_prior,
            )
        if not (0 < target_prior < 1):
            raise ValueError(
                f"true_prior must be in (0, 1); got {target_prior}."
            )

        train_prior = float((y == 1).mean())
        if train_prior <= 0 or train_prior >= 1:
            raise ValueError(
                f"Train P(y=1)={train_prior} is degenerate — Saerens "
                "correction needs both classes in train."
            )

        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y, **fit_params)

        # Cached for inference-time correction in predict_proba.
        self.train_prior_ = train_prior
        self.estimated_prior_ = target_prior

    # ------------------------------------------------------------------
    # Strategy: elkan_noto
    # ------------------------------------------------------------------
    def _fit_elkan_noto(self, X, y, is_unbiased, **fit_params):
        # Construct proxy label: s=1 if biased OR (unbiased AND y=1)
        s = ((y == 1) | (~is_unbiased)).astype(np.int8)

        # Balance proxy training (mandatory in skewed-s regime)
        if self.balance_proxy:
            n0 = int((s == 0).sum()); n1 = int((s == 1).sum())
            if n0 == 0:
                raise ValueError(
                    "elkan_noto requires at least one s=0 row (i.e. an "
                    "unbiased negative). Found none — switch to "
                    "unbiased_only or check is_unbiased flag."
                )
            sample_weight = np.where(s == 1, 1.0 / n1, 1.0 / n0).astype(np.float32)
            sample_weight *= (n0 + n1)  # rescale for numerical stability
        else:
            sample_weight = None

        if "sample_weight" in fit_params:
            extra_sw = np.asarray(fit_params.pop("sample_weight"), dtype=np.float32)
            sample_weight = (sample_weight * extra_sw) if sample_weight is not None else extra_sw

        self.base_estimator_ = clone(self.base_estimator)
        if sample_weight is not None:
            self.base_estimator_.fit(X, s, sample_weight=sample_weight, **fit_params)
        else:
            self.base_estimator_.fit(X, s, **fit_params)

        # Estimate c on truly-positive unbiased rows
        ub_pos_mask = is_unbiased & (y == 1)
        X_ub_pos = self._subset(X, ub_pos_mask)
        proxy_probs_ub_pos = self._predict_proba_pos(self.base_estimator_, X_ub_pos)
        c = estimate_c_from_unbiased_positives(
            proxy_probs_ub_pos, method=self.c_estimation_method,
        )
        if c <= 0 or c > 1:
            raise ValueError(
                f"Estimated c={c:.4g} is outside (0, 1]. Likely causes: "
                "proxy classifier is severely under-fit (g(x) tiny on "
                "true positives), or the unbiased subset is degenerate."
            )
        self.c_ = float(c)

        proxy_all = self._predict_proba_pos(self.base_estimator_, X)
        self.estimated_prior_ = float(np.clip(proxy_all.mean() / self.c_, 0.0, 1.0))

        if self.c_ < self.min_c_warn:
            logger.warning(
                "Estimated c=%.4f is below %.2f — recovered probabilities "
                "f(x)=g(x)/c will be sensitive to noise in g(x). Consider "
                "switching to unbiased_only or importance_weighted; both "
                "tend to beat elkan_noto when the unbiased subset is small.",
                self.c_, self.min_c_warn,
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X: Any) -> np.ndarray:
        """Returns calibrated P(y=1|x) — strategy-specific recovery.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns are P(y=0|x), P(y=1|x).
        """
        if not hasattr(self, "base_estimator_"):
            raise RuntimeError("PULearningWrapper not fitted. Call fit() first.")

        if self.strategy_ == "elkan_noto":
            g = self._predict_proba_pos(self.base_estimator_, X)
            f = np.clip(g / self.c_, 0.0, 1.0)
        elif self.strategy_ == "prior_shift_correction":
            # Saerens-Latinne-Decaestecker 2002: shift the marginal prior
            # at inference time. f(x) ∝ P_train(y=1|x) * P_target(y=1) /
            # P_train(y=1), normalised so positive + negative sum to 1.
            p_train_pos = self._predict_proba_pos(self.base_estimator_, X)
            p_train_neg = 1.0 - p_train_pos
            target = self.estimated_prior_
            train = self.train_prior_
            num = p_train_pos * (target / train)
            den = num + p_train_neg * ((1.0 - target) / (1.0 - train))
            f = np.clip(num / den, 0.0, 1.0)
        else:
            # unbiased_only: base predicts y directly.
            f = self._predict_proba_pos(self.base_estimator_, X)

        return np.column_stack([1.0 - f, f])

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int8)

    def decision_function(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    @staticmethod
    def _subset(X: Any, mask: np.ndarray) -> Any:
        """Container-aware row subset that doesn't copy more than needed."""
        if hasattr(X, "iloc"):
            return X.iloc[mask]
        if hasattr(X, "filter"):
            try:
                import polars as pl
                if isinstance(X, pl.DataFrame):
                    return X.filter(pl.Series(mask))
            except ImportError:
                pass
        return X[mask]

    @staticmethod
    def _predict_proba_pos(estimator: Any, X: Any) -> np.ndarray:
        """Extract P(class=1|x) from a binary classifier robustly."""
        proba = estimator.predict_proba(X)
        if isinstance(proba, list):
            proba = proba[0]
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, -1]
