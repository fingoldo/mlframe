"""Gaussian Mixture classifier: one GMM per class, predicting via posterior likelihood ratio.

COMPETITION / EXPLORATORY ONLY -- NEVER wire this into production defaults.

Source: 5th_instant-gratification.md -- "My model is GMM. The trick is to have 2
gaussian components for each class." That Kaggle dataset was known to be synthetically
generated from a low-dimensional Gaussian-mixture process, so fitting one
``sklearn.mixture.GaussianMixture`` per class and predicting by Bayes' rule (combining
each class's GMM likelihood with the class prior) matched the true data-generating
process exactly and beat generic boosted-tree/logistic baselines on that dataset.

This trick is NARROW: it only wins when the data really is a Gaussian mixture (or very
close to one). Real production tabular data essentially never satisfies this -- see the
"honest negative" test in ``tests/competition/test_biz_val_gmm_classifier.py``, where the
GMM classifier is shown to UNDER-perform a standard GBM/logistic baseline on ordinary
``make_classification``-style informative features. Use this only as a diagnostic/
baseline when a dataset is suspected (e.g. via a QDA/GMM-vs-GBM CV gap) to be sampled
from a low-dimensional Gaussian-mixture generative process -- a known Kaggle synthetic-
data quirk.

This module lives under ``mlframe.competition`` and must never be imported by production
mlframe modules or exported from mlframe's top-level ``__init__.py``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["GaussianMixtureClassifier"]


class GaussianMixtureClassifier(BaseEstimator, ClassifierMixin):
    """Direct classifier fitting one ``GaussianMixture`` per class.

    COMPETITION / EXPLORATORY ONLY -- see module docstring. Predicts via Bayes' rule:

        P(y=k | x)  ~  P(x | y=k) * P(y=k)

    where ``P(x | y=k)`` is the (exponentiated) log-likelihood of ``x`` under class
    ``k``'s fitted ``GaussianMixture`` (with ``n_components_per_class`` components), and
    ``P(y=k)`` is the empirical class prior. Only appropriate when the true
    data-generating process is itself (approximately) a Gaussian mixture per class; on
    generic tabular data it typically underperforms a standard GBM or logistic-regression
    baseline (see the honest-negative biz_value test).

    Parameters
    ----------
    n_components_per_class : int, default=2
        Number of Gaussian mixture components fit independently for each class.
    covariance_type : str, default="full"
        Passed through to each per-class ``sklearn.mixture.GaussianMixture``.
    reg_covar : float, default=1e-6
        Passed through to each per-class ``GaussianMixture``.
    max_iter : int, default=200
        Passed through to each per-class ``GaussianMixture``.
    n_init : int, default=1
        Passed through to each per-class ``GaussianMixture``.
    random_state : int or None, default=None
        Random state used for each per-class ``GaussianMixture``.
    """

    def __init__(
        self,
        n_components_per_class: int = 2,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        n_init: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_components_per_class = n_components_per_class
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "GaussianMixtureClassifier":
        """Fit one ``GaussianMixture`` per class on its own samples and store per-class log priors."""
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = unique_labels(y_arr)
        if self.classes_.shape[0] < 2:
            raise ValueError("GaussianMixtureClassifier requires at least 2 classes.")

        self.gmms_: dict[Any, GaussianMixture] = {}
        self.class_log_priors_: dict[Any, float] = {}
        n_samples = X_arr.shape[0]

        for cls in self.classes_:
            mask = y_arr == cls
            X_cls = X_arr[mask]
            n_components = min(self.n_components_per_class, X_cls.shape[0])
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            gmm.fit(X_cls)
            self.gmms_[cls] = gmm
            self.class_log_priors_[cls] = float(np.log(X_cls.shape[0] / n_samples))

        self.n_features_in_ = X_arr.shape[1]
        return self

    def _joint_log_likelihood(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return per-class joint log-likelihood (per-class GMM log-density plus class log-prior) for each sample."""
        log_probs = np.empty((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            log_probs[:, idx] = self.gmms_[cls].score_samples(X) + self.class_log_priors_[cls]
        return log_probs

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return per-class posterior probabilities via softmax-normalized joint log-likelihoods."""
        check_is_fitted(self, ["gmms_", "classes_"])
        X_arr = check_array(X)
        log_joint = self._joint_log_likelihood(X_arr)
        # log-sum-exp normalization for numerical stability
        max_log = np.max(log_joint, axis=1, keepdims=True)
        shifted = log_joint - max_log
        probs = np.exp(shifted)
        probs /= probs.sum(axis=1, keepdims=True)
        return np.asarray(probs, dtype=np.float64)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict the class label with highest posterior probability for each sample."""
        proba = self.predict_proba(X)
        return np.asarray(self.classes_[np.argmax(proba, axis=1)])
