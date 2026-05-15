import logging

logger = logging.getLogger(__name__)

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_consistent_length
from sklearn.base import RegressorMixin, ClassifierMixin, TransformerMixin, BaseEstimator, clone
from sklearn.utils.estimator_checks import check_estimator, check_transformer_general
from sklearn.utils import check_random_state

from sklearn.model_selection import train_test_split

class EstimatorWithEarlyStopping(BaseEstimator):
    """
        Adds early stopping in pipeline to estimators that only accept fixed evaluation set, like Catboost.
    """
    def __init__(self, base_estimator=None, test_size=0.05, train_size=None, random_state=None, shuffle=True, stratify=None, plot: bool = False):
        self.plot = plot
        self.base_estimator = base_estimator
        self.test_size, self.train_size, self.random_state, self.shuffle, self.stratify = test_size, train_size, random_state, shuffle, stratify

    def fit(self, X, y, **fit_params):

        X = check_array(X)

        random_state = check_random_state(self.random_state)

        self.random_state_ = random_state
        fitted_estimator = clone(self.base_estimator)

        if "CatBoost" in type(fitted_estimator).__name__:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, train_size=self.train_size, random_state=random_state, shuffle=self.shuffle, stratify=self.stratify
            )
            fitted_estimator.fit(X_train, y_train, eval_set=(X_val, y_val), plot=self.plot, **fit_params)
        else:
            logger.warning(f"Early stopping params for estimator of type {type(self.base_estimator)} unknown.")
            fitted_estimator.fit(X, y, **fit_params)

        self.fitted_estimator_=fitted_estimator

        self.n_features_in_=X.shape[1]

        return self

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        return self.fitted_estimator_.predict(X)


class RegressorWithEarlyStopping(EstimatorWithEarlyStopping, RegressorMixin):
    pass


class ClassifierWithEarlyStopping(EstimatorWithEarlyStopping, ClassifierMixin):

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.fitted_estimator_.predict_proba(X)

    def decision_function(self, X):
        check_is_fitted(self)
        if not hasattr(self.fitted_estimator_, "decision_function"):
            raise AttributeError(
                f"Wrapped estimator {type(self.fitted_estimator_).__name__} has no decision_function"
            )
        return self.fitted_estimator_.decision_function(X)
