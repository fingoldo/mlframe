""" Specialized estimators that can be iincluded into sklearn's ML pipelines."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

while True:
    try:

        # ----------------------------------------------------------------------------------------------------------------------------
        # Normal Imports
        # ----------------------------------------------------------------------------------------------------------------------------

        from typing import *
        import pandas as pd, numpy as np
        from scipy.ndimage.interpolation import shift
        from sklearn.preprocessing import KBinsDiscretizer,OrdinalEncoder
        from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, MultiOutputMixin

        from numbers import Number

        from scipy.special import boxcox
        from sklearn.preprocessing import PowerTransformer

        from collections.abc import Iterable
        from sklearn.utils import _safe_indexing, check_array
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.base import BaseEstimator, RegressorMixin, _fit_context, clone        

    except ModuleNotFoundError as e:

        logger.warning(e)

        if "cannot import name" in str(e):
            raise (e)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Packages auto-install
        # ----------------------------------------------------------------------------------------------------------------------------

        from pyutilz.pythonlib import ensure_installed

        ensure_installed("numpy pandas scikit-learn")

    else:
        break       

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

power_transformer_obj = PowerTransformer(method="box-cox")

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


class ESTransformedTargetRegressor(TransformedTargetRegressor):
    """Adds custom early stopping capabilities to vanilla TransformedTargetRegressor."""
    def __init__(
        self,
        regressor=None,
        *,
        transformer=None,
        func=None,
        inverse_func=None,
        check_inverse=True,
        es_fit_param_name:str=None,
    ):
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.es_fit_param_name = es_fit_param_name
        
    def _transform_y(self, y):
        y = check_array(
            y,
            input_name="y",
            accept_sparse=False,
            force_all_finite=True,
            ensure_2d=False,
            dtype="numeric",
            allow_nd=True,
        )
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y            
        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        return y_trans       
    
    def fit(self, X, y, **fit_params):
            """Fit the model according to the given training data.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.

            y : array-like of shape (n_samples,)
                Target values.

            **fit_params : dict
                Parameters passed to the `fit` method of the underlying
                regressor.

            Returns
            -------
            self : object
                Fitted estimator.
            """
            if y is None:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    "requires y to be passed, but the target y is None."
                )
            y = check_array(
                y,
                input_name="y",
                accept_sparse=False,
                force_all_finite=True,
                ensure_2d=False,
                dtype="numeric",
                allow_nd=True,
            )

            # store the number of dimension of the target to predict an array of
            # similar shape at predict
            self._training_dim = y.ndim

            # transformers are designed to modify X which is 2d dimensional, we
            # need to modify y accordingly.
            if y.ndim == 1:
                y_2d = y.reshape(-1, 1)
            else:
                y_2d = y
            self._fit_transformer(y_2d)        
            
            y_trans=self._transform_y(y_2d)
            if self.regressor is None:
                from ..linear_model import LinearRegression

                self.regressor_ = LinearRegression()
            else:
                self.regressor_ = clone(self.regressor)
                
            if self.es_fit_param_name:
                """print(type(fit_params[self.es_fit_param_name]))
                for idx, val_set in enumerate(fit_params[self.es_fit_param_name]):
                    print(type(val_set))"""
                es_param=[]
                multisets=False
                if self.es_fit_param_name in fit_params:
                    for idx, val_set in enumerate(fit_params[self.es_fit_param_name]):
                        if isinstance(val_set,(tuple,list)):
                            # print("isinstance(val_set,(tuple,list))")
                            es_param.append((val_set[0],self._transform_y(val_set[1])))
                            multisets=True
                        else:
                            if idx==1:
                                es_param.append(self._transform_y(val_set))
                            else:
                                es_param.append(val_set)
                            
                if es_param: 
                    """print(type(es_param))
                    for idx, val_set in enumerate(es_param):
                        print('after',type(val_set))"""
                    fit_params[self.es_fit_param_name]=tuple(es_param) if not multisets else es_param
            self.regressor_.fit(X, y_trans, **fit_params)

            if hasattr(self.regressor_, "feature_names_in_"):
                self.feature_names_in_ = self.regressor_.feature_names_in_

            return self

class PdOrdinalEncoder(OrdinalEncoder):
    
    def __init__(self, categories='auto', dtype=np.float32, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, min_frequency=None, max_categories=None):
        super().__init__(categories=categories,dtype=dtype,handle_unknown=handle_unknown,unknown_value=unknown_value,encoded_missing_value=encoded_missing_value,min_frequency=min_frequency,max_categories=max_categories)
    
    def transform(self, X):
        if isinstance(X,pd.DataFrame):
            col_names = X.columns.values.tolist()
        else:
            col_names=None
        
        X = super().transform(X)
        if col_names:
            return pd.DataFrame(data=X, columns=col_names).astype(np.int32)
        else:
            return X.astype(np.int32)


class PdKBinsDiscretizer(KBinsDiscretizer):

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile', dtype=None, subsample='warn', random_state=None):
        super().__init__(n_bins=n_bins,encode=encode, strategy=strategy, dtype=dtype, subsample=subsample, random_state=random_state)

    def transform(self, X):
        if isinstance(X,pd.DataFrame):
            col_names = X.columns.values.tolist()
        else:
            col_names=None
        
        X = super().transform(X)
        if col_names:
            return pd.DataFrame(data=X, columns=col_names).astype(np.int32)
        else:
            return X.astype(np.int32)

class ArithmAvgClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nprobs):
        self.nprobs = nprobs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        posProbs = np.mean(X[:, : self.nprobs], axis=1).reshape(-1, 1)
        return np.concatenate([1 - posProbs, posProbs], axis=1)


class GeomAvgClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nprobs):
        self.nprobs = nprobs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        posProbs = (np.product(X[:, : self.nprobs], axis=1) ** (1 / self.nprobs)).reshape(-1, 1)
        return np.concatenate([1 - posProbs, posProbs], axis=1)


class PureRandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nprobs):
        self.nprobs = nprobs

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        posProbs = np.random.random(len(X)).reshape(-1, 1)
        return np.concatenate([1 - posProbs, posProbs], axis=1)


class MyDecorrelator(BaseEstimator, TransformerMixin):
    """TODO: TEST PROPERLY"""

    def __init__(self, threshold):
        self.threshold = threshold
        self.correlated_columns = None

    def fit(self, X, y=None):
        correlated_features = set()
        X = pd.DataFrame(X)
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    correlated_features.add(colname)
        self.correlated_features = correlated_features
        return self

    def transform(self, X, y=None, **kwargs):
        return (pd.DataFrame(X)).drop(labels=self.correlated_features, axis=1)


def create_dummy_lagged_predictions(y_true: np.ndarray, strategy: str = "constant_lag", lag: int = 1) -> np.ndarray:
    """We can't created such estimator directly, as y_true is never passed during predict().
    So this helper func is just for train set.
    """
    assert strategy in ("constant_lag", "adaptive_lag")
    if strategy == "constant_lag":
        if y_true.ndim == 1:
            shift_params = lag
        elif y_true.ndim == 2:
            shift_params = (lag, 0)
        else:
            raise ValueError("Not supported target dimensionality")

        if lag > 0:
            if y_true.ndim == 1:
                cval = np.median(y_true, axis=0)
            elif y_true.ndim == 2:
                cval = np.median(y_true.flatten(), axis=0)
        else:
            cval = np.NaN

        y_pred = shift(y_true, shift=shift_params, cval=cval)

    return y_pred


# ----------------------------------------------------------------------------------------------------------------------------
# Target transforming functions
# ----------------------------------------------------------------------------------------------------------------------------


def qubed(x):
    return np.power(x, 3)


def log_plus_c(x, c: float = 0.0):
    return np.log(np.clip(x + c, 1e-16, None))


def inv_log_plus_c(x, c: float = 0.0):
    return np.exp(x) - c


def box_cox_plus_c(x, c: float = 50.0, lmbda: float = -1):
    return boxcox(np.clip(x + c, 1e-16, None), lmbda)


def inv_box_cox_plus_c(x, c: float = 50.0, lmbda: float = -1):
    return power_transformer_obj._box_cox_inverse_tranform(x, lmbda=lmbda) - c


def soft_winsorize(
    data: np.ndarray,
    abs_lower_threshold: float,
    rel_lower_limit: float,
    abs_upper_threshold: float,
    rel_upper_limit: float,
    distribution: str = "quantile",
    inplace: bool = False,
) -> None:
    """Analog of np.clip, but soft: does not lose SO much information.
    Instead of simple clipping, applies linear transformation so that max datapoint (subject to clipping to upper_clipping_threshold otherwise)
    becomes upper_clipping_threshold+rel_upper_limit.

    >>arr = np.array([1,2,156,3,4,5,150,],dtype="float32")
    >>soft_winsorize(arr, 2, 0.2, 140, 5, distribution="quantile") # everything above 140 will be distributed between 140 and 145 (+5 is relative), and under 2 betwee 1.8 and 2 (0.2 is relative)
    >>arr
    array([  1.8,   2. , 145. ,   3. ,   4. ,   5. , 142.5], dtype=float32)

    """
    assert distribution in ("linear", "quantile")

    rel_max_real_diff = np.max(data) - abs_upper_threshold
    rel_min_real_diff = abs_lower_threshold - np.min(data)
    assert rel_max_real_diff >= 0
    assert rel_min_real_diff >= 0

    if inplace:
        target = data
    else:
        target = data.copy()

    idx = np.where(target > abs_upper_threshold)[0]
    if len(idx) > 0:
        if distribution == "linear":
            target[idx] = abs_upper_threshold + (target[idx] - abs_upper_threshold) * rel_upper_limit / rel_max_real_diff
        elif distribution == "quantile":
            ordered = np.argsort(target[idx])
            ranks = np.argsort(ordered)
            target[idx] = abs_upper_threshold + (ranks + 1) * rel_upper_limit / len(ranks)

    idx = np.where(target < abs_lower_threshold)[0]

    if len(idx) > 0:
        if distribution == "linear":
            print(abs_lower_threshold - (abs_lower_threshold - target[idx]) * rel_lower_limit / rel_min_real_diff)
            target[idx] = abs_lower_threshold - (abs_lower_threshold - target[idx]) * rel_lower_limit / rel_min_real_diff
        elif distribution == "quantile":
            ordered = np.argsort(target[idx])[::-1]
            ranks = np.argsort(ordered)
            target[idx] = abs_lower_threshold - (ranks + 1) * rel_lower_limit / len(ranks)

    return target


def identity(x):
    return x


def clip_to_quantiles(arr: np.ndarray, quantile: float = 0.01, method: str = "winsor_quantile", winsor_rel_muliplier: float = 0.05) -> np.ndarray:
    """Clips ndarray to its symmetric quantiles either soft (soft_winsorize) or hard (np.clip) way."""
    assert method in ("hard", "winsor_linear", "winsor_quantile")

    assert isinstance(quantile, Number)
    assert 0 <= quantile <= 1

    assert isinstance(winsor_rel_muliplier, Number)
    assert 0 <= winsor_rel_muliplier <= 1

    if quantile > 0.5:
        quantile_from, quantile_to = np.quantile(arr, q=[1 - quantile, quantile])
    else:
        quantile_from, quantile_to = np.quantile(arr, q=[quantile, 1 - quantile])

    if method == "hard":
        return np.clip(arr, quantile_from, quantile_to)
    elif method == "winsor_linear":
        return soft_winsorize(
            data=arr,
            abs_lower_threshold=quantile_from,
            rel_lower_limit=quantile_from * (1 - winsor_rel_muliplier),
            abs_upper_threshold=quantile_to,
            rel_upper_limit=quantile_to * (1 + winsor_rel_muliplier),
            distribution="linear",
        )
    elif method == "winsor_quantile":
        return soft_winsorize(
            data=arr,
            abs_lower_threshold=quantile_from,
            rel_lower_limit=quantile_from * winsor_rel_muliplier,
            abs_upper_threshold=quantile_to,
            rel_upper_limit=quantile_to * winsor_rel_muliplier,
            distribution="quantile",
        )


def clip_to_quantiles_winsor_quantile(arr):
    return clip_to_quantiles(
        arr,
        quantile=0.01,
        method="winsor_quantile",
        winsor_rel_muliplier=0.05,
    )


def clip_to_quantiles_hard(arr):
    return clip_to_quantiles(arr, quantile=0.01, method="hard")
