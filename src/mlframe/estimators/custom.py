""" Specialized estimators that can be iincluded into sklearn's ML pipelines."""

from __future__ import annotations


# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
import pandas as pd, numpy as np
from scipy.ndimage import shift
from sklearn.preprocessing import KBinsDiscretizer,OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin, MultiOutputMixin

from numbers import Number

from scipy.special import boxcox
from sklearn.preprocessing import PowerTransformer

from collections.abc import Iterable
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context, clone

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

# Module-level PowerTransformer removed: it was never fit and unsafe across threads.
# Callers should pass a fitted PowerTransformer to inv_box_cox_plus_c via the `transformer` kwarg.

# sklearn 1.6 renamed check_array's force_all_finite -> ensure_all_finite (force_all_finite
# is deprecated in 1.6 and removed in 1.8). Resolve once at import time so call sites stay
# clean and the same source supports the 1.5-1.8 matrix.
import sklearn as _skl
_skl_ver = tuple(int(p) for p in _skl.__version__.split(".")[:2])
_FORCE_ALL_FINITE_KW = (
    {"ensure_all_finite": True} if _skl_ver >= (1, 6) else {"force_all_finite": True}
)

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
        # Wave 56 (2026-05-20): forward to parent __init__ so any new sklearn
        # TransformedTargetRegressor attrs (e.g. validate_inverse, _n_features_out)
        # added in minor sklearn releases are populated -- the prior manual
        # re-assignment of the 5 known attrs would silently drop new ones from
        # get_params/clone introspection.
        super().__init__(
            regressor=regressor,
            transformer=transformer,
            func=func,
            inverse_func=inverse_func,
            check_inverse=check_inverse,
        )
        self.es_fit_param_name = es_fit_param_name

    def _transform_y(self, y):
        y = check_array(
            y,
            input_name="y",
            accept_sparse=False,
            ensure_2d=False,
            dtype="numeric",
            allow_nd=True,
            **_FORCE_ALL_FINITE_KW,
        )
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y            
        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # Wave 63 (2026-05-20): sklearn FunctionTransformer can return a 1D array
        # even with validate=True (when the wrapped func returns a 1D array
        # itself). The dim-check below handles both shapes; not actionable.
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
                ensure_2d=False,
                dtype="numeric",
                allow_nd=True,
                **_FORCE_ALL_FINITE_KW,
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

    # Wave 50 (2026-05-20): default ``encoded_missing_value`` flipped from
    # ``np.nan`` to ``-1``. Pre-fix, NaN -> int32 produced platform-dependent
    # INT_MIN sentinels that broke downstream code expecting codes in [0, K)
    # AND collided with no real category but obscured the missing-value contract.
    # Callers needing nan-on-missing behaviour should pass it explicitly and
    # NOT use ``.astype(np.int32)`` -- this class always coerces to int.
    def __init__(self, categories='auto', dtype=np.float32, handle_unknown='error', unknown_value=None, encoded_missing_value=-1, min_frequency=None, max_categories=None):
        super().__init__(categories=categories,dtype=dtype,handle_unknown=handle_unknown,unknown_value=unknown_value,encoded_missing_value=encoded_missing_value,min_frequency=min_frequency,max_categories=max_categories)

    def transform(self, X):
        if isinstance(X,pd.DataFrame):
            col_names = X.columns.values.tolist()
        else:
            col_names=None

        X = super().transform(X)
        # Wave 50 (2026-05-20): if caller overrode encoded_missing_value=np.nan,
        # surface NaN explicitly instead of producing platform-dependent INT_MIN
        # via astype(int32). NaN-in-int32 is undefined behaviour.
        if np.any(np.isnan(np.asarray(X, dtype=np.float64))):
            raise ValueError(
                "PdOrdinalEncoder.transform: NaN codes in output -- caller set "
                "encoded_missing_value=np.nan but transform unconditionally casts to int32. "
                "Either keep encoded_missing_value=-1 (default) or change dtype contract."
            )
        if col_names:
            return pd.DataFrame(data=X, columns=col_names).astype(np.int32)
        else:
            return X.astype(np.int32)


class PdKBinsDiscretizer(KBinsDiscretizer):

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile', dtype=None, subsample=200_000, random_state=None):
        super().__init__(n_bins=n_bins,encode=encode, strategy=strategy, dtype=dtype, subsample=subsample, random_state=random_state)

    def transform(self, X):
        if isinstance(X,pd.DataFrame):
            col_names = X.columns.values.tolist()
        else:
            col_names=None

        X = super().transform(X)
        # KBinsDiscretizer with encode='onehot' returns sparse; densify for pandas path
        if hasattr(X, "toarray"):
            X = X.toarray()
        if col_names:
            return pd.DataFrame(data=X, columns=col_names).astype(np.int32)
        else:
            return X.astype(np.int32)

class ArithmAvgClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nprobs):
        self.nprobs = nprobs

    def fit(self, X, y):
        X = check_array(X)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        posProbs = np.mean(X[:, : self.nprobs], axis=1).reshape(-1, 1)
        return np.concatenate([1 - posProbs, posProbs], axis=1)


class GeomAvgClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, nprobs):
        self.nprobs = nprobs

    def fit(self, X, y):
        X = check_array(X)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        # Geometric mean is undefined on negatives, and ``a ** (1/n)`` returns
        # NaN with a RuntimeWarning when ``np.prod(...) < 0``. The estimator
        # documents that X carries pre-computed probability columns (values
        # in [0, 1]) - silently clip into the valid domain so a stray
        # ``-0.0`` from a downstream softmax-eps doesn't poison every row,
        # and use the log-sum-exp form to avoid intermediate over/underflow
        # for nprobs>=2 (a 10-way product of small probs collapses to 0.0
        # in float64). Pre-fix the test cluster around np.prod hit
        # "invalid value encountered in sqrt" on every check_estimator call
        # because sklearn's synthetic X is N(0, 1) including negatives.
        _x = np.clip(X[:, : self.nprobs].astype(np.float64), 0.0, 1.0)
        # log-mean-exp: posProbs = exp(mean(log(x))) so 0-prob rows return 0
        # instead of nan from log(0); use a tiny epsilon to keep the log finite.
        _logx = np.log(np.clip(_x, 1e-300, 1.0))
        posProbs = np.exp(_logx.mean(axis=1)).reshape(-1, 1)
        return np.concatenate([1 - posProbs, posProbs], axis=1)


class PureRandomClassifier(BaseEstimator, ClassifierMixin):
    """Random-prediction baseline. Respects `random_state` for reproducibility.

    Follows sklearn conventions: stores `classes_`, `n_features_in_` in fit,
    and `predict` returns original class labels (not argmax indices).
    """

    def __init__(self, nprobs=2, random_state=None):
        self.nprobs = nprobs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.random_state_ = check_random_state(self.random_state)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self, "random_state_")
        n = len(X)
        n_cls = len(self.classes_)
        if n_cls == 2:
            posProbs = self.random_state_.random_sample(n).reshape(-1, 1) if hasattr(self.random_state_, "random_sample") else self.random_state_.random(n).reshape(-1, 1)
            return np.concatenate([1 - posProbs, posProbs], axis=1)
        # Multiclass: sample Dirichlet-like rows (uniform then normalize).
        raw = self.random_state_.random_sample((n, n_cls)) if hasattr(self.random_state_, "random_sample") else self.random_state_.random((n, n_cls))
        return raw / raw.sum(axis=1, keepdims=True)


class MyDecorrelator(BaseEstimator, TransformerMixin):
    """TODO: TEST PROPERLY"""

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        correlated_features = set()
        X = pd.DataFrame(X)
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:  # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    correlated_features.add(colname)
        self.correlated_features_ = correlated_features
        return self

    def transform(self, X, y=None, **kwargs):
        check_is_fitted(self)
        return (pd.DataFrame(X)).drop(labels=self.correlated_features_, axis=1)


def create_dummy_lagged_predictions(y_true: np.ndarray, strategy: str = "constant_lag", lag: int = 1) -> np.ndarray:
    """We can't created such estimator directly, as y_true is never passed during predict().
    So this helper func is just for train set.
    """
    # Wave 31 (2026-05-20): assert -> ValueError so -O doesn't strip the
    # check. Pre-fix invalid strategy slipped past + ``y_pred`` was
    # never assigned, causing UnboundLocalError far from root.
    if strategy not in ("constant_lag", "adaptive_lag"):
        raise ValueError(
            f"strategy must be 'constant_lag' or 'adaptive_lag'; got {strategy!r}."
        )
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


def inv_box_cox_plus_c(x, c: float = 50.0, lmbda: float = -1, transformer: "PowerTransformer | None" = None):
    if transformer is None:
        raise NotFittedError(
            "inv_box_cox_plus_c requires a fitted PowerTransformer passed via `transformer=`. "
            "The previous module-level instance was removed because it was never fit and was unsafe across threads."
        )
    return transformer._box_cox_inverse_tranform(x, lmbda=lmbda) - c


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
    # Wave 31 (2026-05-20): assert -> ValueError so -O preserves input validation.
    if distribution not in ("linear", "quantile"):
        raise ValueError(
            f"distribution must be 'linear' or 'quantile'; got {distribution!r}."
        )

    rel_max_real_diff = np.max(data) - abs_upper_threshold
    rel_min_real_diff = abs_lower_threshold - np.min(data)
    if rel_max_real_diff < 0:
        raise ValueError(
            f"abs_upper_threshold={abs_upper_threshold} is above the data max "
            f"({np.max(data)}); no data to winsorize above."
        )
    if rel_min_real_diff < 0:
        raise ValueError(
            f"abs_lower_threshold={abs_lower_threshold} is below the data min "
            f"({np.min(data)}); no data to winsorize below."
        )

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
    # Wave 31 (2026-05-20): assert -> ValueError so -O preserves input validation.
    # Pre-fix bad ``method`` slipped past and the elif chain returned None.
    if method not in ("hard", "winsor_linear", "winsor_quantile"):
        raise ValueError(
            f"method must be 'hard', 'winsor_linear', or 'winsor_quantile'; "
            f"got {method!r}."
        )
    if not isinstance(quantile, Number):
        raise TypeError(f"quantile must be a number; got {type(quantile).__name__}.")
    if not (0 <= quantile <= 1):
        raise ValueError(f"quantile must be in [0, 1]; got {quantile!r}.")
    if not isinstance(winsor_rel_muliplier, Number):
        raise TypeError(
            f"winsor_rel_muliplier must be a number; got {type(winsor_rel_muliplier).__name__}."
        )
    if not (0 <= winsor_rel_muliplier <= 1):
        raise ValueError(
            f"winsor_rel_muliplier must be in [0, 1]; got {winsor_rel_muliplier!r}."
        )

    # Wave 39 (2026-05-20): np.quantile on empty input raises an opaque IndexError
    # in numpy>=1.22. Public utility may receive post-filter empty arrays; treat
    # empty as identity (nothing to clip).
    arr_arr = np.asarray(arr)
    if arr_arr.size == 0:
        return arr_arr.copy()

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



class IdentityEstimator(BaseEstimator):
    """Pass-through estimator: returns selected existing feature(s) as-is instead of learning & predicting.

    This is useful for benchmarking: treats a pre-computed column as the model's "prediction", so you can
    compare downstream metrics against real models. For classifier usage (IdentityClassifier), the caller
    must ensure the selected feature already contains values drawn from `self.classes_` (fitted from y).

    Note: `predict` returns the raw feature slice — it is NOT sklearn-standard label-from-argmax behaviour.
    """

    def __init__(self,feature_names:list=None,feature_indices:list=None):
        self.feature_names=feature_names
        self.feature_indices=feature_indices

    def fit(self, X, y, **fit_params):
        if isinstance(self, ClassifierMixin):
            # Wave 61 (2026-05-20): object-dtype label set with mixed types
            # (None + str) would TypeError on Python sorted(); use np.sort
            # when dtype is numeric, str-key fallback otherwise.
            _y_arr = y.unique() if isinstance(y, pd.Series) else np.unique(y)
            if hasattr(_y_arr, "dtype") and _y_arr.dtype != object:
                self.classes_ = np.sort(_y_arr)
            else:
                self.classes_ = np.array(sorted(_y_arr, key=lambda v: (v is None, str(v))))
        return self

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            if self.feature_names:
                return X.loc[:, self.feature_names].values
            else:
                # Wave 31 (2026-05-20): assert -> ValueError so -O doesn't
                # strip the guard. Constructor default is None for both
                # feature_names and feature_indices; calling predict()
                # without setting EITHER would raise an opaque IndexError
                # rather than a clear missing-config message.
                if self.feature_indices is None:
                    raise ValueError(
                        "IdentityEstimator: neither feature_names nor "
                        "feature_indices set on the instance; pass one at "
                        "construction time."
                    )
                return X.iloc[:, self.feature_indices].values
        else:
            if self.feature_indices is None:
                raise ValueError(
                    "IdentityEstimator: feature_indices not set; pass "
                    "feature_indices at construction for ndarray inputs."
                )
            return X[:, self.feature_indices]


class IdentityRegressor(IdentityEstimator, RegressorMixin):
    pass


class IdentityClassifier(IdentityEstimator, ClassifierMixin):
    def predict_proba(self, X):
        last_class_probs = self.predict(X)
        if len(self.classes_) == 2 and last_class_probs.ndim==1:
            return np.vstack([1 - last_class_probs, last_class_probs]).T
        else:
            return last_class_probs
