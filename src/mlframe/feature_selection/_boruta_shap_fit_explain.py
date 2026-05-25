"""``BorutaShap.fit`` + ``BorutaShap.explain`` carved out of
``mlframe.feature_selection.boruta_shap``.

Methods are bound onto the ``BorutaShap`` class at the parent's module
bottom so ``self.fit(...)`` / ``self.explain(...)`` call sites resolve
unchanged.
"""

from __future__ import annotations

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator
from mlframe.utils.misc import get_pipeline_last_element
from pyutilz.system import tqdmu

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.sparse import issparse
try:
    from scipy.stats import binomtest as _binomtest

    def binom_test(x, n, p, alternative="two-sided"):
        # SciPy 1.7+ ``binomtest`` requires ``k`` integer; our hit-count vector is float (np.zeros), so coerce on the boundary.
        return _binomtest(int(x), n=int(n), p=p, alternative=alternative).pvalue
except ImportError:  # SciPy < 1.7 fallback
    from scipy.stats import binom_test  # type: ignore
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from numpy.random import choice
import seaborn as sns
import shap
import os
import re

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

import warnings

# Filters live inside ``fit()`` (scoped via warnings.catch_warnings) so importing this module no longer mutes legitimate sklearn FutureWarning / DeprecationWarning anywhere else in the process.

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)




def fit(self, X, y):
    """
    The main body of the program this method it computes the following

    1. Extend the information system by adding copies of all variables (the information system
    is always extended by at least 5 shadow attributes, even if the number of attributes in
    the original set is lower than 5).

    2. Shuffle the added attributes to remove their correlations with the response.

    3. Run a random forest classifier on the extended information system and gather the
    Z scores computed.

    4. Find the maximum Z score among shadow attributes (MZSA), and then assign a hit to
    every attribute that scored better than MZSA.

    5. For each attribute with undetermined importance perform a two-sided test of equality
    with the MZSA.

    6. Deem the attributes which have importance significantly lower than MZSA as ‘unimportant’
    and permanently remove them from the information system.

    7. Deem the attributes which have importance significantly higher than MZSA as ‘important’.

    8. Remove all shadow attributes.

    9. Repeat the procedure until the importance is assigned for all the attributes, or the
    algorithm has reached the previously set limit of the random forest runs.

    10. Stores results.

    Parameters
    ----------
    X: pandas.DataFrame or polars.DataFrame
        A dataframe of the features. polars frames are converted via a zero-copy Arrow-backed pandas view (``get_pandas_view_of_polars_df``) so downstream code keeps the pandas-only invariants ``check_X`` / ``create_shadow_features`` rely on.

    y: Series/ndarray
        A pandas series or numpy ndarray of the target

    random_state: int
        A random state for reproducibility of results

    Sample: Boolean
        if true then a rowise sample of the data will be used to calculate the feature importance values

    sample_fraction: float
        The sample fraction of the original data used in calculating the feature importance values only
        used if Sample==True.

    train_or_test: string
        Decides whether the feature importance should be calculated on out of sample data see the dicussion here.
        https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html#introduction-to-test-vs.training-data

    normalize: boolean
        if true the importance values will be normalized using the z-score formula

    verbose: Boolean
        a flag indicator to print out all the rejected or accepted features.

    stratify: array
        allows the train test splits to be stratified based on given values.

    """

    # polars input convert-on-the-spot; the rest of BorutaShap calls pandas idioms (``.copy()``, ``.columns.to_numpy()``, ``.apply``, ``.drop(inplace=True)``) and shap.TreeExplainer expects a pandas frame to read ``feature_names_in_``.
    if pl is not None and isinstance(X, pl.DataFrame):
        try:
            from mlframe.training.utils import get_pandas_view_of_polars_df
            X = get_pandas_view_of_polars_df(X)
        except ImportError:
            X = X.to_pandas(use_pyarrow_extension_array=True)
    if pl is not None and isinstance(y, pl.Series):
        y = y.to_pandas()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

        self.starting_X = X.copy()
        self.X = X.copy()
        self.y = y.copy()
        # Ordinal-encode object / pandas-Categorical columns in self.X so
        # the internal surrogate fit (Train_model) and the SHAP step
        # downstream both see numeric features. Pre-fix iter-179 / iter-237
        # path: when the suite's main cat-encoder is bypassed (polars-
        # fastpath models / cat_enc=ordinal where the encoder ran on the
        # polars-pre frame only), BorutaShap is fed a raw pandas frame
        # with object dtype cat cols; LGB / XGB surrogates raise
        # ``ValueError: could not convert string to float: 'A'`` and the
        # entire feature-selection branch is lost. The codes are private
        # to BorutaShap internals: ``transform`` returns ``X.iloc[:,
        # indices]`` of the CALLER-supplied frame (not self.X), so the
        # encoding never leaks into the downstream model's input. The CB
        # path also benefits because cat_features=col_names still works
        # on int codes (CB treats them as ordinal-encoded categories).
        _self_x_encoded_cols = self._ordinal_encode_object_cols_inplace(
            self.X,
        )
        if _self_x_encoded_cols:
            logger.debug(
                "BorutaShap: ordinal-encoded %d object/category col(s) "
                "for internal surrogate fit: %s",
                len(_self_x_encoded_cols), _self_x_encoded_cols,
            )

        self.ncols = self.X.shape[1]
        self.all_columns = self.X.columns.to_numpy()
        self.rejected_columns = []
        self.accepted_columns = []

        self.check_X()
        # self.check_missing_values()

        self.features_to_remove = []
        self.hits = np.zeros(self.ncols)
        self.order = self.create_mapping_between_cols_and_indices()
        self.create_importance_history()

        if self.sample:
            self.preds = self.isolation_forest(self.X)

        pbar = tqdmu(range(self.n_trials), desc="Feature selection", disable=not self.verbose)
        last_ncols = 0
        for trial in pbar:
            self.remove_features_if_rejected()
            self.columns = self.X.columns.to_numpy()
            self.create_shadow_features()

            # early stopping
            if self.X.shape[1] == 0:
                break

            else:
                self.Check_if_chose_train_or_test_and_train_model()

                self.X_feature_import, self.Shadow_feature_import = self.feature_importance(normalize=self.normalize)
                self.update_importance_history()
                hits = self.calculate_hits()
                self.hits += hits
                self.history_hits = np.vstack((self.history_hits, self.hits))
                self.test_features(iteration=trial + 1)

        self.store_feature_importance()
        self.calculate_rejected_accepted_tentative(verbose=self.verbose)
        pbar.set_description(f"Undecided features: {len(self.tentative):_}")
        new_ncols = len(self.columns)
        if new_ncols != last_ncols or trial % 5 == 0:
            logger.info(f"Undecided features: {len(self.tentative):_}")
            last_ncols = new_ncols

    # sklearn-style outputs so callers can treat BorutaShap like any other selector: ``support_`` is the boolean mask aligned with the input column order, ``selected_features_`` is the list of kept names (accepted + tentative when ``optimistic``).
    kept = set(self.accepted)
    if self.optimistic:
        kept |= set(self.tentative)
    self.support_ = np.array([c in kept for c in self.all_columns], dtype=bool)
    self.selected_features_ = [c for c in self.all_columns if c in kept]
    # sklearn convention: feature_names_in_ + n_features_in_ are the canonical
    # discoverable attributes for downstream report builders. Without them the
    # FS report's ``dropped_features`` field is None for BorutaShap and
    # asymmetric vs MRMR / RFECV.
    self.feature_names_in_ = np.asarray(self.all_columns)
    self.n_features_in_ = int(len(self.all_columns))


def explain(self):
    """
    The shap package has numerous variants of explainers which use different assumptions depending on the model
    type this function allows the user to choose explainer

    Returns:
        shap values

    Raise
    ----------
        ValueError:
            if no model type has been specified tree as default
    """
    est_name = type(self.model).__name__
    if est_name == "TransformedTargetRegressor":
        explainer_base = self.model.regressor
    elif est_name == "Pipeline":
        explainer_base = get_pipeline_last_element(self.model)
    else:
        explainer_base = self.model
    explainer = shap.TreeExplainer(explainer_base, feature_perturbation="tree_path_dependent")

    """
    ipdb> explainer_base.feature_names_
    ['1D-Price-arithmetic_mean', '1D-Price-ratio', '1D-Price-npositive', 'shadow_1D-Price-arithmetic_mean', 'shadow_1D-Price-ratio', 'shadow_1D-Price-npositive']
    ipdb> self.X_boruta.columns
    Index(['1D-Price-arithmetic_mean', '1D-Price-quadratic_mean',
        '1D-Price-qubic_mean', '1D-Price-harmonic_mean',
        'shadow_1D-Price-arithmetic_mean', 'shadow_1D-Price-quadratic_mean',
        'shadow_1D-Price-qubic_mean', 'shadow_1D-Price-harmonic_mean'],
        dtype='object')
    """

    if self.sample:
        basis = self.find_sample()
    else:
        basis = self.X_boruta

    # SHAP background must be the TRAIN slice -- self.X_boruta = [self.X | shadow] and self.X was set in fit() from the caller-supplied X (train) via X.copy(). The shadow half is randomized from self.X column-wise so it stays train-distribution-aligned. Both invariants must hold for SHAP TreeExplainer (tree_path_dependent feature_perturbation) to produce attributions on the same distribution the surrogate model was trained on; mixing val/test rows here would let SHAP interpolate against held-out distribution and inflate borderline features' importance.
    if hasattr(self, "X") and hasattr(self.X, "shape") and hasattr(self.X_boruta, "shape"):
        _n_train = int(self.X.shape[0])
        _n_basis = int(self.X_boruta.shape[0]) if not self.sample else int(basis.shape[0])
        if not self.sample:
            assert _n_basis == _n_train, (
                f"BorutaShap: SHAP background row count ({_n_basis}) != train row count ({_n_train}); "
                f"val/test rows must not leak into the explainer basis."
            )
        logger.info(
            "BorutaShap: SHAP TreeExplainer fitted on train background (n_train=%d, n_basis=%d, sampled=%s)",
            _n_train, _n_basis, bool(self.sample),
        )

    # ``self.y.shape[1] > 1`` raises IndexError on 1-D regression targets
    # (shape is ``(n,)``). The intent is "multi-output regression"; guard
    # with ``ndim >= 2``. Pre-fix iter-237 / iter-280: BorutaShap on a 1-D
    # regression target crashed after 50+ minutes of SHAP computation
    # with ``IndexError: tuple index out of range``.
    _y_multi = (
        hasattr(self.y, "shape")
        and getattr(self.y, "ndim", 1) >= 2
        and self.y.shape[1] > 1
    )
    if self.classification or _y_multi:
        # for some reason shap returns values wraped in a list of length 1
        # Wave 29 P1 fix (2026-05-20): pre-fix wrapped the raw return
        # in ``np.array(...)`` BEFORE the ``isinstance(..., list)``
        # check, which made the list branch unreachable on modern
        # SHAP that returns ``list[ndarray]`` for multi-class. As a
        # result multi-class SHAP aggregation silently mis-counted
        # importances (ran the 3-D ndarray branch which sums over
        # axis=0 = classes; the list branch's per-class abs-mean
        # accumulation never fired).
        # Inspect the RAW return type first; only wrap in np.array
        # when we've confirmed it's a single-class single-array case.
        _raw_shap = explainer.shap_values(basis)
        if isinstance(_raw_shap, list):
            # Multi-class SHAP path: list of per-class (n_samples, n_features) arrays.
            self.shap_values = _raw_shap
            class_inds = range(len(self.shap_values))
            shap_imp = np.zeros(self.shap_values[0].shape[1])
            for i, ind in enumerate(class_inds):
                shap_imp += np.abs(self.shap_values[ind]).mean(0)
            # Final aggregated per-feature importance (averaged across classes).
            self.shap_values = shap_imp / len(class_inds)
        else:
            self.shap_values = np.asarray(_raw_shap)
            if self.shap_values.ndim == 3:
                self.shap_values = np.abs(self.shap_values).sum(axis=0)
                self.shap_values = self.shap_values.mean(0)
            else:
                self.shap_values = np.abs(self.shap_values).mean(0)

    else:
        self.shap_values = explainer.shap_values(basis)
        self.shap_values = np.abs(self.shap_values).mean(0)
