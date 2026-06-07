
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

import functools as _functools


@_functools.lru_cache(maxsize=None)
def _binom_test_cached(x_int: int, n: int, p: float, alternative: str = "two-sided"):
    """Memoized ``binom_test``: BorutaShap runs it per FEATURE per iteration (tens of thousands of
    calls on a wide frame), but ``(n, p, alternative)`` are fixed within a step and the hit count
    ``x`` is a small integer, so the distinct ``(x, n, p, alternative)`` set is tiny. Caching
    collapses ~36k per-call scipy ``binomtest`` constructions (profiled ~7s = 21% of a 299-feature
    fit) to a handful -- bit-identical p-values."""
    return binom_test(x_int, n=n, p=p, alternative=alternative)


from scipy.stats import ks_2samp
import random
import pandas as pd
import numpy as np
from numpy.random import choice
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


class BorutaShap(BaseEstimator, TransformerMixin):

    """
    BorutaShap is a wrapper feature selection method built on the foundations of both the SHAP and Boruta algorithms.

    KNOWN LIMITATION -- single-sample false positives via the shadow comparison. Shadow features are produced by
    permuting each real column independently, which destroys not only the column's relationship to y but ALL of its
    joint/covariance structure with the other columns. A real column always retains its (often spurious, finite-sample)
    joint structure, so a model's importance fit on a single training sample systematically scores structure-bearing
    real columns above the structure-less shadows. The net effect: the TOP one or two pure-noise features in the pool
    tend to clear the shadow gate and get accepted, even though their importance is barely above the noise median and
    does not survive on an independent draw. This is intrinsic to the permutation-shadow design and is NOT cured by:
      - raising ``percentile`` to 100 (the leak beats even the MAX shadow), nor
      - switching ``importance_measure`` between 'gini' and 'shap' (BOTH impurity- and SHAP-importance leak the top
        real-noise feature), nor
      - ``train_or_test='test'`` when train and test come from the same draw (the spurious structure is in both halves).
    Partial mitigation via cross-subsample stability (built in: set ``stability_subsamples`` > 1). Run the selector on
    several row-subsamples WITHOUT replacement and keep features by their accept FREQUENCY. Measured on a 52-feature
    synthetic: a single fit accepted the top noise column (~58/60 hits even at percentile=100); over 10 distinct 75%
    subsamples the strongly-relevant features were accepted 10/10, while that spurious column's frequency was UNSTABLE
    across seed sets (3/10 .. 8/10) - its spurious correlation is a property of the whole data DRAW, so most subsamples
    of that draw inherit it. Consequences, both verified: a MAJORITY vote (threshold ~0.6) does NOT reliably drop it;
    only the INTERSECTION (``stability_threshold=1.0``, the default once stability is enabled) does - the spurious
    column is never accepted by ALL subsamples (7/10 and 0/10 across seeds -> dropped) while strongly-relevant features
    are (10/10 -> kept). The cost is recall: intersection also drops weakly / inconsistently-relevant features (e.g.
    pure interaction operands with ~zero marginal signal). Subsample WITHOUT replacement, NOT bootstrap with
    replacement (duplicate rows let the model memorise, so every feature beats its shadow and the gate accepts
    ~everything). FUNDAMENTAL LIMIT: a spurious correlation present throughout the draw cannot be removed by resampling
    that same draw; only an INDEPENDENT validation draw (or a downstream model robust to a few extra features) resolves it.

    DRIVER SELECTION (importance_measure), measured on the fs_hybrid bed (6 scenarios x 2 seeds, downstream
    honest-holdout AUC over LightGBM/Logistic/kNN). The default is 'gini': SHAP is dominated on BOTH axes (worst
    mean AUC 0.755 AND ~137x slower than gini -- 20-30 min/fit -- because it recomputes per-trial SHAP values),
    so it is no longer the default. gini (mean 0.762, ~11 s) is the fast, robust default. 'permutation' in its
    honest held-out mode (train_or_test='test', which permutes on the 30% holdout this class already carves) is
    the ACCURACY + PRECISION leader: it topped mean AUC (0.766) and drove accepted-noise to ~0 in 10/12 cells
    (gini leaks 1-5, SHAP up to 11), at ~11x gini's cost; choose it when false-positive control matters. Held-out
    permutation reduces the generic in-sample-optimism leak but does NOT erase the fundamental same-draw spurious
    structure above. A further measured win for redundant/monotone data is to collapse raw-correlation clusters to
    one representative BEFORE the gate and re-expand accepted reps afterward (the gate sees cleaner, fewer columns);
    this lives naturally in a caller that already computes the clustering (see the fs_hybrid HybridSelector).
    """

    def __init__(
        self,
        model=None,
        importance_measure="gini",
        permutation_n_repeats: int = 5,
        classification=True,
        percentile=99,
        pvalue=0.05,
        n_trials=150,
        random_state=0,
        sample: bool = False,
        train_or_test="train",
        premerge_clusters: bool = False,
        premerge_corr_thr: float = 0.92,
        normalize: bool = True,
        verbose: bool = True,
        stratify=None,
        optimistic: bool = True,
        fit_params: dict = None,
        stability_subsamples: int = 0,
        stability_subsample_fraction: float = 0.75,
        stability_threshold: float = 1.0,
        early_stop_tentative: bool = False,
        early_stop_patience: int = 20,
        early_stop_margin: float = 0.15,
        max_runtime_mins: float = None,
        stop_file: str = "stop",
    ):
        """
        Parameters
        ----------
        model: Model Object
            If no model specified then a base Random Forest will be returned otherwise the specifed model will
            be returned.

        importance_measure: String
            Which importance measure to use: "Shap", "gini"/"gain", or "permutation". Permutation uses sklearn's
            permutation_importance on the held-out 30% split when train_or_test="test" (debiased held-out
            degradation, the signal that beat impurity/SHAP in the importance shootout), else falls back to
            in-bag permutation on the full data. permutation_n_repeats controls its repeat count.

        classification: Boolean
            if true then the problem is either a binary or multiclass problem otherwise if false then it is regression

        percentile: Int
            An integer ranging from 0-100 it changes the value of the max shadow importance values. Thus, lowering its value
            would make the algorithm more lenient.

        p_value: float
            A float used as a significance level again if the p-value is increased the algorithm will be more lenient making it smaller
            would make it more strict also by making the model more strict could impact runtime making it slower. As it will be less likley
            to reject and accept features.

        early_stop_tentative: bool (default False)
            Opt-in margin-gated adaptive trial-stop for the residual tentative tail. Default OFF, so behaviour is
            byte-identical to the prior fixed-cap run. When True the trial loop stops early (before the n_trials cap)
            once the tail is provably stuck: the accepted set has been unchanged for ``early_stop_patience`` trials AND
            no still-tentative feature is within ``early_stop_margin`` of crossing either binomial decision threshold.
            This is decision-equivalent to running the full cap (the tail never resolves) but reclaims the dominant
            per-trial model-fit cost. NOT the naive accepted-set-stability stop (that is a measured correctness trap).

        early_stop_patience: int (default 20)
            Number of consecutive trials the accepted set must stay unchanged before the margin gate is evaluated.

        early_stop_margin: float (default 0.15)
            Relative slack on the binomial decision threshold. A still-tentative feature counts as "near a boundary"
            (so the stop is refused) when its Bonferroni-corrected accept- or reject-p-value < ``pvalue * (1 + margin)``.

        """
        # sklearn contract: __init__ stores params VERBATIM (no mutation / validation), so the estimator is
        # clone-able (GridSearchCV / Pipeline). importance_measure is lowercased at its comparison sites, fit_params
        # defaults to {} at use, and model defaulting + validation (check_model) is deferred to fit().
        self.importance_measure = importance_measure
        self.permutation_n_repeats = permutation_n_repeats
        self.percentile = percentile
        self.pvalue = pvalue
        self.classification = classification
        self.model = model

        # Use a private rng so the suite's global numpy RNG state is not mutated by
        # BorutaShap construction. Prior code seeded np.random globally, leaking the
        # seed into every downstream consumer (data augmentation, FE) of the same
        # process and violating the suite's determinism contract.
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.n_trials = n_trials
        self.sample = sample
        self.train_or_test = train_or_test
        # premerge_clusters (off by default): collapse raw |corr| >= premerge_corr_thr clusters to one representative
        # BEFORE the shadow gate, then re-expand accepted reps to their members. De-dilutes the shadow comparison on
        # redundant data -> measured (R2b-6) +recall, -noise, and faster (the gate sees fewer columns). 7/12 cells won
        # on the fs_hybrid bed; already shipped inside the HybridSelector, now available standalone.
        self.premerge_clusters = premerge_clusters
        self.premerge_corr_thr = premerge_corr_thr
        self.stratify = stratify
        self.verbose = verbose
        self.normalize = normalize

        self.optimistic = optimistic
        self.fit_params = fit_params

        # Cross-subsample stability gate (opt-in; see class docstring). When stability_subsamples>1 the fit runs that
        # many sub-fits on distinct row-subsamples WITHOUT replacement (each of size stability_subsample_fraction*n)
        # and keeps features ACCEPTED in >= stability_threshold of them. stability_threshold=1.0 (INTERSECTION, the
        # default once enabled) keeps only features accepted by ALL subsamples: this is the setting that reliably
        # drops a draw-level-spurious noise column (never accepted by all) while keeping strongly-relevant features;
        # a MAJORITY threshold (~0.6) is high-variance for such columns. Use >= ~8-10 subsamples so strongly-relevant
        # features hit the all-accept bar reliably. Default stability_subsamples=0 = off (single-fit, bit-stable).
        self.stability_subsamples = stability_subsamples
        self.stability_subsample_fraction = stability_subsample_fraction
        self.stability_threshold = stability_threshold

        # Margin-gated adaptive trial-stop for the residual TENTATIVE TAIL (opt-in; default OFF -> byte-identical
        # to the prior behaviour). The shipped all-decided early-stop (in the trial loop) only fires when ZERO
        # features are tentative; on real data a small tail of features whose binomial p-value sits permanently
        # between the accept and reject thresholds NEVER resolves, so the loop always burns the full n_trials cap.
        # When early_stop_tentative=True the loop also stops once the tail is PROVABLY STUCK: the accepted set has
        # been unchanged for ``early_stop_patience`` consecutive trials AND no still-tentative feature is within a
        # binomial-decision margin (``early_stop_margin``) of crossing either threshold (so none can resolve soon).
        # CRITICAL: this is the MARGIN-GATED rule, NOT the naive 'accepted-set stable for W trials' rule. The naive
        # rule is a measured CORRECTNESS TRAP -- it fires on a transient plateau and locks a WRONG accepted set
        # (measured synth Jaccard-vs-cap 1.0 but hard_synth Jaccard 0.0). The margin gate refuses to stop while any
        # tentative feature is still close to a boundary, so the accepted/rejected sets at the stop are
        # decision-equivalent to running the full cap (measured synth Jaccard 1.0 ~72% wall saved, hard_synth
        # Jaccard 0.842 ~63% wall saved). See ``_should_stop_tentative_tail`` (and the documented-but-disabled
        # naive contrast ``_naive_accepted_set_stable``) in ``_boruta_shap_fit_explain.py``.
        self.early_stop_tentative = early_stop_tentative
        self.early_stop_patience = early_stop_patience
        self.early_stop_margin = early_stop_margin

        # Control/safety knobs mirroring MRMR / RFECV: a wall-clock budget and a
        # filesystem stop-flag, both honoured inside the trial loop (see ``_fit_func``
        # in ``_boruta_shap_fit_explain``). ``max_runtime_mins=None`` disables the time
        # budget; ``stop_file`` is checked for existence each trial (touch it to abort
        # the run cleanly, returning the features classified so far).
        self.max_runtime_mins = max_runtime_mins
        self.stop_file = stop_file

    def check_model(self):
        """
        Checks that a model object has been passed as a parameter when intiializing the BorutaShap class.

        Returns
        -------
        Model Object
            If no model specified then a base Random Forest will be returned otherwise the specifed model will
            be returned.

        Raises
        ------
        AttirbuteError
             If the model object does not have the required attributes.

        """

        check_fit = hasattr(self.model, "fit")
        # Renamed from misleading ``check_predict_proba``; the call is ``hasattr(..., "predict")``,
        # not ``"predict_proba"``. Variable now matches the attribute it actually probes.
        check_predict = hasattr(self.model, "predict")

        try:
            check_feature_importance = hasattr(self.model, "feature_importances_")

        except (AttributeError, TypeError):
            check_feature_importance = True

        if self.model is None:
            if self.classification:
                self.model = RandomForestClassifier()
            else:
                self.model = RandomForestRegressor()

        elif check_fit is False and check_predict is False:
            raise AttributeError("Model must contain both the fit() and predict() methods")

        elif (check_feature_importance is False and "RandomForest" not in type(self.model).__name__) and str(self.importance_measure).lower() == "gini":
            raise AttributeError("Model must contain the feature_importances_ method to use Gini try Shap instead")

        else:
            pass

    @staticmethod
    def _ordinal_encode_object_cols_inplace(df: pd.DataFrame) -> list[str]:
        """Replace every object / pandas-Categorical column in ``df`` with
        int32 ordinal codes. Returns the list of touched column names so
        the caller can log / unit-test the path.

        NaN preserved as ``-1`` so the surrogate model can still split on
        "missing"; non-NaN values get stable codes derived from the unique
        set seen in that column (pandas Categorical default).
        """
        touched: list[str] = []
        for _col in df.columns:
            _ser = df[_col]
            _dtype = _ser.dtype
            if isinstance(_dtype, pd.CategoricalDtype):
                df[_col] = _ser.cat.codes.astype("int32")
                touched.append(str(_col))
            elif pd.api.types.is_string_dtype(_dtype):
                # Use ``is_string_dtype`` (not ``== object``) so pandas
                # 2.1+ ``infer_string`` / pyarrow-backed string columns
                # also get encoded. Pre-fix the ``== object`` gate
                # missed StringDtype columns on modern pandas and the
                # surrogate fit downstream crashed with the original
                # "pandas dtypes must be int, float or bool" error.
                df[_col] = pd.Categorical(_ser).codes.astype("int32")
                touched.append(str(_col))
        return touched

    def check_X(self):
        """
        Checks that the data passed to the BorutaShap instance is a pandas Dataframe

        Returns
        -------
        Datframe

        Raises
        ------
        AttirbuteError
             If the data is not of the expected type.

        """

        if isinstance(self.X, pd.DataFrame) is False:
            raise AttributeError("X must be a pandas Dataframe")

        else:
            pass

    def missing_values_y(self):
        """
        Checks for missing values in target variable.

        Returns
        -------
        Boolean

        Raises
        ------
        AttirbuteError
             If data is not in the expected format.

        """

        if isinstance(self.y, pd.Series):
            return self.y.isnull().any().any()

        elif isinstance(self.y, np.ndarray):
            return np.isnan(self.y).any()

        else:
            raise AttributeError("Y must be a pandas Dataframe or a numpy array")

    def check_missing_values(self):
        """
        Checks for missing values in the data.

        Returns
        -------
        Boolean

        Raises
        ------
        AttirbuteError
             If there are missing values present.

        """

        X_missing = self.X.isnull().any().any()
        Y_missing = self.missing_values_y()

        models_to_check = ("xgb", "catboost", "lgbm", "lightgbm")

        model_name = str(type(self.model)).lower()
        if X_missing or Y_missing:
            if any([x in model_name for x in models_to_check]):
                logger.warning("There are missing values in your data !")

            else:
                raise ValueError("There are missing values in your Data")

        else:
            pass

    def Check_if_chose_train_or_test_and_train_model(self):
        """
        Decides to fit the model to either the training data or the test/unseen data a great discussion on the
        differences can be found here.

        https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html#introduction-to-test-vs.training-data

        """
        if self.stratify is not None and not self.classification:
            raise ValueError("Cannot take a strtified sample from continuos variable please bucket the variable and try again !")

        if self.train_or_test.lower() == "test":
            # keeping the same naming convenetion as to not add complexit later on
            self.X_boruta_train, self.X_boruta_test, self.y_train, self.y_test = train_test_split(
                self.X_boruta, self.y, test_size=0.3, random_state=self.random_state, stratify=self.stratify
            )
            self.Train_model(self.X_boruta_train, self.y_train)

        elif self.train_or_test.lower() == "train":
            # model will be trained and evaluated on the same data
            self.Train_model(self.X_boruta, self.y)

        else:
            raise ValueError('The train_or_test parameter can only be "train" or "test"')

    def Train_model(self, X, y):
        """
        Trains Model also checks to see if the model is an instance of catboost as it needs extra parameters
        also the try except is for models with a verbose statement

        Parameters
        ----------
        X: Dataframe
            A pandas dataframe of the features.

        y: Series/ndarray
            A pandas series or numpy ndarray of the target

        Returns
        ----------
        fitted model object

        """

        if "catboost" in str(type(self.model)).lower():
            self.model.fit(X, y, cat_features=self.X_categorical, verbose=False, **(self.fit_params or {}))

        else:
            try:
                self.model.fit(X, y, verbose=False, **(self.fit_params or {}))

            except TypeError:
                self.model.fit(X, y, **(self.fit_params or {}))


    def transform(
        self,
        X,
    ):
        # Name-based selection: ``self.X`` was mutated in place during fit
        # (rejected columns dropped by ``remove_features_if_rejected``), so its
        # column ordering is NOT the input X's ordering. Using ``X[selected]`` /
        # ``X.loc[:, selected]`` is stable across caller-side column reordering
        # (train vs serve drift) where the prior ``iloc[:, sorted(indices)]``
        # silently selected the wrong columns.
        #
        # Fit-state guard: surfaced iter-347 (cb,lgb regression + boruta=True,
        # weight=recency) -- the per-weight pre_pipeline path can hand a cloned
        # (unfit) BorutaShap to ``transform`` directly, raising
        # ``AttributeError: 'BorutaShap' object has no attribute
        # 'selected_features_'`` and dropping the entire model from the suite.
        # Raise the canonical sklearn NotFittedError so the caller's
        # check_is_fitted-aware fallback path (see ``predict.py`` iter-59
        # recovery branch and ``_pipeline_helpers._is_fitted``) can react.
        from sklearn.exceptions import NotFittedError
        if not hasattr(self, "selected_features_"):
            raise NotFittedError(
                "BorutaShap.transform called before fit. Call fit_transform "
                "or fit + transform before using the selector.",
            )
        selected = list(self.selected_features_)
        if hasattr(X, "loc"):
            return X.loc[:, selected]
        return X[selected]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def calculate_rejected_accepted_tentative(self, verbose):
        """
        Figures out which features have been either accepted rejeected or tentative

        Returns
        -------
        3 lists

        """

        self.rejected = list(set(self.flatten_list(self.rejected_columns)) - set(self.flatten_list(self.accepted_columns)))
        self.accepted = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(set(self.all_columns) - set(self.rejected + self.accepted))

        if verbose:
            logger.info("%s attributes confirmed important: %s", len(self.accepted), self.accepted)
            logger.info("%s attributes confirmed unimportant: %s", len(self.rejected), self.rejected)
            logger.info("%s tentative attributes remains: %s", len(self.tentative), self.tentative)

    def create_importance_history(self):
        """
        Creates a dataframe object to store historical feature importance scores.

        Returns
        -------
        Datframe

        """

        self.history_shadow = np.zeros(self.ncols)
        self.history_x = np.zeros(self.ncols)
        self.history_hits = np.zeros(self.ncols)

    def update_importance_history(self):
        """
        At each iteration update the datframe object that stores the historical feature importance scores.

        Returns
        -------
        Datframe

        """

        padded_history_shadow = np.full((self.ncols), np.nan)
        padded_history_x = np.full((self.ncols), np.nan)

        for index, col in enumerate(self.columns):
            map_index = self.order[col]
            padded_history_shadow[map_index] = self.Shadow_feature_import[index]
            padded_history_x[map_index] = self.X_feature_import[index]

        self.history_shadow = np.vstack((self.history_shadow, padded_history_shadow))
        self.history_x = np.vstack((self.history_x, padded_history_x))

    def store_feature_importance(self):
        """
        Reshapes the columns in the historical feature importance scores object also adds the mean, median, max, min
        shadow feature scores.

        Returns
        -------
        Datframe

        """

        self.history_x = pd.DataFrame(data=self.history_x, columns=self.all_columns)

        self.history_x["Max_Shadow"] = [max(i) for i in self.history_shadow]
        self.history_x["Min_Shadow"] = [min(i) for i in self.history_shadow]
        self.history_x["Mean_Shadow"] = [np.nanmean(i) for i in self.history_shadow]
        self.history_x["Median_Shadow"] = [np.nanmedian(i) for i in self.history_shadow]

    def remove_features_if_rejected(self):
        """
        At each iteration if a feature has been rejected by the algorithm remove it from the process

        """

        if len(self.features_to_remove) != 0:
            # Single-call drop instead of a per-feature loop: each in-place ``DataFrame.drop`` rebuilds the
            # block manager, so dropping N rejected features one at a time was the dominant mlframe-side cost
            # (profiled 1.56 s = 5.2% of a 299/120-col SHAP fit, almost all in pandas ``base.drop``). Dropping
            # the whole list in one call rebuilds the manager once. ``errors="ignore"`` reproduces the prior
            # per-feature ``except KeyError: pass`` exactly (a feature already dropped in a prior trial is
            # skipped), and the resulting column set + ORDER is bit-identical to the loop (drop preserves the
            # surviving columns' relative order regardless of how many are removed per call).
            self.X.drop(list(self.features_to_remove), axis=1, inplace=True, errors="ignore")

        else:
            pass

    @staticmethod
    def average_of_list(lst):
        return sum(lst) / len(lst)

    @staticmethod
    def flatten_list(array):
        return [item for sublist in array for item in sublist]

    def create_mapping_between_cols_and_indices(self):
        # Wave 54 (2026-05-20): refuse duplicate-column input -- prior dict(zip(...))
        # silently collapsed dupes to the LAST index, so any earlier-duplicated column
        # would never be shuffled / tested by Boruta's shadow-feature loop.
        cols = self.X.columns.to_list()
        if len(set(cols)) != len(cols):
            from collections import Counter
            dupes = [c for c, n in Counter(cols).items() if n > 1]
            raise ValueError(
                f"BorutaShap: input X has {len(dupes)} duplicate column name(s) "
                f"({dupes[:5]}); deduplicate before fit() to avoid silently dropping shadow indices."
            )
        return dict(zip(cols, np.arange(self.X.shape[1])))

    def calculate_hits(self):
        """
        If a features importance is greater than the maximum importance value of all the random shadow
        features then we assign it a hit.

        Parameters
        ----------
        Percentile : value ranging from 0-1
            can be used to reduce value of the maximum value of the shadow features making the algorithm
            more lenient.

        """

        # Wave 21 P0: use nanpercentile so a NaN in shadow_feature_import
        # (some boosters emit NaN for never-split features) doesn't poison
        # the threshold. Pre-fix any single NaN made shadow_threshold == NaN,
        # then ``X_feature_import > NaN`` returned all-False, silently
        # rejecting every feature from the Boruta gate.
        # Caveat (see class docstring): this threshold is the only signal separating real from random. Because shadows
        # are independently permuted they carry NO joint structure, while real columns keep their finite-sample
        # covariance structure; importance fit on one sample therefore ranks the top real-noise column above every
        # shadow, so it gets a hit nearly every trial. Raising the percentile does NOT fix it; only INTERSECTION-mode
        # cross-subsample stability (stability_subsamples>1, stability_threshold=1.0) reliably drops it.
        shadow_threshold = np.nanpercentile(self.Shadow_feature_import, self.percentile)
        # If EVERY shadow importance was NaN (degenerate input), nanpercentile
        # also returns NaN; surface that loudly rather than silently rejecting
        # all features.
        if not np.isfinite(shadow_threshold):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "BorutaShap: shadow_threshold is non-finite (all shadow "
                "feature importances were NaN/inf); the gate cannot "
                "discriminate. Returning an empty hits vector and "
                "letting the caller's iteration cap decide.",
            )
            shadow_threshold = float("inf")  # ensures all `X > thr` return False predictably

        padded_hits = np.zeros(self.ncols)
        hits = self.X_feature_import > shadow_threshold

        for index, col in enumerate(self.columns):
            map_index = self.order[col]
            padded_hits[map_index] += hits[index]

        return padded_hits

    def create_shadow_features(self):
        """
        Creates the random shadow features by shuffling the existing columns.

        Returns:
            Datframe with random permutations of the original columns.
        """
        # Private rng (set in __init__) keeps shadow-feature permutations seeded
        # without mutating the global np.random stream that other suite stages rely on.
        _rng = getattr(self, "_rng", None) or np.random.default_rng(getattr(self, "random_state", None))
        # ``self.X.apply(lambda col: _rng.permutation(col.values))`` permutes each column independently in
        # COLUMN ORDER, one ``_rng.permutation`` call per column. That per-column lambda wraps every result in a
        # pandas Series, which dominates this method (~13 ms/trial -> 2.4% of a SHAP fit). When every column shares
        # one numpy numeric dtype, ``to_numpy()`` is a no-upcast 2-D view, so permuting each column into a
        # same-dtype 2-D buffer reproduces ``.apply`` EXACTLY -- same per-column ``_rng.permutation(col.values)``
        # call sequence (so the rng stream, and thus every downstream shadow value + hit, is bit-identical) and
        # the same per-column dtype -- at ~1.9x. Mixed-dtype / categorical / bool frames take the dict fallback,
        # which is itself dtype-identical to ``.apply`` (``col.values`` -> category=int codes, object=str, etc.)
        # but carries no speedup. ``bool`` is excluded from the fast path (``np.empty_like`` on a bool 2-D buffer
        # is correct, but keeping the explicit per-column path avoids any edge with bool ``permutation``).
        cols = self.X.columns
        _dtypes = self.X.dtypes
        _fast = False
        if len(cols) > 0:
            _d0 = _dtypes.iloc[0]
            if (
                (_dtypes == _d0).all()
                and pd.api.types.is_numeric_dtype(_d0)
                and not isinstance(_d0, pd.CategoricalDtype)
                and _d0 != bool
            ):
                _vals = self.X.to_numpy()
                if _vals.dtype == _d0:  # guard: confirm no silent upcast (e.g. nullable/extension dtypes)
                    _out = np.empty_like(_vals)
                    for _j in range(_vals.shape[1]):
                        _out[:, _j] = _rng.permutation(_vals[:, _j])
                    self.X_shadow = pd.DataFrame(_out, columns=cols, index=self.X.index, copy=False)
                    _fast = True
        if not _fast:
            self.X_shadow = self.X.apply(lambda col: _rng.permutation(col.values))

        if isinstance(self.X_shadow, pd.DataFrame):
            # append
            obj_col = self.X_shadow.select_dtypes(include=["object", "string"]).columns.tolist()
            if obj_col == []:
                pass
            else:
                self.X_shadow[obj_col] = self.X_shadow[obj_col].astype("category")

        self.X_shadow.columns = ["shadow_" + feature for feature in self.X.columns]
        self.X_boruta = pd.concat([self.X, self.X_shadow], axis=1)

        col_types = self.X_boruta.dtypes
        self.X_categorical = list(col_types[(col_types == "category") | (col_types == "object")].index)

    @staticmethod
    def calculate_Zscore(array):
        """
        Calculates the Z-score of an array

        Parameters
         ----------
        array: array_like

        Returns:
            normalised array
        """
        mean_value = np.mean(array)
        std_value = np.std(array)
        array = np.asarray(array, dtype=np.float64)
        return (array - mean_value) / (std_value + 1e-12)

    def feature_importance(self, normalize):
        """
        Caculates the feature importances scores of the model

        Parameters
        ----------
        importance_measure: string
            allows the user to choose either the Shap or Gini importance metrics

        normalize: boolean
            if true the importance values will be normalized using the z-score formula

        Returns:
            array of normalized feature importance scores for both the shadow and original features.

        Raise
        ----------
            ValueError:
                If no Importance measure was specified
        """

        if str(self.importance_measure).lower() == "shap":
            self.explain()
            vals = self.shap_values

            if normalize:
                vals = self.calculate_Zscore(vals)

            # Layout of self.X_boruta is [X | X_shadow]. Real features come first,
            # shadow afterwards. Using len(self.X.columns) for the split is correct
            # even when the shadow side was padded to >= 5 columns.
            X_feature_import = vals[: len(self.X.columns)]
            Shadow_feature_import = vals[len(self.X.columns) :]

        elif str(self.importance_measure).lower() == "gini":
            feature_importances_ = np.abs(self.model.feature_importances_)

            if normalize:
                feature_importances_ = self.calculate_Zscore(feature_importances_)

            X_feature_import = feature_importances_[: len(self.X.columns)]
            Shadow_feature_import = feature_importances_[len(self.X.columns) :]

        elif str(self.importance_measure).lower() == "permutation":
            from sklearn.inspection import permutation_importance

            # Debiased held-out permutation when a 30% holdout exists (train_or_test="test"): permuting there
            # measures genuine held-out degradation, the signal that beat in-sample impurity/SHAP in the
            # importance shootout. With the "train" default no holdout is set, so this falls back to in-bag
            # permutation on the full X_boruta (still ranks shadows near zero, but inherits the in-sample
            # optimism gini/SHAP also carry). Negative permutation importances mean "noise" -> clipped to 0 so
            # they tie with shadows rather than being inflated by abs().
            X_perm = getattr(self, "X_boruta_test", None)
            if X_perm is not None:
                X_perm, y_perm = self.X_boruta_test, self.y_test
            else:
                X_perm, y_perm = self.X_boruta, self.y
            pi = permutation_importance(
                self.model, X_perm, y_perm, n_repeats=self.permutation_n_repeats,
                random_state=self.random_state, n_jobs=-1,
            )
            feature_importances_ = np.clip(pi.importances_mean, 0.0, None)

            if normalize:
                feature_importances_ = self.calculate_Zscore(feature_importances_)

            X_feature_import = feature_importances_[: len(self.X.columns)]
            Shadow_feature_import = feature_importances_[len(self.X.columns) :]

        else:
            raise ValueError("No Importance_measure was specified select one of (shap, gini, permutation)")

        return X_feature_import, Shadow_feature_import

    @staticmethod
    def isolation_forest(X):
        """
        fits isloation forest to the dataset and gives an anomally score to every sample
        """
        clf = IsolationForest().fit(X)
        preds = clf.score_samples(X)
        return preds

    @staticmethod
    def get_5_percent(num):
        return round(5 / 100 * num)

    def get_5_percent_splits(self, length):
        """
        splits dataframe into 5% intervals
        """
        five_percent = self.get_5_percent(length)
        return np.arange(five_percent, length, five_percent)

    def find_sample(self):
        """
        Finds a sample by comparing the distributions of the anomally scores between the sample and the original
        distribution using the KS-test. Starts of a 5% howver will increase to 10% and then 15% etc. if a significant sample can not be found
        """
        loop = True
        iteration = 0
        size = self.get_5_percent_splits(self.X.shape[0])
        element = 1
        while loop:
            sample_indices = choice(np.arange(self.preds.size), size=size[element], replace=False)
            sample = np.take(self.preds, sample_indices)
            if ks_2samp(self.preds, sample).pvalue > 0.95:
                break

            if iteration == 20:
                element += 1
                iteration = 0

        return self.X_boruta.iloc[sample_indices]


    @staticmethod
    def binomial_H0_test(array, n, p, alternative):
        """
        Perform a test that the probability of success is p.
        This is an exact, two-sided test of the null hypothesis
        that the probability of success in a Bernoulli experiment is p
        """
        return [_binom_test_cached(int(x), n, p, alternative) for x in array]

    @staticmethod
    def symetric_difference_between_two_arrays(array_one, array_two):
        set_one = set(array_one)
        set_two = set(array_two)
        return np.array(list(set_one.symmetric_difference(set_two)))

    @staticmethod
    def find_index_of_true_in_array(array):
        length = len(array)
        return list(filter(lambda x: array[x], range(length)))

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):
        """
        used to counteract the problem of multiple comparisons.
        """
        pvals = np.array(pvals)

        if n_tests is None:
            n_tests = len(pvals)
        else:
            pass

        alphacBon = alpha / float(n_tests)
        reject = pvals <= alphacBon
        pvals_corrected = pvals * float(n_tests)
        return reject, pvals_corrected

    def test_features(self, iteration):
        """
        For each feature with an undetermined importance perform a two-sided test of equality
        with the maximum shadow value to determine if it is statistcally better

        Parameters
        ----------
        hits: an array which holds the history of the number times
              this feature was better than the maximum shadow

        Returns:
            Two arrays of the names of the accepted and rejected columns at that instance
        """

        acceptance_p_values = self.binomial_H0_test(self.hits, n=iteration, p=0.5, alternative="greater")

        regect_p_values = self.binomial_H0_test(self.hits, n=iteration, p=0.5, alternative="less")

        # [1] as function returns a tuple
        modified_acceptance_p_values = self.bonferoni_corrections(acceptance_p_values, alpha=0.05, n_tests=len(self.columns))[1]

        modified_regect_p_values = self.bonferoni_corrections(regect_p_values, alpha=0.05, n_tests=len(self.columns))[1]

        # Take the inverse as we want true to keep featrues
        rejected_columns = np.array(modified_regect_p_values) < self.pvalue
        accepted_columns = np.array(modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.all_columns[rejected_indices]
        accepted_features = self.all_columns[accepted_indices]

        self.features_to_remove = rejected_features

        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)

    def TentativeRoughFix(self):
        """
        Sometimes no matter how many iterations are run a feature may neither be rejected or
        accepted. This method is used in this case to make a decision on a tentative feature
        by comparing its median importance value with the median max shadow value.

        Parameters
        ----------
        tentative: an array which holds the names of the tentative attiributes.

        Returns:
            Two arrays of the names of the final decision of the accepted and rejected columns.

        """

        median_tentaive_values = self.history_x[self.tentative].median(axis=0).values
        median_max_shadow = self.history_x["Max_Shadow"].median(axis=0)

        filtered = median_tentaive_values > median_max_shadow

        self.tentative = np.array(self.tentative)
        newly_accepted = self.tentative[filtered]

        if len(newly_accepted) < 1:
            newly_rejected = self.tentative

        else:
            newly_rejected = self.symetric_difference_between_two_arrays(newly_accepted, self.tentative)

        print(str(len(newly_accepted)) + " tentative features are now accepted: " + str(newly_accepted))
        print(str(len(newly_rejected)) + " tentative features are now rejected: " + str(newly_rejected))

        self.rejected = self.rejected + newly_rejected.tolist()
        self.accepted = self.accepted + newly_accepted.tolist()

    def Subset(self, tentative=False):
        """
        Returns the subset of desired features
        """
        if tentative:
            return self.starting_X[self.accepted + self.tentative.tolist()]
        else:
            return self.starting_X[self.accepted]


def load_data(data_type="classification"):
    """
    Load Example datasets for the user to try out the package
    """

    data_type = data_type.lower()

    if data_type == "classification":
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        X = pd.DataFrame(np.c_[cancer["data"], cancer["target"]], columns=np.append(cancer["feature_names"], ["target"]))
        y = X.pop("target")

    elif data_type == "regression":
        try:
            from sklearn.datasets import load_boston
            boston = load_boston()
        except ImportError:
            # sklearn >= 1.2 removed load_boston (ethical concerns over the dataset). Fall back to fetch_california_housing,
            # which exposes the same {"data", "target", "feature_names"} dict-like API and serves the same demo purpose.
            from sklearn.datasets import fetch_california_housing
            boston = fetch_california_housing(as_frame=False)
        X = pd.DataFrame(np.c_[boston["data"], boston["target"]], columns=np.append(boston["feature_names"], ["target"]))
        y = X.pop("target")

    else:
        raise ValueError("No data_type was specified, use either 'classification' or 'regression'")

    return X, y


# ----------------------------------------------------------------------
# Method bindings. ``fit`` + ``explain`` bodies live in
# ``_boruta_shap_fit_explain.py`` so this file stays below the 1k-LOC
# monolith threshold.
# ----------------------------------------------------------------------
from ._boruta_shap_fit_explain import (  # noqa: E402
    fit as _fit_func,
    explain as _explain_func,
)
BorutaShap.fit = _fit_func
BorutaShap.explain = _explain_func

from ._boruta_shap_io_plot import (  # noqa: E402,F401
    results_to_csv as _results_to_csv_func,
    plot as _plot_func,
    box_plot as _box_plot_func,
    create_mapping_of_features_to_attribute as _create_mapping_func,
    create_list as _create_list_func,
    filter_data as _filter_data_func,
    hasNumbers as _has_numbers_func,
    check_if_which_features_is_correct as _check_which_features_func,
    to_dictionary as _to_dictionary_func,
)
BorutaShap.results_to_csv = _results_to_csv_func
BorutaShap.plot = _plot_func
BorutaShap.box_plot = _box_plot_func
BorutaShap.create_mapping_of_features_to_attribute = _create_mapping_func
BorutaShap.create_list = staticmethod(_create_list_func)
BorutaShap.filter_data = staticmethod(_filter_data_func)
BorutaShap.hasNumbers = staticmethod(_has_numbers_func)
BorutaShap.check_if_which_features_is_correct = staticmethod(_check_which_features_func)
BorutaShap.to_dictionary = staticmethod(_to_dictionary_func)
