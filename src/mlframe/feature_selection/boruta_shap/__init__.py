from __future__ import annotations

from typing import Optional

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


@_functools.lru_cache(maxsize=100_000)
def _binom_test_cached(x_int: int, n: int, p: float, alternative: str = "two-sided"):
    """Memoized ``binom_test``: BorutaShap runs it per FEATURE per iteration (tens of thousands of
    calls on a wide frame), but ``(n, p, alternative)`` are fixed within a step and the hit count
    ``x`` is a small integer, so the distinct ``(x, n, p, alternative)`` set is tiny. Caching
    collapses ~36k per-call scipy ``binomtest`` constructions (profiled ~7s = 21% of a 299-feature
    fit) to a handful -- bit-identical p-values. ``maxsize`` is bounded (not ``None``) so a
    long-lived process that runs BorutaShap across many differently-sized frames cannot pin the
    cache unbounded; 100k dwarfs the distinct-key count of any single fit, so hit rate is unaffected."""
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


# Pure selector (shadow features are internal, transform emits a column subset), but it intentionally hand-rolls the
# SelectorMixin surface (get_support / get_feature_names_out) rather than inheriting SelectorMixin, to keep full control
# over the fitted-mask semantics; SelectorMixin is deliberately NOT added to avoid clashing with those hand-rolled methods.
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

    AUTO DISPATCH (importance_measure='auto', opt-in): one cheap RandomForest probe on (X, y) at fit
    start picks the driver per-fit -- permutation on noisy / overfit-prone / small-n-relative-to-p beds
    (where gini over-credits noise via split-frequency bias), gini on clean / large-n beds (where the
    ~11x permutation cost buys nothing). Signals: n/p ratio, train-vs-OOB gap, real-vs-shadow impurity
    fraction. The resolution + signals are exposed on ``auto_dispatch_diagnostics_`` after fit. The
    default stays 'gini': on the fs_hybrid bed auto matched gini's holdout on clean beds without paying
    permutation there and tied/won permutation on noisy beds, but did NOT beat the static gini default
    on a REPLICATED majority of scenarios+seeds, so auto remains opt-in (see _benchmarks bench). See
    ``_auto_dispatch.py``.
    """

    # history_x accumulates as an ndarray across trials (run()), then is promoted to a DataFrame
    # in store_feature_importance() once accumulation is done and column-labeled stats are added.
    history_x: "np.ndarray | pd.DataFrame"
    # tentative starts as a plain list (set difference at fit entry), then TentativeRoughFix() rebinds it to an ndarray for boolean-mask filtering.
    tentative: "list | np.ndarray"

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
        fit_params: Optional[dict] = None,
        stability_subsamples: int = 0,
        stability_subsample_fraction: float = 0.75,
        stability_threshold: float = 1.0,
        early_stop_tentative: bool = False,
        early_stop_patience: int = 20,
        early_stop_margin: float = 0.15,
        max_runtime_mins: Optional[float] = None,
        stop_file: str = "stop",
        shadow_min_pad: int = 5,
    ):
        """
        Parameters
        ----------
        model: Model Object
            If no model specified then a base Random Forest will be returned otherwise the specified model will
            be returned.

        importance_measure: String
            Which importance measure to use: "Shap", "gini"/"gain", "permutation", or "auto". "auto" runs a cheap
            noise/overfit probe on (X, y) at fit start and routes to permutation on noisy/small-n-per-feature beds and
            gini on clean/large-n beds (resolution stored on auto_dispatch_diagnostics_; see _auto_dispatch.py).
            Permutation uses sklearn's
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
            would make it more strict also by making the model more strict could impact runtime making it slower. As it will be less likely
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

        shadow_min_pad: int (default 5)
            Minimum number of shadow attributes the information system is extended by (the canonical Boruta >=5 pad).
            On frames with fewer real columns the shadow side is padded with recycled real columns (each independently
            re-permuted, so still uncorrelated with y) up to this count, so the per-trial shadow-importance MAX (the
            gate threshold) is estimated from a well-populated null rather than 1-2 noisy draws. Wide frames
            (n_features >= shadow_min_pad) are unaffected. Set 0 for the legacy exactly-one-shadow-per-column null.

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

        # sklearn contract: __init__ stores params verbatim and derives nothing that mutates them. The
        # resolved-model (``model_``) is built in ``check_model`` so ``get_params``/``clone`` round-trip the
        # constructor args unchanged (``model=None`` stays None on a fitted instance). The private RNG
        # (``_rng``, A-P0-004) is a fresh ``default_rng(random_state)`` -- an INDEPENDENT Generator that does
        # NOT touch the process-global ``np.random`` stream, so building it here keeps both contracts: clone
        # re-runs __init__ and re-derives the same RNG from the unchanged seed.
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.n_trials = n_trials
        # Canonical Boruta extends the system by AT LEAST ``shadow_min_pad`` shadow attributes even when the real
        # frame is narrower, so the per-trial shadow-importance MAX (the gate threshold) is not estimated from 1-2
        # draws. On narrow frames the extra shadows are recycled real columns (re-permuted), each still independent
        # of y, only widening the null pool. Default 5 (the canonical value, bench-validated neutral-or-better on
        # narrow frames). Set 0 to opt out (legacy exactly-one-shadow-per-column behaviour); wide frames (>= pad) are
        # unaffected. See ``create_shadow_features``.
        self.shadow_min_pad = shadow_min_pad
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
        # naive contrast ``_naive_accepted_set_stable``) in ``_fit_explain.py``.
        self.early_stop_tentative = early_stop_tentative
        self.early_stop_patience = early_stop_patience
        self.early_stop_margin = early_stop_margin

        # Control/safety knobs mirroring MRMR / RFECV: a wall-clock budget and a
        # filesystem stop-flag, both honoured inside the trial loop (see ``_fit_func``
        # in ``_fit_explain``). ``max_runtime_mins=None`` disables the time
        # budget; ``stop_file`` is checked for existence each trial (touch it to abort
        # the run cleanly, returning the features classified so far).
        self.max_runtime_mins = max_runtime_mins
        self.stop_file = stop_file

    def _active_importance_measure(self) -> str:
        """The concrete importance measure in effect for this fit. Equals ``importance_measure`` for
        the explicit choices; for ``"auto"`` it is the per-fit resolution (gini/permutation) computed
        once in ``fit`` via the cheap noise/overfit probe (``_resolved_importance_measure_``). Falls
        back to the raw value if accessed before resolution (e.g. a direct helper call in a test)."""
        return str(getattr(self, "_resolved_importance_measure_", None) or self.importance_measure).lower()

    def _resolve_auto_importance_measure(self, X, y) -> None:
        """When ``importance_measure='auto'``, probe (X, y) ONCE and pin the concrete driver for this
        fit into ``_resolved_importance_measure_`` (+ diagnostics in ``auto_dispatch_diagnostics_``).
        permutation needs a holdout to be the honest noise-leader, so auto also pins the resolved
        ``_train_or_test_`` to 'test' for that branch (only when the user left the default 'train'); the
        constructor param ``train_or_test`` itself is never mutated (sklearn clone contract). Explicit measures untouched."""
        # Resolved working copy of the train/test choice. Never mutate the verbatim ``self.train_or_test`` param.
        self._train_or_test_ = str(self.train_or_test)
        if str(self.importance_measure).lower() != "auto":
            self._resolved_importance_measure_ = str(self.importance_measure).lower()
            self.auto_dispatch_diagnostics_ = None
            return
        from ._auto_dispatch import resolve_auto_importance_measure

        measure, diag = resolve_auto_importance_measure(
            X, y, classification=bool(self.classification), random_state=int(self.random_state),
        )
        self._resolved_importance_measure_ = measure
        self.auto_dispatch_diagnostics_ = diag
        # Honest held-out permutation requires the 30% holdout; only override the *default* 'train'.
        if measure == "permutation" and str(self.train_or_test).lower() == "train":
            self._auto_forced_test_split_ = True
            self._train_or_test_ = "test"
        if self.verbose:
            logger.info(
                "BorutaShap importance_measure='auto' -> '%s' (n/p=%.1f, oob_gap=%.3f, shadow_gap=%.3f; %s)",
                measure, diag.get("np_ratio", float("nan")), diag.get("oob_gap", float("nan")),
                diag.get("shadow_gap", float("nan")), ", ".join(diag.get("reasons", [])) or "clean",
            )

    def check_model(self):
        """
        Resolve the working surrogate model into the learned attribute ``self.model_`` (sklearn contract:
        the constructor param ``self.model`` is stored verbatim and NEVER mutated, so ``get_params`` on a
        fitted instance still returns the value the caller passed -- ``None`` stays ``None`` -- and ``clone``
        reconstructs from the original args). When ``self.model is None`` the default RandomForest surrogate
        is built into ``model_``; otherwise the caller-supplied estimator is used as-is.

        Raises
        ------
        AttributeError
             If the supplied model object does not have the required attributes.
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
            # The default surrogate is refit every trial and the tree fit is ~82% of wall (profiled); sklearn's
            # default n_jobs=None runs it single-threaded. n_jobs=-1 parallelizes the independent trees across
            # cores -- selection is bit-identical (fixed random_state makes the forest deterministic regardless
            # of worker count). Resolved into ``model_`` so the verbatim ``self.model`` param stays None.
            if self.classification:
                self.model_ = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
            else:
                self.model_ = RandomForestRegressor(n_jobs=-1, random_state=self.random_state)

        elif check_fit is False and check_predict is False:
            raise AttributeError("Model must contain both the fit() and predict() methods")

        elif (check_feature_importance is False and "RandomForest" not in type(self.model).__name__) and str(self.importance_measure).lower() == "gini":
            raise AttributeError("Model must contain the feature_importances_ method to use Gini try Shap instead")

        else:
            self.model_ = self.model

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

        # Inspect the surrogate's type to decide warn-vs-raise on NaN. This method is callable standalone
        # (before ``check_model`` resolves ``model_``), so fall back to the verbatim constructor param when
        # the learned attribute is not yet built.
        resolved_model = getattr(self, "model_", None)
        if resolved_model is None:
            resolved_model = self.model
        model_name = str(type(resolved_model)).lower()
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
            raise ValueError("Cannot take a strtified sample from continuous variable please bucket the variable and try again !")

        # Read the RESOLVED working copy (``_train_or_test_``), not the verbatim param: 'auto' importance may
        # pin it to 'test' for held-out permutation without mutating the clone-able constructor arg. Falls back
        # to the param for direct helper calls that bypass ``_resolve_auto_importance_measure``.
        train_or_test = getattr(self, "_train_or_test_", None) or self.train_or_test
        if train_or_test.lower() == "test":
            # keeping the same naming convenetion as to not add complexit later on
            self.X_boruta_train, self.X_boruta_test, self.y_train, self.y_test = train_test_split(
                self.X_boruta, self.y, test_size=0.3, random_state=self.random_state, stratify=self.stratify
            )
            self.Train_model(self.X_boruta_train, self.y_train)

        elif train_or_test.lower() == "train":
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

        if "catboost" in str(type(self.model_)).lower():
            self.model_.fit(X, y, cat_features=self.X_categorical, verbose=False, **(self.fit_params or {}))

        else:
            try:
                self.model_.fit(X, y, verbose=False, **(self.fit_params or {}))

            except TypeError:
                self.model_.fit(X, y, **(self.fit_params or {}))

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
                "BorutaShap.transform called before fit. Call fit_transform " "or fit + transform before using the selector.",
            )
        # sklearn contract: validate the transform-time feature space against what fit saw. A width mismatch
        # (n_features_in_) or, for named frames, a column-name mismatch means the caller is transforming a frame
        # the selector was not fitted on -- name-based selection would then silently pull the wrong / missing columns.
        n_in = getattr(self, "n_features_in_", None)
        _width = X.shape[1] if getattr(X, "ndim", 1) >= 2 else None
        if n_in is not None and _width is not None and _width != n_in:
            raise ValueError(f"BorutaShap.transform: X has {_width} features, but the selector was fitted on {n_in}.")
        names_in = getattr(self, "feature_names_in_", None)
        if names_in is not None and hasattr(X, "columns"):
            missing = [c for c in self.selected_features_ if c not in set(X.columns)]
            if missing:
                raise ValueError(f"BorutaShap.transform: X is missing {len(missing)} selected feature(s) " f"present at fit time, e.g. {missing[:5]}.")
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

        # Empty acceptance is a defensible-but-silent outcome (single-class target -> RF fits, importances ~0, the
        # shadow gate accepts nothing; or every importance collapsed to ~0). Emit a logger warning so the caller gets
        # a signal instead of an unexplained empty support_. Always on (corrective), independent of ``verbose``: a
        # downstream selector silently keeping zero features is a failure the user must see. The two probed causes are
        # the ones a caller can act on (degenerate target / no discriminative signal); other empty-accept cases still
        # warn generically. ``getattr`` keeps the check robust for partially-built / pickled instances.
        if not self.accepted:
            _y = getattr(self, "y", None)
            _single_class = False
            if _y is not None:
                try:
                    _single_class = bool(getattr(self, "classification", False)) and len(np.unique(np.asarray(_y))) < 2
                except (TypeError, ValueError):
                    _single_class = False
            _imp = getattr(self, "X_feature_import", None)
            _all_zero_imp = False
            if _imp is not None and len(_imp):
                _imp_arr = np.asarray(_imp, dtype=float)
                _all_zero_imp = bool(np.all(np.abs(_imp_arr[np.isfinite(_imp_arr)]) < 1e-12))
            if _single_class:
                logger.warning(
                    "BorutaShap accepted 0 features: the target has a single class (n_unique<2), so the surrogate "
                    "cannot rank any feature above the shadow null. Check the target / split before trusting an empty selection."
                )
            elif _all_zero_imp:
                logger.warning(
                    "BorutaShap accepted 0 features: every real-feature importance collapsed to ~0, so none cleared "
                    "the shadow gate. The features may carry no signal for this target, or the surrogate failed to fit."
                )
            else:
                logger.warning("BorutaShap accepted 0 features (all features rejected or tentative); the selection is empty.")

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

        self.history_x["Max_Shadow"] = [max(i) for i in self.history_shadow]  # type: ignore[call-overload]  # numpy stubs narrow the per-row iteration element to a scalar float64; each row is actually a 1D ndarray
        self.history_x["Min_Shadow"] = [min(i) for i in self.history_shadow]  # type: ignore[call-overload]
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

        # history_x is promoted from ndarray to DataFrame by run() before TentativeRoughFix is ever called.
        hx: pd.DataFrame = self.history_x  # type: ignore[assignment]
        median_tentaive_values = hx[self.tentative].median(axis=0).values
        median_max_shadow = hx["Max_Shadow"].median(axis=0)

        filtered = median_tentaive_values > median_max_shadow

        self.tentative = np.array(self.tentative)
        newly_accepted = self.tentative[filtered]

        if len(newly_accepted) < 1:
            newly_rejected = np.asarray(self.tentative)

        else:
            newly_rejected = self.symetric_difference_between_two_arrays(newly_accepted, self.tentative)

        # logger (not print): a non-ASCII feature name in the array repr would crash cp1251 stdout on Windows, and
        # this should honour the verbose channel like the rest of the class rather than writing to stdout directly.
        logger.info("%s tentative features are now accepted: %s", len(newly_accepted), list(newly_accepted))
        logger.info("%s tentative features are now rejected: %s", len(newly_rejected), list(newly_rejected))

        self.rejected = self.rejected + newly_rejected.tolist()
        self.accepted = self.accepted + newly_accepted.tolist()

    def Subset(self, tentative=False):
        """
        Returns the subset of desired features
        """
        if tentative:
            return self.starting_X[self.accepted + list(self.tentative)]
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
# Method bindings. ``fit`` + ``explain`` bodies live in ``_fit_explain.py`` so the package facade stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._fit_explain import (  # noqa: E402
    fit as _fit_func,
    explain as _explain_func,
)
from mlframe.utils.misc import rng_hygienic_fit  # noqa: E402
BorutaShap.fit = rng_hygienic_fit(_fit_func)
BorutaShap.explain = _explain_func

from ._io_plot import (  # noqa: E402,F401
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

# Shadow-feature construction + statistical / hit-test helpers live in ``_shadow_stats.py`` (same LOC-budget method-binding split). Instance methods take ``self`` and bind directly; the previously-``@staticmethod`` helpers are re-wrapped with ``staticmethod`` here, exactly as the IO/plot helpers above.
from ._shadow_stats import (  # noqa: E402,F401
    calculate_hits as _calculate_hits_func,
    create_shadow_features as _create_shadow_features_func,
    calculate_Zscore as _calculate_zscore_func,
    feature_importance as _feature_importance_func,
    isolation_forest as _isolation_forest_func,
    get_5_percent as _get_5_percent_func,
    get_5_percent_splits as _get_5_percent_splits_func,
    find_sample as _find_sample_func,
    binomial_H0_test as _binomial_h0_test_func,
    symetric_difference_between_two_arrays as _symetric_difference_func,
    find_index_of_true_in_array as _find_index_of_true_func,
    bonferoni_corrections as _bonferoni_corrections_func,
    test_features as _test_features_func,
)
BorutaShap.calculate_hits = _calculate_hits_func
BorutaShap.create_shadow_features = _create_shadow_features_func
BorutaShap.calculate_Zscore = staticmethod(_calculate_zscore_func)
BorutaShap.feature_importance = _feature_importance_func
BorutaShap.isolation_forest = staticmethod(_isolation_forest_func)
BorutaShap.get_5_percent = staticmethod(_get_5_percent_func)
BorutaShap.get_5_percent_splits = _get_5_percent_splits_func
BorutaShap.find_sample = _find_sample_func
BorutaShap.binomial_H0_test = staticmethod(_binomial_h0_test_func)
BorutaShap.symetric_difference_between_two_arrays = staticmethod(_symetric_difference_func)
BorutaShap.find_index_of_true_in_array = staticmethod(_find_index_of_true_func)
BorutaShap.bonferoni_corrections = staticmethod(_bonferoni_corrections_func)
BorutaShap.test_features = _test_features_func
