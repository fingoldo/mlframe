"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Callable, Sequence, Union

import copy
import random
import textwrap
import warnings
from contextlib import nullcontext
from enum import Enum
from os.path import exists
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import clean_ram, tqdmu
from pyutilz.numbalib import set_numba_random_seed  # noqa: F401  (kept for downstream callers)
from pyutilz.pythonlib import (
    get_parent_func_args,
    store_params_in_object,
    suppress_stdout_stderr,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: F401  (kept for downstream callers)
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,  # noqa: F401  (re-exported for callers)
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,  # noqa: F401  (re-exported for callers)
)
from sklearn.pipeline import Pipeline

# Project imports - was 4 star imports, now explicit. Star imports forced
# every reader to grep for symbol origin and broke linters; replace with the
# concrete set of names actually consumed below.
from mlframe.config import CATBOOST_MODEL_TYPES
from mlframe.optimization import (
    CandidateSamplingMethod,
    MBHOptimizer,
    OptimizationDirection,
    OptimizationProgressPlotting,
)
from mlframe.votenrank import Leaderboard
from mlframe.utils import set_random_seed
from mlframe.baselines import get_best_dummy_score
from mlframe.helpers import has_early_stopping_support
from mlframe.training.helpers import compute_cb_text_processing
from mlframe.preprocessing import pack_val_set_into_fit_params
from mlframe.metrics import compute_probabilistic_multiclass_error

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------


class OptimumSearch(str, Enum):
    ScipyLocal = "ScipyLocal"  # Brent
    ScipyGlobal = "ScipyGlobal"  # direct, diff evol, shgo
    ModelBasedHeuristic = "ModelBasedHeuristic"  # GaussianProcess or Catboost with uncertainty, or quantile regression
    ExhaustiveRandom = "ExhaustiveRandom"
    ExhaustiveDichotomic = "ExhaustiveDichotomic"


class VotesAggregation(str, Enum):
    Minimax = "Minimax"
    OG = "OG"
    Borda = "Borda"
    Plurality = "Plurality"
    Dowdall = "Dowdall"
    Copeland = "Copeland"
    AM = "AM"
    GM = "GM"


# ----------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------


def suppress_irritating_3rdparty_warnings() -> None:

    for message in [r"Can't optimze method \"evaluate\" because self argument is used"]:
        # Filter out the specific warning message using a substring or regex pattern.
        warnings.filterwarnings("ignore", category=UserWarning, message=message)


# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


class RFECV(BaseEstimator, TransformerMixin):
    """Finds subset of features having best CV score, by iterative narrowing down set of top_n candidates having highest importance, as per estimator's FI scores.

    Optimizes mean CV scores (possibly accounting for variation, possibly translated into ranks) divided by the features number.

    Uses several optimization methods:
        exhaustive search
        random search
        model-based heuristic search.

    Problems:
        Impactful, but correlated factors all get low importance and will be thrown away (probably only for forests, not boostings?).
        confirmed for boostings also! adding more predictors to original features worsens scores, whereas in theory it at least should not be worse!

        Due to noise some random features can become "important".

    Solution:
        use CV to calculate fold FI, then combine across folds (by voting).
        When estimating featureset quality at another TopN, use different splits & combine new FIs with all known before, to mitigate noise even more.

    Optionally plots (and saves) the optimization path - checked nfeatures and corresponding scores.
    If surrogate models are used, also shows predicted scores along with confidence bounds.

    Challenges:
        CV performance itself can be a multi-component value! Say, both ROC AUC and CALIB metrics can be considered. Voting can be a solution.
        Estimator might itself be a HPT search instance. Or a pipeline.
        It could be good to have several estimators. Their importance evaluations must be accounted for simultaneously (voting).
        Estimator might need eval_set or similar (eval_frac).
        Different folds invocations could benefit from generating all possible hyper parameters. Even if FS does not care, collected info could be used further at the HPT step.

    Parameters
    ----------
        cv : int, cross-validation generator or an iterable, default=None

    Attributes
    ----------

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    cv_results_ : dict of ndarrays
        A dict with keys:

        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.

        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.


    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    ranking_ ?: narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    """

    def __init__(
        self,
        estimator: BaseEstimator,
        fit_params: dict = None,
        max_nfeatures: int = None,
        mean_perf_weight: float = 1.0,
        std_perf_weight: float = 0.1,
        feature_cost: float = 0.0,
        smooth_perf: int = 0,
        # stopping conditions
        max_runtime_mins: float = None,
        max_refits: int = None,
        best_desired_score: float = None,
        max_noimproving_iters: int = 30,
        # CV
        cv: Union[object, int, None] = 3,
        cv_shuffle: bool = False,
        min_train_size: int = None,
        # Other
        early_stopping_val_nsplits: Union[int, None] = 10,
        early_stopping_rounds: Union[int, None] = None,
        scoring: Union[object, None] = None,
        nofeatures_dummy_scoring: bool = True,
        top_predictors_search_method: OptimumSearch = OptimumSearch.ModelBasedHeuristic,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        use_all_fi_runs: bool = True,
        use_last_fi_run_only: bool = False,
        use_one_freshest_fi_run: bool = False,
        use_fi_ranking: bool = False,
        importance_getter: Union[str, Callable, None] = None,
        random_state: int = None,
        leave_progressbars: bool = True,
        verbose: Union[bool, int] = 1,
        show_plot: bool = False,
        optimizer_plotting: Union[str, None] = None,  # Controls Optimizer plotting: 'No', 'Final', 'OnScoreImprovement', 'Regular'
        cat_features: Union[Sequence, None] = None,
        keep_estimators: bool = False,
        estimators_save_path: str = None,  # fitted estimators get saved into join(estimators_save_path,estimator_type_name,nestimator_nfeatures_nfold.dump)
        # Required features and achieved ml metrics get saved in a dict join(estimators_save_path,required_features.dump).
        frac: float = None,
        skip_retraining_on_same_shape: bool = True,
        stop_file: str = "stop",
        report_ndigits: int = 4,
        #
        special_feature_indices: list = None,
        conduct_final_voting: bool = False,
        # Phase 4 N5: must_include hybrid. List of feature names (or
        # integer indices for ndarray X) that MUST end up in support_.
        # The optimiser only searches over the remaining features; the
        # final support_ is the union of must_include and the optimiser's
        # pick. Differs from special_feature_indices which forces a fixed
        # subset and short-circuits the search after one iteration.
        must_include: Union[Sequence, None] = None,
    ):

        # checks
        if frac is not None:
            if not (frac > 0.0 and frac < 1.0):
                raise ValueError(f"frac must be between 0 and 1, got {frac}")
            if verbose:
                logger.info("Using %s fraction of the training dataset.", frac)

        # assert isinstance(estimator, (BaseEstimator,))

        # save params

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)
        self.signature = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, **fit_params):

        # ----------------------------------------------------------------------------------------------------------------------------
        # Polars -> pandas at entry. The whole RFECV path uses pandas /
        # numpy idioms (KFold.split, current_features.index(...), pd-style
        # passthrough_cols handling). Inner estimators run via .fit(X)
        # also expect pandas; CatBoost on a polars Enum raises
        # ``Unsupported data type Enum(categories=['', 'B', 'C',
        # '__MISSING__']) for a numerical feature column`` (fuzz c0103 /
        # c0102 / c0114 / c0147). Convert once here so every downstream
        # caller sees pandas.
        # ----------------------------------------------------------------------------------------------------------------------------
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        if isinstance(y, pl.Series):
            y = y.to_pandas()

        # ----------------------------------------------------------------------------------------------------------------------------
        # Zero-variance / all-null column filter (2026-04-28 batch 4 followup)
        # ----------------------------------------------------------------------------------------------------------------------------
        # Drop columns RFECV cannot meaningfully evaluate BEFORE we record
        # ``feature_names_in_``. Without this, an all-NaN or single-value
        # column entered ``feature_names_in_`` and could end up in
        # ``support_`` (the inner CB ranker doesn't reject zero-info
        # columns deterministically). Then a downstream pipeline step
        # whose ``transform`` silently dropped the column (e.g.
        # ``SimpleImputer(strategy='mean')`` with ``keep_empty_features=False``,
        # the sklearn 1.x default) left ``RFECV.transform`` looking for a
        # column the caller's frame no longer has - which surfaced as
        # ``RFECV.transform: 1/N selected columns missing from input X``
        # on fuzz seed=42 c0093 / c0095 (polars_nullable + cb_rfecv +
        # inject_all_nan_col=True). The fix at the wrapper level is
        # immune to whether upstream ``remove_constant_columns`` ran;
        # zero-info columns simply never enter the selector's universe.
        if isinstance(X, pd.DataFrame) and X.shape[1] > 0:
            # F14 fix + Perf #5 fix: vectorised across ALL dtypes (was only
            # ``select_dtypes(include="number")`` so a constant categorical /
            # string / bool column slipped through and could leak into
            # ``support_``). ``DataFrame.nunique(dropna=True)`` handles every
            # dtype pandas exposes and is one C-level pass instead of the
            # per-column Python loop. Replaces the prior loop's ~30-300 s
            # overhead on 10k-col x 1M-row inputs with a single nunique scan.
            try:
                nunique = X.nunique(dropna=True)
                degenerate = nunique[nunique <= 1].index.tolist()
            except TypeError:
                # Fallback for exotic dtypes that nunique() can't hash; loop
                # column-by-column and skip non-hashable ones quietly.
                degenerate = []
                for col in X.columns:
                    series = X[col]
                    if series.isna().all():
                        degenerate.append(col)
                    else:
                        try:
                            if series.nunique(dropna=True) <= 1:
                                degenerate.append(col)
                        except TypeError:
                            continue
            if degenerate:
                if getattr(self, "verbose", 0):
                    logger.info(
                        "RFECV: dropping %d zero-variance / all-null column(s) "
                        "before fit so they cannot leak into ``support_`` or "
                        "trip a transform-time column-set drift later: %s",
                        len(degenerate), degenerate,
                    )
                X = X.drop(columns=degenerate)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Compute inputs/outputs signature
        # ----------------------------------------------------------------------------------------------------------------------------

        # Shape alone is not sufficient -- two datasets with identical (n, p) but
        # different column identities must trigger a retrain; otherwise
        # `self.support_` silently applies stale column selections.
        if isinstance(X, pd.DataFrame):
            columns_key = tuple(map(str, X.columns.tolist()))
        else:
            columns_key = ("__ndarray__", int(X.shape[1]))
        signature = (X.shape, y.shape, columns_key)
        # F6 fix: invalidate stale support_/cache at fit entry so a partial-
        # fit failure cannot leave a previous-fit's selection silently in
        # place. The cache is rebuilt below only on a successful path.
        self._selected_cols_cache = None
        if self.skip_retraining_on_same_shape:
            if signature == self.signature:
                if self.verbose:
                    logger.info("Skipping retraining on the same inputs signature %s", signature)
                return self

        # ---------------------------------------------------------------------------------------------------------------
        # Inits
        # ---------------------------------------------------------------------------------------------------------------

        estimator = self.estimator
        fit_params = copy.copy(self.fit_params) if self.fit_params else {}
        max_runtime_mins = self.max_runtime_mins
        max_refits = self.max_refits
        cv = self.cv
        cv_shuffle = self.cv_shuffle
        early_stopping_val_nsplits = self.early_stopping_val_nsplits
        early_stopping_rounds = self.early_stopping_rounds
        scoring = self.scoring
        top_predictors_search_method = self.top_predictors_search_method
        votes_aggregation_method = self.votes_aggregation_method
        use_all_fi_runs = self.use_all_fi_runs
        use_last_fi_run_only = self.use_last_fi_run_only
        use_one_freshest_fi_run = self.use_one_freshest_fi_run
        use_fi_ranking = self.use_fi_ranking
        importance_getter = self.importance_getter
        random_state = self.random_state
        self._rng = np.random.default_rng(random_state)
        leave_progressbars = self.leave_progressbars
        verbose = self.verbose
        show_plot = self.show_plot
        cat_features = self.cat_features
        # Strip cat_features whose columns have already been numerically
        # encoded by an upstream pipeline step (e.g. CatBoostEncoder in
        # init_common_params turns cat_0 -> float). With cat_0 still in
        # cat_features, the inner CatBoost.fit hits the encoded float
        # column with cat_features=['cat_0'] and raises ``Invalid type
        # for cat_feature ... =0.49...`` (fuzz c0102 / c0114 / c0147 /
        # c0056 / c0070 / c0151). Restrict to columns whose dtype is
        # still categorical/object -- those are the ones CB/XGB can
        # actually consume as cat_features. LOCAL only -- never mutate
        # self.cat_features (back-to-back fits across encoded/un-encoded
        # frames must each pick the right subset for their X).
        if cat_features and isinstance(X, pd.DataFrame):
            try:
                _consumable_kinds = {"O", "U", "S"}
                _consumable = []
                for _c in cat_features:
                    if _c not in X.columns:
                        continue
                    _dt = X[_c].dtype
                    if str(_dt).startswith("category") or getattr(_dt, "kind", "") in _consumable_kinds:
                        _consumable.append(_c)
                if len(_consumable) != len(cat_features):
                    if verbose:
                        logger.info(
                            "wrappers.fit: %d/%d cat_features kept after dtype check; "
                            "the rest (%s) appear numerically encoded upstream and "
                            "are skipped for the inner estimator.",
                            len(_consumable), len(cat_features),
                            [c for c in cat_features if c not in _consumable],
                        )
                    cat_features = _consumable
            except Exception:
                pass
        keep_estimators = self.keep_estimators
        feature_cost = self.feature_cost
        smooth_perf = self.smooth_perf
        frac = self.frac
        best_desired_score = self.best_desired_score
        max_noimproving_iters = self.max_noimproving_iters
        ndigits = self.report_ndigits

        start_time = timer()
        ran_out_of_time = False
        if max_runtime_mins:
            if verbose:
                logger.info("max_runtime_mins=%.2f", max_runtime_mins)

        if random_state is not None:
            set_random_seed(random_state)

        feature_importances = {}
        evaluated_scores_std = {}
        evaluated_scores_mean = {}

        if isinstance(X, pd.DataFrame):
            original_features = X.columns.tolist()
        else:
            original_features = np.arange(X.shape[1])

        # Phase 4 N5: must_include partition. The optimiser only sees the
        # complement; pinned features are glued back into support_ at the
        # end (see select_optimal_nfeatures_ ``must_include_resolved``).
        # Validate that every requested name is present.
        must_include_resolved: list = []
        if self.must_include:
            if isinstance(X, pd.DataFrame):
                missing = [m for m in self.must_include if m not in original_features]
            else:
                # ndarray: must_include must be integer indices in [0, n_cols)
                p = X.shape[1]
                missing = [m for m in self.must_include if not (isinstance(m, (int, np.integer)) and 0 <= int(m) < p)]
            if missing:
                raise ValueError(
                    f"must_include contains entries not in X: {missing}. "
                    f"Available: {list(original_features)[:20]}..."
                )
            must_include_resolved = list(self.must_include)
            # Remove from search universe; the optimiser explores only the
            # COMPLEMENT. Final support_ = must_include + optimiser's pick.
            original_features = [c for c in original_features if c not in must_include_resolved]
            if verbose:
                logger.info(
                    "must_include: %d feature(s) pinned, %d searched.",
                    len(must_include_resolved), len(original_features),
                )
            if len(original_features) == 0:
                logger.warning(
                    "must_include exhausts every feature in X; nothing for "
                    "the optimiser to pick. Fitting on the pinned set only."
                )
        self._must_include_resolved = must_include_resolved

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init cv
        # ----------------------------------------------------------------------------------------------------------------------------

        if cv is None or str(cv).isnumeric():
            if cv is None:
                cv = 3
            if is_classifier(estimator):
                if groups is not None:
                    cv = StratifiedGroupKFold(n_splits=cv, shuffle=cv_shuffle, random_state=random_state if cv_shuffle else None)
                else:
                    if cv_shuffle:
                        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                    else:
                        cv = StratifiedKFold(n_splits=cv, shuffle=False)
            else:
                if groups is not None:
                    cv = GroupKFold(n_splits=cv)  # GroupKFold doesn't support shuffle/random_state
                else:
                    if cv_shuffle:
                        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                    else:
                        cv = KFold(n_splits=cv, shuffle=False)
            if verbose:
                logger.info("Using cv=%s", cv)

        if early_stopping_val_nsplits:
            try:
                val_cv = type(cv)(**{**cv.get_params(), "n_splits": early_stopping_val_nsplits})
            except (AttributeError, TypeError):
                # F11 fix: warn loudly when we hit the fallback path. For
                # LeaveOneOut / iterator-based custom CVs, get_params() may
                # not exist or n_splits is computed from the data, not a
                # constructor arg. The setattr-style fallback below silently
                # writes to a meaningless attribute and the CV runs with its
                # original (often n_samples) split count, ignoring the user's
                # ``early_stopping_val_nsplits`` request.
                logger.warning(
                    "RFECV: cv=%r does not accept n_splits via get_params(); "
                    "falling back to copy.copy + attribute assignment. The user's "
                    "early_stopping_val_nsplits=%s may be IGNORED if this CV "
                    "computes its split count from the data (e.g. LeaveOneOut). "
                    "Pass an explicit val_cv-compatible splitter to silence this.",
                    type(cv).__name__, early_stopping_val_nsplits,
                )
                val_cv = copy.copy(cv)
                val_cv.n_splits = early_stopping_val_nsplits
            if not early_stopping_rounds:
                early_stopping_rounds = 20  # TODO(2026-04-28): derive as 1/5 of n_estimators
        else:
            val_cv = None

        progressbar_prefix = "RFECV:"
        iters_pbar = tqdmu(
            desc=progressbar_prefix,
            leave=leave_progressbars,
            total=min(len(original_features) + 1, max_refits) if max_refits else len(original_features) + 1,
        )

        suppress_irritating_3rdparty_warnings()

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init scoring
        # ----------------------------------------------------------------------------------------------------------------------------

        if scoring is None:
            if is_classifier(estimator):
                logger.info(f"Scoring omitted, using probabilistic_multiclass_error by default.")
                # Use response_method='predict_proba' for sklearn 1.4+
                # (needs_proba is deprecated)
                scoring = make_scorer(score_func=compute_probabilistic_multiclass_error, response_method="predict_proba", greater_is_better=False)
            elif is_regressor(estimator):
                logger.info(f"Scoring omitted, using mean_squared_error by default.")
                scoring = make_scorer(score_func=mean_squared_error, greater_is_better=False)
            else:
                raise ValueError(f"Appropriate scoring not known for estimator type: {estimator}")
            self.scoring = scoring

        if verbose:
            logger.info("Scoring=%s", scoring)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Init importance_getter
        # ----------------------------------------------------------------------------------------------------------------------------

        if isinstance(estimator, Pipeline):
            estimator_type = type(estimator.steps[-1][1]).__name__
        else:
            estimator_type = type(estimator).__name__

        if importance_getter is None or importance_getter == "auto":
            # F38 fix: defer the dispatch to get_feature_importances so it can
            # look at the FITTED model's attributes. The prior hardcoded
            # ``LogisticRegression -> coef_, else -> feature_importances_``
            # crashed on LinearRegression / Ridge / Lasso / SVC(linear) /
            # SGDClassifier and any other sklearn linear model that exposes
            # only ``coef_``.
            importance_getter = "auto"

        # ----------------------------------------------------------------------------------------------------------------------------
        # Start evaluating different nfeatures, being guided by the selected search method
        # ----------------------------------------------------------------------------------------------------------------------------

        nsteps = 0
        dummy_scores = []
        fitted_estimators = {}
        selected_features_per_nfeatures = {}

        if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
            # Default plotting mode: 'No'. The previous default of
            # OnScoreImprovement called plt.show() inside the optimizer on
            # every score improvement, which blocks pytest / headless runs
            # indefinitely (Qt event loop). Users who want plotting must opt
            # in explicitly via ``optimizer_plotting='OnScoreImprovement'``,
            # 'Regular', or 'Final'.
            _plotting_map = {
                "No": OptimizationProgressPlotting.No,
                "Final": OptimizationProgressPlotting.Final,
                "OnScoreImprovement": OptimizationProgressPlotting.OnScoreImprovement,
                "Regular": OptimizationProgressPlotting.Regular,
            }
            if self.optimizer_plotting is None:
                plotting_mode = OptimizationProgressPlotting.No
            else:
                plotting_mode = _plotting_map.get(
                    self.optimizer_plotting, OptimizationProgressPlotting.No
                )

            Optimizer = MBHOptimizer(
                search_space=(
                    np.array(np.arange(min(self.max_nfeatures, len(original_features)) + 1).tolist() + [len(original_features)])
                    if self.max_nfeatures
                    else np.arange(len(original_features) + 1)
                ),
                direction=OptimizationDirection.Maximize,
                init_sampling_method=CandidateSamplingMethod.Equidistant,
                init_evaluate_ascending=False,
                init_evaluate_descending=True,
                plotting=plotting_mode,
                seeded_inputs=[min(2, len(original_features))],
            )
        else:
            Optimizer = None

        prev_score, prev_nfeatures = -np.inf, 0
        n_noimproving_iters = 0
        best_nfeatures = 0
        best_iter = 0
        # F21 fix: -1e6 floor was too high for high-error scorers (negative MSE
        # on a noisy regression target hits -1e8 routinely). With the old floor
        # ``final_score > best_score`` was False forever, n_noimproving_iters
        # incremented every iter, and max_noimproving_iters fired prematurely.
        # -inf is the only safe initial value for an argmax-style accumulator.
        best_score = -np.inf

        while nsteps < len(original_features):

            # ----------------------------------------------------------------------------------------------------------------------------
            # Select current set of features to work on, based on ranking received so far, and the optimum search method
            # ----------------------------------------------------------------------------------------------------------------------------

            # Perf hotspot fix: clean_ram() (== gc.collect()) cost ~290ms per
            # call in the cProfile baseline (4.3s / 11% of total wall-clock
            # on a 15-iter run). Run only every 5th iter; sklearn estimators
            # don't leak meaningfully between iters and the per-fold clones
            # are released by Python's reference counting before this point.
            if nsteps % 5 == 0:
                clean_ram()

            if self.special_feature_indices is not None and len(self.special_feature_indices) > 0:
                current_features = self.special_feature_indices
            else:

                current_features = get_next_features_subset(
                    nsteps=nsteps,
                    original_features=original_features,
                    feature_importances=feature_importances,
                    evaluated_scores_mean=evaluated_scores_mean,
                    evaluated_scores_std=evaluated_scores_std,
                    use_all_fi_runs=use_all_fi_runs,
                    use_last_fi_run_only=use_last_fi_run_only,
                    use_one_freshest_fi_run=use_one_freshest_fi_run,
                    use_fi_ranking=use_fi_ranking,
                    top_predictors_search_method=top_predictors_search_method,
                    votes_aggregation_method=votes_aggregation_method,
                    Optimizer=Optimizer,
                )

            if current_features is None or len(current_features) == 0:
                break  # nothing more to try
            if self.stop_file and exists(self.stop_file):
                logger.warning(f"Stop file {self.stop_file} detected, quitting.")
                break

            desc = f"{progressbar_prefix} trying {len(current_features):_}F"
            if nsteps > 0:
                desc += f", had {prev_nfeatures:_}F with score {prev_score:.{ndigits}f}, best was {best_nfeatures:_}F with score {best_score:.{ndigits}f} @iter={best_iter:_} "
            iters_pbar.set_description(
                desc,
                refresh=True,
            )

            # F35 fix: defer the selected_features_per_nfeatures write until
            # after we know whether this exploration's score beats the
            # existing best at the same N. The prior unconditional write
            # silently downgraded a winning subset whenever MBH revisited
            # the same N with a worse one.

            # ----------------------------------------------------------------------------------------------------------------------------
            # Each split better be different. so, even if random_state is provided, random_state to the cv is generated separately
            # (and deterministically) each time based on the original random_state.
            # ----------------------------------------------------------------------------------------------------------------------------

            scores = []

            splitter = cv.split(X=X, y=y, groups=groups)
            if verbose:
                splitter = tqdmu(splitter, desc="CV folds", leave=False, total=cv.n_splits)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Evaluate currently selected set of features on CV
            # ----------------------------------------------------------------------------------------------------------------------------

            for nfold, (train_index, test_index) in enumerate(splitter):

                if self.min_train_size and len(train_index) < self.min_train_size:
                    continue
                if frac:
                    size = int(len(train_index) * frac)
                    if size > 10:
                        train_index = self._rng.choice(train_index, size=size, replace=False)

                # Phase 4 N5: actual fit/score uses must_include + optimiser's
                # pick. current_features already lives in the search-universe
                # complement (must_include filtered out at fit entry), so
                # concatenation never duplicates.
                if must_include_resolved:
                    fit_features = list(must_include_resolved) + list(current_features)
                else:
                    fit_features = current_features
                X_train, y_train, X_test, y_test = split_into_train_test(
                    X=X, y=y, train_index=train_index, test_index=test_index, features_indices=fit_features
                )  # this splits both dataframes & ndarrays in the same fashion
                if verbose > 2:
                    print(f"Train set size={len(y_train):_}, train idx sum={train_index.sum():_}")

                if val_cv and has_early_stopping_support(estimator_type):

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Make additional early stopping split from X_train
                    # ----------------------------------------------------------------------------------------------------------------------------

                    if groups is not None:
                        if isinstance(groups, pd.Series):
                            train_groups = groups.iloc[train_index]
                        else:
                            train_groups = groups[train_index]
                    else:
                        train_groups = None

                    splits = list(val_cv.split(X=X_train, y=y_train, groups=train_groups))
                    true_train_index, val_index = splits[-1]

                    X_train, y_train, X_val, y_val = split_into_train_test(X=X_train, y=y_train, train_index=true_train_index, test_index=val_index)
                    if verbose > 2:
                        print(f"Val set size={len(y_val):_}, val idx sum={val_index.sum():_}")

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # If estimator is known, apply early stopping to its fit params
                    # ----------------------------------------------------------------------------------------------------------------------------

                    temp_cat_features = [current_features.index(var) for var in cat_features if var in current_features] if cat_features else None

                    temp_fit_params = pack_val_set_into_fit_params(
                        model=estimator,
                        X_val=X_val,
                        y_val=y_val,
                        early_stopping_rounds=early_stopping_rounds,
                        cat_features=temp_cat_features,
                    )  # crafts fit params with early stopping tailored to particular model type.
                    # Filter feature-list keys (cat_features / text_features /
                    # embedding_features) coming in via fit_params to only
                    # columns present in the current selector iteration.
                    # Without this, names from the outer call reference
                    # columns dropped by the current iteration, and CB
                    # raises ``Error while processing column for feature
                    # 'cat_0'`` (fuzz c0103 / c0102 / c0147 / c0114).
                    #
                    # cat_features handling: ``pack_val_set_into_fit_params``
                    # above already injected index-based temp_cat_features
                    # IFF that list was non-empty. When it's empty
                    # (current_features doesn't intersect self.cat_features)
                    # we MUST still pass a name-list filtered to current_-
                    # features so CB doesn't fall back to auto-detect on
                    # numerically-encoded category columns (target-encoded
                    # cats look like floats and trip CB's "Invalid type for
                    # cat_feature ... =0.49..." c0102 / c0147).
                    # F17 fix: when current_features holds integer indices
                    # (ndarray X path), set-membership against string column
                    # names always misses and silently drops every cat_features
                    # / text_features / embedding_features entry from
                    # fit_params. In that case the user-supplied lists pass
                    # through unfiltered (the inner estimator can deal with
                    # missing column references on its own; we shouldn't
                    # second-guess via empty filtering).
                    _features_are_integer = (
                        len(current_features) > 0
                        and isinstance(current_features[0], (int, np.integer))
                    )
                    _current_set = set(current_features) if not _features_are_integer else None
                    _filtered_fit_params = {}
                    for _k, _v in fit_params.items():
                        if _k == "cat_features":
                            if "cat_features" in temp_fit_params:
                                # temp's index-based cat_features wins; drop outer name list.
                                continue
                            if _v and _current_set is not None:
                                _filtered = [c for c in _v if c in _current_set]
                                _filtered_fit_params[_k] = _filtered or None
                            elif _v:
                                _filtered_fit_params[_k] = _v
                            continue
                        if _k in ("text_features", "embedding_features") and _v:
                            if _current_set is not None:
                                _filtered = [c for c in _v if c in _current_set]
                                _filtered_fit_params[_k] = _filtered or None
                            else:
                                _filtered_fit_params[_k] = _v
                            continue
                        _filtered_fit_params[_k] = _v
                    temp_fit_params.update(_filtered_fit_params)

                else:

                    temp_fit_params = {}
                    X_val = None

                # ----------------------------------------------------------------------------------------------------------------------------
                # Fit our estimator on current train fold. Score on test & and get its feature importances.
                # ----------------------------------------------------------------------------------------------------------------------------

                # TODO(2026-04-28): invoke different hyper parameters generation here

                # F32 + F33 fix: always clone per fold via sklearn.base.clone.
                # The prior ``copy.copy`` was a SHALLOW copy that shared mutable
                # state (cat_features list, set_params side effects, warm_start
                # buffers) across folds. The keep_estimators=False branch was
                # even worse: it reused the SAME instance fold after fold, so
                # any set_params(text_processing=...) mutation persisted into
                # the next iter's estimator. clone() returns an unfitted
                # estimator with the same constructor params and NO fitted
                # state - the canonical sklearn idiom.
                fitted_estimator = clone(estimator)

                # Dynamic CB ``text_processing`` calibration for THIS fold's
                # clone (not the outer estimator). RFECV folds are typically
                # much smaller than the outer training set; with CB's default
                # ``occurrence_lower_bound=50`` words that occur < 50 times
                # in the fold are pruned, leaving an empty dictionary and
                # HANGING CB's C++ ``_train`` loop (fuzz c0056 / c0070).
                # ``compute_cb_text_processing`` returns a config that scales
                # the floor proportionally to fold rows, or None (no-op) when
                # the fold is large enough for CB's defaults to work.
                if val_cv and has_early_stopping_support(estimator_type):
                    _temp_text_feats = _filtered_fit_params.get("text_features") or []
                    if _temp_text_feats and "CatBoost" in type(fitted_estimator).__name__:
                        _fold_rows = X_train.shape[0] if hasattr(X_train, "shape") else None
                        _tp = compute_cb_text_processing(_fold_rows) if _fold_rows is not None else None
                        if _tp is not None and hasattr(fitted_estimator, "set_params"):
                            try:
                                fitted_estimator.set_params(text_processing=_tp)
                            except Exception as _tp_exc:
                                logger.warning(
                                    "RFECV inner fold: failed to set CB text_processing "
                                    "(fold_rows=%s, exc=%s).", _fold_rows, _tp_exc,
                                )

                model_type_name = type(fitted_estimator).__name__ if fitted_estimator is not None else ""
                # Empty-train guard: heavy upstream filtering (small n +
                # outlier_detection + trainset_aging_limit) can collapse
                # X_train / y_train to 0 rows on a CV fold; CatBoost then
                # raises "Labels variable is empty" deep in C++ Pool init
                # (fuzz c0079). Skip the fold cleanly with a sentinel
                # score so the FS loop continues; this matches sklearn's
                # behaviour for degenerate folds.
                _x_n = X_train.shape[0] if hasattr(X_train, "shape") else None
                _y_n = len(y_train) if y_train is not None and hasattr(y_train, "__len__") else None
                if (_x_n is not None and _x_n == 0) or (_y_n is not None and _y_n == 0):
                    # Always-on ERROR (was verbose-gated WARNING) so the
                    # operator sees the empty-fold collapse without
                    # extra verbosity. Continuing with NaN-score is the
                    # right behavior -- sklearn's RFECV does the same on
                    # degenerate inner folds. Root cause is upstream
                    # filter aggression (OD + trainset_aging_limit
                    # together can shrink the inner-CV training fraction
                    # below cv splits); fix that, not this guard.
                    logger.error(
                        "wrappers.fit: skipping fold %s - empty train slice "
                        "(rows=%s, target_len=%s). Upstream filters reduced "
                        "the train batch to zero. Investigate "
                        "outlier_detection contamination + "
                        "trainset_aging_limit interactions.",
                        nfold, _x_n, _y_n,
                    )
                    scores.append(np.nan)
                    feature_importances[f"{len(current_features)}_{nfold}"] = {}
                    continue
                ctx = suppress_stdout_stderr() if model_type_name in CATBOOST_MODEL_TYPES else nullcontext()
                with ctx:
                    fitted_estimator.fit(X=X_train, y=y_train, **temp_fit_params)

                score = scoring(fitted_estimator, X_test, y_test)
                scores.append(score)
                # Phase 4 N5: FI is computed on the actual fit_features (which
                # includes must_include); we only persist the importances for
                # the optimiser-searchable features so voting / ranking remain
                # over the same universe the optimiser explores.
                fi_full = get_feature_importances(
                    model=fitted_estimator, current_features=fit_features, data=X_test, reference_data=X_val, importance_getter=importance_getter
                )
                if must_include_resolved:
                    must_set = set(must_include_resolved)
                    fi = {k: v for k, v in fi_full.items() if k not in must_set}
                else:
                    fi = fi_full

                key = f"{len(current_features)}_{nfold}"
                feature_importances[key] = fi

                if keep_estimators:
                    fitted_estimators[key] = fitted_estimator

                if 0 not in evaluated_scores_mean:

                    # ----------------------------------------------------------------------------------------------------------------------------
                    # Dummy baselines must serve as fitness @ 0 features.
                    # ----------------------------------------------------------------------------------------------------------------------------

                    if not self.nofeatures_dummy_scoring:
                        # F5 fix: sign-direction agnostic "worse than model" placeholder.
                        # The previous `score/10` path silently put the dummy ABOVE the model when
                        # `_sign==1` and the score was negative (e.g. R^2 < 0 on a bad fold). Both
                        # sklearn conventions agree that subtracting a positive number makes the
                        # score worse: for sign=+1 lower is worse; for sign=-1 (sklearn-negated)
                        # more negative is worse. Use magnitude-relative fudge so it scales with
                        # whatever metric is at play (log-loss ~0.5, MSE ~1e6, R^2 ~1).
                        fudge = max(abs(score), 1e-3) * 9.0
                        dummy_scores.append(score - fudge)
                    else:
                        dummy_scores.append(
                            get_best_dummy_score(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring)
                        )

            if 0 not in evaluated_scores_mean:
                scores_mean, scores_std, final_score, _ = store_averaged_cv_scores(
                    pos=0, scores=dummy_scores, evaluated_scores_mean=evaluated_scores_mean, evaluated_scores_std=evaluated_scores_std, self=self
                )
                nofeatures_score = final_score
                if verbose:
                    logger.info(f"Baseline with 0 features, score={scores_mean:.{ndigits}f} +/- {scores_std:.{ndigits}f} ~ {final_score:.{ndigits}f}")
                if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                    Optimizer.submit_evaluations(candidates=[0], evaluations=[final_score], durations=[None])

            scores_mean, scores_std, final_score, was_stored = store_averaged_cv_scores(
                pos=len(current_features),
                scores=scores,
                evaluated_scores_mean=evaluated_scores_mean,
                evaluated_scores_std=evaluated_scores_std,
                self=self,
            )
            # F35: only commit selected_features when this run actually won at its N.
            if was_stored:
                selected_features_per_nfeatures[len(current_features)] = current_features

            if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                Optimizer.submit_evaluations(candidates=[len(current_features)], evaluations=[final_score], durations=[None])

                if verbose:
                    logger.info(
                        f"Tried {len(current_features):_} features ({textwrap.shorten(', '.join(map(str, current_features[:40])), 150)}), score={scores_mean:.{ndigits}f} +/- {scores_std:.{ndigits}f} ~ {final_score:.{ndigits}f}"
                    )

            prev_nfeatures, prev_score = len(current_features), final_score
            iters_pbar.update(1)

            # ----------------------------------------------------------------------------------------------------------------------------
            # Checking exit conditions
            # ----------------------------------------------------------------------------------------------------------------------------

            nsteps += 1

            if len(evaluated_scores_mean) == 2:
                # F41 followup: the comment used to claim "0 features & all features",
                # but iter 1 explores whatever MBH seeded (default seed: 2 features),
                # NOT the full feature set. The early-stop is really "if the FIRST
                # explored point is worse than the dummy baseline at 0 features,
                # there's no point continuing". Keep the safety check, but don't
                # mislabel what's being compared.
                if final_score < nofeatures_score:
                    logger.info(
                        f"Stopping RFECV early: dummy baseline at 0 features ({nofeatures_score:.{ndigits}f}) "
                        f"already beats the first explored subset of {len(current_features)} features "
                        f"({final_score:.{ndigits}f}). The model can't learn anything from this data."
                    )
                    break

            if max_runtime_mins and not ran_out_of_time:
                delta = timer() - start_time
                ran_out_of_time = delta > (max_runtime_mins * 60)
                if ran_out_of_time:
                    if verbose:
                        logger.info(f"max_runtime_mins={max_runtime_mins:_.1f} reached.")
                    break

            if max_refits and nsteps >= max_refits:
                if verbose:
                    logger.info(f"max_refits={max_refits:_} reached.")
                break

            if final_score > best_score:
                best_score = final_score
                best_iter = nsteps
                best_nfeatures = len(current_features)
                n_noimproving_iters = 0
            else:
                n_noimproving_iters += 1

            if best_desired_score is not None and final_score > best_desired_score:
                if verbose:
                    logger.info(f"best_desired_score {best_desired_score:_.{ndigits}f} reached.")
                break

            if max_noimproving_iters and n_noimproving_iters >= max_noimproving_iters:
                if verbose:
                    logger.info("Max # of noimproved iters reached: %s", n_noimproving_iters)
                break

            if self.special_feature_indices is not None:
                if verbose:
                    logger.info(f"Quitting as special_feature_indices were checked.")
                break

        # ----------------------------------------------------------------------------------------------------------------------------
        # Saving best result found so far as final
        # ----------------------------------------------------------------------------------------------------------------------------

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(map(str, np.arange(self.n_features_in_)))

        self.estimators_ = fitted_estimators  # a dict with key=nfeatures_nfold
        self.feature_importances_ = feature_importances  # a dict with key=nfeatures_nfold
        self.selected_features_ = selected_features_per_nfeatures  # a dict with key=nfeatures

        checked_nfeatures = sorted(evaluated_scores_mean.keys())
        cv_std_perf = [evaluated_scores_std[n] for n in checked_nfeatures]
        cv_mean_perf = [evaluated_scores_mean[n] for n in checked_nfeatures]
        self.cv_results_ = {"nfeatures": checked_nfeatures, "cv_mean_perf": cv_mean_perf, "cv_std_perf": cv_std_perf}

        self.select_optimal_nfeatures_(
            checked_nfeatures=checked_nfeatures,
            cv_mean_perf=cv_mean_perf,
            cv_std_perf=cv_std_perf,
            feature_cost=feature_cost,
            smooth_perf=smooth_perf,
            use_all_fi_runs=use_all_fi_runs,
            use_last_fi_run_only=use_last_fi_run_only,
            use_one_freshest_fi_run=use_one_freshest_fi_run,
            use_fi_ranking=use_fi_ranking,
            votes_aggregation_method=votes_aggregation_method,
            verbose=verbose,
            show_plot=show_plot,
        )

        # Phase 4 N5: glue must_include into the final support_. The optimiser
        # produced a support_ over the search-universe complement only;
        # must_include features are always in the final selection regardless
        # of what the optimiser picked.
        if must_include_resolved and hasattr(self, "support_") and len(self.support_) > 0:
            must_set = set(must_include_resolved)
            if isinstance(self.support_[0], (bool, np.bool_)):
                # support_ is bool-mask aligned with feature_names_in_;
                # set the must_include positions to True.
                support_mask = np.asarray(self.support_, dtype=bool)
                for col in must_include_resolved:
                    if col in self.feature_names_in_:
                        support_mask[self.feature_names_in_.index(col)] = True
                self.support_ = support_mask
            else:
                # support_ is integer indices; prepend must_include positions.
                idx_must = [self.feature_names_in_.index(c) for c in must_include_resolved if c in self.feature_names_in_]
                merged = list(idx_must) + [i for i in self.support_ if i not in idx_must]
                self.support_ = np.asarray(merged)
            self.n_features_ = int(np.sum(self.support_)) if isinstance(self.support_[0], (bool, np.bool_)) else len(self.support_)

        self.signature = signature

        # Cache resolved column list so transform() avoids per-call reconstruction.
        self._selected_cols_cache = None
        support = getattr(self, "support_", None)
        if support is not None and len(support) > 0:
            if isinstance(support[0], (bool, np.bool_)):
                self._selected_cols_cache = [col for col, selected in zip(self.feature_names_in_, support) if selected]
            else:
                self._selected_cols_cache = [self.feature_names_in_[i] for i in support]

        return self

    def select_optimal_nfeatures_(
        self,
        checked_nfeatures: np.ndarray,
        cv_mean_perf: np.ndarray,
        cv_std_perf: np.ndarray,
        feature_cost: float = 0.0,
        smooth_perf: int = 3,
        use_all_fi_runs: bool = True,
        use_last_fi_run_only: bool = False,
        use_one_freshest_fi_run: bool = False,
        use_fi_ranking: bool = False,
        votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
        verbose: bool = False,
        show_plot: bool = False,
        plot_file=None,
        font_size: int = 12,
        figsize: tuple = (10, 7),
    ):

        base_perf = np.array(cv_mean_perf) * self.mean_perf_weight - np.array(cv_std_perf) * self.std_perf_weight
        if smooth_perf:
            smoothed_perf = pd.Series(base_perf).rolling(smooth_perf, center=True).mean().values
            idx = np.isnan(smoothed_perf)
            smoothed_perf[idx] = base_perf[idx]
            base_perf = smoothed_perf

        ultimate_perf = base_perf - np.array(checked_nfeatures) * feature_cost

        sorted_idx = np.argsort(ultimate_perf)[::-1]
        # F8 fix: defensive defaults so an all-zero checked_nfeatures input
        # (only the dummy at 0 was ever evaluated, e.g. the search loop
        # broke on iter 1 with empty current_features) doesn't leave
        # ``best_idx``/``best_top_n`` unbound and crash 30 lines below at
        # ``base_perf[best_idx]`` / ``self.n_features_ = best_top_n``.
        best_idx = None
        best_top_n = 0
        for idx in sorted_idx:
            if checked_nfeatures[idx] != 0:
                best_idx = idx
                best_top_n = checked_nfeatures[best_idx]
                break
            else:
                logger.warning(f"Can't allow nfeatures to be zero. Using first non-zero value.")
        if best_idx is None:
            logger.warning(
                "select_optimal_nfeatures_: only nfeatures==0 was evaluated; "
                "no features can be selected. Returning empty support_."
            )
            self.n_features_ = 0
            self.support_ = np.array([])
            return

        if show_plot or plot_file:
            plt.rcParams.update({"font.size": font_size})
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = ax1.twinx()

            ax1.set_xlabel("Number of features selected")
            ax1.set_ylabel("Mean CV score", c="b")

            ax1.errorbar(checked_nfeatures, cv_mean_perf, yerr=cv_std_perf, c="b", alpha=0.4)

            ax2.plot(checked_nfeatures, ultimate_perf, c="g")
            ax1.plot(checked_nfeatures[best_idx], base_perf[best_idx], "ro")
            ax2.set_ylabel("Adj CV score", c="g")

            plt.title("Performance by nfeatures")
            plt.tight_layout()

            if plot_file:
                plt.savefig(plot_file)
            if show_plot:
                plt.show()

        # after making a cutoff decision:

        self.n_features_ = best_top_n
        if best_top_n == 0:
            self.support_ = np.array([])
        else:

            if not self.conduct_final_voting:

                # An obvious solution is to return exact features that we used when measuring scores.
                # Convert feature_name to string if feature_names_in_ contains strings (ndarray case)
                selected = self.selected_features_[best_top_n]
                # Represent support_ as a boolean mask for consistency with sklearn's RFE API.
                if self.feature_names_in_ and isinstance(self.feature_names_in_[0], str):
                    selected_set = {str(feature_name) for feature_name in selected}
                    self.support_ = np.array([str(f) in selected_set for f in self.feature_names_in_])
                else:
                    selected_set = set(selected)
                    self.support_ = np.array([f in selected_set for f in self.feature_names_in_])

            else:

                # ----------------------------------------------------------------------------------------------------------------------------
                # A more advanced alternative would be to last time vote for feature_importances using all info up to date
                # ----------------------------------------------------------------------------------------------------------------------------

                fi_to_consider = select_appropriate_feature_importances(
                    feature_importances=self.feature_importances_,
                    nfeatures=best_top_n,
                    n_original_features=self.n_features_in_,
                    use_all_fi_runs=use_all_fi_runs,
                    use_last_fi_run_only=use_last_fi_run_only,
                    use_one_freshest_fi_run=use_one_freshest_fi_run,
                    use_fi_ranking=use_fi_ranking,
                )

                self.ranking_ = get_actual_features_ranking(
                    feature_importances=fi_to_consider,
                    votes_aggregation_method=votes_aggregation_method,
                )

                self.support_ = np.array([(i in self.ranking_[:best_top_n]) for i in self.feature_names_in_])

        if verbose:
            dummy_gain = (base_perf[0] / base_perf[best_idx] - 1) if base_perf[best_idx] != 0 else np.inf
            allfeat_gain = (base_perf[-1] / base_perf[best_idx] - 1) if base_perf[best_idx] != 0 else np.inf
            logger.info(
                f"{self.n_features_:_} predictive factors selected out of {self.n_features_in_:_} during {len(self.selected_features_):_} rounds. Gain vs dummy={dummy_gain*100:.1f}%, gain vs all features={allfeat_gain*100:.1f}%"
            )

    def get_feature_names_out(self, input_features=None):
        """sklearn-1.x transformer protocol. Returns the names of the selected
        features as an ndarray of str, matching what ``transform`` will produce
        as columns. Compatible with sklearn Pipelines that call this method
        for downstream feature naming (ColumnTransformer, set_output)."""
        if not hasattr(self, "support_"):
            raise ValueError("RFECV is not fitted; call fit() first.")
        cache = getattr(self, "_selected_cols_cache", None)
        if cache is not None:
            return np.asarray(cache, dtype=object)
        if len(self.support_) == 0:
            return np.array([], dtype=object)
        if isinstance(self.support_[0], (bool, np.bool_)):
            return np.asarray(
                [c for c, s in zip(self.feature_names_in_, self.support_) if s],
                dtype=object,
            )
        return np.asarray([self.feature_names_in_[i] for i in self.support_], dtype=object)

    def selection_stability_(self, metric: str = "jaccard") -> float:
        """Mean pairwise feature-selection stability across CV folds at the
        chosen ``n_features_``. Free signal extracted from feature_importances_,
        no extra fits required.

        Args:
            metric: 'jaccard' (default), 'dice', or 'kuncheva'.

        Returns:
            Float in [0, 1] (1 = identical selections across folds, 0 = disjoint).
            Returns NaN when fewer than 2 folds have FI data at n_features_.
        """
        if not hasattr(self, "feature_importances_") or not hasattr(self, "n_features_"):
            raise ValueError("RFECV is not fitted; call fit() first.")
        if self.n_features_ == 0:
            return float("nan")
        # Pull per-fold FI runs at the chosen N: keys are 'N_fold' strings.
        target_prefix = f"{self.n_features_}_"
        per_fold_top: list[set] = []
        for key, fi in self.feature_importances_.items():
            if not key.startswith(target_prefix):
                continue
            if len(fi) < self.n_features_:
                continue
            # Top-N features in this fold by importance value
            top_ids = sorted(fi.keys(), key=lambda k: fi[k], reverse=True)[: self.n_features_]
            per_fold_top.append(set(top_ids))
        if len(per_fold_top) < 2:
            return float("nan")

        def _pair_stability(a: set, b: set) -> float:
            inter = len(a & b)
            if metric == "jaccard":
                union = len(a | b)
                return inter / union if union else 1.0
            if metric == "dice":
                denom = len(a) + len(b)
                return (2 * inter) / denom if denom else 1.0
            if metric == "kuncheva":
                # Kuncheva's index normalises by chance overlap; needs the
                # universe size N (total features). Range: [-1, 1] but
                # clamped to [0, 1] here for a uniform interpretation.
                k = len(a)  # |a| == |b| == n_features_ by construction
                N = self.n_features_in_
                if k == 0 or N == 0 or k == N:
                    return 1.0 if a == b else 0.0
                expected = k * k / N
                ki = (inter - expected) / (k - expected)
                return max(0.0, ki)
            raise ValueError(f"Unknown stability metric: {metric!r}")

        pairs = [
            _pair_stability(per_fold_top[i], per_fold_top[j])
            for i in range(len(per_fold_top))
            for j in range(i + 1, len(per_fold_top))
        ]
        return float(np.mean(pairs)) if pairs else float("nan")

    def n_features_one_se_(self) -> int:
        """1-SE rule: smallest N whose CV mean is within one standard error
        of the best CV mean. Often selected over n_features_ when the operator
        wants the most parsimonious model that's not statistically distinguishable
        from the optimum.

        Returns the integer count, or n_features_ as a fallback if cv_results_
        is unavailable.
        """
        if not hasattr(self, "cv_results_") or not self.cv_results_.get("nfeatures"):
            return getattr(self, "n_features_", 0)
        nfeatures = np.asarray(self.cv_results_["nfeatures"], dtype=int)
        means = np.asarray(self.cv_results_["cv_mean_perf"], dtype=float)
        stds = np.asarray(self.cv_results_["cv_std_perf"], dtype=float)
        if len(means) == 0:
            return getattr(self, "n_features_", 0)
        # Exclude the 0-features dummy from selection
        nonzero = nfeatures > 0
        if not nonzero.any():
            return getattr(self, "n_features_", 0)
        nf, m, s = nfeatures[nonzero], means[nonzero], stds[nonzero]
        # mean_perf_weight + std_perf_weight already baked into final_score;
        # for 1-SE we need the *unadjusted* mean. cv_mean_perf is the raw mean.
        best_idx = int(np.argmax(m))
        threshold = m[best_idx] - s[best_idx]
        # Smallest N whose mean >= threshold
        eligible = nf[m >= threshold]
        if len(eligible) == 0:
            return int(nf[best_idx])
        return int(eligible.min())

    def transform(self, X, y=None):
        # 2026-04-28: when ``X`` arrives as a polars DataFrame (callers
        # like ``_passthrough_cols_fit_transform`` keep the native
        # frame), the legacy ``X[:, self.support_]`` mask path raises
        # ``expected N values when selecting columns by boolean mask,
        # got M`` if the polars schema has more cols than the fit-time
        # ``support_`` (because ``RFECV.fit`` dropped zero-variance cols
        # at entry, see line 256). Convert to pandas so the name-keyed
        # transform path below kicks in and the column-set drift becomes
        # a clear ``RuntimeError`` instead of an opaque polars index
        # mismatch. Surfaced default-seed c0016
        # (cb_hgb_xgb / pl_nullable / confidence_analysis_cfg=True).
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        # Use getattr to handle unfitted RFECV (support_ not yet set)
        support = getattr(self, 'support_', None)
        if support is None:
            return X
        if len(support) == 0:
            # Return empty DataFrame/array with same rows but no columns
            # This signals that feature selection found no useful features
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, []]
            else:
                return X[:, np.array([], dtype=np.intp)]
        if isinstance(X, pd.DataFrame):
            # Use column names (not .iloc) to support Arrow-backed DataFrames
            # from polars zero-copy conversion - they don't support
            # .iloc[:, integer_array] reliably.
            selected_cols = getattr(self, "_selected_cols_cache", None)
            if selected_cols is None:
                if len(self.support_) > 0 and isinstance(self.support_[0], (bool, np.bool_)):
                    selected_cols = [col for col, selected in zip(self.feature_names_in_, self.support_) if selected]
                else:
                    selected_cols = [self.feature_names_in_[i] for i in self.support_]
            # Column-set drift between fit-time and transform-time is a hard
            # error: the fit-time zero-variance filter ensures
            # feature_names_in_ never contains columns sklearn pipeline
            # steps may silently drop. If we still see drift, an upstream
            # step is mutating the schema between fit and transform - a
            # real pipeline-order bug we want surfaced loud, not masked.
            missing = [c for c in selected_cols if c not in X.columns]
            if missing:
                raise RuntimeError(
                    f"RFECV.transform: {len(missing)}/{len(selected_cols)} "
                    f"selected columns missing from input X ({missing}); "
                    f"the fitted support_ mask no longer reflects the "
                    f"physical columns. The zero-variance filter at "
                    f"RFECV.fit already excludes constant / all-null "
                    f"columns from feature_names_in_, so this drift means "
                    f"an upstream step (constant-col-removal / imputer-drop "
                    f"/ OD filter) is mutating the column set BETWEEN fit "
                    f"and transform. Investigate."
                )
            return X[selected_cols]
        else:
            return X[:, self.support_]


def split_into_train_test(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray], train_index: np.ndarray, test_index: np.ndarray, features_indices: np.ndarray = None
) -> tuple:
    """Split X & y according to indices & dtypes. Basically this accounts for diffeent dtypes (pd.DataFrame, np.ndarray) to perform the same."""

    if isinstance(X, pd.DataFrame):
        # Perf #2 fix: the prior integer-index path did
        # ``X.iloc[rows].iloc[:, cols]`` (chained), which materialises the
        # full row-slab BEFORE column selection. On wide tables (200k rows x
        # 10k cols) that's ~16 GB of intermediate write when only the
        # K-feature subset (~8 GB) was needed. ``X.iloc[np.ix_(rows, cols)]``
        # mirrors the numpy branch and selects both axes in one shot, cutting
        # 2-7% off total wall-clock on wide-table runs.
        if features_indices is None:
            X_train = X.iloc[train_index, :]
            X_test = X.iloc[test_index, :]
        else:
            tr_arr = np.asarray(train_index)
            te_arr = np.asarray(test_index)
            if isinstance(features_indices[0], (int, np.integer)):
                fi_arr = np.asarray(features_indices)
                X_train = X.iloc[np.ix_(tr_arr, fi_arr)]
                X_test = X.iloc[np.ix_(te_arr, fi_arr)]
            else:
                # Name-based selection: .loc is the fastest single-shot path
                # for column NAMES; .iloc[rows, :] then .loc[:, cols] also
                # works but is two passes.
                X_train = X.loc[X.index[tr_arr], list(features_indices)]
                X_test = X.loc[X.index[te_arr], list(features_indices)]
        y_train = y.iloc[train_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index])
        y_test = y.iloc[test_index, :] if isinstance(y, pd.DataFrame) else (y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index])
    elif isinstance(X, pl.DataFrame):
        # Polars branch (2026-04-21 fix 9.8): the generic numpy path below
        # used ``X[np.ix_(rows, cols)]``, which passes a 2-D ndarray to
        # ``polars.DataFrame.__getitem__`` and raises
        # ``TypeError: multi-dimensional NumPy arrays not supported as
        # index`` at polars/_utils/getitem.py. Polars selects rows+cols in
        # two steps: ``df[rows_seq]`` or ``df[rows_seq, cols]`` requires
        # 1-D row index + columns list.
        tr_idx = list(np.asarray(train_index))
        te_idx = list(np.asarray(test_index))
        if features_indices is None:
            X_train = X[tr_idx]
            X_test = X[te_idx]
        else:
            fi = np.asarray(features_indices)
            if fi.dtype.kind in ("i", "u"):
                cols_sel = [X.columns[int(i)] for i in fi]
            else:
                cols_sel = [str(c) for c in fi]
            # ``df.select(cols)[rows]`` = column-first then row-slice; avoids
            # materialising an (n_rows x n_all_cols) intermediate.
            X_train = X.select(cols_sel)[tr_idx]
            X_test = X.select(cols_sel)[te_idx]
        # y for polars CV usually arrives as numpy already, but handle
        # pl.Series defensively.
        if hasattr(y, "to_numpy") and not isinstance(y, np.ndarray):
            y_np = y.to_numpy()
        else:
            y_np = y
        y_train = y_np[train_index, :] if hasattr(y_np, "shape") and len(y_np.shape) > 1 else y_np[train_index]
        y_test = y_np[test_index, :] if hasattr(y_np, "shape") and len(y_np.shape) > 1 else y_np[test_index]
    else:
        if features_indices is None:
            X_train = X[train_index, :]
            X_test = X[test_index, :]
        else:
            # One-shot fancy indexing via np.ix_; avoids the intermediate row slab.
            fi = np.asarray(features_indices)
            X_train = X[np.ix_(np.asarray(train_index), fi)]
            X_test = X[np.ix_(np.asarray(test_index), fi)]
        y_train = y[train_index, :] if len(y.shape) > 1 else y[train_index]
        y_test = y[test_index, :] if len(y.shape) > 1 else y[test_index]

    return X_train, y_train, X_test, y_test


def store_averaged_cv_scores(pos: int, scores: list, evaluated_scores_mean: dict, evaluated_scores_std: dict, self: object) -> tuple:
    """Compute (mean, std, final_score) and store at ``pos`` ONLY if the new score
    beats any existing score at the same key. Returns (mean, std, final_score, was_stored).

    F35 fix: the prior version unconditionally overwrote evaluated_scores_mean[pos],
    so when MBH re-explored the same nfeatures-count with a worse subset, both
    the curve and the cached selected_features were silently downgraded to the
    last evaluation rather than the best. The Optimizer still sees every probe
    via submit_evaluations() at the call site (its surrogate must learn from
    all data), but cv_results_ now reflects the *best-known* score per N.
    """

    scores = np.array(scores)

    # Observability: a NaN fold score (e.g., scorer hit a single-class CV
    # fold and returned NaN) poisons ``scores_mean`` / ``final_score``.
    # Downstream ``final_score > best_score`` with NaN is always False, so
    # RFECV's early-stop patience counter (n_noimproving_iters) gets
    # consumed every iteration -- the search eventually terminates via
    # max_noimproving_iters, but spends many CV rounds producing no
    # signal. Surface this explicitly so operators can fix the scorer
    # (e.g., switch to stratified CV) rather than silently eating it.
    n_nan = int(np.isnan(scores).sum()) if scores.size else 0
    if n_nan:
        logger.warning(
            "store_averaged_cv_scores @ pos=%d: %d / %d CV fold score(s) are NaN. "
            "Likely cause: single-class CV fold (stratified split would fix it) "
            "or scorer returning NaN on degenerate folds.",
            pos, n_nan, scores.size,
        )

    # F23 fix: nanmean/nanstd so a single degenerate fold doesn't poison the
    # entire iter's final_score. The warning above is preserved so operators
    # still see the underlying issue (single-class fold, NaN-returning scorer).
    if scores.size and n_nan and n_nan < scores.size:
        scores_mean, scores_std = np.nanmean(scores), np.nanstd(scores)
    else:
        scores_mean, scores_std = np.mean(scores), np.std(scores)
    final_score = scores_mean * self.mean_perf_weight - scores_std * self.std_perf_weight

    existing_mean = evaluated_scores_mean.get(pos)
    existing_std = evaluated_scores_std.get(pos)
    if existing_mean is None:
        existing_final = -np.inf
    else:
        existing_final = existing_mean * self.mean_perf_weight - existing_std * self.std_perf_weight
    # Treat NaN incoming as "never beats existing"; keep the prior best.
    was_stored = (not np.isnan(final_score)) and final_score > existing_final
    if was_stored:
        evaluated_scores_mean[pos] = scores_mean
        evaluated_scores_std[pos] = scores_std

    return scores_mean, scores_std, final_score, was_stored


def get_feature_importances(
    model: object,
    current_features: list,
    importance_getter: Union[str, Callable],
    data: Union[pd.DataFrame, np.ndarray, None] = None,
    reference_data: Union[pd.DataFrame, np.ndarray, None] = None,
) -> dict:

    if isinstance(importance_getter, str):
        # F38 fix: 'auto' resolution looks at what the fitted model exposes.
        # Tree/boosting estimators expose feature_importances_; linear models
        # (LinearRegression, Ridge, Lasso, SVC(kernel='linear'), SGDClassifier,
        # ElasticNet, ...) expose only coef_. The prior dispatch hardcoded
        # LogisticRegression as the only coef_ estimator and crashed on the
        # rest with AttributeError.
        if importance_getter == "auto":
            if hasattr(model, "feature_importances_"):
                getter_attr = "feature_importances_"
            elif hasattr(model, "coef_"):
                getter_attr = "coef_"
            else:
                raise AttributeError(
                    f"importance_getter='auto' could not find ``feature_importances_`` "
                    f"or ``coef_`` on a fitted {type(model).__name__}. Pass an explicit "
                    f"``importance_getter='attr_name'`` or a callable."
                )
        else:
            getter_attr = importance_getter
        res = getattr(model, getter_attr)
        if getter_attr == "coef_":
            res = np.abs(res)
        if res.ndim > 1:
            res = res.sum(axis=0)
    else:
        res = importance_getter(model=model, data=data, reference_data=reference_data)

    if len(res) != len(current_features):
        raise ValueError(f"Feature importances length {len(res)} doesn't match current_features length {len(current_features)}")

    # Observability for degenerate folds (2026-04-19 probe finding).
    # When a model fits on a single-class / all-constant CV fold, its
    # ``feature_importances_`` attribute can legitimately contain NaN
    # (e.g. CatBoost on a single-class target, LightGBM on constant y).
    # Downstream ``get_actual_features_ranking`` then folds NaN into
    # the per-feature aggregate, poisoning the rank for every feature
    # that appeared in that fold -- silent, indistinguishable from "zero
    # importance". We already warn on NaN scoring; pair it here.
    try:
        res_arr = np.asarray(res, dtype=float)
        n_nan = int(np.isnan(res_arr).sum()) if res_arr.size else 0
    except (TypeError, ValueError):
        n_nan = 0  # non-numeric result; let downstream raise on use
    if n_nan:
        logger.warning(
            "get_feature_importances: %d / %d importance value(s) are NaN "
            "from %s. Likely cause: degenerate CV fold (single-class target, "
            "zero-variance features). Downstream RFECV ranking aggregation "
            "will fold these NaNs in, silently poisoning the rank for the "
            "affected features.",
            n_nan, res_arr.size, type(model).__name__,
        )
    return {feature_index: feature_importance for feature_index, feature_importance in zip(current_features, res)}


def get_next_features_subset(
    nsteps: int,
    original_features: list,
    feature_importances: pd.DataFrame,
    evaluated_scores_mean: dict,
    evaluated_scores_std: dict,
    use_all_fi_runs: bool,
    use_last_fi_run_only: bool,
    use_one_freshest_fi_run: bool,
    use_fi_ranking: bool,
    top_predictors_search_method: OptimumSearch = OptimumSearch.ScipyLocal,
    votes_aggregation_method: VotesAggregation = VotesAggregation.Borda,
    Optimizer: object = None,
) -> list:
    """Generates a "next_nfeatures_to_check" candidate to evaluate.
    Decides on a subset of FIs to use (all, freshest preceeding, all preceeding).
    Combines FIs from different runs into ranks using voting.
    Selects next_nfeatures_to_check best ranked features as candidates for the upcoming FI evaluation.
    The whole idea of this approach is that we don't need to go all the way from len(original_features) up to 0 and evaluate
    EVERY nfeatures. for 10k features and 1TB datast it's a waste.
    """

    # ----------------------------------------------------------------------------------------------------------------------------
    # First step is to try all features.
    # ----------------------------------------------------------------------------------------------------------------------------

    if nsteps == 0:
        return original_features
    else:
        # F41 fix: range upper-bound was len(original_features), so the
        # all-features candidate (k == len(original_features)) was NEVER
        # included in remaining and the optimizer could not re-evaluate it.
        # +1 makes the all-features point a legitimate candidate alongside
        # every smaller k.
        remaining = list(set(np.arange(1, len(original_features) + 1)) - set(evaluated_scores_mean.keys()))
        if len(remaining) == 0:
            return []
        else:

            if top_predictors_search_method == OptimumSearch.ExhaustiveRandom:
                next_nfeatures_to_check = random.choice(remaining)
            elif top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
                next_nfeatures_to_check = Optimizer.suggest_candidate()
            else:
                # F1 fix: ScipyLocal / ScipyGlobal / ExhaustiveDichotomic are declared in
                # OptimumSearch but the dispatch was never wired here, so any user picking
                # them hit UnboundLocalError on iter>=1. Surface it loudly at call time so
                # the failure points at the search-method choice instead of a name-binding
                # mystery 30 lines down.
                raise NotImplementedError(
                    f"top_predictors_search_method={top_predictors_search_method!r} "
                    f"is declared in OptimumSearch but not implemented in "
                    f"get_next_features_subset. Currently supported: "
                    f"OptimumSearch.ExhaustiveRandom, OptimumSearch.ModelBasedHeuristic."
                )

            if next_nfeatures_to_check is not None:

                # ----------------------------- -----------------------------------------------------------------------------------------------
                # At each step, feature importances must be recalculated in light of recent training on a smaller subset.
                # The features already thrown away all receive constant importance update of the same size, to keep up with number of trains (?)
                # ----------------------------------------------------------------------------------------------------------------------------

                fi_to_consider = select_appropriate_feature_importances(
                    feature_importances=feature_importances,
                    nfeatures=next_nfeatures_to_check,
                    n_original_features=len(original_features),
                    use_all_fi_runs=use_all_fi_runs,
                    use_last_fi_run_only=use_last_fi_run_only,
                    use_one_freshest_fi_run=use_one_freshest_fi_run,
                    use_fi_ranking=use_fi_ranking,
                )
                ranks = get_actual_features_ranking(feature_importances=fi_to_consider, votes_aggregation_method=votes_aggregation_method)
                return ranks[:next_nfeatures_to_check]
            else:
                return []


def select_appropriate_feature_importances(
    feature_importances: dict,
    nfeatures: int,
    n_original_features: int,
    use_all_fi_runs: bool = True,
    use_last_fi_run_only: bool = False,
    use_one_freshest_fi_run: bool = False,
    use_fi_ranking: bool = False,
) -> dict:

    if use_last_fi_run_only:
        # use train folds with specific length. key is nfeatures_nfold
        fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) == n_original_features}
    else:
        if use_all_fi_runs:
            # use all fi data collected so far
            fi_to_consider = {key: value for key, value in feature_importances.items() if len(value) > 1} if n_original_features > 1 else feature_importances
        else:
            # can only use runs preceeding nfeatures here.
            if use_one_freshest_fi_run:
                # freshest preceeding
                # F25 fix: range upper-bound was n_original_features (exclusive),
                # so the FI run on the full feature set (key length ==
                # n_original_features) was never picked as "freshest preceding"
                # for any smaller nfeatures. +1 includes it.
                fi_to_consider = {}
                for possible_nfeatures in range(nfeatures + 1, n_original_features + 1):
                    for key, value in feature_importances.items():
                        if len(value) == possible_nfeatures:

                            fi_to_consider[key] = value
                    if fi_to_consider:
                        print(f"using freshest FI of {possible_nfeatures} features for nfeatures={nfeatures}")
                        break
            else:
                # all preceeding
                fi_to_consider = {key: value for key, value in feature_importances.items() if (len(value) > nfeatures and len(value) != 1)}
    if use_fi_ranking:
        fi_to_consider = {key: pd.Series(value).rank(ascending=True, pct=True).to_dict() for key, value in fi_to_consider.items()}
    return fi_to_consider


def get_actual_features_ranking(feature_importances: dict, votes_aggregation_method: VotesAggregation) -> list:
    """Absolute FIs from estimators trained on CV for each nfeatures are stored separatly.
    They can be used to recompute final voted importances using any desired voting algo.
    But of course the exploration path was already lead by specific voting algo active at the fitting time.

    GM, and esp Minimax & Plurality are suboptimal for FS.

    Caching note: an earlier optimisation attempt cached by frozenset of
    feature_importances.keys(), but the same key strings ("{N}_{fold}") can
    map to DIFFERENT importance values across fits, so the cache returned
    stale ranks across test runs. Since cache hit rate within one fit() is
    near-zero (the dict grows monotonically each iter so the key set is
    different every call), caching at this layer is removed - the proper
    fix is making Leaderboard incremental, deferred to a later PR.
    """

    lb = Leaderboard(table=pd.DataFrame(feature_importances))
    if votes_aggregation_method == VotesAggregation.Borda:
        ranks = lb.borda_ranking()
    elif votes_aggregation_method == VotesAggregation.AM:
        ranks = lb.mean_ranking(mean_type="arithmetic")
    elif votes_aggregation_method == VotesAggregation.GM:
        ranks = lb.mean_ranking(mean_type="geometric")
    elif votes_aggregation_method == VotesAggregation.Copeland:
        ranks = lb.copeland_ranking()
    elif votes_aggregation_method == VotesAggregation.Dowdall:
        ranks = lb.dowdall_ranking()
    elif votes_aggregation_method == VotesAggregation.Minimax:
        ranks = lb.minimax_ranking()
    elif votes_aggregation_method == VotesAggregation.OG:
        ranks = lb.optimality_gap_ranking(gamma=1)
    elif votes_aggregation_method == VotesAggregation.Plurality:
        ranks = lb.plurality_ranking()
    else:
        # Defensive: F27 finding flagged unbound `ranks` if a future enum
        # value falls through. Surface clearly instead of NameError.
        raise NotImplementedError(
            f"votes_aggregation_method={votes_aggregation_method!r} not handled"
        )
    return ranks.index.values.tolist()
