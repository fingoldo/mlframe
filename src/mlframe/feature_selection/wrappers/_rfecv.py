"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

from __future__ import annotations


import copy
import hashlib
import logging
import textwrap
from contextlib import nullcontext
from os.path import exists
from timeit import default_timer as timer
from typing import Callable, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from pyutilz.system import tqdmu
from pyutilz.numbalib import set_numba_random_seed  # noqa: F401
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
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: F401
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,  # noqa: F401
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,  # noqa: F401
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline

from mlframe.config import CATBOOST_MODEL_TYPES
from mlframe.models.optimization import (
    CandidateSamplingMethod,
    MBHOptimizer,
    OptimizationDirection,
    OptimizationProgressPlotting,
)
from mlframe.utils.misc import set_random_seed
from mlframe.estimators.baselines import get_best_dummy_score
from mlframe.core.helpers import has_early_stopping_support
from mlframe.training.helpers import compute_cb_text_processing
from mlframe.preprocessing.transforms import pack_val_set_into_fit_params
from mlframe.metrics.core import compute_probabilistic_multiclass_error

from ._enums import OptimumSearch, VotesAggregation
from ._helpers import (
    _detect_multithreaded,
    _pin_threads_to_one,
    get_feature_importances,
    get_next_features_subset,
    get_actual_features_ranking,
    select_appropriate_feature_importances,
    split_into_train_test,
    store_averaged_cv_scores,
    suppress_irritating_3rdparty_warnings,
)

logger = logging.getLogger(__name__)


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

    Notes on ``nofeatures_dummy_scoring`` (default True)
    ----------------------------------------------------
    With this flag on, the "0-feature" anchor point of the CV curve is a
    ``DummyClassifier`` / ``DummyRegressor`` baseline rather than skipped. On
    AUROC / log-loss scorers the dummy reference is informative. On accuracy /
    F1 with severely imbalanced binary targets the prior-strategy DummyClassifier
    can score within a few points of the real model, which makes the marginal
    gain of adding the first real feature look small and biases the chosen
    optimum toward fewer features. Disable on imbalanced datasets if you score
    on accuracy / F1.

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
        estimator: Union[BaseEstimator, None] = None,
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
        # must_include: feature names (or integer indices for ndarray X) that MUST end up in support_. The optimiser only searches over the remaining features;
        # the final support_ is the union of must_include and the optimiser's pick. Differs from special_feature_indices which forces a fixed subset and short-circuits search.
        must_include: Union[Sequence, None] = None,
        # n_jobs>1 spawns joblib workers, one per fold. CRITICAL: gradient-boosting estimators (CatBoost, LightGBM, XGBoost) and tree ensembles (RandomForest)
        # already use native multi-threading; parallelising folds on top over-subscribes cores and SLOWS DOWN the run. When n_jobs>1 AND a multi-threaded estimator
        # is detected, we either auto-fallback to sequential (force_parallel=False) or pin the estimator's thread_count/n_jobs/n_threads to 1 (force_parallel=True).
        n_jobs: int = 1,
        force_parallel: bool = False,
        # must_exclude: symmetric counterpart of must_include. Named features are dropped at fit entry so they never enter the optimiser's universe and cannot
        # end up in support_. Use case: known target-leak columns (IDs, timestamps, post-hoc enrichments) the operator wants guaranteed excluded.
        must_exclude: Union[Sequence, None] = None,
        # leakage_corr_threshold: at fit entry, check |Pearson(X_i, y)| against this. Catches the most common leak (post-hoc enrichments, ID columns that encode
        # the target) before the model sees the leaked column. Set None to disable.
        leakage_corr_threshold: Union[float, None] = 0.95,
        # leakage_action: 'warn' only logs; 'exclude' auto-drops the column (treats it like must_exclude); 'raise' aborts the fit.
        leakage_action: str = "warn",
        # mbh_adaptive_threshold: cutoff (in MBH evaluation budget) below which the surrogate switches from CatBoost (~500ms fixed overhead) to sklearn ExtraTreesRegressor (~20ms).
        # The historical hardcoded value was 30; tune up when the outer estimator is so cheap that CB's fixed cost dominates even at larger budgets, tune down when ETR's 20-tree noise hurts selection.
        mbh_adaptive_threshold: int = 30,
        # feature_groups: maps group_name -> list of column names; support_ then reflects an all-or-nothing decision at the group level (all members in, or all out).
        # Resolves the "5 collinear copies" caveat at configuration level when the operator knows the groups (e.g. one-hot expansions).
        feature_groups: Union[dict, None] = None,
        # n_features_selection_rule: rule for picking n_features_ from cv_results_.
        #   'argmax' - argmax of (mean - lambda*std - feature_cost*N). On FLAT score curves around the optimum this collapses to the FIRST N visited near-max, often under-selecting.
        #   'one_se_max' - LARGEST N within 1 SE of the best mean; more robust on plateau, less likely to drop marginally-informative features.
        #   'one_se_min' - sklearn-canonical smallest N within 1 SE; parsimonious but vulnerable to plateau collapse.
        #   'auto' - 'one_se_max' when estimators= is a list (multi-estimator is plateau-prone), else 'argmax'.
        n_features_selection_rule: str = "auto",
        # Stability Selection (Meinshausen & Buhlmann 2010, JRSS-B). When True, replaces MBH+CV-fold-voting with bootstrap subsampling: B replicates of n/2 (no
        # replacement), fit estimator on each, count how often each feature appears in the top-K importance ranks. Feature is selected if frequency >= stability_threshold.
        # Provable family-wise error rate control. Preferred over CV-fold voting on small n / high p.
        stability_selection: bool = False,
        stability_n_bootstrap: int = 50,
        stability_threshold: float = 0.6,
        stability_top_k: Union[int, None] = None,  # default n_features // 4
        # estimators: list of BaseEstimators; on each CV fold fit ALL of them, gather FI from each, aggregate via the existing voting layer (Leaderboard treats each
        # per-estimator FI run as a separate column). Robust to single-estimator FI bias (LR favours scale, RF favours high-cardinality, CB favours continuous).
        # Supersedes ``estimator`` when set. Must all be the same type-family (classifier or regressor).
        # Do NOT parallelise across estimators - they use native multi-threading, and parallel folds is the layer where joblib lives.
        estimators: Union[Sequence, None] = None,
        # checkpoint_path: when set, RFECV pickles outer-loop state (evaluated_scores_*, optimizer, counters, best-so-far) after every iter; on a subsequent fit()
        # with a matching (X.shape, y.shape, columns) signature the loop resumes where it left off. The fitted-estimators dict is NOT persisted (CB / RF ensembles
        # would dominate file size). Atomic write: tmpfile + os.replace, so a crash mid-write cannot corrupt the previous checkpoint.
        checkpoint_path: Union[str, None] = None,
        # swap_top_k: after the main MBH loop converges, run K paired swap evaluations on the best subset - replace each of the K worst-FI features kept with each
        # of the K best-FI features dropped, accept any swap that improves the CV score. Cost: O(K) extra CV evaluations at the END only (classical SFFS would run
        # after every backward step but that's O(K)*iter_count, often impractical). Default 0 = disabled.
        # Swap evaluations use sklearn.cross_val_score directly and do NOT honour fit_params / val_cv / early stopping; use as a final-mile refinement.
        swap_top_k: int = 0,
        # optimizer_config: MBH fits an internal surrogate to predict score-per-nfeatures and pick the next candidate. On small problems (p<=30 with cheap outer
        # estimators like Ridge / LR) a 150-tree CatBoost surrogate dominates wall-clock. Auto-tune: when left None and the max-evaluations budget is small, use
        # a right-sized surrogate (ETR n_estimators=20 for budgets up to 30; CB iterations=50 up to 100; CB iterations=150 above).
        # Escape hatch: pass an explicit dict (e.g. ``{"model_name": "CBQ", "model_params": {"iterations": 50}}`` or any other MBHOptimizer kwarg subset) to override.
        optimizer_config: Union[dict, None] = None,
    ):

        # checks
        if frac is not None:
            if not (frac > 0.0 and frac < 1.0):
                raise ValueError(f"frac must be between 0 and 1, got {frac}")
            if verbose:
                logger.info("Using %s fraction of the training dataset.", frac)

        # max_refits=0 would be silently ignored by ``if max_refits and ...`` (0 is falsy). Reject explicitly.
        if max_refits is not None and max_refits < 1:
            raise ValueError(
                f"max_refits must be >= 1 (or None for unlimited); got {max_refits}. "
                f"To run zero iterations, just don't call fit()."
            )

        # cv=1 is degenerate (no train/test split possible).
        if isinstance(cv, int) and cv < 2:
            raise ValueError(
                f"cv must be >= 2 (or a CV splitter object); got cv={cv}. "
                f"k-fold CV requires at least 2 splits."
            )

        if stability_selection:
            if not (0.0 < stability_threshold <= 1.0):
                raise ValueError(
                    f"stability_threshold must be in (0, 1]; got {stability_threshold}."
                )
            if stability_n_bootstrap < 10 and verbose:
                logger.warning(
                    "RFECV: stability_n_bootstrap=%d is below the recommended "
                    "minimum of 10. Bootstrap voting is statistically meaningful "
                    "only with B >= 10; expect noisy / unstable selection.",
                    stability_n_bootstrap,
                )
            if stability_n_bootstrap < 1:
                raise ValueError(
                    f"stability_n_bootstrap must be >= 1; got {stability_n_bootstrap}."
                )

        if feature_groups:
            for _gname, _gmembers in feature_groups.items():
                if not _gmembers:
                    if verbose:
                        logger.warning(
                            "RFECV: feature_groups[%r] is empty; this group "
                            "will have no effect on selection.", _gname,
                        )

        if leakage_action not in ("warn", "exclude", "raise"):
            raise ValueError(
                f"leakage_action must be 'warn', 'exclude', or 'raise'; "
                f"got {leakage_action!r}."
            )

        if n_features_selection_rule not in ("auto", "argmax", "one_se_min", "one_se_max"):
            raise ValueError(
                f"n_features_selection_rule must be 'auto', 'argmax', "
                f"'one_se_min', or 'one_se_max'; got {n_features_selection_rule!r}."
            )

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)
        self.signature = None

    def _sffs_swap_pass(
        self, X, y, estimator, cv, scoring,
        best_nfeatures: int, best_score_ref: float,
        selected_features_per_nfeatures: dict, feature_importances: dict,
        original_features, evaluated_scores_mean: dict,
        evaluated_scores_std: dict, verbose: int, ndigits: int,
    ) -> None:
        """Replace each of the K worst-FI kept features with each of the K best-FI dropped features and accept any swap that improves the
        CV score. Mutates selected_features_per_nfeatures / evaluated_scores_* in place. Cost: O(K) extra CV evaluations executed via
        sklearn.model_selection.cross_val_score.
        """
        from sklearn.model_selection import cross_val_score

        K = int(self.swap_top_k)
        best_set = list(selected_features_per_nfeatures.get(best_nfeatures, []))
        if len(best_set) < 1:
            return

        # Aggregate FI across all runs (mean of non-NaN values per feature).
        from collections import defaultdict
        fi_acc = defaultdict(list)
        for _key, _fi in feature_importances.items():
            for feat, val in _fi.items():
                try:
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        fi_acc[feat].append(float(val))
                except (TypeError, ValueError):
                    continue
        fi_mean = {f: float(np.mean(v)) for f, v in fi_acc.items() if v}

        # Worst K kept (features with no FI history get 0).
        kept_sorted = sorted(best_set, key=lambda f: fi_mean.get(f, 0.0))
        swap_out = kept_sorted[:K]
        # Best K dropped.
        not_in_set = [f for f in original_features if f not in set(best_set)]
        not_sorted = sorted(not_in_set, key=lambda f: fi_mean.get(f, 0.0), reverse=True)
        swap_in = not_sorted[:K]

        cur_set = list(best_set)
        cur_score = float(best_score_ref)
        n_swaps_accepted = 0
        for out_f, in_f in zip(swap_out, swap_in):
            trial_set = [in_f if f == out_f else f for f in cur_set]
            try:
                if isinstance(X, pd.DataFrame):
                    trial_X = X[trial_set]
                else:
                    idx = [list(original_features).index(f) for f in trial_set]
                    trial_X = X[:, idx]
                trial_scores = cross_val_score(
                    clone(estimator), trial_X, y, cv=cv,
                    scoring=scoring, n_jobs=1,
                )
            except Exception as _exc:
                if verbose:
                    logger.warning(
                        "SFFS swap %s -> %s evaluation failed (%s); skipping pair.",
                        out_f, in_f, _exc,
                    )
                continue
            if trial_scores is None or len(trial_scores) == 0:
                continue
            trial_mean = float(np.nanmean(trial_scores))
            trial_std = float(np.nanstd(trial_scores))
            if trial_mean > cur_score:
                if verbose:
                    logger.info(
                        f"SFFS swap accepted: {out_f} -> {in_f} improved "
                        f"{cur_score:.{ndigits}f} -> {trial_mean:.{ndigits}f}"
                    )
                cur_set = trial_set
                cur_score = trial_mean
                n_swaps_accepted += 1
                # |trial_set| == |best_set|, so this overwrites the entry for that nfeatures count.
                _n = len(trial_set)
                selected_features_per_nfeatures[_n] = trial_set
                evaluated_scores_mean[_n] = trial_mean
                evaluated_scores_std[_n] = trial_std

        if verbose:
            logger.info(
                f"SFFS swap pass: {n_swaps_accepted}/{len(swap_out)} paired "
                f"swaps accepted (best score {cur_score:.{ndigits}f})."
            )

    # cv_results_ stays a dict-of-arrays for sklearn parity. The DataFrame property below is purely additive and avoids the silent semantics
    # change that "0 in series" introduces vs "0 in list" (Series.__contains__ checks the INDEX, not values).
    @property
    def cv_results_df_(self) -> "pd.DataFrame":
        """Return cv_results_ as a pd.DataFrame for tabular operations (sort_values, query, plot, to_csv). Built lazily on access; raises if fit() has not run."""
        if not hasattr(self, "cv_results_") or "nfeatures" not in self.cv_results_:
            raise ValueError(
                "cv_results_df_ requires fit() to have been called and "
                "cv_results_ to be populated."
            )
        return pd.DataFrame(self.cv_results_)

    # sklearn 1.6 deprecated _get_tags / _more_tags in favour of __sklearn_tags__, which returns a sklearn.utils.Tags dataclass carrying
    # classifier/regressor type info, input contract, and request metadata. Without overriding it, downstream sklearn helpers
    # (estimator_html_repr, check_is_fitted, set_config(transform_output=...)) see RFECV as a generic transformer with no estimator-type
    # tag and may mis-route routing requests. Delegate to the wrapped estimator's tags.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Multi-estimator path uses self.estimators[0] as the type-determining estimator; single-estimator path uses self.estimator.
        inner = None
        if getattr(self, "estimators", None):
            try:
                inner = list(self.estimators)[0]
            except (TypeError, IndexError):
                inner = None
        if inner is None:
            inner = getattr(self, "estimator", None)
        if inner is not None and hasattr(inner, "__sklearn_tags__"):
            try:
                inner_tags = inner.__sklearn_tags__()
                tags.estimator_type = inner_tags.estimator_type
                tags.classifier_tags = inner_tags.classifier_tags
                tags.regressor_tags = inner_tags.regressor_tags
                tags.target_tags = inner_tags.target_tags
            except (AttributeError, TypeError):
                pass
        return tags

    # Schema version of the on-disk checkpoint dict. Bump on any breaking change to the keys saved by _save_checkpoint; _load_checkpoint
    # refuses mismatched versions and starts fresh.
    _CHECKPOINT_VERSION = 1

    def _save_checkpoint(self, state: dict) -> None:
        """Atomically dump RFECV outer-loop state to ``self.checkpoint_path``.

        Atomicity: write to a sibling tempfile then ``os.replace`` it onto the target path. ``os.replace`` is atomic on POSIX and on
        Windows (Python >=3.3), so a crash mid-write cannot corrupt the prior checkpoint.
        """
        import os
        import pickle
        import tempfile

        path = self.checkpoint_path
        if not path:
            return
        dir_name = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".rfecv_ckpt_", dir=dir_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _load_checkpoint(self) -> Union[dict, None]:
        """Return the checkpoint dict iff present, version-compatible, and signature-matching self.signature; otherwise return None.

        On any pickle error (truncated file, missing class, etc.) log a warning and return None so the caller starts fresh.
        """
        import os
        import pickle

        path = self.checkpoint_path
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                state = pickle.load(fh)
        except (pickle.PickleError, EOFError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "RFECV: checkpoint at %s could not be loaded (%s); starting from scratch.",
                path, exc,
            )
            return None
        if not isinstance(state, dict):
            logger.warning(
                "RFECV: checkpoint at %s is not a dict (got %s); starting from scratch.",
                path, type(state).__name__,
            )
            return None
        if state.get("version") != self._CHECKPOINT_VERSION:
            logger.warning(
                "RFECV: checkpoint at %s has version %s but expected %s; starting from scratch.",
                path, state.get("version"), self._CHECKPOINT_VERSION,
            )
            return None
        return state

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray], groups: Union[pd.Series, np.ndarray] = None, sample_weight: Union[np.ndarray, pd.Series, None] = None, **fit_params):
        # sample_weight, when provided, is sliced per CV fold and threaded into both the cloned estimator's
        # ``fit(..., sample_weight=fold_train_w)`` call (if the estimator advertises support) and into the
        # sklearn scorer's ``__call__(..., sample_weight=fold_test_w)`` (if the scorer accepts the kwarg).
        # Default None preserves the legacy code path byte-for-byte (regression sentry); the gating flag
        # ``FeatureSelectionConfig.use_sample_weights_in_fs`` decides whether the caller forwards weights here.
        self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if self._fit_sample_weight_ is not None:
            _n_rows_for_sw = X.shape[0] if hasattr(X, "shape") else len(X)
            if self._fit_sample_weight_.ndim != 1:
                raise ValueError(f"RFECV.fit sample_weight must be 1-D, got shape {self._fit_sample_weight_.shape}")
            if self._fit_sample_weight_.shape[0] != _n_rows_for_sw:
                raise ValueError(f"RFECV.fit sample_weight length {self._fit_sample_weight_.shape[0]} != n_rows {_n_rows_for_sw}")
            if not np.all(np.isfinite(self._fit_sample_weight_)) or (self._fit_sample_weight_ < 0).any():
                raise ValueError("RFECV.fit sample_weight must be finite and non-negative")

        # Polars -> pandas at entry. RFECV uses pandas / numpy idioms throughout (KFold.split, current_features.index(...), passthrough_cols).
        # Inner estimators (notably CatBoost) crash on polars Enum columns, so convert once here and let every downstream caller see pandas.
        # Before lossy conversion, stash a "polars-detected monotonic datetime axis" hint so the CV auto-detect block below can route
        # to TimeSeriesSplit. After to_pandas() the original polars schema is gone and the .index becomes a plain RangeIndex.
        _polars_time_series_hint = False
        if isinstance(X, pl.DataFrame):
            try:
                _dt_cols = [
                    n for n, d in X.schema.items()
                    if d in (pl.Datetime, pl.Date) or str(d).startswith(("Datetime", "Date"))
                ]
                if len(_dt_cols) == 1:
                    _col = X.get_column(_dt_cols[0])
                    if _col.is_sorted(descending=False) and _col.null_count() == 0:
                        _polars_time_series_hint = True
            except Exception:
                pass
            try:
                # Polars 1.x: extension-array kwarg only; library forwards split_blocks/self_destruct
                # internally. Passing them explicitly on 1.x raises TypeError("got multiple values for
                # keyword argument 'split_blocks'") and falls through to bare .to_pandas() (lossy:
                # pl.Enum / pl.Categorical degrade to object).
                X = X.to_pandas(use_pyarrow_extension_array=True)
            except TypeError:
                try:
                    # Legacy polars 0.x signature.
                    X = X.to_pandas(use_pyarrow_extension_array=True, split_blocks=True, self_destruct=True)
                except TypeError:
                    X = X.to_pandas()
        if isinstance(y, pl.Series):
            y = y.to_pandas()

        # Reject pathological y / X early instead of letting sklearn raise opaque errors deep in the splitter or estimator.
        try:
            y_arr = np.asarray(y)
        except Exception as exc:
            raise ValueError(f"y must be array-like; got {type(y).__name__}: {exc}") from exc
        if y_arr.size == 0:
            raise ValueError("y is empty; nothing to fit.")
        # NaN / Inf in y are silent miscompute traps in sklearn folds.
        if y_arr.dtype.kind in "fc":
            n_nan_y = int(np.isnan(y_arr).sum())
            n_inf_y = int(np.isinf(y_arr).sum())
            if n_nan_y or n_inf_y:
                raise ValueError(
                    f"y contains {n_nan_y} NaN and {n_inf_y} +/-inf values. "
                    f"sklearn CV splitters silently mishandle these. Drop or "
                    f"impute these rows before passing y to RFECV."
                )
        # Single-class y for classification is a fold-collapse trap.
        if is_classifier(self.estimator if self.estimator is not None
                         else (self.estimators[0] if self.estimators else None)):
            unique_y = np.unique(y_arr)
            if len(unique_y) < 2:
                raise ValueError(
                    f"y has only {len(unique_y)} unique class(es) "
                    f"({unique_y.tolist()}). Classification CV requires at "
                    f"least 2 classes. Check that y is not constant or that "
                    f"upstream filtering didn't drop the minority class."
                )
            # Minority-class size must support the requested CV.
            class_counts = np.bincount(y_arr.astype(int)) if y_arr.dtype.kind in "iu" else None
            if class_counts is not None and len(class_counts) > 0:
                min_class = int(class_counts[class_counts > 0].min())
                cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)
                if min_class < cv_n:
                    raise ValueError(
                        f"Minority class has {min_class} samples but cv={cv_n}. "
                        f"StratifiedKFold requires at least n_splits samples per "
                        f"class. Reduce cv or oversample the minority class."
                    )

        # must_include + must_exclude intersection is a confusing config error.
        if self.must_include and self.must_exclude:
            mi_set = set(self.must_include)
            me_set = set(self.must_exclude)
            overlap = mi_set & me_set
            if overlap:
                raise ValueError(
                    f"must_include and must_exclude both contain {sorted(overlap)}. "
                    f"Resolve the conflict in your config."
                )

        # X-side input checks. Run after y validation so common operator mistakes surface clearly at fit entry.
        if isinstance(X, pd.DataFrame):
            # Estimator behaviour on Inf is undefined - LR crashes, CB silently treats as huge.
            try:
                _num = X.select_dtypes(include="number")
                if _num.size > 0:
                    _inf_mask = np.isinf(_num.to_numpy())
                    if _inf_mask.any():
                        _inf_cols = _num.columns[_inf_mask.any(axis=0)].tolist()
                        raise ValueError(
                            f"X contains +/-Inf values in column(s) {_inf_cols[:10]}. "
                            f"Estimator behaviour on Inf is undefined. Drop or "
                            f"clip these values before fit()."
                        )
            except (TypeError, ValueError) as exc:
                if "+/-Inf" in str(exc):
                    raise

            # Tree-based estimators handle NaN; linear models don't.
            if getattr(self, "verbose", 0):
                _nan_count = int(X.isna().to_numpy().sum())
                if _nan_count > 0:
                    logger.warning(
                        "RFECV: X has %d NaN cells. Tree-based estimators "
                        "(RF/CB/XGB/HGBM) handle NaN; linear models (LR, "
                        "Ridge, Lasso) do NOT and will crash on .fit(). "
                        "Pre-impute via SimpleImputer / KNNImputer if using "
                        "linear estimators.", _nan_count,
                    )

            _obj_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            if _obj_cols:
                _user_cats = set(self.cat_features or [])
                _unhandled = [c for c in _obj_cols if c not in _user_cats]
                if _unhandled and getattr(self, "verbose", 0):
                    logger.warning(
                        "RFECV: %d object/string/category column(s) %s have "
                        "NOT been listed in cat_features=. CB/XGB will crash "
                        "on string columns; LR will fail on .fit(). Either "
                        "encode them upstream or pass via cat_features.",
                        len(_unhandled), _unhandled[:10],
                    )

        # n_samples < 2 * cv breaks every k-fold split.
        cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)
        if X.shape[0] < 2 * cv_n:
            raise ValueError(
                f"n_samples={X.shape[0]} < 2 * cv ({cv_n}); each fold would "
                f"have <2 train samples. Reduce cv or get more data."
            )

        # p >> n with no max_nfeatures cap - MBH search space is O(p) and each iter is a CV fit, so runtime is unbounded.
        if X.shape[1] >= 5000 and self.max_nfeatures is None and getattr(self, "verbose", 0):
            logger.warning(
                "RFECV: p=%d features with no max_nfeatures cap. The "
                "MBH search space is O(p) and each iter is a CV fit; runtime "
                "is unbounded. Set max_nfeatures=300 (or use "
                "stability_selection=True / importance_getter='knockoff' "
                "for sub-second selection on this scale).",
                X.shape[1],
            )

        # max_runtime_mins below 1 second is almost certainly a units-confusion mistake.
        if self.max_runtime_mins is not None:
            if self.max_runtime_mins < 0:
                raise ValueError(f"max_runtime_mins must be >= 0; got {self.max_runtime_mins}")
            if 0 < self.max_runtime_mins < (1.0 / 60.0) and getattr(self, "verbose", 0):
                logger.warning(
                    "RFECV: max_runtime_mins=%.4f is < 1 second. Did you mean "
                    "to pass seconds instead of minutes? RFECV will likely "
                    "exit after iter 1 with an under-fit selection.",
                    self.max_runtime_mins,
                )

        # cv >= n_samples (LeaveOneOut on small data) gives 1-sample test folds where most metrics are undefined.
        if isinstance(self.cv, int) and self.cv >= X.shape[0] and getattr(self, "verbose", 0):
            logger.warning(
                "RFECV: cv=%d >= n_samples=%d (effectively LeaveOneOut). "
                "Per-fold test set has 1 sample; ROC AUC and many other "
                "metrics are undefined and will produce NaN scores. Use "
                "cv <= n_samples / 5 for stable scoring.",
                self.cv, X.shape[0],
            )

        # Drop zero-variance / all-null columns BEFORE recording ``feature_names_in_``. Otherwise a constant column can land in support_ and
        # downstream pipeline steps that silently drop it (e.g. SimpleImputer with keep_empty_features=False) cause transform-time column-set
        # drift. Vectorised across ALL dtypes via DataFrame.nunique() so constant categorical / string / bool columns are also caught.
        if isinstance(X, pd.DataFrame) and X.shape[1] > 0:
            try:
                nunique = X.nunique(dropna=True)
                degenerate = nunique[nunique <= 1].index.tolist()
            except TypeError:
                # Fallback for exotic dtypes that nunique() can't hash.
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

        # Drop exact-duplicate columns (numeric AND categorical). Without this, RFECV's voting splits the importance of a duplicated feature across all copies,
        # biasing selection toward isolated noise features whose FI isn't diluted. ``pandas.util.hash_array`` is dtype-agnostic and treats NaN as a single sentinel,
        # so a real ``-1.234e308`` value no longer collides with NaN the way the prior ``np.nan_to_num(...).tobytes()`` path did; categorical columns are factorised
        # to integer codes before hashing so two semantically-identical category sequences with different ordered category dictionaries still dedup.
        if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
            try:
                from pandas.util import hash_array as _hash_array

                _hashes: dict[bytes, str] = {}
                _to_drop = []
                for _col in X.columns:
                    _ser = X[_col]
                    _dtype = _ser.dtype
                    if pd.api.types.is_numeric_dtype(_dtype) or pd.api.types.is_bool_dtype(_dtype):
                        _arr = _ser.to_numpy()
                    elif isinstance(_dtype, pd.CategoricalDtype) or _dtype == object or pd.api.types.is_string_dtype(_dtype):
                        # ``factorize`` is the canonical pandas pathway for collapsing label semantics into a code sequence dedup can compare; ``use_na_sentinel=True`` keeps NaN distinct from any real label without forcing a fillna copy.
                        _codes, _ = pd.factorize(_ser, use_na_sentinel=True)
                        _arr = _codes
                    else:
                        continue
                    _key = bytes(_hash_array(_arr))
                    if _key in _hashes:
                        _to_drop.append(_col)
                    else:
                        _hashes[_key] = _col
                if _to_drop:
                    if getattr(self, "verbose", 0):
                        logger.info(
                            "RFECV: dropping %d duplicate column(s) (exact-equal "
                            "to another column already kept) before fit: %s. "
                            "Pass them via ``feature_groups`` if you want "
                            "all-or-nothing group decisions.",
                            len(_to_drop), _to_drop,
                        )
                    X = X.drop(columns=_to_drop)
            except (TypeError, ValueError):
                # Non-hashable dtype - skip dedup.
                pass

        # must_exclude: drop named columns at fit entry so they never enter the optimiser's universe.
        if self.must_exclude and isinstance(X, pd.DataFrame):
            _drop = [c for c in self.must_exclude if c in X.columns]
            if _drop:
                if getattr(self, "verbose", 0):
                    logger.info(
                        "RFECV: must_exclude drops %d column(s) at fit entry: %s",
                        len(_drop), _drop,
                    )
                X = X.drop(columns=_drop)
            _missing = [c for c in self.must_exclude if c not in (list(X.columns) + _drop)]
            # Missing names are silently ignored: the column is already absent so the exclusion goal is satisfied.

        # Target-leakage early warning: Pearson correlation between each numeric feature and y.
        # Common leak shapes: ID columns that encode the target, post-hoc enrichments, target-encoded categoricals computed on the full set.
        _suspicious: list = []
        if self.leakage_corr_threshold is not None and isinstance(X, pd.DataFrame) and X.shape[0] >= 30:
            try:
                _y_arr = np.asarray(y, dtype=float).ravel()
                if _y_arr.size == X.shape[0] and not np.all(np.isnan(_y_arr)):
                    _y_std = float(np.nanstd(_y_arr))
                    if _y_std > 1e-12:
                        # Use is_numeric_dtype so pandas nullable extension dtypes (Int8/.../Float64) and boolean dtypes also enter the
                        # leakage scan; select_dtypes(include="number") silently skips them.
                        from pandas.api.types import is_numeric_dtype, is_bool_dtype
                        _numeric_cols = [c for c in X.columns
                                         if is_numeric_dtype(X[c]) or is_bool_dtype(X[c])]
                        for _c in _numeric_cols:
                            _x_arr = np.asarray(X[_c].values, dtype=float)
                            _mask = np.isfinite(_x_arr) & np.isfinite(_y_arr)
                            if _mask.sum() < 10:
                                continue
                            _x_std = float(np.nanstd(_x_arr[_mask]))
                            if _x_std < 1e-12:
                                continue
                            _corr = float(np.corrcoef(_x_arr[_mask], _y_arr[_mask])[0, 1])
                            if abs(_corr) >= float(self.leakage_corr_threshold):
                                _suspicious.append((_c, round(_corr, 4)))
                        # The outer try/except catches only TypeError/ValueError raised by the corr computation, NOT our intentional 'raise'
                        # action - so we collect findings here and raise OUTSIDE the try.
            except (TypeError, ValueError):
                pass
            if _suspicious:
                _msg = (
                    f"RFECV: {len(_suspicious)} feature(s) have "
                    f"|Pearson(X, y)| >= {self.leakage_corr_threshold}, "
                    f"likely target leakage. Inspect: {_suspicious[:20]}. "
                    f"To suppress, set leakage_corr_threshold=None or "
                    f"list these in must_exclude."
                )
                _action = getattr(self, "leakage_action", "warn")
                if _action == "raise":
                    raise ValueError(_msg + " (leakage_action='raise')")
                elif _action == "exclude":
                    _leaky_cols = [c for c, _ in _suspicious if c in X.columns]
                    if _leaky_cols:
                        logger.warning(
                            _msg + " (leakage_action='exclude' - dropping these columns)"
                        )
                        X = X.drop(columns=_leaky_cols)
                else:
                    logger.warning(_msg)

        # Inputs/outputs signature. Shape alone isn't enough - two datasets with identical (n, p) but different column identities must
        # trigger a retrain, otherwise self.support_ silently applies stale column selections. y-content is folded in via a blake2b
        # 16-byte digest because two semantically-different targets of the same length and shape (e.g. column-A binary vs column-B
        # binary picked off the same frame) used to replay the prior fit's support_; without the y-hash a per-target FS loop reusing
        # one RFECV instance silently selected features for whichever target arrived first.
        if isinstance(X, pd.DataFrame):
            columns_key = tuple(map(str, X.columns.tolist()))
        else:
            columns_key = ("__ndarray__", int(X.shape[1]))
        try:
            _y_arr = np.ascontiguousarray(
                y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            )
            _y_hash = hashlib.blake2b(_y_arr.tobytes(), digest_size=16).hexdigest()
        except (TypeError, ValueError):
            # Object-dtype y or otherwise non-bytes-castable; fall back to a stringified per-element hash that still discriminates content.
            _y_hash = hashlib.blake2b(
                ",".join(map(str, np.asarray(y).ravel().tolist())).encode("utf-8"),
                digest_size=16,
            ).hexdigest()
        # X-content sample (10 evenly-spaced rows, all columns) so two RUNS on
        # different X with identical shape + column names cannot collide on the
        # checkpoint signature. Cost is O(n_cols * 10) value reads -- negligible
        # vs a single RFECV iteration. Without this, a user reusing a checkpoint
        # across experiments with same schema gets silent stale resume.
        try:
            _n = int(X.shape[0])
            if _n > 0:
                _step = max(1, _n // 10)
                _positions = [i * _step for i in range(10) if i * _step < _n]
                if isinstance(X, pd.DataFrame):
                    _x_sample_arr = X.iloc[_positions].to_numpy()
                else:
                    _x_sample_arr = np.asarray(X)[_positions]
                _x_hash = hashlib.blake2b(
                    np.ascontiguousarray(_x_sample_arr.astype(np.float64, copy=False, casting="unsafe") if _x_sample_arr.dtype.kind in "biufc" else np.asarray(_x_sample_arr, dtype=str)).tobytes(),
                    digest_size=12,
                ).hexdigest()
            else:
                _x_hash = "empty"
        except (TypeError, ValueError):
            _x_hash = hashlib.blake2b(
                repr(np.asarray(X).ravel()[:100].tolist()).encode("utf-8"),
                digest_size=12,
            ).hexdigest()
        signature = (X.shape, y.shape, columns_key, _y_hash, _x_hash)
        # Invalidate stale support_/cache at fit entry so a partial-fit failure cannot leave a previous-fit's selection silently in place.
        # The cache is rebuilt below only on a successful path.
        self._selected_cols_cache = None
        if self.skip_retraining_on_same_shape:
            if signature == self.signature:
                if self.verbose:
                    logger.info("Skipping retraining on the same inputs signature %s", signature)
                return self

        # Stability selection branch: uses bootstrap voting instead of the MBH+CV-fold-voting search. Returns early after setting
        # support_ / n_features_ / cv_results_-shim / feature_names_in_ etc.
        if self.stability_selection:
            return self._fit_stability_selection(X=X, y=y, signature=signature)

        # ``estimators`` (list) supersedes the singular ``estimator``. Work with a list internally; singular path is a len-1 list.
        # Score per fold = mean across estimators; FI runs stored under separate keys so the voting layer treats each estimator's
        # importance as an independent "run".
        estimators_list = list(self.estimators) if self.estimators else (
            [self.estimator] if self.estimator is not None else []
        )
        if not estimators_list:
            raise ValueError("RFECV requires either estimator= or estimators=.")
        # ``estimator`` retained for legacy code paths inside fit() that need a single object for type-dispatch (CV stratification,
        # scoring, importance_getter resolution). First estimator is the representative.
        estimator = estimators_list[0]
        # Extract suite-level ``timestamps`` hint BEFORE we shadow ``fit_params`` with ``self.fit_params``;
        # the time-series auto-detect block below picks it up to upgrade plain int cv -> TimeSeriesSplit
        # for callers whose X has no DatetimeIndex / polars datetime column but who know the row order
        # is temporal (e.g. epoch-seconds in a separate array).
        _ts_hint_from_caller = fit_params.pop("timestamps", None) if isinstance(fit_params, dict) else None
        fit_params = copy.copy(self.fit_params) if self.fit_params else {}
        if _ts_hint_from_caller is not None:
            fit_params["timestamps"] = _ts_hint_from_caller
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
        # When random_state is None, derive a stable seed from the signature so re-fits on the SAME data are deterministic. Otherwise
        # self._rng is reseeded from system entropy on every fit, breaking reproducibility silently.
        if random_state is None:
            _seed = abs(hash(signature)) % (2 ** 32)
            self._rng = np.random.default_rng(_seed)
        else:
            self._rng = np.random.default_rng(random_state)
        leave_progressbars = self.leave_progressbars
        verbose = self.verbose
        show_plot = self.show_plot
        cat_features = self.cat_features
        # Strip cat_features whose columns have already been numerically encoded by an upstream pipeline step (e.g. CatBoostEncoder turning
        # cat_0 -> float). With cat_0 still in cat_features, inner CatBoost.fit raises ``Invalid type for cat_feature``. Restrict to columns
        # whose dtype is still categorical/object - those are the ones CB/XGB can actually consume. LOCAL only - never mutate
        # self.cat_features (back-to-back fits across encoded/un-encoded frames must each pick the right subset for their X).
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

        # Resolve effective n_jobs. Multi-threaded estimators (CB/LGB/XGB/RF/...) already use all cores natively; parallelising folds on top
        # oversubscribes and SLOWS DOWN. Auto-fallback to sequential unless force_parallel=True (then pin inner threads to 1 in _eval_fold).
        n_jobs_effective = int(self.n_jobs) if self.n_jobs else 1
        if n_jobs_effective < 0:
            # joblib semantics: -1 = all cores.
            try:
                import os as _os
                n_jobs_effective = max(1, (_os.cpu_count() or 1))
            except Exception:
                n_jobs_effective = 1
        _is_multithreaded = _detect_multithreaded(estimator)
        if n_jobs_effective > 1 and _is_multithreaded and not self.force_parallel:
            if verbose:
                logger.info(
                    "RFECV: n_jobs=%d requested, but %s already uses native "
                    "multi-threading. Auto-falling back to sequential CV folds "
                    "to avoid core oversubscription. Pass ``force_parallel=True`` "
                    "to override (will pin inner threads to 1).",
                    n_jobs_effective, type(estimator).__name__,
                )
            n_jobs_effective = 1
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

        # must_include partition: the optimiser only sees the complement; pinned features are glued back into support_ at the end.
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

        if cv is None or str(cv).isnumeric():
            if cv is None:
                cv = 3
            # Time-series auto-detect: a monotonic datetime axis means KFold-style shuffles would leak future into past; TimeSeriesSplit is
            # the correct choice. Triggers ONLY when groups is None (group-aware splits already handle temporal grouping) and the
            # detected datetime axis is strictly increasing (avoid TSS on randomly-shuffled datetime data, which has no ordering meaning).
            # Pandas path: DatetimeIndex.is_monotonic_increasing. Polars path: scan for a single datetime / date column that is
            # monotonically increasing (polars has no row index; the time axis is necessarily a column).
            _is_time_series = False
            # Suite-level timestamps hint: callers pass ``timestamps=`` via fit_params (1-D monotonic
            # array-like). This catches the case where X has no DatetimeIndex / no polars datetime col
            # but the suite knows the row order is temporal (e.g. integer epoch seconds in a separate
            # array). Honour the hint regardless of X's schema.
            _ts_hint = fit_params.pop("timestamps", None) if isinstance(fit_params, dict) else None
            if groups is None and _ts_hint is not None:
                try:
                    _ts_arr = np.asarray(_ts_hint)
                    if _ts_arr.ndim == 1 and _ts_arr.size == (X.shape[0] if hasattr(X, "shape") else len(X)):
                        # Monotonic-non-decreasing -> time axis.
                        if bool(np.all(_ts_arr[1:] >= _ts_arr[:-1])):
                            _is_time_series = True
                except (TypeError, ValueError):
                    pass
            # Polars-input path: the schema-level monotonic-datetime check happens BEFORE the to_pandas() at fit entry; the hint is set
            # there so the conversion doesn't erase the per-column polars dtype information needed for unambiguous detection.
            if not _is_time_series and groups is None and _polars_time_series_hint:
                _is_time_series = True
            elif groups is None and isinstance(X, pd.DataFrame):
                _idx = X.index
                if isinstance(_idx, pd.DatetimeIndex) and _idx.is_monotonic_increasing:
                    _is_time_series = True
            elif groups is None:
                try:
                    import polars as _pl
                    if isinstance(X, _pl.DataFrame):
                        _dt_cols = [
                            n for n, d in X.schema.items()
                            if d in (_pl.Datetime, _pl.Date) or str(d).startswith(("Datetime", "Date"))
                        ]
                        # Exactly one datetime column = unambiguous time axis; multiple datetimes would require the
                        # caller to disambiguate via an explicit cv= (we won't guess which column orders the rows).
                        if len(_dt_cols) == 1:
                            _col = X.get_column(_dt_cols[0])
                            if _col.is_sorted(descending=False) and _col.null_count() == 0:
                                _is_time_series = True
                except ImportError:
                    pass
            if _is_time_series:
                cv = TimeSeriesSplit(n_splits=cv)
                if verbose:
                    logger.info(
                        "Using cv=%s (auto-detected from monotonic DatetimeIndex; "
                        "pass cv=KFold(...) explicitly to override).", cv,
                    )
            elif is_classifier(estimator):
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
            if verbose and not _is_time_series:
                logger.info("Using cv=%s", cv)

        # Expose the resolved splitter for introspection (auto-detect tests / callers checking which CV strategy was picked).
        self.cv_ = cv

        if early_stopping_val_nsplits:
            try:
                # sklearn KFold-family raises ValueError if random_state is set while shuffle=False. Drop random_state in that case.
                _cv_params = dict(cv.get_params())
                if _cv_params.get("shuffle") is False and _cv_params.get("random_state") is not None:
                    _cv_params["random_state"] = None
                _cv_params["n_splits"] = early_stopping_val_nsplits
                val_cv = type(cv)(**_cv_params)
            except (AttributeError, TypeError, ValueError):
                # Fallback for LeaveOneOut / iterator-based custom CVs whose get_params() may not exist or whose n_splits is computed
                # from the data. The setattr-style fallback below may silently write to a meaningless attribute and the CV runs with its
                # original split count, ignoring the user's early_stopping_val_nsplits request - warn loudly.
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
                early_stopping_rounds = 20
        else:
            val_cv = None

        progressbar_prefix = "RFECV:"
        iters_pbar = tqdmu(
            desc=progressbar_prefix,
            leave=leave_progressbars,
            total=min(len(original_features) + 1, max_refits) if max_refits else len(original_features) + 1,
        )

        suppress_irritating_3rdparty_warnings()

        if scoring is None:
            if is_classifier(estimator):
                logger.info(f"Scoring omitted, using probabilistic_multiclass_error by default.")
                # response_method='predict_proba' for sklearn 1.4+ (needs_proba is deprecated).
                scoring = make_scorer(score_func=compute_probabilistic_multiclass_error, response_method="predict_proba", greater_is_better=False)
            elif is_regressor(estimator):
                logger.info(f"Scoring omitted, using mean_squared_error by default.")
                scoring = make_scorer(score_func=mean_squared_error, greater_is_better=False)
            else:
                raise ValueError(f"Appropriate scoring not known for estimator type: {estimator}")
            self.scoring = scoring

        if verbose:
            logger.info("Scoring=%s", scoring)

        if isinstance(estimator, Pipeline):
            estimator_type = type(estimator.steps[-1][1]).__name__
        else:
            estimator_type = type(estimator).__name__

        if importance_getter is None or importance_getter == "auto":
            # Defer the dispatch to get_feature_importances so it can look at the FITTED model's attributes; a hardcoded
            # ``LogisticRegression -> coef_, else -> feature_importances_`` crashes on Ridge / Lasso / SVC(linear) / SGDClassifier etc.
            importance_getter = "auto"

        nsteps = 0
        dummy_scores = []
        fitted_estimators = {}
        selected_features_per_nfeatures = {}

        if top_predictors_search_method == OptimumSearch.ModelBasedHeuristic:
            # Default plotting mode is 'No': OnScoreImprovement calls plt.show() inside the optimizer on every score improvement, which
            # blocks pytest / headless runs indefinitely (Qt event loop). Users must opt in explicitly.
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

            # Adaptive MBH surrogate. CatBoost has a fixed ~500ms per-fit overhead (Python<->C++ marshalling, roughly independent of
            # n_estimators), which dominates wall-clock on tiny RFECV problems where the outer estimator fits in <10ms. Switch to a sklearn
            # ExtraTreesRegressor surrogate when the evaluation budget is small: ETR n_estimators=20 fits in ~20ms with equivalent quality
            # on the 1D score curve. Quantile-style uncertainty via per-tree-prediction std (Breiman 2001 OOB variance estimate).
            #
            # Decision tree:
            #   evaluation budget <= 30:  ETR n_estimators=20
            #   31..100:                  CatBoost iterations=50
            #   >100:                     CatBoost iterations=150
            #
            # Users override via ``optimizer_config={"model_name":..., "model_params": {...}}``.
            _search_space_size = (
                min(self.max_nfeatures, len(original_features)) + 1
                if self.max_nfeatures
                else len(original_features) + 1
            )
            _max_evals_budget = (
                min(max_refits, _search_space_size)
                if max_refits
                else _search_space_size
            )
            _user_cfg = dict(self.optimizer_config) if self.optimizer_config else {}
            _user_model_name = _user_cfg.pop("model_name", None)
            _user_model_params = dict(_user_cfg.pop("model_params", {}) or {})
            # Adaptive surrogate threshold is operator-tunable: 30 was the historical hardcoded crossover, but very-cheap outer estimators benefit from higher cutoffs (CB overhead still dominates at budgets in the 30..60 range) while heavy outer estimators may prefer the noisier ETR even lower.
            _mbh_adaptive_threshold = getattr(self, "mbh_adaptive_threshold", 30)
            if _user_model_name is None:
                if _max_evals_budget <= _mbh_adaptive_threshold:
                    _auto_model_name = "ETR"
                else:
                    _auto_model_name = "CBQ"
                    if "iterations" not in _user_model_params:
                        _user_model_params["iterations"] = 50 if _max_evals_budget <= 100 else 150
                _model_name = _auto_model_name
            else:
                _model_name = _user_model_name
                # Only fill iterations default when the user picked a CatBoost-family surrogate.
                if _model_name in ("CBQ", "CB") and "iterations" not in _user_model_params:
                    _user_model_params["iterations"] = 50 if _max_evals_budget <= 100 else 150
            _mbh_kwargs = dict(
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
                model_name=_model_name,
                model_params=_user_model_params,
            )
            # Apply the rest of optimizer_config last so user's explicit kwargs (plotting=..., direction=..., etc.) override our defaults.
            _mbh_kwargs.update(_user_cfg)
            Optimizer = MBHOptimizer(**_mbh_kwargs)
        else:
            Optimizer = None

        prev_score, prev_nfeatures = -np.inf, 0
        n_noimproving_iters = 0
        best_nfeatures = 0
        best_iter = 0
        # -inf is the only safe initial value for an argmax-style accumulator: any finite floor (e.g. -1e6) is breached routinely by
        # negative-MSE scorers on noisy regression, leaving ``final_score > best_score`` False forever and tripping max_noimproving_iters
        # prematurely.
        best_score = -np.inf

        # Resume-from-checkpoint: restore mutable outer-loop state iff the checkpoint signature matches the current
        # (X.shape, y.shape, columns_key). Mismatch silently starts fresh, so users can keep the same checkpoint_path across experiments.
        if self.checkpoint_path is not None:
            _state = self._load_checkpoint()
            if _state is not None and _state.get("signature") == signature:
                nsteps = int(_state.get("nsteps", 0))
                evaluated_scores_mean = dict(_state.get("evaluated_scores_mean", {}))
                evaluated_scores_std = dict(_state.get("evaluated_scores_std", {}))
                feature_importances = dict(_state.get("feature_importances", {}))
                selected_features_per_nfeatures = dict(_state.get("selected_features_per_nfeatures", {}))
                prev_score = _state.get("prev_score", prev_score)
                prev_nfeatures = _state.get("prev_nfeatures", prev_nfeatures)
                n_noimproving_iters = int(_state.get("n_noimproving_iters", 0))
                best_nfeatures = int(_state.get("best_nfeatures", 0))
                best_iter = int(_state.get("best_iter", 0))
                best_score = _state.get("best_score", -np.inf)
                dummy_scores = list(_state.get("dummy_scores", []))
                # MBHOptimizer carries its own evaluation history; restore if present and current search method is MBH.
                _saved_optimizer = _state.get("optimizer")
                if _saved_optimizer is not None and Optimizer is not None:
                    Optimizer = _saved_optimizer
                if verbose:
                    logger.info(
                        "RFECV: resumed from checkpoint at %s (nsteps=%d, best_score=%s).",
                        self.checkpoint_path, nsteps,
                        f"{best_score:.4f}" if np.isfinite(best_score) else str(best_score),
                    )
            elif _state is not None:
                if verbose:
                    logger.info(
                        "RFECV: checkpoint at %s does not match current "
                        "(X, y, columns) signature; starting from scratch.",
                        self.checkpoint_path,
                    )

        # Establish a baseline RSS + best-effort frame footprint so the RAM-aware ``maybe_clean_ram_and_gpu`` short-circuit can decide whether a ``gc.collect()`` is justified each iter. The old "every 5th iter" trigger ran a ~290ms gc.collect even when nothing had accumulated, dominating wall on small problems; the helper only fires when RSS actually grew past a threshold or free RAM gets tight relative to frame size.
        from mlframe.training._ram_helpers import (
            estimate_df_size_mb as _estimate_df_size_mb,
            get_process_rss_mb as _get_process_rss_mb,
            maybe_clean_ram_and_gpu as _maybe_clean_ram_and_gpu,
        )
        _ram_baseline_mb = _get_process_rss_mb()
        try:
            _ram_df_size_mb = _estimate_df_size_mb(X)
        except Exception:
            _ram_df_size_mb = 0.0

        while nsteps < len(original_features):

            # Select current set of features to work on, based on ranking received so far and the optimum search method.

            # RAM-aware GC: ``maybe_clean_ram_and_gpu`` checks RSS-vs-baseline and free-vs-frame BEFORE invoking gc.collect, so the ~290ms cost is paid only when something actually accumulated. The old unconditional ``clean_ram()`` every-5-iters trigger fired even on small problems where no garbage existed, dominating wall.
            _ram_baseline_mb = _maybe_clean_ram_and_gpu(
                _ram_baseline_mb, _ram_df_size_mb, verbose=False, reason=f"RFECV iter {nsteps}"
            )

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

            # Defer the selected_features_per_nfeatures write until after we know whether this exploration's score beats the existing best
            # at the same N. An unconditional write silently downgrades a winning subset whenever MBH revisits the same N with a worse one.

            # Each split must be different: even with a fixed random_state, the per-iter CV random_state is derived deterministically.
            scores = []

            splitter = cv.split(X=X, y=y, groups=groups)

            # Pre-materialise fold args so we can dispatch sequentially or in parallel from the same code path. Each fold gets its own
            # pre-derived RNG seed rather than sharing self._rng, which would race in a parallel context.
            _fold_args: list = []
            for _nfold, (_tr_idx, _te_idx) in enumerate(splitter):
                _fold_seed = int(self._rng.integers(0, 2**31 - 1))
                _fold_args.append((_nfold, _tr_idx, _te_idx, _fold_seed))

            def _eval_fold(nfold, train_index, test_index, fold_seed, current_features=current_features, scores=scores):
                """Per-fold evaluation. Returns a dict or None on skip. Fold-local state (RNG, estimator clone, fit_params) is built fresh inside.

                ``current_features`` and ``scores`` are passed as default args so they bind at def-time to the current outer-iter values; this is
                safe because the closure is created fresh each outer iter and consumed within that iter (sequentially or via joblib.Parallel).
                """
                if self.min_train_size and len(train_index) < self.min_train_size:
                    return None
                if frac:
                    size = int(len(train_index) * frac)
                    if size > 10:
                        # Per-fold local RNG seeded deterministically; avoids races on self._rng when joblib runs folds in parallel.
                        local_rng = np.random.default_rng(fold_seed)
                        train_index = local_rng.choice(train_index, size=size, replace=False)

                # Actual fit/score uses must_include + optimiser's pick. current_features already lives in the search-universe complement
                # (must_include filtered out at fit entry), so concatenation never duplicates.
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

                    # Additional early-stopping split from X_train.
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

                    # If the estimator type is known, craft early-stopping fit params tailored to it.
                    temp_cat_features = [current_features.index(var) for var in cat_features if var in current_features] if cat_features else None

                    temp_fit_params = pack_val_set_into_fit_params(
                        model=estimator,
                        X_val=X_val,
                        y_val=y_val,
                        early_stopping_rounds=early_stopping_rounds,
                        cat_features=temp_cat_features,
                    )
                    # Filter feature-list keys (cat_features / text_features / embedding_features) coming in via fit_params to only columns
                    # present in the current selector iteration. Otherwise names from the outer call reference columns dropped by the
                    # current iteration and CB raises ``Error while processing column for feature 'cat_0'``.
                    #
                    # cat_features: pack_val_set_into_fit_params above already injected index-based temp_cat_features IFF that list was
                    # non-empty. When empty (current_features doesn't intersect self.cat_features) we MUST still pass a name-list filtered
                    # to current_features so CB doesn't fall back to auto-detect on numerically-encoded category columns (target-encoded
                    # cats look like floats and trip CB's "Invalid type for cat_feature").
                    # When current_features holds integer indices (ndarray X path), set-membership against string column names always
                    # misses; in that case the user-supplied lists pass through unfiltered (the inner estimator can deal with missing
                    # column references on its own).
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

                # Per-fold sample_weight slicing. Estimator-side: pass via temp_fit_params only when the estimator's
                # fit signature accepts sample_weight (sklearn convention). Scorer-side: detected at score-call time
                # below (sklearn's _BaseScorer.__call__ accepts sample_weight=).
                _fold_train_sw = None
                _fold_test_sw = None
                if self._fit_sample_weight_ is not None:
                    _fold_train_sw = self._fit_sample_weight_[train_index]
                    _fold_test_sw = self._fit_sample_weight_[test_index]

                # Fit on current train fold, score on test, get FI.

                # Always clone per fold via sklearn.base.clone. copy.copy is a SHALLOW copy that shares mutable state (cat_features list,
                # set_params side effects, warm_start buffers) across folds. clone() returns an unfitted estimator with the same
                # constructor params and NO fitted state.
                fitted_estimator = clone(estimator)

                # Dynamic CB ``text_processing`` calibration for THIS fold's clone (not the outer estimator). RFECV folds are typically
                # much smaller than the outer training set; with CB's default ``occurrence_lower_bound=50`` words that occur < 50 times in
                # the fold are pruned, leaving an empty dictionary and HANGING CB's C++ ``_train`` loop. ``compute_cb_text_processing``
                # returns a config that scales the floor proportionally to fold rows, or None when the fold is large enough.
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

                # Empty-train guard: heavy upstream filtering (small n + outlier_detection + trainset_aging_limit) can collapse X_train /
                # y_train to 0 rows on a CV fold; CatBoost then raises "Labels variable is empty" deep in C++ Pool init. Skip the fold
                # cleanly with a NaN score; sklearn's RFECV does the same on degenerate inner folds.
                _x_n = X_train.shape[0] if hasattr(X_train, "shape") else None
                _y_n = len(y_train) if y_train is not None and hasattr(y_train, "__len__") else None
                if (_x_n is not None and _x_n == 0) or (_y_n is not None and _y_n == 0):
                    # Always-on ERROR (not verbose-gated) so the operator sees the empty-fold collapse. Root cause is upstream filter
                    # aggression (OD + trainset_aging_limit together can shrink the inner-CV training fraction below cv splits) - fix
                    # that, not this guard.
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
                    # The body has since been wrapped in a nested ``_eval_fold`` function so an outer-loop ``continue`` is no longer
                    # legal; return None to skip the fold (matches the function's documented "or None on skip" contract).
                    return None
                # Multi-estimator: fit ALL estimators (singular case is a len-1 list). Score per fold = mean across estimators; FI runs
                # stored under separate keys ("{N}_{fold}" for singular, "{N}_{fold}_e{idx}" for multi) so the voting layer treats each
                # estimator's importances as an independent run.
                _est_scores = []
                _est_fi_runs = []  # list of (key, fi_dict)
                for _est_idx, _est_proto in enumerate(estimators_list):
                    if _est_idx == 0:
                        # First estimator was already cloned + text_processing tuned above as ``fitted_estimator``.
                        _fitted = fitted_estimator
                    else:
                        _fitted = clone(_est_proto)
                        # Apply CB text_processing if applicable for THIS clone.
                        if val_cv and has_early_stopping_support(estimator_type):
                            _temp_text_feats = _filtered_fit_params.get("text_features") or []
                            if _temp_text_feats and "CatBoost" in type(_fitted).__name__:
                                _fold_rows = X_train.shape[0] if hasattr(X_train, "shape") else None
                                _tp = compute_cb_text_processing(_fold_rows) if _fold_rows is not None else None
                                if _tp is not None and hasattr(_fitted, "set_params"):
                                    try:
                                        _fitted.set_params(text_processing=_tp)
                                    except Exception:
                                        pass

                    _model_type_name = type(_fitted).__name__
                    _ctx = suppress_stdout_stderr() if _model_type_name in CATBOOST_MODEL_TYPES else nullcontext()
                    # Build per-call fit kwargs: only attach sample_weight when the estimator's fit signature accepts it.
                    # CatBoost / LGBM / sklearn estimators all accept it; some custom shims may not, so introspect.
                    _per_est_fit_params = dict(temp_fit_params)
                    if _fold_train_sw is not None and "sample_weight" not in _per_est_fit_params:
                        try:
                            import inspect as _inspect
                            _sig = _inspect.signature(_fitted.fit)
                            if "sample_weight" in _sig.parameters or any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in _sig.parameters.values()):
                                _per_est_fit_params["sample_weight"] = _fold_train_sw
                        except (TypeError, ValueError):
                            pass
                    with _ctx:
                        _fitted.fit(X=X_train, y=y_train, **_per_est_fit_params)
                    # Scorer-side: forward fold-test sample_weight when sklearn scorer accepts it. make_scorer-wrapped
                    # callables expose sample_weight via _BaseScorer.__call__; bare callables may not.
                    if _fold_test_sw is not None:
                        try:
                            _score = scoring(_fitted, X_test, y_test, sample_weight=_fold_test_sw)
                        except TypeError:
                            _score = scoring(_fitted, X_test, y_test)
                    else:
                        _score = scoring(_fitted, X_test, y_test)
                    _est_scores.append(_score)

                    # FI is computed on the actual fit_features.
                    _fi_full = get_feature_importances(
                        model=_fitted, current_features=fit_features,
                        data=X_test, reference_data=X_val, target=y_test,
                        importance_getter=importance_getter,
                    )
                    if must_include_resolved:
                        must_set = set(must_include_resolved)
                        _fi = {k: v for k, v in _fi_full.items() if k not in must_set}
                    else:
                        _fi = _fi_full

                    _est_suffix = f"_e{_est_idx}" if len(estimators_list) > 1 else ""
                    _key = f"{len(current_features)}_{nfold}{_est_suffix}"
                    _est_fi_runs.append((_key, _fi))
                    if keep_estimators:
                        fitted_estimators[_key] = _fitted

                # Aggregate fold score via worst-case (min) across estimators given sklearn's "higher is better". Mean would let one strong
                # estimator (e.g. RF on 2 informative features) mask the fact that another (e.g. LR) needs more features to converge;
                # worst-case forces N to be where ALL estimators agree it's sufficient. For the singular path (len-1 list), min == mean.
                if _est_scores:
                    valid_scores = [s for s in _est_scores if not np.isnan(s)]
                    score = float(np.min(valid_scores)) if valid_scores else float("nan")
                else:
                    score = float("nan")
                scores.append(score)
                # Persist every estimator's FI run.
                for _k, _fi in _est_fi_runs:
                    feature_importances[_k] = _fi

                if 0 not in evaluated_scores_mean:

                    # Dummy baselines serve as fitness @ 0 features.
                    if not self.nofeatures_dummy_scoring:
                        # Sign-direction-agnostic "worse than model" placeholder. ``score/10`` silently put the dummy ABOVE the model when
                        # _sign==+1 and score was negative (e.g. R^2 < 0). Subtracting a positive number always makes the score worse
                        # under both sklearn conventions; use a magnitude-relative fudge so it scales with the metric in play (log-loss
                        # ~0.5, MSE ~1e6, R^2 ~1).
                        fudge = max(abs(score), 1e-3) * 9.0
                        dummy_scores.append(score - fudge)
                    else:
                        dummy_scores.append(
                            get_best_dummy_score(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring)
                        )

            # Dispatch _eval_fold sequentially or in parallel. n_jobs>1 uses prefer="threads" so we don't pickle X/y across workers
            # (datasets stay in shared memory) and the closure can keep mutating outer state under the GIL. When n_jobs>1 AND the
            # estimator is multi-threaded AND force_parallel=True, pin inner threads to 1.
            if n_jobs_effective > 1 and _is_multithreaded and self.force_parallel:
                _orig_eval_fold = _eval_fold
                def _eval_fold_pinned(*args, _orig=_orig_eval_fold):
                    # The closure clones the estimator inside its body so we can't reach in. Pin once on the OUTER estimator; clone()
                    # preserves params so each fold's clone inherits thread_count=1 / n_jobs=1.
                    _pin_threads_to_one(estimator)
                    return _orig(*args)
                _fold_runner = _eval_fold_pinned
            else:
                _fold_runner = _eval_fold

            if n_jobs_effective > 1 and len(_fold_args) > 1:
                from joblib import Parallel, delayed
                # prefer="threads": sklearn / CB / LGB / XGB all release GIL during fit, so threads give true parallelism without the
                # serialisation cost of multiprocessing.
                Parallel(n_jobs=n_jobs_effective, prefer="threads")(
                    delayed(_fold_runner)(*a) for a in _fold_args
                )
            else:
                if verbose:
                    _iter = tqdmu(_fold_args, desc="CV folds", leave=False, total=len(_fold_args))
                else:
                    _iter = _fold_args
                for a in _iter:
                    _fold_runner(*a)

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
            # Only commit selected_features when this run actually won at its N.
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

            nsteps += 1

            # Persist outer-loop state so a crash mid-run is recoverable. fitted_estimators is intentionally NOT pickled (CB / RF
            # ensembles dominate file size); they are re-fit on resume when needed. Save errors are logged but do not abort the fit.
            if self.checkpoint_path is not None:
                try:
                    self._save_checkpoint({
                        "version": self._CHECKPOINT_VERSION,
                        "signature": signature,
                        "nsteps": nsteps,
                        "evaluated_scores_mean": dict(evaluated_scores_mean),
                        "evaluated_scores_std": dict(evaluated_scores_std),
                        "feature_importances": dict(feature_importances),
                        "selected_features_per_nfeatures": dict(selected_features_per_nfeatures),
                        "prev_score": prev_score,
                        "prev_nfeatures": prev_nfeatures,
                        "n_noimproving_iters": n_noimproving_iters,
                        "best_nfeatures": best_nfeatures,
                        "best_iter": best_iter,
                        "best_score": best_score,
                        "dummy_scores": list(dummy_scores),
                        "optimizer": Optimizer,
                    })
                except Exception as _ckpt_exc:
                    if verbose:
                        logger.warning(
                            "RFECV: checkpoint save at nsteps=%d failed: %s",
                            nsteps, _ckpt_exc,
                        )

            if len(evaluated_scores_mean) == 2:
                # If the first explored subset (whatever MBH seeded; default seed = 2 features) is already worse than the dummy at 0
                # features, there's no point continuing.
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

            # Abort early if every iter so far produced a NaN final_score. The most common cause is a custom scorer returning NaN on every
            # fold (e.g. ROC AUC on single-class CV folds). Without this, the noimproving counter would consume max_noimproving_iters
            # worth of useless CV fits. Detect 5 consecutive NaN iters and bail.
            if np.isnan(final_score):
                if not hasattr(self, "_consecutive_nan_iters"):
                    self._consecutive_nan_iters = 0
                self._consecutive_nan_iters += 1
                if self._consecutive_nan_iters >= 5:
                    raise RuntimeError(
                        "RFECV: scoring returned NaN on 5 consecutive iters. "
                        "Likely cause: custom scorer NaN-ing on degenerate folds, "
                        "or single-class fold with a binary-only metric. Switch "
                        "to a NaN-safe scorer or pass cv=StratifiedKFold(...)."
                    )
            else:
                self._consecutive_nan_iters = 0

            if self.special_feature_indices is not None:
                if verbose:
                    logger.info(f"Quitting as special_feature_indices were checked.")
                break

        # Truncated SFFS final-pass swap: run K paired swaps on the best subset found - replace each of the K worst-FI kept features with
        # each of the K best-FI dropped features, accept any swap that improves the CV score. Uses sklearn.cross_val_score directly so it
        # does NOT honour fit_params / val_cv / early stopping.
        if self.swap_top_k and self.swap_top_k > 0 and best_nfeatures > 0:
            try:
                self._sffs_swap_pass(
                    X=X, y=y, estimator=estimator, cv=cv, scoring=scoring,
                    best_nfeatures=best_nfeatures, best_score_ref=best_score,
                    selected_features_per_nfeatures=selected_features_per_nfeatures,
                    feature_importances=feature_importances,
                    original_features=original_features,
                    evaluated_scores_mean=evaluated_scores_mean,
                    evaluated_scores_std=evaluated_scores_std,
                    verbose=verbose, ndigits=ndigits,
                )
                # After the swap pass best_nfeatures may have changed.
                if evaluated_scores_mean:
                    new_best_nf = max(evaluated_scores_mean, key=evaluated_scores_mean.get)
                    if evaluated_scores_mean[new_best_nf] > best_score:
                        best_nfeatures = new_best_nf
                        best_score = evaluated_scores_mean[new_best_nf]
            except Exception as _swap_exc:
                if verbose:
                    logger.warning("RFECV: SFFS swap pass failed (%s); continuing.", _swap_exc)

        # Save best result found so far as final.
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(map(str, np.arange(self.n_features_in_)))

        self.estimators_ = fitted_estimators  # dict keyed by nfeatures_nfold
        self.feature_importances_ = feature_importances  # dict keyed by nfeatures_nfold
        self.selected_features_ = selected_features_per_nfeatures  # dict keyed by nfeatures

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

        # Glue must_include into the final support_. The optimiser produced support_ over the search-universe complement only;
        # must_include features are always in the final selection regardless of what the optimiser picked.
        if must_include_resolved and hasattr(self, "support_") and len(self.support_) > 0:
            if isinstance(self.support_[0], (bool, np.bool_)):
                # support_ is bool-mask aligned with feature_names_in_; set the must_include positions to True.
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

        # feature_groups: all-or-nothing decision per group. If ANY member of group G is in support_, ALL members are added; if NONE, all
        # stay out. Resolves the "5 collinear copies" caveat at config level when the operator knows the group structure.
        if self.feature_groups and hasattr(self, "support_") and len(self.support_) > 0:
            # Convert support_ to bool-mask form for uniform handling.
            if isinstance(self.support_[0], (bool, np.bool_)):
                support_mask = np.asarray(self.support_, dtype=bool)
            else:
                support_mask = np.zeros(len(self.feature_names_in_), dtype=bool)
                for i in self.support_:
                    support_mask[i] = True
            _name_to_idx = {n: i for i, n in enumerate(self.feature_names_in_)}
            _expanded = 0
            for _group_name, _members in self.feature_groups.items():
                _idx = [_name_to_idx[m] for m in _members if m in _name_to_idx]
                if not _idx:
                    continue
                _any_selected = any(support_mask[i] for i in _idx)
                if _any_selected:
                    for i in _idx:
                        if not support_mask[i]:
                            _expanded += 1
                            support_mask[i] = True
            if _expanded > 0:
                if verbose:
                    logger.info(
                        "RFECV: feature_groups expanded support_ by %d column(s) "
                        "for all-or-nothing group decisions.", _expanded,
                    )
                self.support_ = support_mask
                self.n_features_ = int(support_mask.sum())

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

    def _fit_stability_selection(self, X, y, signature):
        """Stability Selection (Meinshausen & Buhlmann 2010, JRSS-B).

        Bootstrap-based feature selection. For each of B bootstrap subsamples (n/2, no replacement), fit the estimator(s) and record which
        features appeared in the top-K by importance. A feature is selected if its appearance frequency >= ``stability_threshold``
        (typically 0.6-0.9). Provable error control: E[V] <= q^2 / ((2*pi - 1) * p), where q is the average number of selected features
        per bootstrap and pi is the threshold.

        Particularly robust on small-n / high-p problems where per-fold CV voting is dominated by sampling noise. If ``self.estimators`` is
        set, FI is averaged across them inside each bootstrap.
        """
        estimators_list = list(self.estimators) if self.estimators else [self.estimator]
        importance_getter = self.importance_getter or "auto"
        rng = np.random.default_rng(self.random_state)
        is_df = isinstance(X, pd.DataFrame)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        feature_names = X.columns.tolist() if is_df else [str(i) for i in range(n_features)]

        top_k = self.stability_top_k
        if top_k is None:
            # Top quartile - generous enough that informative features clearly above the noise floor will hit threshold.
            top_k = max(1, n_features // 4)
        top_k = min(int(top_k), n_features)

        # Subsample size: n/2, the standard Meinshausen-Buhlmann choice.
        sub_size = max(2, n_samples // 2)
        selection_counts = np.zeros(n_features, dtype=int)

        if self.verbose:
            logger.info(
                "stability_selection: B=%d bootstraps, sub_size=%d (of %d), "
                "top_k=%d (of %d), threshold=%.2f, estimators=%d",
                self.stability_n_bootstrap, sub_size, n_samples,
                top_k, n_features, self.stability_threshold, len(estimators_list),
            )

        for b in range(int(self.stability_n_bootstrap)):
            idx = rng.choice(n_samples, size=sub_size, replace=False)
            if is_df:
                X_sub = X.iloc[idx]
            else:
                X_sub = X[idx]
            y_arr = np.asarray(y)
            y_sub = y_arr[idx]

            # Aggregate FI across estimators within this bootstrap.
            per_feature_score_sum = np.zeros(n_features, dtype=float)
            for est in estimators_list:
                est_clone = clone(est)
                try:
                    est_clone.fit(X_sub, y_sub)
                except Exception as exc:
                    if self.verbose:
                        logger.warning(
                            "stability_selection: bootstrap %d, %s.fit failed: %s. "
                            "Skipping this estimator for this bootstrap.",
                            b, type(est_clone).__name__, exc,
                        )
                    continue
                try:
                    fi_dict = get_feature_importances(
                        model=est_clone, current_features=feature_names,
                        data=X_sub, target=y_sub,
                        importance_getter=importance_getter,
                    )
                except Exception as exc:
                    if self.verbose:
                        logger.warning(
                            "stability_selection: bootstrap %d, get_feature_importances failed: %s.",
                            b, exc,
                        )
                    continue
                # Align with feature_names.
                fi_arr = np.array([float(fi_dict.get(n, 0.0)) for n in feature_names])
                fi_arr = np.where(np.isnan(fi_arr), 0.0, fi_arr)
                per_feature_score_sum += fi_arr

            # Top-K from this bootstrap (across-estimator mean importance).
            if per_feature_score_sum.sum() <= 0:
                continue
            top_idx = np.argsort(per_feature_score_sum)[::-1][:top_k]
            selection_counts[top_idx] += 1

        selection_freq = selection_counts / max(1, int(self.stability_n_bootstrap))
        support_mask = selection_freq >= float(self.stability_threshold)

        # must_include: pinned features always in support_.
        must_include_resolved = list(self.must_include) if self.must_include else []
        if must_include_resolved:
            for c in must_include_resolved:
                if c in feature_names:
                    support_mask[feature_names.index(c)] = True

        # Public state: same shape as the regular path so transform / get_feature_names_out / selection_stability_ all work.
        self.feature_names_in_ = list(feature_names)
        self.n_features_in_ = n_features
        self.support_ = support_mask
        self.n_features_ = int(support_mask.sum())
        self.estimators_ = {}
        self.feature_importances_ = {}
        self.selected_features_ = {}
        self.cv_results_ = {
            "nfeatures": [self.n_features_],
            "cv_mean_perf": [float(selection_freq[support_mask].mean()) if support_mask.any() else 0.0],
            "cv_std_perf": [0.0],
        }
        # Per-feature stability frequencies for inspection / downstream weighting, aligned with feature_names_in_.
        self.stability_selection_freq_ = selection_freq

        self._selected_cols_cache = [c for c, s in zip(feature_names, support_mask) if s]
        self.signature = signature

        if self.verbose:
            logger.info(
                "stability_selection: selected %d / %d features at threshold=%.2f. "
                "Top-10 by frequency: %s",
                self.n_features_, n_features, self.stability_threshold,
                [(feature_names[i], round(float(selection_freq[i]), 3))
                 for i in np.argsort(selection_freq)[::-1][:10]],
            )
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

        # Resolve n_features selection rule.
        #   multi-estimator -> 'one_se_max' (avoids the multi-estimator collapse-to-2: a strong estimator can mask a weak estimator's need for more features)
        #   singular -> 'argmax' (legacy; on small data the 1-SE band can be dominated by accidental small-N points due to MBH's sparse N exploration)
        # Users who want parsimonious selection on wide score plateaus should opt into 'one_se_min' explicitly.
        rule = getattr(self, "n_features_selection_rule", "auto")
        if rule == "auto":
            rule = "one_se_max" if getattr(self, "estimators", None) else "argmax"

        nfeatures_arr = np.array(checked_nfeatures)
        nonzero_mask = nfeatures_arr > 0
        # Honour max_nfeatures as a HARD cap on the final pick. The optimiser may evaluate larger N during search (e.g. an all-features
        # baseline at iter=0), but the final selection must NEVER exceed self.max_nfeatures when set.
        max_nf = getattr(self, "max_nfeatures", None)
        if max_nf is not None:
            nonzero_mask = nonzero_mask & (nfeatures_arr <= max_nf)
        if not nonzero_mask.any():
            logger.warning(
                "select_optimal_nfeatures_: only nfeatures==0 was evaluated; "
                "no features can be selected. Returning empty support_."
            )
            self.n_features_ = 0
            self.support_ = np.array([])
            return

        if rule == "argmax":
            # Pick the index with the highest ultimate_perf among the candidate N values; nonzero_mask honours max_nfeatures.
            sorted_idx = np.argsort(ultimate_perf)[::-1]
            best_idx = None
            for idx in sorted_idx:
                if nonzero_mask[idx]:
                    best_idx = idx
                    break
        else:
            # one_se_max / one_se_min: build the SE band around the best mean (cv_mean_perf - the *unadjusted* score, so 1-SE has its
            # standard interpretation), then pick the largest or smallest N within the band.
            mean_arr = np.array(cv_mean_perf)
            std_arr = np.array(cv_std_perf)
            nz_idx = np.where(nonzero_mask)[0]
            best_mean_idx = nz_idx[np.argmax(mean_arr[nz_idx])]
            threshold = mean_arr[best_mean_idx] - std_arr[best_mean_idx]
            in_band = [i for i in nz_idx if mean_arr[i] >= threshold]
            if not in_band:
                in_band = [int(best_mean_idx)]
            if rule == "one_se_max":
                best_idx = max(in_band, key=lambda i: nfeatures_arr[i])
            elif rule == "one_se_min":
                best_idx = min(in_band, key=lambda i: nfeatures_arr[i])
            else:
                raise ValueError(
                    f"n_features_selection_rule={rule!r} not supported. "
                    f"Use 'auto', 'argmax', 'one_se_max', or 'one_se_min'."
                )
        best_top_n = int(nfeatures_arr[best_idx])

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
                # Non-blocking show: plt.show(block=True) (the default) freezes the script behind a modal Qt window. Pair with a tiny pause
                # to flush the GUI event loop so the figure actually renders before training continues / exits.
                try:
                    plt.show(block=False)
                    plt.pause(0.001)
                except Exception:
                    pass

        self.n_features_ = best_top_n
        if best_top_n == 0:
            self.support_ = np.array([])
        else:

            if not self.conduct_final_voting:

                # Return exactly the features used when measuring scores.
                selected = self.selected_features_[best_top_n]
                # Represent support_ as a boolean mask for consistency with sklearn's RFE API.
                if self.feature_names_in_ and isinstance(self.feature_names_in_[0], str):
                    selected_set = {str(feature_name) for feature_name in selected}
                    self.support_ = np.array([str(f) in selected_set for f in self.feature_names_in_])
                else:
                    selected_set = set(selected)
                    self.support_ = np.array([f in selected_set for f in self.feature_names_in_])

            else:

                # Advanced alternative: vote for feature_importances using all info up to date.
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
        """sklearn-1.x transformer protocol. Returns the names of the selected features as an ndarray of str, matching what ``transform``
        produces as columns. Compatible with sklearn Pipelines that call this method for downstream feature naming
        (ColumnTransformer, set_output).
        """
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
        """Mean pairwise feature-selection stability across CV folds at the chosen ``n_features_``. Free signal extracted from
        feature_importances_, no extra fits required.

        Args:
            metric: 'jaccard' (default), 'dice', or 'kuncheva'.

        Returns:
            Float in [0, 1] (1 = identical selections across folds, 0 = disjoint). Returns NaN when fewer than 2 folds have FI data
            at n_features_.
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
                # Kuncheva's index normalises by chance overlap; needs the universe size N. Range [-1, 1] but clamped to [0, 1] here.
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
        """1-SE rule: smallest N whose CV mean is within one standard error of the best CV mean. Often selected over n_features_ when the
        operator wants the most parsimonious model that's not statistically distinguishable from the optimum.

        Returns the integer count, or n_features_ as a fallback if cv_results_ is unavailable.
        """
        if not hasattr(self, "cv_results_") or not self.cv_results_.get("nfeatures"):
            return getattr(self, "n_features_", 0)
        nfeatures = np.asarray(self.cv_results_["nfeatures"], dtype=int)
        means = np.asarray(self.cv_results_["cv_mean_perf"], dtype=float)
        stds = np.asarray(self.cv_results_["cv_std_perf"], dtype=float)
        if len(means) == 0:
            return getattr(self, "n_features_", 0)
        # Exclude the 0-features dummy.
        nonzero = nfeatures > 0
        if not nonzero.any():
            return getattr(self, "n_features_", 0)
        nf, m, s = nfeatures[nonzero], means[nonzero], stds[nonzero]
        # mean_perf_weight + std_perf_weight are baked into final_score; for 1-SE we need the unadjusted mean - cv_mean_perf is raw.
        best_idx = int(np.argmax(m))
        threshold = m[best_idx] - s[best_idx]
        # Smallest N whose mean >= threshold.
        eligible = nf[m >= threshold]
        if len(eligible) == 0:
            return int(nf[best_idx])
        return int(eligible.min())

    def n_features_bootstrap_ci_(self, n_bootstrap: int = 200, ci: float = 0.9,
                                  random_state: Union[int, None] = None) -> tuple:
        """Parametric bootstrap CI on the optimal n_features_.

        Draws B bootstrap replicates of the cv_results_ score curve by sampling each (mean, std) pair as Normal(mean, std), recomputes
        argmax over the non-zero N values for each replicate, returns (low_pct, n_features_, high_pct) where the percentiles bracket
        ``ci`` mass of the bootstrap distribution.

        Use this to gauge whether n_features_=N is meaningfully different from N+/-5; a wide CI suggests caution about the exact N
        choice. Parametric bootstrap (no raw per-fold scores retained), so it under-estimates true uncertainty when fold scores are
        non-Normal.
        """
        if not hasattr(self, "cv_results_") or not self.cv_results_.get("nfeatures"):
            n = getattr(self, "n_features_", 0)
            return (n, n, n)
        nfeatures = np.asarray(self.cv_results_["nfeatures"], dtype=int)
        means = np.asarray(self.cv_results_["cv_mean_perf"], dtype=float)
        stds = np.asarray(self.cv_results_["cv_std_perf"], dtype=float)
        nonzero = nfeatures > 0
        if not nonzero.any():
            n = getattr(self, "n_features_", 0)
            return (n, n, n)
        nf, m, s = nfeatures[nonzero], means[nonzero], stds[nonzero]
        rng = np.random.default_rng(random_state)
        choices = []
        for _ in range(int(n_bootstrap)):
            sampled = rng.normal(loc=m, scale=np.maximum(s, 1e-12))
            best_idx = int(np.argmax(sampled))
            choices.append(int(nf[best_idx]))
        choices_arr = np.asarray(choices)
        alpha = (1.0 - float(ci)) / 2.0
        low = int(np.percentile(choices_arr, 100.0 * alpha))
        high = int(np.percentile(choices_arr, 100.0 * (1.0 - alpha)))
        n = getattr(self, "n_features_", int(np.median(choices_arr)))
        return (low, int(n), high)

    def transform(self, X, y=None):
        # Polars X (callers like _passthrough_cols_fit_transform keep the native frame) breaks the legacy ``X[:, self.support_]`` mask
        # path with ``expected N values when selecting columns by boolean mask, got M`` when the polars schema has more cols than the
        # fit-time support_ (because RFECV.fit dropped zero-variance cols at entry). Convert to pandas so the name-keyed transform path
        # kicks in and column-set drift becomes a clear RuntimeError instead of an opaque polars index mismatch.
        if isinstance(X, pl.DataFrame):
            X = X.to_pandas()
        # transform on an unfitted estimator must raise NotFittedError; silently returning X unchanged masquerades a config bug as a
        # successful transform and lets downstream pipelines run on the wrong column set.
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This RFECV instance is not fitted yet. Call 'fit' before "
                "using 'transform'."
            )
        support = self.support_
        if len(support) == 0:
            # Empty DataFrame/array with same rows but no columns: feature selection found no useful features.
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, []]
            else:
                return X[:, np.array([], dtype=np.intp)]
        if isinstance(X, pd.DataFrame):
            # Use column names (not .iloc) to support Arrow-backed DataFrames from polars zero-copy conversion - they don't support
            # .iloc[:, integer_array] reliably.
            selected_cols = getattr(self, "_selected_cols_cache", None)
            if selected_cols is None:
                if len(self.support_) > 0 and isinstance(self.support_[0], (bool, np.bool_)):
                    selected_cols = [col for col, selected in zip(self.feature_names_in_, self.support_) if selected]
                else:
                    selected_cols = [self.feature_names_in_[i] for i in self.support_]
            # Column-set drift between fit-time and transform-time is a hard error: the fit-time zero-variance filter ensures
            # feature_names_in_ never contains columns sklearn pipeline steps may silently drop. If we still see drift, an upstream
            # step is mutating the schema between fit and transform.
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


