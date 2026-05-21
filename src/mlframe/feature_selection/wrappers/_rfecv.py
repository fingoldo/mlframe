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
        # Wave 57 (2026-05-20): add feature name as deterministic tiebreaker --
        # many features tie at fi_mean=0 (no FI history); without the tiebreak,
        # which feature wins the swap depends on Python set/list iteration
        # order, silently flipping selection across runs.
        kept_sorted = sorted(best_set, key=lambda f: (fi_mean.get(f, 0.0), str(f)))
        swap_out = kept_sorted[:K]
        # Best K dropped.
        not_in_set = [f for f in original_features if f not in set(best_set)]
        # For the reverse-desc side, negate score then ascend alphabetically
        # so the deterministic tiebreaker is uniform-direction.
        not_sorted = sorted(not_in_set, key=lambda f: (-fi_mean.get(f, 0.0), str(f)))
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
        # Wave 36 Low fix (2026-05-20): mirror the ``_fd_adopted`` flag
        # pattern used canonically across the project
        # (``training/io.py:atomic_write_bytes``,
        # ``composite_cache.py._save_lru``). If ``os.fdopen(fd, "wb")``
        # raises BEFORE the BufferedWriter adopts ``fd`` (rare: MemoryError
        # on the buffered-writer allocation, future refactor with an
        # invalid mode), the raw ``fd`` is never adopted by a
        # context-manager and leaks. Track adoption explicitly + close
        # the raw fd in the failure branch.
        fd, tmp = tempfile.mkstemp(prefix=".rfecv_ckpt_", dir=dir_name)
        _fd_adopted = False
        try:
            with os.fdopen(fd, "wb") as fh:
                _fd_adopted = True  # fdopen returned -> the with-block owns fd
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            if not _fd_adopted:
                try:
                    os.close(fd)
                except OSError:
                    pass
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
        if not path:
            return None
        # Wave 48 (2026-05-20): the prior exists-then-open pattern was a TOCTOU race
        # with concurrent RFECV runs sharing checkpoint_path (or an external cleanup
        # cron); FileNotFoundError/OSError would propagate uncaught and abort the fit.
        # Drop the redundant exists check; add OSError/FileNotFoundError to the except.
        try:
            with open(path, "rb") as fh:
                state = pickle.load(fh)
        except FileNotFoundError:
            return None
        except (pickle.PickleError, EOFError, AttributeError, TypeError, ValueError, OSError) as exc:
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




    def get_feature_names_out(self, input_features=None):
        """sklearn-1.x transformer protocol. Returns the names of the selected features as an ndarray of str, matching what ``transform``
        produces as columns. Compatible with sklearn Pipelines that call this method for downstream feature naming
        (ColumnTransformer, set_output).
        """
        if not hasattr(self, "support_"):
            # Wave 37 P1 fix (2026-05-20): sklearn convention is
            # NotFittedError (ValueError-compatible subclass), so existing
            # ``except ValueError`` chains stay green AND
            # ``except NotFittedError`` discriminators work.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("RFECV is not fitted; call fit() first.")
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
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError as _NFE
            raise _NFE("RFECV is not fitted; call fit() first.")
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
            # Top-N features in this fold by importance value.
            # Wave 57 (2026-05-20): secondary key on feature name so tied
            # zero-importance features don't shift across runs and the
            # downstream Jaccard / Dice between fold-sets stays stable.
            top_ids = sorted(fi.keys(), key=lambda k: (-fi[k], str(k)))[: self.n_features_]
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
        # Wave 21 P0: same shape as the other winner-picker above -- mask
        # NaN candidates before argmax. Pre-fix argmax picks NaN slot when
        # any candidate's cv_mean_perf is all-NaN-folds.
        _finite_mask = np.isfinite(m)
        if not _finite_mask.any():
            # No usable candidates -- fall back to the cached n_features_.
            return getattr(self, "n_features_", 0)
        if not _finite_mask.all():
            nf, m, s = nf[_finite_mask], m[_finite_mask], s[_finite_mask]
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
        # Wave 39 (2026-05-20): n_bootstrap<=0 degenerates to empty choices_arr,
        # then int(np.median([])) raises ValueError with a RuntimeWarning.
        if int(n_bootstrap) <= 0:
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


# ----------------------------------------------------------------------
# Sibling-module bindings. Methods are defined in sibling files because
# they're too large to keep inline. Each ``RFECV.<name>`` rebind happens
# AFTER the class body has loaded so the sibling can reference RFECV-via-
# self with no cycle.
# ----------------------------------------------------------------------
from ._rfecv_fit import fit as _fit_func  # noqa: E402
RFECV.fit = _fit_func

from ._rfecv_stability_select import (  # noqa: E402
    _fit_stability_selection as _fit_stability_selection_func,
    select_optimal_nfeatures_ as _select_optimal_nfeatures_func,
)
RFECV._fit_stability_selection = _fit_stability_selection_func
RFECV.select_optimal_nfeatures_ = _select_optimal_nfeatures_func
