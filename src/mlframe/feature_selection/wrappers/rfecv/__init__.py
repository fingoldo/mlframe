"""Feature selection within ML pipelines. Wrappers methods. Currently includes recursive feature elimination."""

from __future__ import annotations

import copy
import functools
import hashlib
import logging
import textwrap
from contextlib import nullcontext
from os.path import exists
from timeit import default_timer as timer
from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
from pyutilz.numbalib import set_numba_random_seed  # noqa: F401
from pyutilz.pythonlib import (
    get_parent_func_args,
    store_params_in_object,
    suppress_stdout_stderr,
)
from pyutilz.system import tqdmu
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: F401
from sklearn.metrics import make_scorer  # noqa: F401
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
from mlframe.core.helpers import has_early_stopping_support
from mlframe.estimators.baselines import get_best_dummy_score
from mlframe.metrics.core import compute_probabilistic_multiclass_error
from mlframe.models.optimization import (
    CandidateSamplingMethod,
    MBHOptimizer,
    OptimizationDirection,
    OptimizationProgressPlotting,
)
from mlframe.preprocessing.transforms import pack_val_set_into_fit_params
from mlframe.utils.misc import set_random_seed

from .._enums import OptimumSearch, VotesAggregation
from .._helpers import (
    _detect_multithreaded,
    _pin_threads_to_one,
    get_actual_features_ranking,
    get_feature_importances,
    get_next_features_subset,
    select_appropriate_feature_importances,
    split_into_train_test,
    store_averaged_cv_scores,
    suppress_irritating_3rdparty_warnings,
)
from ._configs import FIConfig, RobustnessConfig, SearchConfig

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

    # RFECV rejects duplicate input column names at fit entry (_fit_init guard); the GroupAwareMRMR wrapper reads this flag to surface that rejection when RFECV is the inner selector (the inner only sees the deduped cluster medoids, so the wrapper must guard on its behalf).
    rejects_duplicate_feature_names = True

    def __init__(
        self,
        estimator: Union[BaseEstimator, None] = None,
        # 2026-05-28: grouped pydantic configs. When passed, their non-None fields override matching flat kwargs.
        # All flat kwargs are kept for back-compat AND because some power-users want flat call-sites.
        # See ``_rfecv_configs.py`` and USAGE.md for the canonical mlframe pattern.
        search_config: "Union[SearchConfig, None]" = None,
        fi_config: "Union[FIConfig, None]" = None,
        robustness_config: "Union[RobustnessConfig, None]" = None,
        fit_params: Union[dict, None] = None,
        max_nfeatures: Union[int, None] = None,
        mean_perf_weight: float = 1.0,
        std_perf_weight: float = 0.1,
        feature_cost: float = 0.0,
        smooth_perf: int = 0,
        # stopping conditions
        max_runtime_mins: Union[float, None] = None,
        max_refits: Union[int, None] = None,
        best_desired_score: Union[float, None] = None,
        max_noimproving_iters: int = 30,
        # CV
        cv: Union[object, int, None] = 3,
        cv_shuffle: bool = False,
        min_train_size: Union[int, None] = None,
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
        # importance_getter: source of the per-feature importance that drives elimination. 'auto'/'feature_importances_'
        # use the estimator's (impurity/gain) importance - fast but biased toward high-cardinality/structure-bearing
        # columns and under-rank low-marginal signals. 'permutation' (OOF permutation importance) debiases this and
        # measured a CONSISTENT downstream gain on a controlled synthetic (+0.029 mean LightGBM holdout AUC across 3
        # seeds, positive every seed, incl. full base-recall on one) at ~4-5x fit cost - prefer it for QUALITY when the
        # budget allows. 'conditional_permutation' (Strobl) was measured to HURT here (splits credit across correlated
        # copies, keeps more noise). Also: 'coef_', 'shap', 'drop_column', 'boruta', 'boruta_shap', or a callable.
        importance_getter: Union[str, Callable, None] = None,
        random_state: Union[int, None] = None,
        leave_progressbars: bool = True,
        verbose: Union[bool, int] = 0,
        show_plot: bool = False,
        optimizer_plotting: Union[str, None] = None,  # Controls Optimizer plotting: 'No', 'Final', 'OnScoreImprovement', 'Regular'
        cat_features: Union[Sequence, None] = None,
        keep_estimators: bool = False,
        estimators_save_path: Union[str, None] = None,  # fitted estimators get saved into join(estimators_save_path,estimator_type_name,nestimator_nfeatures_nfold.dump)
        # Required features and achieved ml metrics get saved in a dict join(estimators_save_path,required_features.dump).
        frac: Union[float, None] = None,
        # Skip the full re-fit when fit() is called again on identical inputs. Despite the legacy "same_shape" name, the
        # skip keys on CONTENT: it invalidates on (a) X content / column-name change, (b) y / TARGET content change, AND
        # (c) ANY selector- or wrapped-estimator-parameter change (set_params or direct attribute assignment alike;
        # params are re-read at every fit call) -- so it never replays a stale support_ for a changed target or settings.
        skip_retraining_on_same_shape: bool = True,
        stop_file: str = "stop",
        report_ndigits: int = 4,
        #
        special_feature_indices: Union[list, None] = None,
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
        # n_features_selection_rule: rule for picking n_features_ from cv_results_ (resolved in select_optimal_nfeatures_).
        #   'argmax' - argmax of (mean - lambda*std - feature_cost*N). On FLAT score curves around the optimum this collapses to the FIRST N visited near-max, often under-selecting.
        #   'one_se_max' - LARGEST N within 1 SE of the best mean; robust on plateaus, but NOT parsimonious: on noise-robust learners (GBM / RF) the
        #       whole N-range can sit inside the 1-SE band, so this keeps ~all features. Set feature_cost>0 (it biases the band toward fewer
        #       features) or use 'one_se_min' when you want a compact set.
        #   'one_se_min' - sklearn-canonical SMALLEST N within 1 SE; parsimonious (drops redundant / marginally-informative features) but can under-select on flat curves.
        #   'auto' (default) - resolves to 'one_se_max' for ALL estimators (single AND multi); deliberately recall-oriented after 'one_se_min' was found to
        #       under-select on plateau-prone curves. Pass 'one_se_min' explicitly for parsimony.
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
        # ----- Wave 1 ML-correctness knobs (2026-05-28) -----
        # F9: when False (NEW default), the FI runs of a re-explored same-N subset that LOSES the best-of-N gate are popped from feature_importances
        # so they don't contaminate the next voting round. Set True to restore the pre-2026-05-28 behaviour (every explored subset votes equally).
        keep_loser_subset_fi: bool = False,
        # F1+F2+F3 missing-entry policy for voting. The historical default left ragged-NaN tables intact, which made Borda/Dowdall/Copeland/Minimax silently
        # bias toward late-surviving features (a feature present in 30/30 runs sums over 30 columns; a feature eliminated at iter 3, only over 3). Options:
        #   'worst'  (NEW default): impute missing per-run FI with min(FI)-eps for that run -> features eliminated early get LAST place in those runs
        #            uniformly across every voting rule. Equivalent to "treated as eliminated" rather than "treated as absent".
        #   'median': impute with the run's median FI -> features eliminated early get average treatment (older project default for AM/GM/OG via fillna).
        #   'skip'  : pre-2026-05-28 raw behaviour (ragged NaN). Documented as biased but exposed for back-compat A/B benches.
        fi_missing_policy: str = "worst",
        # C2: dummy-baseline-at-N=0 is fed to the MBH surrogate when True. Bench (2026-05-28) showed that on small p (~8) problems
        # removing this anchor halves the explored-N count -- MBH can't extrapolate to low N without a low-anchor and converges to
        # even-N only. So the safer default is True until S9 (proper low-N init design) lands in Wave 2. Set False on imbalanced
        # accuracy/F1 datasets where the dummy ~= model score and the optimizer biases toward small N.
        submit_dummy_to_optimizer: bool = True,
        # auto rule rationale rewrite (C3): the legacy 'auto' = ('one_se_max' for multi-estimator else 'argmax') uses INVERTED logic - multi-estimator uses
        # min(scores) across estimators, which has HIGHER variance, WIDER 1-SE band, and 'one_se_max' over-selects MORE. NEW default: plain 'argmax' for
        # both single and multi. Users wanting parsimony pass 'one_se_min' explicitly. Kept as: rule resolution moved into one block in
        # select_optimal_nfeatures_, see that file for the dispatch.
        # E2 escape hatch: keep swap_top_k active even when val_cv is set (compare ES vs non-ES scores, accepting potential overfit-swaps).
        swap_top_k_allow_no_es: bool = False,
        # ----- Wave 2 search-strategy knobs (2026-05-28) -----
        # S8: what target value to feed the MBH surrogate. 'mean' (NEW default) submits raw cv_mean_perf -> aligned with the 1-SE rule
        #   semantics in select_optimal_nfeatures_; 'final_score' (legacy) submits mean*w_mean - std*w_std (a UCB-of-noise) which can
        #   disagree with the post-processing rule when std_perf_weight or feature_cost are non-default.
        optimizer_target: str = "mean",
        # S7: tolerance-based convergence. When set, break when max(last K final_scores) - min(...) < tol * |best_score|. None disables.
        convergence_tol: Union[float, None] = None,
        convergence_tol_window: int = 10,
        # S9+S10: richer init design seeding the MBH optimizer with multiple low/mid/high N anchors. None = legacy single seed;
        # int K = exactly K equidistant anchors; 'auto' (NEW default) scales K by p AND evaluation budget so init-seed wall stays
        # small relative to user-driven exploration: p<=10 -> K=2, p<=50 OR budget<30 -> K=3, else -> K=5.
        init_design_size: Union[int, str, None] = "auto",
        # S6: epsilon random kick for ExhaustiveDichotomic to avoid getting stuck in the first local maximum on multi-modal score-vs-N
        # curves. Default 0.1 (NEW) sets ~10% iters to pick a random unevaluated N outside the best-known neighbourhood. Set 0.0 to
        # restore pure dichotomic behaviour.
        dichotomic_epsilon: float = 0.1,
        # dichotomic_step: elimination-pace schedule for ExhaustiveDichotomic. 'midpoint' (default) is the legacy fixed
        # bisection. 'auto' is the adaptive coarse-to-fine schedule: stride by max(1, floor(frac*n_remaining)) away from the
        # best while the unevaluated N-pool is large AND the CV curve is flat, then collapse to step=1 midpoint refinement near
        # the knee. bench-attempt-rejected-as-default (see _benchmarks/bench_dichotomic_adaptive_step.py): selection is exactly
        # equivalent (Jaccard 1.00, held-out delta 0.0000 across 5 scenarios x 3 seeds, p in 30..600) but NO replicated wall win
        # (median 1.00x, 3/15 wins) -- the outer-loop early-stop terminates the dichotomic search before the pool is ever large+
        # flat enough for a coarse stride to SAVE an iteration (iters identical auto/midpoint every row). Kept as an opt-in for
        # future re-test on harder curves / different stopping budgets.
        dichotomic_step: str = "midpoint",
        # ----- Wave 3 FI-semantics knobs (2026-05-28) -----
        # F8: exponential decay of FI history weights. With rate=r>0, a K-iter-old FI run weighs (1-r)^K vs the freshest run = 1.0.
        # Without decay, voting treats iter-1 FI (on the full feature set) and iter-30 FI (on the narrowed-down survivor set) equally,
        # even though late-iter FI is generally more reliable (fewer correlated co-features). Recommended 0.02-0.1 for runs >=30 iters.
        # Default 0.0 = no decay (legacy).
        fi_decay_rate: float = 0.0,
        # F5: how to collapse multi-class one-vs-rest |coef_| across classes. 'max' (NEW default) -> a feature important for ANY class is
        # important; 'sum' (legacy) -> mixes class-specific signals into a noisy aggregate (a 1-class discriminator equals a mid-relevance
        # feature for every class). 'max' matches what permutation/SHAP would produce; 'sum' is documented for back-compat A/B.
        multiclass_coef_aggregation: str = "max",
        # F4: which set to use for coef_ scale correction. 'train' (NEW default) -> stds computed from X_train of the fold (the data the
        # model was fitted on); 'test' (legacy) -> stds from X_test (leaks test variance into FI). 'none' -> skip the rescale entirely.
        coef_scale_source: str = "train",
        # F10: max_depth for the conditional-permutation auxiliary tree. None (NEW default) lets the tree grow until min_samples_leaf
        # constraint kicks in (more reliable conditioning on high-d feature sets). Integer caps depth at that value (legacy was 5).
        cpi_max_depth: Union[int, None] = None,
        # F10: min_samples_leaf for the conditional-permutation auxiliary tree. Default 10 (NEW) follows Strobl 2008 (>=5 recommended).
        cpi_min_samples_leaf: int = 10,
        # repeats for 'permutation' / 'conditional_permutation' importance (forwarded to get_feature_importances at each
        # fold + the stability bootstraps). Surfaced for tuning; default 5 keeps prior behaviour.
        n_repeats: int = 5,
        # Wide-data perm-FI cost guard (2026-06-04). Permutation / conditional-permutation importance rescore the model
        # O(p * n_repeats) times PER FOLD. On wide frames a single RFECV iteration can exceed the whole runtime budget
        # (measured madelon p=500, n_repeats=5 -> ~208s/iter > a 180s budget), so only 2-3 iters complete, the CV curve
        # is ~3 points, and the N-rule lands at the over-selection. When True (NEW default) and the search universe
        # exceeds ``wide_data_fi_threshold``, RFECV falls back to the estimator's native (gain/impurity) importance for
        # the elimination ranking so the outer loop can build a REAL multi-point curve in budget; ``wide_data_fi_n_repeats``
        # caps n_repeats just under the threshold to soften the cliff. False = exact permutation FI regardless of p.
        wide_data_fi_fallback: bool = True,
        wide_data_fi_threshold: int = 200,
        wide_data_fi_n_repeats: int = 2,
        # F6: when False (NEW default), multi-estimator + AM/GM auto-falls-back to Borda. Set True to keep the user's choice for
        # benchmark / A-B purposes.
        allow_unsafe_aggregation: bool = False,
        # F14: when True (NEW default), drop per-fold-per-estimator FI runs whose score was NaN (estimator failed / degenerate fold). Set
        # False to keep them (legacy contaminates voting with garbage importances from collapsed fits).
        drop_nan_score_fi: bool = True,
        # ----- TODO A / Wave 6 prelim (2026-05-28): auto-parameter tuning -----
        # When True, ``fit`` first computes a DataFingerprint from (X, y) and
        # picks SearchConfig + FIConfig + RobustnessConfig from a rule-based
        # table (later: ML classifier trained on synthetic-bench sweep). The
        # chosen combo is stored in ``self.auto_tune_decision_`` for inspection.
        # Any flat kwarg or config explicitly passed by the caller is PRESERVED
        # (auto-tune only fills the unspecified slots).
        auto_tune: bool = False,
        # ----- Wave 4 robustness knobs (2026-05-28) -----
        # E15: when True (NEW default), raise if must_exclude contains names not in X (catches typos). False = silently ignore (legacy).
        must_exclude_strict: bool = True,
        # C8: when False (NEW default), the no-improve counter only ticks when the iter actually stored a new best subset. Multi-revisit
        # of the same N with a worse subset no longer trips max_noimproving_iters prematurely. True = legacy.
        noimprove_counts_revisit: bool = False,
        # ----- Wave 5 / L4-L7 (2026-05-28) -----
        # L7: optional prescreen pass run BEFORE the MBH outer loop. Reduces the universe original_features to a smaller candidate set
        # so MBH explores a tighter space. Supported values:
        #   None              (default) : no prescreen
        #   'univariate_ht'              : in-tree native Mann-Whitney / Kruskal-Wallis / Kendall / chi-squared + BY-FDR
        #                                  (see _univariate_ht.py; numba-compiled rank/U/H/tau kernels).
        #   callable(X, y) -> list       : user-supplied prescreen returning the kept feature names/indices
        prescreen: Union[str, Callable, None] = None,
        # L7: top-K cap on the prescreen output. None = keep every feature that passes the prescreen's own significance filter.
        prescreen_top_k: Union[int, None] = None,
        # L7: relevance p-value FDR-level (Benjamini-Yekutieli). 0.05 is the standard default.
        prescreen_fdr_level: float = 0.05,
        # audit4-C (2026-07-03): honest nested prescreen. The full-data prescreen still defines the SEARCH universe
        # (legitimate for the final model), but when True the per-fold cv_mean_perf is computed against a prescreen
        # re-derived on that fold's TRAIN rows only, so a feature that survived the global prescreen purely via
        # test-fold leakage is dropped in folds where it fails the train-only prescreen -- eliminating the
        # selection-metric optimism. Default True (honest); set False for the cheaper legacy in-universe estimate.
        prescreen_nested: bool = True,
        # multioutput_strategy: how to handle a 2D y (multilabel / multi-target regression). sklearn RFE/RFECV is single-target, so we fit one
        # single-target RFECV per output column and aggregate the per-column support_. Default 'union' (OR) just works out of the box -- keeps a
        # feature selected for ANY output (recall-oriented, never drops a feature useful to one target). 'intersect' (AND) keeps only features
        # selected for EVERY output (precision-oriented). Set None to opt OUT and get the historical clear NotImplementedError on a 2D y. Each
        # sub-fit clones the full configured RFECV on one y column.
        multioutput_strategy: Union[str, None] = "union",
        # drop_id_like_sequences: when True (default), drop near-unique columns whose sorted distinct values are (near-)perfectly affine-spaced at fit entry --
        # an enumerated row-id / index / counter (or an affine rescale of one). Such a column is pure sample-order with ZERO generalisable signal yet a tree
        # estimator memorises it via split-frequency bias and admits it into support_. The guard is deliberately NARROW: it fires only on the structureless
        # affine-sequence shape (spacing coefficient-of-variation <= id_like_spacing_cv), so a continuous real signal (spacing CV ~O(1)) and a hash-style random
        # id (irregular spacing) are NEVER touched -- it cannot drop a weak recoverable signal. Set False to disable.
        drop_id_like_sequences: bool = True,
        id_like_ratio_threshold: float = 0.999,
        id_like_spacing_cv: float = 1e-3,
        # drop_near_dup_corr: when True (default), drop columns that are a (near-)monotone copy of another already kept -- a scaled / shifted / tiny-noise replica
        # (``100*x``, ``x + 1e-3*eps``) the exact-dup hash misses bit-for-bit. RFECV's voting otherwise splits the replica's importance across the copies and admits
        # the redundant copy into support_. Uses |Spearman| (rank) so any monotone rescale is caught regardless of slope/offset; keeps the FIRST of each pair. NARROW
        # BY CONSTRUCTION: the 0.999 default fires only on a near-perfect monotone replica -- a legitimately-distinct correlated pair (corr ~0.7, even ~0.95 cluster
        # mates) sits far below it and BOTH survive, so it cannot drop a weak recoverable signal. Reducing a genuine high-VIF cluster to a representative remains the
        # redundancy-aware GroupAwareMRMR wrapper's job (cluster_reduce=True). Set False to disable.
        drop_near_dup_corr: bool = True,
        near_dup_corr_threshold: float = 0.999,
        # nan_in_X_policy: how to handle NaN cells in X at fit entry, mirroring MRMR's native-NaN contract for cross-selector
        # consistency. The default sklearn linear cores (LogisticRegression / Ridge) raise ``ValueError: Input X contains NaN``,
        # so ordinary real-world missing data used to hard-crash RFECV. Options:
        #   'impute' (NEW default): graceful median-impute NaN per column on the local working frame so the linear core fits.
        #       When the core estimator NATIVELY tolerates NaN (HistGradientBoosting / CatBoost / LightGBM / XGBoost - detected via
        #       the sklearn ``allow_nan`` input tag + a known-name allowlist) imputation is SKIPPED and NaN passes through untouched.
        #       The impute path is a strict no-op on a NaN-free frame, so non-NaN selection is byte-identical to before.
        #   'raise': preserve the strict legacy crash on any NaN (benchmarks / replay).
        nan_in_X_policy: str = "impute",
        # nan_indicator_cols: optional column names for which an ``is_missing__{col}`` 0/1 indicator is appended at fit entry
        # (built from the PRE-impute mask) so a missingness-carried signal (MNAR) stays capturable even after the value is imputed.
        # Mirrors MRMR's Layer-37 ``fe_missingness_indicator`` emitter, which is itself opt-in, so this defaults to () (OFF) for
        # cross-selector parity. Indicators are pandas-only (they need column names); ignored for raw ndarray X.
        nan_indicator_cols: Union[Sequence, None] = (),
        # importance_agg: how per-fold importances are aggregated across CV folds into the elimination ranking.
        #   'legacy'     : historical mean+vote (Leaderboard / votes_aggregation_method) on abs'd per-fold FI.
        #   'dispatched' (default): estimator-type-aware. TREE/GBM -> mean down-weighted by cross-fold CV
        #       (mean/(1+cv)); LINEAR -> sign-harmony on SIGNED coef (|mean signed| * sign-agreement) so a
        #       feature whose sign flips across folds is demoted; KERNEL / no native FI -> defers to the legacy
        #       vote. Flipped to default after a multi-scenario x multi-seed honest-holdout win (see
        #       _benchmarks/bench_rfecv_importance_agg.py). Falls back to legacy automatically when family info
        #       or signed coef are unavailable.
        importance_agg: str = "dispatched",
        # k_cv: tree-family variance penalty strength in importance_agg='dispatched'; score=mean/(1+k_cv*cv).
        importance_agg_k_cv: float = 1.0,
        # elimination_rule: how the per-iteration elimination ranking is formed from the cross-fold FI table.
        #   'importance' (default): rank by aggregated importance (legacy / importance_agg path).
        #   'stability'  (opt-in) : rank by mean_importance * fold_selection_frequency, where frequency is the
        #       fraction of folds in which the feature lands in the top-N (survives the cut in that fold alone).
        #       Protects steady-mid-rank features from one-fold-noise eviction. Operates on the raw per-fold
        #       table independently of importance_agg (no double-count). Kept opt-in pending a replicated win.
        elimination_rule: str = "importance",
    ):

        # checks
        if frac is not None:
            if not (frac > 0.0 and frac < 1.0):
                raise ValueError(f"frac must be between 0 and 1, got {frac}")
            if verbose:
                logger.info("Using %s fraction of the training dataset.", frac)

        # max_refits=0 would be silently ignored by ``if max_refits and ...`` (0 is falsy). Reject explicitly.
        if max_refits is not None and max_refits < 1:
            raise ValueError(f"max_refits must be >= 1 (or None for unlimited); got {max_refits}. " f"To run zero iterations, just don't call fit().")

        # cv=1 is degenerate (no train/test split possible).
        if isinstance(cv, int) and cv < 2:
            raise ValueError(f"cv must be >= 2 (or a CV splitter object); got cv={cv}. " f"k-fold CV requires at least 2 splits.")

        if stability_selection:
            if not (0.0 < stability_threshold <= 1.0):
                raise ValueError(f"stability_threshold must be in (0, 1]; got {stability_threshold}.")
            if stability_n_bootstrap < 10 and verbose:
                logger.warning(
                    "RFECV: stability_n_bootstrap=%d is below the recommended "
                    "minimum of 10. Bootstrap voting is statistically meaningful "
                    "only with B >= 10; expect noisy / unstable selection.",
                    stability_n_bootstrap,
                )
            if stability_n_bootstrap < 1:
                raise ValueError(f"stability_n_bootstrap must be >= 1; got {stability_n_bootstrap}.")

        if feature_groups:
            for _gname, _gmembers in feature_groups.items():
                if not _gmembers:
                    if verbose:
                        logger.warning(
                            "RFECV: feature_groups[%r] is empty; this group " "will have no effect on selection.",
                            _gname,
                        )

        if nan_in_X_policy not in ("impute", "raise"):
            raise ValueError(f"nan_in_X_policy must be 'impute' or 'raise'; got {nan_in_X_policy!r}.")

        if leakage_action not in ("warn", "exclude", "raise"):
            raise ValueError(f"leakage_action must be 'warn', 'exclude', or 'raise'; " f"got {leakage_action!r}.")

        if n_features_selection_rule not in ("auto", "argmax", "one_se_min", "one_se_max", "plateau"):
            raise ValueError(
                f"n_features_selection_rule must be 'auto', 'argmax', " f"'one_se_min', 'one_se_max', or 'plateau'; got {n_features_selection_rule!r}."
            )

        if fi_missing_policy not in ("worst", "median", "skip"):
            raise ValueError(f"fi_missing_policy must be 'worst', 'median', or 'skip'; " f"got {fi_missing_policy!r}.")

        if multioutput_strategy not in (None, "union", "intersect"):
            raise ValueError(f"multioutput_strategy must be None, 'union', or 'intersect'; got {multioutput_strategy!r}.")
        if importance_agg not in ("legacy", "dispatched"):
            raise ValueError(f"importance_agg must be 'legacy' or 'dispatched'; got {importance_agg!r}.")

        if elimination_rule not in ("importance", "stability"):
            raise ValueError(f"elimination_rule must be 'importance' or 'stability'; got {elimination_rule!r}.")

        if optimizer_target not in ("mean", "final_score"):
            raise ValueError(f"optimizer_target must be 'mean' or 'final_score'; got {optimizer_target!r}.")

        if not (0.0 <= dichotomic_epsilon <= 1.0):
            raise ValueError(f"dichotomic_epsilon must be in [0, 1]; got {dichotomic_epsilon}.")

        if dichotomic_step not in ("auto", "midpoint"):
            raise ValueError(f"dichotomic_step must be 'auto' or 'midpoint'; got {dichotomic_step!r}.")

        if not (0.0 <= fi_decay_rate < 1.0):
            raise ValueError(f"fi_decay_rate must be in [0, 1); got {fi_decay_rate}.")

        if multiclass_coef_aggregation not in ("max", "sum"):
            raise ValueError(f"multiclass_coef_aggregation must be 'max' or 'sum'; got {multiclass_coef_aggregation!r}.")

        if coef_scale_source not in ("train", "test", "none"):
            raise ValueError(f"coef_scale_source must be 'train', 'test', or 'none'; got {coef_scale_source!r}.")

        # E9 (Wave 4, 2026-05-28): assert all entries in ``estimators=`` share the
        # same type-family (classifier vs regressor). Mixing silently picks the
        # first estimator's family for CV stratification / scoring and the other
        # estimator crashes mid-fold with cryptic errors.
        if estimators:
            from sklearn.base import is_classifier as _is_clf
            from sklearn.base import is_regressor as _is_reg
            _est_list = list(estimators)
            if len(_est_list) >= 2:
                _is_clf_flags = [_is_clf(e) for e in _est_list]
                _is_reg_flags = [_is_reg(e) for e in _est_list]
                # Allow estimators that are NEITHER (custom transformers used as importers)
                # but reject mixing detectable classifier+regressor.
                if any(_is_clf_flags) and any(_is_reg_flags):
                    raise ValueError(
                        "RFECV: estimators=[...] mixes classifier and regressor "
                        "families. All entries must share type-family for CV "
                        "stratification + scoring to be consistent. Got: "
                        f"{[type(e).__name__ for e in _est_list]}"
                    )

        # F6 (Wave 3, 2026-05-28): multi-estimator + AM/GM is unsafe.
        # LR coef ~ 0.01..1, RF Gini ~ 0..0.1, CB split-gain ~ thousands -- raw
        # arithmetic / geometric mean is dominated by the largest-magnitude
        # estimator (AM) or zeroed by any single zero (GM). Force rank-based
        # rules (Borda / Copeland) for multi-estimator. User can opt back in
        # via allow_unsafe_aggregation=True for benchmarks.
        if estimators and votes_aggregation_method in (VotesAggregation.AM, VotesAggregation.GM):
            if verbose:
                logger.warning(
                    "RFECV: multi-estimator + votes_aggregation_method=%s mixes raw "
                    "FI values from incomparable scales (LR coef ~ 0.01-1, RF Gini ~ "
                    "0-0.1, CB split-gain ~ 1000s). Switching to Borda (rank-based). "
                    "Pass allow_unsafe_aggregation=True to keep the original choice.",
                    votes_aggregation_method,
                )

        # E3 (Wave 1, 2026-05-28): feature_groups overlap is silently expanded into a contradictory all-or-nothing rule that grows
        # the support_ by every group member of every overlapping group whenever ANY shared member is picked. Reject upfront.
        if feature_groups:
            _seen: dict = {}
            for _gname, _gmembers in feature_groups.items():
                for _m in _gmembers or []:
                    if _m in _seen:
                        raise ValueError(
                            f"feature_groups: column {_m!r} appears in BOTH "
                            f"group {_seen[_m]!r} and group {_gname!r}. Groups "
                            f"must be disjoint - the all-or-nothing rule "
                            f"otherwise expands to the union of every "
                            f"overlapping group when any shared member is "
                            f"picked, producing a wider support_ than asked."
                        )
                    _seen[_m] = _gname

        params = get_parent_func_args()
        # 2026-05-28: grouped-config merge. When a SearchConfig / FIConfig /
        # RobustnessConfig is passed, its EXPLICITLY-SET fields (per pydantic
        # v2's ``model_fields_set``) override the matching flat kwarg before
        # ``store_params_in_object`` writes them onto ``self``. Default fields
        # of a config object do NOT clobber explicit flat values, so a caller
        # who passes ``RFECV(estimator=lr, max_refits=20)`` AND a default
        # ``SearchConfig()`` keeps max_refits=20.
        for _cfg in (search_config, fi_config, robustness_config):
            if _cfg is None:
                continue
            _set_fields = getattr(_cfg, "model_fields_set", None)
            if _set_fields is not None:
                _dump = {k: getattr(_cfg, k) for k in _set_fields}
            elif hasattr(_cfg, "model_dump"):
                _dump = _cfg.model_dump()
            else:
                _dump = {k: v for k, v in vars(_cfg).items() if not k.startswith("_")}
            for _k, _v in _dump.items():
                if _k in params:
                    params[_k] = _v
        store_params_in_object(obj=self, params=params)
        self.signature = None

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
        import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file
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
            # Write the sha256 sidecar so _load_checkpoint's safe_load round-trips. The sidecar is an integrity/corruption gate, not an
            # authenticity control: an attacker with write access to checkpoint_path rewrites both payload and sidecar (see safe_pickle docs).
            from mlframe.utils.safe_pickle import write_sidecar
            write_sidecar(path)
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
        import pickle  # nosec B403 - pickle used only for trusted same-process/dev-local round-trips, see call sites in this file

        path = self.checkpoint_path
        if not path:
            return None
        # Wave 48 (2026-05-20): the prior exists-then-open pattern was a TOCTOU race
        # with concurrent RFECV runs sharing checkpoint_path (or an external cleanup
        # cron); FileNotFoundError/OSError would propagate uncaught and abort the fit.
        # Drop the redundant exists check; add OSError/FileNotFoundError to the except.
        from mlframe.utils.safe_pickle import PickleVerificationError, safe_load
        try:
            # Route the caller-supplied resume checkpoint through the sidecar-gated loader: a missing/mismatched .sha256 fails closed
            # (refuse to unpickle, start fresh) rather than running arbitrary pickle from a tampered checkpoint file.
            state = safe_load(path)
        except FileNotFoundError:
            return None
        except PickleVerificationError as exc:
            logger.warning(
                "RFECV: checkpoint at %s failed sha256 sidecar verification (%s); starting from scratch.",
                path, exc,
            )
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
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("RFECV is not fitted; call fit() first.")
        # sklearn ``_check_feature_names_in`` contract: when the caller passes input_features it MUST match the fitted width, else raise (column-drift detection). A correct-length input_features overrides the stored feature_names_in_ -- this lets a caller re-inject real names after an ndarray fit (which synthesized x0..xN placeholders).
        names_in = getattr(self, "feature_names_in_", None)
        if input_features is not None:
            input_features = list(input_features)
            n_in = int(getattr(self, "n_features_in_", len(input_features)))
            if len(input_features) != n_in:
                raise ValueError(
                    f"input_features has {len(input_features)} elements, expected {n_in} "
                    f"(n_features_in_). The names passed to get_feature_names_out must match the "
                    f"feature set RFECV was fit on (sklearn column-drift contract)."
                )
            names_in = input_features
        else:
            cache = getattr(self, "_selected_cols_cache", None)
            if cache is not None:
                return np.asarray(cache, dtype=object)
        if len(self.support_) == 0:
            return np.array([], dtype=object)
        if names_in is None:
            names_in = [f"x{i}" for i in range(int(getattr(self, "n_features_in_", len(self.support_))))]
        if isinstance(self.support_[0], (bool, np.bool_)):
            return np.asarray(
                [c for c, s in zip(names_in, self.support_) if s],
                dtype=object,
            )
        return np.asarray([names_in[i] for i in self.support_], dtype=object)

    def get_support(self, indices: bool = False):
        """sklearn ``SelectorMixin`` protocol. Returns a bool mask of length
        ``n_features_in_`` (``indices=False``) or the integer indices of the
        selected features (``indices=True``). ``support_`` is stored as either a
        bool mask or an int-index array depending on the fit path, so normalise
        both shapes here -- sklearn tooling that introspects selection via
        ``get_support`` (SelectFromModel-style pipelines, set_output column
        derivation) relies on this method existing.
        """
        if not hasattr(self, "support_"):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("RFECV is not fitted; call fit() first.")
        n = int(getattr(self, "n_features_in_", len(self.support_)))
        support = np.asarray(self.support_)
        if support.size and isinstance(self.support_[0], (bool, np.bool_)):
            mask = support.astype(bool)
        else:
            mask = np.zeros(n, dtype=bool)
            if support.size:
                mask[support.astype(int)] = True
        return np.where(mask)[0] if indices else mask

    def transform(self, X, y=None):
        # Polars X (callers like _passthrough_cols_fit_transform keep the native frame) breaks the legacy ``X[:, self.support_]`` mask
        # path with ``expected N values when selecting columns by boolean mask, got M`` when the polars schema has more cols than the
        # fit-time support_ (because RFECV.fit dropped zero-variance cols at entry). Convert to pandas so the name-keyed transform path
        # kicks in and column-set drift becomes a clear RuntimeError instead of an opaque polars index mismatch.
        if isinstance(X, pl.DataFrame):
            # Mirror the sibling fit-path bridge kwargs (_rfecv_fit.py:102): use_pyarrow_extension_array+split_blocks+self_destruct keep numeric
            # columns zero-copy through the Arrow split-blocks bridge instead of densifying into a single block on transform.
            try:
                X = X.to_pandas(use_pyarrow_extension_array=True, split_blocks=True, self_destruct=True)
            except TypeError:
                X = X.to_pandas()
        # transform on an unfitted estimator must raise NotFittedError; silently returning X unchanged masquerades a config bug as a
        # successful transform and lets downstream pipelines run on the wrong column set.
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("This RFECV instance is not fitted yet. Call 'fit' before " "using 'transform'.")
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
from ._fit import fit as _fit_func  # noqa: E402


@functools.wraps(_fit_func)
def _fit_with_rng_hygiene(self, *args, **kwargs):
    # _rfecv_fit.fit calls set_random_seed(random_state) to make sub-estimators
    # with random_state=None reproducible WITHIN the fit; that clobbers the
    # caller's process-global numpy/random RNG (the violation set_random_seed's
    # own docstring forbids). Snapshot+restore around fit so within-fit
    # determinism is bit-identical while the caller's global RNG resumes
    # untouched (RFECV's only global-RNG touch is that seed call).
    from mlframe.utils.misc import preserve_global_rng
    with preserve_global_rng():
        return _fit_func(self, *args, **kwargs)


RFECV.fit = _fit_with_rng_hygiene

from ._stability_select import (  # noqa: E402
    _fit_stability_selection as _fit_stability_selection_func,
)
from ._stability_select import (
    select_optimal_nfeatures_ as _select_optimal_nfeatures_func,
)

RFECV._fit_stability_selection = _fit_stability_selection_func
RFECV.select_optimal_nfeatures_ = _select_optimal_nfeatures_func

from ._diagnostics import (  # noqa: E402,F401
    cv_results_df_ as _cv_results_df_func,
)
from ._diagnostics import (
    n_features_bootstrap_ci_ as _n_features_bootstrap_ci_func,
)
from ._diagnostics import (
    n_features_one_se_ as _n_features_one_se_func,
)
from ._diagnostics import (
    n_stability_elbow_ as _n_stability_elbow_func,
)
from ._diagnostics import (
    pareto_front_ as _pareto_front_func,
)
from ._diagnostics import (
    pareto_knee_ as _pareto_knee_func,
)
from ._diagnostics import (
    selection_stability_ as _selection_stability_func,
)
from ._diagnostics import (
    stability_vs_n_curve_ as _stability_vs_n_curve_func,
)

# cv_results_df_ is exposed as a property for sklearn parity; the rest are plain methods.
RFECV.cv_results_df_ = property(_cv_results_df_func)
RFECV.selection_stability_ = _selection_stability_func
RFECV.n_features_one_se_ = _n_features_one_se_func
RFECV.stability_vs_n_curve_ = _stability_vs_n_curve_func
RFECV.n_stability_elbow_ = _n_stability_elbow_func
RFECV.pareto_front_ = _pareto_front_func
RFECV.pareto_knee_ = _pareto_knee_func
RFECV.n_features_bootstrap_ci_ = _n_features_bootstrap_ci_func

from ._sffs import _sffs_swap_pass as _sffs_swap_pass_func  # noqa: E402

RFECV._sffs_swap_pass = _sffs_swap_pass_func
