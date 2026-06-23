"""``BorutaShap.fit`` + ``BorutaShap.explain`` carved out of the ``mlframe.feature_selection.boruta_shap`` package facade.

Methods are bound onto the ``BorutaShap`` class in the package ``__init__`` so ``self.fit(...)`` / ``self.explain(...)`` call sites resolve unchanged.
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


def _premerge_collapse(X, thr):
    """premerge_clusters: collapse |Pearson| >= thr clusters of X to one representative (first column of each
    cluster). Returns (X_reps, rep_members) where rep_members maps representative -> [member columns incl. rep]."""
    cols = list(X.columns)
    # Factor-code non-numeric columns: ``np.corrcoef`` divides/means over the raw
    # values and would raise on string categorical levels (e.g. ``"NA"`` / ``""``).
    _num = np.empty((len(X), len(cols)), dtype=np.float64)
    for _j, _c in enumerate(cols):
        _s = X[_c]
        if pd.api.types.is_numeric_dtype(_s.dtype):
            _num[:, _j] = _s.to_numpy(dtype=np.float64, na_value=np.nan)
        else:
            _num[:, _j] = pd.factorize(_s, use_na_sentinel=True)[0]
    C = np.nan_to_num(np.corrcoef(_num, rowvar=False))
    if C.ndim == 0:
        return X, {cols[0]: [cols[0]]}
    reps, members, seen = [], {}, set()
    for i, c in enumerate(cols):
        if c in seen:
            continue
        reps.append(c); members[c] = [c]; seen.add(c)
        for j in range(i + 1, len(cols)):
            if cols[j] not in seen and abs(C[i, j]) >= thr:
                members[c].append(cols[j]); seen.add(cols[j])
    return X[reps], members


def _premerge_expand(accepted_reps, rep_members, original_cols):
    """Re-expand accepted representative columns to all their cluster members. Returns (kept_set, support_mask)
    over the ORIGINAL column order."""
    kept = set()
    for r in accepted_reps:
        kept.update(rep_members.get(r, [r]))
    return kept, np.array([c in kept for c in original_cols], dtype=bool)


def _naive_accepted_set_stable(accepted_history, patience):
    """REJECTED naive trial-stop -- kept as a DOCUMENTED CONTRAST, deliberately NOT wired into the loop.

    The naive rule stops as soon as the cumulative accepted set has been unchanged for ``patience`` consecutive
    trials. It is a measured CORRECTNESS TRAP: on the fs_hybrid bed it fired on a TRANSIENT plateau while weak
    features were still slowly crossing -- synth Jaccard(vs full cap)=1.0 but hard_synth Jaccard=0.0 (it locked a
    WRONG accepted set). Do NOT enable this. The safe stop (``_should_stop_tentative_tail``) additionally requires
    that no still-tentative feature is near a decision boundary, so it only stops when the tail is provably stuck.

    ``accepted_history`` is the list of cumulative accepted-set snapshots (one frozenset per completed trial).
    Returns True when the last ``patience``+1 snapshots are identical. Exposed for unit-testing the contrast only.
    """
    if len(accepted_history) <= patience:
        return False
    recent = accepted_history[-(patience + 1):]
    return all(s == recent[0] for s in recent[1:])


def _tentative_near_boundary(hits, tentative_indices, iteration, n_tests, pvalue, margin, null_p=0.5):
    """Margin gate primitive: is ANY still-tentative feature within ``margin`` of crossing a binomial threshold?

    Replays the EXACT prod decision statistic used in ``test_features``: a two-sided-style pair of one-sided exact
    binomial tests (greater -> accept, less -> reject) at the null hit probability ``null_p`` over ``iteration``
    trials, Bonferroni-corrected by multiplying by ``n_tests`` (capped at 1.0). ``null_p`` MUST match the value
    ``test_features`` uses (percentile-derived, ~ (100 - percentile)/100), otherwise the margin gate replays a
    different statistic than the actual decision and the early-stop is no longer decision-equivalent. A tentative
    feature is "near a boundary" when either corrected p-value < ``pvalue * (1 + margin)``. Returns True if any
    tentative feature is near (so it is NOT yet safe to stop); False when the whole tail is provably stuck.
    """
    thr = pvalue * (1.0 + margin)
    for i in tentative_indices:
        h = hits[i]
        acc_pc = min(binom_test(h, n=iteration, p=null_p, alternative="greater") * n_tests, 1.0)
        if acc_pc < thr:
            return True
        rej_pc = min(binom_test(h, n=iteration, p=null_p, alternative="less") * n_tests, 1.0)
        if rej_pc < thr:
            return True
    return False


def _should_stop_tentative_tail(accepted_history, hits, tentative_indices, iteration, n_tests, pvalue, patience, margin, null_p=0.5):
    """MARGIN-GATED adaptive trial-stop (the SHIPPED safe rule).

    Stop the trial loop ONLY when BOTH hold:
      (a) the cumulative accepted set has been unchanged for ``patience`` consecutive trials (a plateau), AND
      (b) NO still-tentative feature is within ``margin`` of crossing either binomial decision threshold given the
          hits accumulated so far (the tail is provably stuck, not transiently flat).

    Condition (b) is what makes this decision-equivalent to running the full n_trials cap: a feature that could
    still cross keeps the loop alive, so the accepted/rejected partition at the stop matches the full-cap partition
    (measured synth Jaccard 1.0, hard_synth Jaccard 0.842). The naive rule drops (b) and locks a wrong set -- see
    ``_naive_accepted_set_stable``. Returns True iff it is safe to stop now.
    """
    if not _naive_accepted_set_stable(accepted_history, patience):
        return False
    # Plateau reached; refuse to stop while any tentative feature is still near a boundary.
    return not _tentative_near_boundary(hits, tentative_indices, iteration, n_tests, pvalue, margin, null_p=null_p)


def _fit_with_subsample_stability(self, X, y):
    """Run BorutaShap on several distinct row-subsamples (WITHOUT replacement) and keep only features
    ACCEPTED in >= stability_threshold of them. Removes the finite-sample-spurious top real-noise column
    that a single fit leaks past the shadow gate (see class docstring), while keeping genuinely-relevant
    features. Returns self with the usual accepted/rejected/tentative/support_/selected_features_ set.
    """
    from sklearn.base import clone as _sk_clone

    # Normalise to pandas for positional subsampling (the per-subsample fit re-handles dtypes/cats).
    if pl is not None and isinstance(X, pl.DataFrame):
        try:
            from mlframe.training.utils import get_pandas_view_of_polars_df
            X = get_pandas_view_of_polars_df(X)
        except ImportError:
            X = X.to_pandas(use_pyarrow_extension_array=True)
    if pl is not None and isinstance(y, pl.Series):
        y = y.to_pandas()
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    all_columns = list(X.columns)
    n = len(X)
    n_sub = int(self.stability_subsamples)
    frac = float(getattr(self, "stability_subsample_fraction", 0.75) or 0.75)
    thr = float(getattr(self, "stability_threshold", 0.6) or 0.6)
    # Cap the >=10-row floor by n: with replace=False, np.random.Generator.choice raises when size>n,
    # so on a tiny frame (n<10) the bare max(10, ...) floor would request more rows than exist and crash.
    # min(n, max(10, ...)) keeps the intended floor when the data allows it, never exceeding the population.
    size = min(n, max(10, int(round(frac * n))))
    base_seed = int(getattr(self, "random_state", 0) or 0)

    params = self.get_params(deep=False)  # shallow: deep=True yields model__* nested keys __init__ rejects
    params["stability_subsamples"] = 0  # recursion guard: each sub-fit is a single pass
    params["verbose"] = False

    # ``get_params`` copies ``stratify`` verbatim, so every sub-fit would carry the original length-n
    # stratify array while running on a ``size``-row subsample. When train_or_test='test',
    # Check_if_chose_train_or_test_and_train_model passes it to train_test_split(X_sub, y_sub, stratify=...)
    # and the length mismatch raises ValueError. Detect a per-row stratify array here so each sub-fit can
    # slice it positionally with its own ``idx`` (scalars / None / mismatched lengths are passed through).
    _stratify = getattr(self, "stratify", None)
    _stratify_arr = None
    if _stratify is not None:
        _strat_candidate = np.asarray(_stratify.to_numpy() if hasattr(_stratify, "to_numpy") else _stratify)
        if _strat_candidate.ndim == 1 and _strat_candidate.shape[0] == n:
            _stratify_arr = _strat_candidate

    def _run_one_subfit(k: int):
        rng = np.random.default_rng(base_seed + 1 + k)
        idx = rng.choice(n, size=size, replace=False)
        Xk = X.iloc[idx].reset_index(drop=True)
        yk = pd.Series(y_arr[idx]).reset_index(drop=True)
        sub_params = dict(params)
        _model_k = _sk_clone(self.model) if self.model is not None else None
        # Vary the estimator's own randomness per subsample too (not just the rows). Without this the cloned
        # model keeps its fixed random_state, so the finite-sample-spurious noise importance stays CORRELATED
        # across subsamples and survives the vote; varying it (as in Meinshausen-Buhlmann stability selection)
        # decorrelates the spurious structure so the lucky-noise column changes per subsample and is voted out.
        if _model_k is not None:
            try:
                if "random_state" in _model_k.get_params(deep=False):
                    _model_k.set_params(random_state=base_seed + 1 + k)
            except (TypeError, ValueError):
                pass
        sub_params["model"] = _model_k
        sub_params["random_state"] = base_seed + 1 + k
        # Subsample the per-row stratify array to match this sub-fit's rows; leave scalar/None as-is.
        if _stratify_arr is not None:
            sub_params["stratify"] = _stratify_arr[idx]
        sub = self.__class__(**sub_params)
        sub.fit(Xk, yk)
        return list(getattr(sub, "accepted", []) or [])

    # The n_sub sub-fits are fully independent (distinct row-subsamples, distinct seeds, own cloned model)
    # and each is a full BorutaShap run (the dominant FS cost: up to n_trials model fits + a SHAP
    # TreeExplainer per trial). Run them concurrently via joblib's in-process threading backend so the
    # native estimator/SHAP work (which releases the GIL) overlaps; threading avoids the loky pickle /
    # matplotlib-in-worker hazards of the process backend while still parallelising the dominant cost.
    _n_jobs = int(getattr(self, "stability_n_jobs", -1) or -1)
    if n_sub > 1 and _n_jobs != 1:
        from joblib import Parallel, delayed
        _per_sub_accepted = Parallel(n_jobs=_n_jobs, backend="threading")(delayed(_run_one_subfit)(k) for k in range(n_sub))
    else:
        _per_sub_accepted = [_run_one_subfit(k) for k in range(n_sub)]

    accept_counts: dict = {c: 0 for c in all_columns}
    for _accepted in _per_sub_accepted:
        for c in _accepted:
            if c in accept_counts:
                accept_counts[c] += 1

    need = max(1, int(np.ceil(thr * n_sub)))
    self.accepted = [c for c in all_columns if accept_counts[c] >= need]
    self.rejected = [c for c in all_columns if accept_counts[c] == 0]
    self.tentative = [c for c in all_columns if 0 < accept_counts[c] < need]
    # Intersection mode (stability_threshold==1.0, the default once stability is enabled) keeps ONLY
    # features accepted by ALL subsamples -- that is exactly set(self.accepted) with need==n_sub. The
    # 'tentative' bucket here is 0<count<need, i.e. features accepted by SOME but not all subsamples:
    # precisely the draw-level-spurious columns intersection is meant to drop. So in intersection mode we
    # must NOT let optimistic re-add tentative (which would silently keep the spurious column the class
    # docstring promises to remove). optimistic only applies to majority-vote thresholds (<1.0).
    if thr >= 1.0:
        kept = set(self.accepted)
    else:
        kept = set(self.accepted) | (set(self.tentative) if self.optimistic else set())
    if getattr(self, "_premerge_active_", False):  # re-expand accepted representatives to their cluster members
        kept, self.support_ = _premerge_expand(kept, self._premerge_members_, self._premerge_original_cols_)
        self.selected_features_ = [c for c in self._premerge_original_cols_ if c in kept]
        self.feature_names_in_ = np.asarray(self._premerge_original_cols_)
    else:
        self.support_ = np.array([c in kept for c in all_columns], dtype=bool)
        self.selected_features_ = [c for c in all_columns if c in kept]
        self.feature_names_in_ = np.asarray(all_columns)
    self.n_features_in_ = int(len(all_columns))
    self.stability_accept_counts_ = dict(accept_counts)  # diagnostic: per-feature accept frequency
    if self.verbose:
        logger.info(
            "BorutaShap stability: %d subsamples @%.0f%%, threshold>=%d/%d -> %d accepted, %d tentative, %d rejected",
            n_sub, 100 * frac, need, n_sub, len(self.accepted), len(self.tentative), len(self.rejected),
        )
    return self


def fit(self, X, y):
    """
    The main body of the program this method it computes the following

    1. Extend the information system by adding one shadow copy per real attribute (a value-permutation of that
    column). On frames with fewer than 5 real columns the shadow side is, by default, additionally padded by
    recycling real columns up to 5 shadows so the per-trial shadow-max null is not estimated from 1-2 draws (opt
    out via ``shadow_min_pad=0``; see ``create_shadow_features`` and ``shadow_min_pad`` in ``__init__``).

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

    # Deferred from __init__ (sklearn clone-ability): resolve model=None -> default RF and validate fit/predict here,
    # before either the stability orchestration (which clones self.model) or the single-pass body uses it.
    self.check_model()

    # importance_measure='auto': probe (X, y) ONCE here and pin the concrete driver (gini/permutation)
    # for this fit. No-op for explicit measures. Done before stability orchestration / single-pass body
    # so every sub-fit and the trial loop read the resolved value via _active_importance_measure().
    self._resolve_auto_importance_measure(X, y)

    # premerge_clusters (opt-in, off by default): collapse raw-corr clusters to one representative BEFORE the shadow
    # gate, so BOTH the stability path and the single-pass body run on the de-duplicated representative columns; the
    # finalization re-expands accepted reps to their members. Stored state is consumed at both finalization sites.
    self._premerge_active_ = bool(getattr(self, "premerge_clusters", False)) and getattr(X, "shape", (0, 0))[1] >= 2
    if self._premerge_active_:
        self._premerge_original_cols_ = list(X.columns)
        X, self._premerge_members_ = _premerge_collapse(X, float(getattr(self, "premerge_corr_thr", 0.92)))

    # Cross-subsample stability gate (opt-in). See BorutaShap class docstring: a single-sample shadow
    # comparison leaks the top finite-sample-spurious real-noise column; majority vote across distinct
    # row-subsamples (WITHOUT replacement) removes it. >1 guards against a degenerate single-subsample.
    if int(getattr(self, "stability_subsamples", 0) or 0) > 1:
        return _fit_with_subsample_stability(self, X, y)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

        # Memory: frames here can be 100+ GB, so the algorithm keeps exactly ONE full-frame copy. ``self.X``
        # is that single mutable working frame -- BorutaShap ordinal-encodes its object/category columns,
        # drops rejected columns, and appends shadow features into it. ``self.X = X.copy()`` makes that copy
        # independent of the caller, so neither the encode nor the drops mutate the caller's ``X``.
        # ``starting_X`` keeps the ORIGINAL-dtype frame for ``Subset()`` (__init__.py): since encoding now
        # happens only on the independent ``self.X`` copy, the untouched caller ``X`` already IS that
        # snapshot, so we reference it rather than taking a second copy. ``self.y`` is never mutated (only
        # read for splits / model fit), so it is referenced, not copied.
        self.starting_X = X
        self.X = X.copy()
        self.y = y
        # Duplicate column names make ``df[label]`` return a DataFrame (not a Series), whose ``.dtype`` access raises in the object-col encoder below, and break the column -> accepted-feature mapping. Surface a clear error before the encoder runs.
        if hasattr(self.X, "columns") and self.X.columns.has_duplicates:
            dup_names = self.X.columns[self.X.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"BorutaShap.fit: duplicate column names not supported: {dup_names[:10]}. "
                f"De-duplicate (e.g. ``X.loc[:, ~X.columns.duplicated()]`` or rename) before fitting."
            )
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

        # Control/safety budget (parity with MRMR / RFECV): a wall-clock timer + a
        # filesystem stop-flag, both checked at the top of each trial below so an
        # over-budget or stop-flagged run returns the features classified so far rather
        # than running the remaining (dominant-cost) trials. ``getattr`` keeps old
        # pickled instances (without the new attrs) working.
        from os.path import exists as _stop_file_exists
        from time import perf_counter as _budget_timer
        _fit_start_t = _budget_timer()
        _max_runtime_mins = getattr(self, "max_runtime_mins", None)
        _stop_file = getattr(self, "stop_file", None)
        self.n_trials_run_ = 0  # defensive: every early-break path leaves this set

        # Margin-gated adaptive trial-stop (opt-in; see __init__ doc + _should_stop_tentative_tail). Off by default
        # so the loop is byte-identical to the fixed-cap run. When on we track the cumulative accepted set after
        # each trial and stop once it has plateaued AND the residual tentative tail is provably stuck.
        _early_stop_tentative = bool(getattr(self, "early_stop_tentative", False))
        _early_stop_patience = int(getattr(self, "early_stop_patience", 20))
        _early_stop_margin = float(getattr(self, "early_stop_margin", 0.15))
        _accepted_history: list = []
        _n_total_cols = len(self.all_columns)  # full original count -> matches the validated bench Bonferroni base

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
                self.n_trials_run_ = trial + 1  # actual trials executed (< n_trials when early-terminated)

                # Early-termination: once NO feature is still tentative (every feature is confirmed or rejected)
                # the remaining trials only re-test already-confirmed features and cannot change the outcome.
                # Canonical Boruta stops here; this loop previously always ran all n_trials (the dominant cost).
                # Mirrors calculate_rejected_accepted_tentative's accepted/rejected/tentative partition exactly,
                # so the final support_ is identical to running every trial - this is a pure speedup.
                _acc = set(self.flatten_list(self.accepted_columns))
                _rej = set(self.flatten_list(self.rejected_columns)) - _acc
                if len(self.all_columns) - len(_acc) - len(_rej) == 0:
                    if self.verbose:
                        logger.info("BorutaShap: all features decided after %d/%d trials; stopping early.", trial + 1, self.n_trials)
                    break

                # Margin-gated adaptive trial-stop (opt-in) for the residual TENTATIVE TAIL that the all-decided
                # stop above never resolves. Stop only when the accepted set has plateaued for the patience window
                # AND no still-tentative feature can still cross a binomial threshold within the margin (the tail is
                # provably stuck). This is decision-equivalent to running the full cap; the naive 'plateau-only' rule
                # is deliberately NOT used (it locks a wrong set on a transient plateau -- see helper docstrings).
                if _early_stop_tentative:
                    _accepted_history.append(frozenset(_acc))
                    _decided = _acc | _rej
                    _tentative_idx = [idx for idx, col in enumerate(self.all_columns) if col not in _decided]
                    _null_hit_p = max(min((100.0 - float(self.percentile)) / 100.0, 1.0), 1e-9)
                    if _should_stop_tentative_tail(
                        _accepted_history, self.hits, _tentative_idx, iteration=trial + 1, n_tests=_n_total_cols,
                        pvalue=self.pvalue, patience=_early_stop_patience, margin=_early_stop_margin, null_p=_null_hit_p,
                    ):
                        if self.verbose:
                            logger.info(
                                "BorutaShap: margin-gated stop after %d/%d trials (accepted set plateaued for %d "
                                "trials and %d tentative feature(s) provably stuck).",
                                trial + 1, self.n_trials, _early_stop_patience, len(_tentative_idx),
                            )
                        break

            # Control/safety budget + stop-flag (parity with MRMR / RFECV), checked at the END of each
            # trial so the just-completed trial's state is valid (>=1 trial always runs): an over-budget
            # or stop-flagged run returns the features decided so far rather than starting the next
            # (dominant-cost) trial. ``n_trials_run_`` was set above to the count completed.
            if _max_runtime_mins and (_budget_timer() - _fit_start_t) > _max_runtime_mins * 60.0:
                if self.verbose:
                    logger.info("BorutaShap: runtime budget %.1f min exceeded after trial %d/%d; stopping with features decided so far.", _max_runtime_mins, trial + 1, self.n_trials)
                break
            if _stop_file and _stop_file_exists(_stop_file):
                if self.verbose:
                    logger.info("BorutaShap: stop_file %r found after trial %d/%d; stopping with features decided so far.", _stop_file, trial + 1, self.n_trials)
                break

        self.store_feature_importance()
        self.calculate_rejected_accepted_tentative(verbose=self.verbose)
        pbar.set_description(f"Undecided features: {len(self.tentative):_}")
        new_ncols = len(self.columns)
        if new_ncols != last_ncols or trial % 5 == 0:
            logger.info("Undecided features: %s", f"{len(self.tentative):_}")
            last_ncols = new_ncols

    # sklearn-style outputs so callers can treat BorutaShap like any other selector: ``support_`` is the boolean mask aligned with the input column order, ``selected_features_`` is the list of kept names (accepted + tentative when ``optimistic``).
    kept = set(self.accepted)
    if self.optimistic:
        kept |= set(self.tentative)
    if getattr(self, "_premerge_active_", False):  # re-expand accepted representatives to their cluster members
        kept, self.support_ = _premerge_expand(kept, self._premerge_members_, self._premerge_original_cols_)
        _final_columns = self._premerge_original_cols_
    else:
        self.support_ = np.array([c in kept for c in self.all_columns], dtype=bool)
        _final_columns = self.all_columns
    self.selected_features_ = [c for c in _final_columns if c in kept]
    # sklearn convention: feature_names_in_ + n_features_in_ are the canonical
    # discoverable attributes for downstream report builders. Without them the
    # FS report's ``dropped_features`` field is None for BorutaShap and
    # asymmetric vs MRMR / RFECV.
    self.feature_names_in_ = np.asarray(_final_columns)
    self.n_features_in_ = int(len(_final_columns))

    # sklearn fit-returns-self contract: the stability path already returns self, so this single-fit path
    # must too, otherwise ``selector.fit(X, y).transform(X)`` (used across this codebase) yields None ->
    # AttributeError, and breaks only when stability is off -- a config-dependent behavioral divergence.
    return self


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
    est_name = type(self.model_).__name__
    if est_name == "TransformedTargetRegressor":
        explainer_base = self.model_.regressor
    elif est_name == "Pipeline":
        explainer_base = get_pipeline_last_element(self.model_)
    else:
        explainer_base = self.model_
    # perf note (2026-06-08): profiled on n=2407/p=120/LGBM-50tree/n_trials=30, a SHAP-driven fit is 98%
    # third-party -- model .fit ~69%, TreeSHAP ~29% (TreeExplainer.__init__ ~1.35s reading the just-refit model
    # + shap_values ~6.76s of C++ tree traversal). Both are intrinsic per-trial: the model is REFIT every trial,
    # so the explainer must be rebuilt and cannot be cached. GPU TreeSHAP was considered and is OUT OF SCOPE here:
    # (a) the wrapped model is caller-supplied (a CPU booster), so we cannot move it to GPU without changing the
    # caller's estimator, and (b) GPU contrib reorders the float reductions, perturbing shap values ~1e-6, which
    # can flip a borderline hit at the nanpercentile/`>` gate -> selection-altering. The mlframe-side per-trial
    # overhead (shadow build, hit counting, binomial test, history) is now <0.1s total after the single-call drop
    # + homogeneous-numeric shadow fast path; nothing further is actionable without changing the wrapped model.
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
                # SHAP's 3-D array layout is version-dependent: legacy
                # TreeExplainer returns (classes, samples, features); modern
                # SHAP (>=0.43) returns (samples, features, classes). The old
                # ``abs.sum(axis=0).mean(0)`` hard-coded classes-first, so on the
                # modern layout it reduced over (samples, features) and collapsed
                # to length n_CLASSES instead of n_features -- a too-short
                # importance vector that left ``Shadow_feature_import`` empty and
                # crashed update_importance_history with an IndexError (fuzz
                # c0095 hgb_xgb). Identify the FEATURE axis by matching
                # ``X_boruta``'s column count (preferring the non-leading axes,
                # where the feature dim lives in both layouts) and mean |shap|
                # over the other two axes, so the result is always length
                # n_features regardless of SHAP's axis order.
                _arr = np.abs(self.shap_values)
                _nfeat = self.X_boruta.shape[1]
                _fax = next((ax for ax in (1, 2, 0) if _arr.shape[ax] == _nfeat), 1)
                _other = tuple(ax for ax in range(3) if ax != _fax)
                self.shap_values = _arr.mean(axis=_other)
            else:
                self.shap_values = np.abs(self.shap_values).mean(0)

    else:
        self.shap_values = explainer.shap_values(basis)
        self.shap_values = np.abs(self.shap_values).mean(0)
