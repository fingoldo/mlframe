"""BorutaShap shadow-feature construction + statistical / hit-test helpers.

Carved out of the ``boruta_shap`` package facade (LOC-budget submodule split, see mlframe/CLAUDE.md "Monolith split"), mirroring the sibling
``_fit_explain.py`` / ``_io_plot.py`` method-binding split. These functions are bound onto ``BorutaShap`` in the package ``__init__``
(instance methods take ``self``; the ``@staticmethod`` ones are re-wrapped with ``staticmethod`` at bind time, exactly like the IO/plot helpers), so the class's public method surface is unchanged.

``_binom_test_cached`` stays defined in the package ``__init__`` (its public name must remain importable there); ``binomial_H0_test`` lazy-imports it in-body to avoid the facade<->submodule circular import at module-import time.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numpy.random import choice
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest


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

def _column_tie_fraction(values: np.ndarray) -> float:
    """Fraction of entries that share their value with at least one other entry (the "tied mass" of a column).

    0.0 == all-distinct (continuous); 1.0 == fully discrete with every value repeated. Used as the gate predicate
    for any tie-sensitive shadow kernel: a value-PERMUTATION shadow (the shipped default) is tie-agnostic and keeps
    the fast path at any tie fraction, but an argsort/rank-based shadow would break ties POSITIONALLY and bias the
    shadow MI low on columns above ~0.20 tie fraction, so such a kernel must gate out (stable path) above the
    threshold. Kept O(n) and dependency-free so it is free to call per column. See the shadow-kernel docstring."""
    v = np.asarray(values)
    n = v.size
    if n <= 1:
        return 0.0
    _, counts = np.unique(v, return_counts=True)
    tied = int(counts[counts > 1].sum())
    return tied / n


# Tie fraction above which a rank/argsort-based shadow kernel would bias shadow MI low (ties broken positionally).
# The shipped value-permutation kernel is tie-agnostic and ignores this; it exists so any future fast argsort path
# can gate on the SAME measured predicate. Exposed as a module constant + override env var (no hardcoded magic).
SHADOW_TIE_GATE_FRACTION = float(os.environ.get("MLFRAME_BORUTA_SHADOW_TIE_GATE", "0.20"))


def create_shadow_features(self):
    """
    Creates the random shadow features by shuffling the existing columns.

    The shadow kernel is a value-PERMUTATION (``_rng.permutation`` per column): it reorders each column's existing
    values, so the shadow's value-multiset is IDENTICAL to the real column's and its marginal distribution is exactly
    preserved -- only the row alignment with y (and the other columns) is destroyed. This is tie-agnostic BY
    CONSTRUCTION: permuting values never compares or sorts them, so a discrete / heavily-tied column is shuffled with
    no positional bias and its shadow MI is unbiased. (Contrast: a rank/argsort-based shadow breaks ties positionally
    and biases the shadow MI low on tied columns -- the failure class ``_column_tie_fraction`` /
    ``SHADOW_TIE_GATE_FRACTION`` guard, should anyone replace this with such a fast path.)

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

    # Canonical >=N shadow pad on narrow frames: when there are fewer real columns than ``shadow_min_pad`` the
    # shadow-importance MAX (the per-trial gate threshold) would be estimated from only 1-2 draws and is noisy.
    # Recycle real columns (each independently re-permuted, so still uncorrelated with y) as extra shadows up to the
    # pad, widening the null pool without adding any real-vs-real comparison. Wide frames (n_cols >= pad) are
    # untouched. Opt out with ``shadow_min_pad=0``. Suffixed names (``shadow_<col>__pad<k>``) stay unique and the
    # real/shadow split in ``feature_importance`` (``len(self.X.columns)``) already counts every extra as a shadow.
    _pad = int(getattr(self, "shadow_min_pad", 0) or 0)
    _n_real = self.X.shape[1]
    if _pad > _n_real > 0 and isinstance(self.X_shadow, pd.DataFrame):
        _extra = {}
        _real_cols = list(self.X.columns)
        for _k in range(_pad - _n_real):
            _src = _real_cols[_k % _n_real]
            _extra[f"__shadowpad{_k}__{_src}"] = _rng.permutation(self.X[_src].values)
        self.X_shadow = pd.concat([self.X_shadow, pd.DataFrame(_extra, index=self.X.index)], axis=1)

    if isinstance(self.X_shadow, pd.DataFrame):
        # append
        obj_col = self.X_shadow.select_dtypes(include=["object", "string"]).columns.tolist()
        if obj_col == []:
            pass
        else:
            self.X_shadow[obj_col] = self.X_shadow[obj_col].astype("category")

    # Prefix each shadow's name; the pad columns already carry a distinct ``__shadowpad{k}__`` infix so all
    # shadow names stay unique even when a real column is recycled.
    self.X_shadow.columns = ["shadow_" + str(feature) for feature in self.X_shadow.columns]
    self.X_boruta = pd.concat([self.X, self.X_shadow], axis=1)

    col_types = self.X_boruta.dtypes
    self.X_categorical = list(col_types[(col_types == "category") | (col_types == "object")].index)

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

    _measure = self._active_importance_measure() if hasattr(self, "_active_importance_measure") else str(self.importance_measure).lower()
    if _measure == "shap":
        self.explain()
        vals = self.shap_values

        if normalize:
            vals = self.calculate_Zscore(vals)

        # Layout of self.X_boruta is [X | X_shadow]. Real features come first,
        # shadow afterwards. Using len(self.X.columns) for the split is correct
        # even when the shadow side was padded to >= 5 columns.
        X_feature_import = vals[: len(self.X.columns)]
        Shadow_feature_import = vals[len(self.X.columns) :]

    elif _measure == "gini":
        feature_importances_ = np.abs(self.model_.feature_importances_)

        if normalize:
            feature_importances_ = self.calculate_Zscore(feature_importances_)

        X_feature_import = feature_importances_[: len(self.X.columns)]
        Shadow_feature_import = feature_importances_[len(self.X.columns) :]

    elif _measure == "permutation":
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
            self.model_, X_perm, y_perm, n_repeats=self.permutation_n_repeats,
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

def isolation_forest(X):
    """
    fits isloation forest to the dataset and gives an anomally score to every sample
    """
    clf = IsolationForest().fit(X)
    preds = clf.score_samples(X)
    return preds

def get_5_percent(num):
    return round(5 / 100 * num)

def get_5_percent_splits(self, length):
    """
    splits dataframe into 5% intervals
    """
    five_percent = self.get_5_percent(length)
    # On small frames (length <= 18) ``round(0.05*length)`` is 0, and ``np.arange(0, length, 0)`` raises
    # ZeroDivisionError, aborting the whole fit before any trial when ``sample=True``. Floor the step at 1
    # so the split grid degrades gracefully to single-row increments instead of crashing.
    step = max(1, int(five_percent))
    return np.arange(step, length, step)

def find_sample(self):
    """
    Finds a sample by comparing the distributions of the anomally scores between the sample and the original
    distribution using the KS-test. Starts of a 5% howver will increase to 10% and then 15% etc. if a significant sample can not be found
    """
    iteration = 0
    size = self.get_5_percent_splits(self.X.shape[0])
    element = 1
    # ``iteration`` bounds the search per sample size; on the 20th miss we grow the
    # sample (next ``size`` element). Without incrementing it the bound never fired and a
    # frame where no sub-sample reaches the KS p>0.95 threshold looped forever.
    while True:
        sample_indices = choice(np.arange(self.preds.size), size=size[element], replace=False)
        sample = np.take(self.preds, sample_indices)
        if ks_2samp(self.preds, sample).pvalue > 0.95:
            break

        iteration += 1
        if iteration == 20:
            element += 1
            iteration = 0
            # Exhausted every sample size without a KS match -- return the last (largest) draw.
            if element >= len(size):
                break

    return self.X_boruta.iloc[sample_indices]


def binomial_H0_test(array, n, p, alternative):
    """
    Perform a test that the probability of success is p.
    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment is p
    """
    # Lazy import of the package facade's memoized binom_test to avoid the facade<->submodule circular import at module-import time (this submodule is imported at the bottom of the package ``__init__``, after ``_binom_test_cached`` is defined there).
    from . import _binom_test_cached
    return [_binom_test_cached(int(x), n, p, alternative) for x in array]

def symetric_difference_between_two_arrays(array_one, array_two):
    set_one = set(array_one)
    set_two = set(array_two)
    return np.array(list(set_one.symmetric_difference(set_two)))

def find_index_of_true_in_array(array):
    length = len(array)
    return list(filter(lambda x: array[x], range(length)))

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

    # ``self.hits`` is full-length (indexed by ``all_columns``), so this re-tests already-removed features every
    # trial. That is CORRECT, not a bug: a removed feature's hit count is frozen at removal, the Bonferroni base is
    # the constant full count (see ``_n_tests`` below), so its decision never changes -- re-testing it is a no-op on
    # the final partition. The only residual cost is the extra binomtests, which the ``_binom_test_cached`` LRU
    # collapses to O(1) per distinct ``(x, n, p, alternative)``, so the waste is negligible (B4, subsumed by the B3 base fix).
    # Null hit probability. A "hit" is X_feature_import > percentile-th percentile of the shadow pool. Under H0 (a real
    # column is exchangeable with the shadows) a real importance exceeds the q-th percentile of the shadow values with
    # probability ~ (100 - q)/100, NOT 0.5. The legacy p=0.5 corresponds to a median-shadow gate; with the canonical
    # MAX-of-shadows gate (percentile=100) the true per-trial null hit rate is ~1/(m+1) for m shadows, far below 0.5, so
    # p=0.5 is grossly anti-conservative (accepts noise) on the accept side and over-conservative on the reject side.
    # Derive the calibrated p from self.percentile.
    null_hit_p = max(min((100.0 - float(self.percentile)) / 100.0, 1.0), 1e-9)

    acceptance_p_values = self.binomial_H0_test(self.hits, n=iteration, p=null_hit_p, alternative="greater")

    regect_p_values = self.binomial_H0_test(self.hits, n=iteration, p=null_hit_p, alternative="less")

    # [1] as function returns a tuple. Bonferroni base is the FULL original feature count (all_columns), not the
    # shrinking current column set: self.hits and the accept/reject indexing are full-length, and using the live
    # (post-removal) count would weaken the correction trial-over-trial as features drop out -- a leniency drift.
    # This matches the base the shipped margin-gated stop already uses (_n_total_cols = len(all_columns)).
    _n_tests = len(self.all_columns)
    modified_acceptance_p_values = self.bonferoni_corrections(acceptance_p_values, alpha=0.05, n_tests=_n_tests)[1]

    modified_regect_p_values = self.bonferoni_corrections(regect_p_values, alpha=0.05, n_tests=_n_tests)[1]

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
