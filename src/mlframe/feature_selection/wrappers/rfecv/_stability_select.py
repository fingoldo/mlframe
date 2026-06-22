"""``RFECV._fit_stability_selection`` + ``RFECV.select_optimal_nfeatures_`` carved
out of ``mlframe.feature_selection.wrappers._rfecv``.

Methods are bound onto the ``RFECV`` class at the parent's module bottom
so ``self._fit_stability_selection(...)`` / ``self.select_optimal_nfeatures_(...)``
call sites resolve unchanged.
"""
from __future__ import annotations

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone

from pyutilz.system import tqdmu

from .._enums import VotesAggregation
from .._helpers import (
    get_actual_features_ranking,
    get_feature_importances,
    select_appropriate_feature_importances,
)

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _rank_features_by_importance(
    self,
    *,
    use_all_fi_runs: bool,
    use_last_fi_run_only: bool,
    use_one_freshest_fi_run: bool,
    use_fi_ranking: bool,
    votes_aggregation_method,
    nfeatures: int,
):
    """Vote-aggregate ``self.feature_importances_`` into a ranked feature list (best first).

    Shared by the ``conduct_final_voting`` final-pick path and the p>=n FP-control cap. Returns ``None``
    when no usable feature importances were captured (so callers can fall back to a positional pick).
    """
    fi_to_consider = select_appropriate_feature_importances(
        feature_importances=self.feature_importances_,
        nfeatures=nfeatures,
        n_original_features=self.n_features_in_,
        use_all_fi_runs=use_all_fi_runs,
        use_last_fi_run_only=use_last_fi_run_only,
        use_one_freshest_fi_run=use_one_freshest_fi_run,
        use_fi_ranking=use_fi_ranking,
    )
    if not fi_to_consider:
        return None

    _imp_agg = getattr(self, "importance_agg", "legacy")
    _fi_family = getattr(self, "_fi_family", None)
    if _imp_agg == "dispatched" and _fi_family in ("tree", "linear"):
        from .._helpers_importance_agg import aggregate_importances_dispatched
        _signed = getattr(self, "_signed_importances", None)
        _signed_subset = None
        if _signed and _fi_family == "linear":
            _signed_subset = {k: v for k, v in _signed.items() if k in fi_to_consider}
        return aggregate_importances_dispatched(
            feature_importances=fi_to_consider,
            family=_fi_family,
            votes_aggregation_method=votes_aggregation_method,
            signed_importances=_signed_subset,
            k_cv=float(getattr(self, "importance_agg_k_cv", 1.0)),
            fi_missing_policy=getattr(self, "fi_missing_policy", "worst"),
        )
    return get_actual_features_ranking(
        feature_importances=fi_to_consider,
        votes_aggregation_method=votes_aggregation_method,
        fi_missing_policy=getattr(self, "fi_missing_policy", "worst"),
    )


def _fit_stability_selection(self, X, y, signature):
    """Stability Selection (Meinshausen & Buhlmann 2010, JRSS-B).

    Bootstrap-based feature selection. For each of B bootstrap subsamples (n/2, no replacement), fit the estimator(s) and record which
    features appeared in the top-K by importance. A feature is selected if its appearance frequency >= ``stability_threshold``
    (typically 0.6-0.9). Provable error control: E[V] <= q^2 / ((2*pi - 1) * p), where q is the average number of selected features
    per bootstrap and pi is the threshold.

    Particularly robust on small-n / high-p problems where per-fold CV voting is dominated by sampling noise. If ``self.estimators`` is
    set, FI is averaged across them inside each bootstrap.
    """
    # Docstring must be the FIRST statement to bind to __doc__; the n_samples
    # guard below was previously placed above it, leaving __doc__ unset and
    # the literal evaluated-and-discarded on every call.
    # E12 (Wave 4, 2026-05-28): floor n_samples for stability selection. With
    # n<20 the n/2 sub_size becomes <=10 and per-bootstrap FI is noise; the
    # threshold-based selection then picks essentially-random features.
    if X.shape[0] < 20:
        raise ValueError(
            f"stability_selection requires n_samples >= 20; got n={X.shape[0]}. "
            f"With n/2 sub_size below 10 the bootstrap FI signal is dominated "
            f"by sampling noise. Use the regular MBH path or increase n."
        )
    estimators_list = list(self.estimators) if self.estimators else [self.estimator]
    importance_getter = self.importance_getter or "auto"
    rng = np.random.default_rng(self.random_state)
    is_df = isinstance(X, pd.DataFrame)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    feature_names = X.columns.tolist() if is_df else [str(i) for i in range(n_features)]

    # Wide-data perm-FI cost guard (2026-06-04): mirror RFECV.fit's MBH-path guard on the stability-selection path.
    # Permutation / conditional-permutation importance rescore the model O(p * n_repeats) times PER BOOTSTRAP; over
    # stability_n_bootstrap replicates of a wide frame this is prohibitive. When wide_data_fi_fallback (default True) and
    # p exceeds wide_data_fi_threshold, fall back to native (gain/impurity) importance for the per-bootstrap top-K ranking;
    # below the threshold cap n_repeats. Stability selection ranks by importance regardless of estimator, so the native
    # importance is a valid substitute here too.
    _eff_n_repeats = int(getattr(self, "n_repeats", 5))
    if (
        getattr(self, "wide_data_fi_fallback", True)
        and isinstance(importance_getter, str)
        and importance_getter in ("permutation", "conditional_permutation")
    ):
        _threshold = int(getattr(self, "wide_data_fi_threshold", 200))
        if n_features > _threshold:
            if self.verbose:
                logger.info(
                    "stability_selection wide-data guard: %d features > wide_data_fi_threshold=%d; "
                    "falling back from importance_getter=%r to native 'auto'.",
                    n_features, _threshold, importance_getter,
                )
            importance_getter = "auto"
        else:
            _cap = int(getattr(self, "wide_data_fi_n_repeats", 2))
            if _eff_n_repeats > _cap and n_features > max(1, _threshold // 4):
                _eff_n_repeats = _cap
    self._effective_n_repeats = _eff_n_repeats

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
                    n_repeats=int(getattr(self, "_effective_n_repeats", None) or getattr(self, "n_repeats", 5)),
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
        # np.lexsort with feature-index tiebreaker so tied scores don't make the top-K pick depend on feature_names insertion order.
        top_idx = np.lexsort(
            (np.arange(len(per_feature_score_sum)), -per_feature_score_sum)
        )[:top_k]
        # Only count features that actually carry positive importance this bootstrap. When fewer than top_k features have nonzero FI,
        # padding the top-K with zero-importance (pure-noise) columns lets them accrue selection counts every bootstrap and cross the
        # stability threshold -- admitting noise into support_. A zero-FI feature is, by definition, not "selected" by this bootstrap.
        top_idx = top_idx[per_feature_score_sum[top_idx] > 0]
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
    # The stability path ranks features by IN-SAMPLE bootstrap selection frequency, NOT an honest
    # held-out CV score. Exposing that frequency under ``cv_mean_perf`` (the key consumers read as a
    # genuine OOF score in select_optimal_nfeatures_ / diagnostics) would mislabel it. Surface the
    # frequency under its own key and leave cv_mean_perf NaN, since no held-out score is computed here.
    self.cv_results_ = {
        "nfeatures": [self.n_features_],
        "cv_mean_perf": [float("nan")],
        "cv_std_perf": [0.0],
        "selection_frequency": [float(selection_freq[support_mask].mean()) if support_mask.any() else 0.0],
    }
    # Per-feature stability frequencies for inspection / downstream weighting, aligned with feature_names_in_.
    self.stability_selection_freq_ = selection_freq

    self._selected_cols_cache = [c for c, s in zip(feature_names, support_mask) if s]
    # Refresh the params slot with POST-fit values before storing (fit may resolve params in place,
    # e.g. ``scoring=None -> make_scorer(...)``); see ``_rfecv_fit_init._current_params_signature``.
    from ._fit_init import _current_params_signature

    self.signature = signature[:-1] + (_current_params_signature(self),)

    if self.verbose:
        logger.info(
            "stability_selection: selected %d / %d features at threshold=%.2f. "
            "Top-10 by frequency: %s",
            self.n_features_, n_features, self.stability_threshold,
            [(feature_names[i], round(float(selection_freq[i]), 3))
             # Wave 57: lexsort with feature-index tiebreaker for deterministic
             # log output across runs.
             for i in np.lexsort((np.arange(len(selection_freq)), -selection_freq))[:10]],
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
        # C4 (Wave 4, 2026-05-28): rolling.mean smooths by INDEX, not by N
        # value. On sparse N exploration ({2, 10, 30, 60}) adjacent rows
        # mix physically-unrelated regimes -> garbage smoothing. Warn loud.
        _nf_arr = np.array(checked_nfeatures)
        if len(_nf_arr) >= 2:
            _gaps = np.diff(np.sort(_nf_arr))
            _med_gap = float(np.median(_gaps))
            _max_gap = float(_gaps.max())
            if _max_gap > 3 * max(1.0, _med_gap):
                logger.warning(
                    "select_optimal_nfeatures_: smoothing across sparse N "
                    "(max gap %.0f, median gap %.0f). Rolling-mean averages "
                    "by index, not by N value, so adjacent rows may mix "
                    "unrelated regimes. Either set smooth_perf=0 or "
                    "evaluate more N values.", _max_gap, _med_gap,
                )
        # ``.rolling().mean().values`` returns a read-only ndarray on
        # recent pandas (the underlying BlockManager exposes an immutable
        # view of its memory). ``.to_numpy(copy=True)`` forces a writeable
        # buffer so the in-place NaN backfill on the next line doesn't
        # raise ``ValueError: assignment destination is read-only``.
        smoothed_perf = pd.Series(base_perf).rolling(smooth_perf, center=True).mean().to_numpy(copy=True)
        idx = np.isnan(smoothed_perf)
        smoothed_perf[idx] = base_perf[idx]
        base_perf = smoothed_perf

    ultimate_perf = base_perf - np.array(checked_nfeatures) * feature_cost

    # C3 (Wave 1, 2026-05-28; revised post-bench 2026-05-28):
    # Pre-Wave-1 'auto' = ('one_se_max' for multi-estimator else 'argmax')
    # had inverted multi-vs-singular logic. The original Wave 1 fix changed
    # 'auto' -> 'argmax' uniformly, but the synthetic-bench (n=8000, p=200,
    # 30 informative, flat score curve) showed argmax catastrophically
    # underselects on plateau (recall 0.30) where 'one_se_max' takes the
    # full band and recovers recall=1.0. On non-flat curves both rules
    # converge to the same N within ±1 feature. So NEW default 'auto' =
    # 'one_se_max' is strictly safer for the workloads RFECV sees:
    # plateau-prone score curves benefit, real-signal curves see no change.
    # Users wanting argmax-greedy or 1-SE-parsimony pass explicit
    # 'argmax' / 'one_se_min'.
    rule = getattr(self, "n_features_selection_rule", "auto")
    if rule == "auto":
        rule = "one_se_max"
    # Surface the resolved rule (after 'auto' expansion) so the FS report can show which selection rule
    # actually picked n_features_ without re-deriving the auto logic.
    self.resolved_n_features_rule_ = rule

    # bench-attempt-rejected (2026-06-11): a flat-curve / pure-noise reject in the 'auto' branch -- when the best evaluated subset
    # cannot beat (or is SE-significantly worse than) the no-features N=0 dummy, select NOTHING instead of all features -- was
    # measured as a TRADEOFF, NOT a clean win, and REVERTED. On pure noise it correctly flipped selection 15 -> 0 (and STRONG /
    # WEAK detectable-signal recall stayed bit-identical), BUT it also rejected RECOVERABLE signal on two real-signal fixtures
    # where the FULL feature set scores below the dummy while an UNEXPLORED smaller subset would beat it: (1) 6-informative
    # multi-estimator min-aggregation (knockoffs K3): 12 -> 0; (2) recency sample-weighted 2-feature (A predictive under the
    # weighting): 2 -> 0. Root cause: RFECV's "all-features can't beat the dummy" early-exit (_rfecv_fit_outer_loop.py ~351)
    # stops the search at {N=0, N=full} on BOTH pure noise AND noise-diluted-but-recoverable signal, so the two are
    # INDISTINGUISHABLE at this rule-resolution layer -- any reject here sacrifices recoverable signal. The only safe noise
    # rejection requires the search to explore smaller subsets BEFORE concluding (an outer-loop change). Harness +
    # measured numbers: _benchmarks/bench_auto_rule_noise_fp.py.

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

    # p>=n FP-control gate. On p>>n the elimination search routinely collapses to evaluating ONLY {N=0 dummy, N=full}: the per-step
    # CV score is non-decreasing out to the full set, so every selection rule (argmax / 1-SE / plateau) is forced to pick N=full --
    # i.e. ALL p features, which gives ZERO multiple-comparison control. When that collapse happens AND the full set is SE-significantly
    # WORSE than the no-features dummy, selecting all p is never right. Instead of selecting the full set we CAP the selection at the
    # FP-control ceiling ``max(20, p//3)`` features chosen by importance ranking: that keeps multiple-comparison inflation bounded
    # (a controlled selector must not return ~all p) WHILE still recovering the genuinely-informative columns (the top-ceiling by
    # importance includes them), so a noise-diluted-but-recoverable signal is no longer sacrificed. This is GATED to the p>=n regime
    # (n_features_in_ >= n_samples) and to the collapsed-search case (the only non-zero candidate IS the full set), so the well-powered
    # p<n path -- where the search explores intermediate N and the 1-SE rule legitimately recovers signal -- is untouched. An earlier
    # version of this gate ABSTAINED (returned empty support_) here; that broke signal-bearing high-dim selection (it fired on
    # overfit-but-recoverable high-dim too, not just pure noise), so it was replaced by the bounded cap. The even-earlier unconditional
    # below-dummy reject (bench-attempt-rejected 2026-06-11, see below) was worse still -- it fired on p<n collapses too.
    n_samples_fit = getattr(self, "_n_samples_fit_", None)
    p_in = int(getattr(self, "n_features_in_", 0))
    nz_candidates = nfeatures_arr[nonzero_mask]
    if (
        n_samples_fit is not None
        and p_in >= int(n_samples_fit)
        and nz_candidates.size == 1
        and int(nz_candidates[0]) == p_in
        and (nfeatures_arr == 0).any()
    ):
        _full_pos = int(np.flatnonzero(nonzero_mask)[0])
        _zero_pos = int(np.flatnonzero(nfeatures_arr == 0)[0])
        _mean = np.array(cv_mean_perf)
        _std = np.array(cv_std_perf)
        if np.isfinite(_mean[_full_pos]) and np.isfinite(_mean[_zero_pos]):
            _full_se = _std[_full_pos] if np.isfinite(_std[_full_pos]) else 0.0
            if _mean[_full_pos] + _full_se < _mean[_zero_pos]:
                _cap = min(p_in, max(20, p_in // 3))
                _ranking = _rank_features_by_importance(
                    self,
                    use_all_fi_runs=use_all_fi_runs,
                    use_last_fi_run_only=use_last_fi_run_only,
                    use_one_freshest_fi_run=use_one_freshest_fi_run,
                    use_fi_ranking=use_fi_ranking,
                    votes_aggregation_method=votes_aggregation_method,
                    nfeatures=p_in,
                )
                if _ranking is not None and len(_ranking) > 0:
                    _capped = _cap if _cap <= len(_ranking) else len(_ranking)
                    _keep = set(_ranking[:_capped])
                    self.support_ = np.array([f in _keep for f in self.feature_names_in_])
                    self.n_features_ = int(self.support_.sum())
                    self.ranking_ = _ranking
                else:
                    # Importance ranking unavailable (no FI captured): keep the first ``_cap`` features by position.
                    _capped = min(_cap, len(self.feature_names_in_))
                    self.support_ = np.array([i < _capped for i in range(len(self.feature_names_in_))])
                    self.n_features_ = int(self.support_.sum())
                logger.warning(
                    "select_optimal_nfeatures_: p>=n collapsed search ({N=0, N=%d}); the full set scores SE-worse than the no-features "
                    "dummy. Capping selection at %d features (ceiling max(20, p//3)) by importance for multiple-comparison control "
                    "instead of selecting all %d.",
                    p_in, self.n_features_, p_in,
                )
                self.resolved_n_features_rule_ = f"{rule}+p_ge_n_fp_control_cap"
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
        # 2026-05-28: when ``feature_cost > 0`` use the cost-adjusted ``ultimate_perf`` for both the band reference and the band
        # values, so the cost penalty actually bites under 1-SE rules too. Without this, feature_cost was silently a no-op under
        # the new ``auto='one_se_max'`` default and the user's "shrink toward fewer features" hint was ignored.
        if feature_cost and feature_cost > 0:
            mean_arr = np.array(ultimate_perf)
        else:
            mean_arr = np.array(cv_mean_perf)
        std_arr = np.array(cv_std_perf)
        nz_idx = np.where(nonzero_mask)[0]
        # Wave 21 P0: mask out NaN candidates before argmax. cv_mean_perf
        # can hold NaN when every fold for that N was degenerate
        # (_helpers.py:337). Pre-fix np.argmax picks the NaN slot ->
        # ``best_mean_idx`` refers to a never-evaluated N and the caller
        # silently returns a bogus support_ for that count.
        _finite_mask = np.isfinite(mean_arr[nz_idx])
        if not _finite_mask.any():
            raise ValueError(
                "RFECV: cv_mean_perf is all-NaN across non-zero "
                "candidates; cannot pick a winner. Re-run with more "
                "data or check that the inner-CV folds aren't all "
                "degenerate (all-one-class)."
            )
        _finite_nz_idx = nz_idx[_finite_mask]
        best_mean_idx = _finite_nz_idx[np.argmax(mean_arr[_finite_nz_idx])]
        threshold = mean_arr[best_mean_idx] - std_arr[best_mean_idx]
        in_band = [i for i in nz_idx if mean_arr[i] >= threshold]
        if not in_band:
            in_band = [int(best_mean_idx)]
        if rule == "one_se_max":
            best_idx = max(in_band, key=lambda i: nfeatures_arr[i])
        elif rule == "one_se_min":
            best_idx = min(in_band, key=lambda i: nfeatures_arr[i])
        elif rule == "plateau":
            # Plateau-onset (round-2 R2r-6): smallest N whose mean is within 1 SE of the BEST mean achievable
            # at >= that N -- i.e. the point past which adding features yields no SE-significant gain. Sits
            # between one_se_max (keeps ~all on flat tails) and one_se_min/knee (over-prunes a flat curve to a
            # tiny N): it stops where the curve flattened, capturing the full achievable score parsimoniously.
            _se = std_arr[best_mean_idx]
            _order = sorted(nz_idx, key=lambda i: nfeatures_arr[i])
            best_idx = _order[-1]
            for _pos, _i in enumerate(_order):
                _future_best = max(mean_arr[j] for j in _order[_pos:])
                if mean_arr[_i] >= _future_best - _se:
                    best_idx = _i
                    break
        else:
            raise ValueError(
                f"n_features_selection_rule={rule!r} not supported. "
                f"Use 'auto', 'argmax', 'one_se_max', 'one_se_min', or 'plateau'."
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
            self.ranking_ = _rank_features_by_importance(
                self,
                use_all_fi_runs=use_all_fi_runs,
                use_last_fi_run_only=use_last_fi_run_only,
                use_one_freshest_fi_run=use_one_freshest_fi_run,
                use_fi_ranking=use_fi_ranking,
                votes_aggregation_method=votes_aggregation_method,
                nfeatures=best_top_n,
            )
            self.support_ = np.array([(i in self.ranking_[:best_top_n]) for i in self.feature_names_in_])

    if verbose:
        # base_perf[0]/[-1] are the smallest/largest EVALUATED N on the curve, not necessarily N=0 (dummy) or N=full;
        # report the gain against those actual curve endpoints so the figure is honest when neither extreme was probed.
        _nf = np.asarray(checked_nfeatures)
        min_n, max_n = int(_nf[0]), int(_nf[-1])
        low_gain = (base_perf[0] / base_perf[best_idx] - 1) if base_perf[best_idx] != 0 else np.inf
        high_gain = (base_perf[-1] / base_perf[best_idx] - 1) if base_perf[best_idx] != 0 else np.inf
        logger.info(
            "%d predictive factors selected out of %d during %d rounds. Gain vs smallest evaluated N=%d: %.1f%%, gain vs largest evaluated N=%d: %.1f%%",
            self.n_features_, self.n_features_in_, len(self.selected_features_), min_n, low_gain * 100, max_n, high_gain * 100,
        )
