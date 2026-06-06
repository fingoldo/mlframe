"""Self-contained FE-step sub-blocks carved out of ``_mrmr_fe_step._run_fe_step``.

Two blocks with explicit inputs/outputs (no shared-local threading with the surrounding orchestrator beyond their declared parameters):

* ``compute_pair_maxt_floor`` -- the order-2 Westfall-Young permutation-null joint-MI floor over the whole candidate-pair pool (returns a single float, applied as an extra gate in both the XOR and uplift branches of the pair loop).
* ``run_cluster_aggregate_emission`` -- the opt-in clustered-feature aggregation emission block (runs once on the first FE step; returns the updated ``(data, cols, nbins, X, selected_vars, n_recommended_features)`` tuple and stamps the fitted summary onto ``self``).

Both take the ``MRMR`` instance as ``self`` so they read the frozen ``fe_*`` / ``cluster_aggregate_*`` config off it exactly as the inlined blocks did. The heavy kernels stay in their own modules (``_permutation_null``, ``_cluster_aggregate``) and are lazy-imported in-body to avoid import cycles.
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def apply_synergy_bootstrap(
    self,
    *,
    num_fs_steps: int,
    data,
    cols,
    target_indices,
    categorical_vars,
    numeric_vars_to_consider: set,
    non_numeric_idx: set,
    verbose,
):
    """Synergy bootstrap: widen the FE pair pool with unselected raw numeric columns so zero-marginal synergy pairs get joint-MI screened.

    Pure-synergy interactions (a*d, sign products, log(c)*sin(d)) have ~zero marginal MI per factor, so neither factor is screened-in and the pair never reaches the prospective-pair
    screen -- even though that screen ALREADY keeps a zero-individual-MI pair whose JOINT MI is positive (the canonical XOR branch). The fix is purely to widen the POOL: when the raw
    numeric feature count is within ``fe_synergy_screen_max_features``, add the UNSELECTED raw numeric columns so the all-pairs joint-MI sweep screens the synergy pairs. Two cost/quality
    guards live downstream: (1) synergy pairs (>=1 bootstrap-added operand) must clear the STRICTER ``fe_synergy_min_prevalence`` uplift bar (rejects finite-sample-bias noise pairs), and
    (2) the surviving synergy pairs are budget-capped to ``fe_synergy_max_pairs`` by joint MI. Runs only on the FIRST FE step.

    Gated by a MIN-ROWS guard (``fe_synergy_min_rows``: a 2-D joint-MI estimate is finite-sample-bias-dominated at tiny n, admitting pure-NOISE pairs) and an N-AWARE COST GATE
    (``fe_synergy_max_sweep_cost`` ~ n*p^2: the O(p^2*n) sweep grows super-linearly in n; default 5e8 fires on the measured wins but SKIPS large-n blow-ups; set to inf to disable).

    bench-rejected (2026-06-03) -- GAP-3 "poly-feature synergy re-entry": feeding ENGINEERED univariate poly features (He2(a) etc.) back into this bootstrap pool so a synergy like
    y=He2(a)*b surfaces was benchmarked and REJECTED. Pool entry is NOT the blocker -- the raw (a,b) pair already reaches the prospective-pair screen as the top-joint-MI pair, and the
    per-pair unary search builds He2(a)*b from it. poly x raw classification is already recovered 6/6 via the default-on prewarp operand; the genuine miss is poly x raw regression (1/6),
    blocked simultaneously by four noise-control gates, and forcing recovery would loosen the gates ``test_biz_val_mrmr_default_filtering.py`` pins for a fragile 2/6. Don't re-add it.

    Returns ``(numeric_vars_to_consider, synergy_added_idx)`` -- the (possibly widened) pool and the set of bootstrap-added operand indices (empty when the bootstrap did not fire).
    """
    synergy_cap = int(getattr(self, "fe_synergy_screen_max_features", 0) or 0)
    synergy_added_idx: set = set()
    synergy_min_rows = int(getattr(self, "fe_synergy_min_rows", 300) or 0)
    n_rows_for_synergy = int(data.shape[0]) if hasattr(data, "shape") else 0
    synergy_max_sweep_cost = float(getattr(self, "fe_synergy_max_sweep_cost", 5e8) or float("inf"))
    if synergy_cap > 0 and num_fs_steps == 0 and n_rows_for_synergy >= synergy_min_rows:
        _raw_names = set(getattr(self, "feature_names_in_", []) or [])
        _target_idx_set = {int(t) for t in np.atleast_1d(target_indices)}
        _cat_set = set(categorical_vars)
        _raw_numeric_idx = {
            i for i, nm in enumerate(cols)
            if nm in _raw_names and i not in _target_idx_set and i not in _cat_set
        }
        _raw_numeric_idx -= non_numeric_idx
        if self.factors_to_use is not None:
            _raw_numeric_idx &= set(self.factors_to_use)
        if self.factors_names_to_use is not None:
            _raw_numeric_idx &= {cols.index(n) for n in self.factors_names_to_use if n in cols}
        _n_raw = len(_raw_numeric_idx)
        _sweep_cost = n_rows_for_synergy * (_n_raw ** 2)
        if 0 < _n_raw <= synergy_cap and _sweep_cost <= synergy_max_sweep_cost:
            _added = _raw_numeric_idx - numeric_vars_to_consider
            if _added:
                synergy_added_idx = set(_added)
                numeric_vars_to_consider = numeric_vars_to_consider | _raw_numeric_idx
                if verbose:
                    logger.info(
                        "MRMR FE synergy bootstrap: augmented pair pool with %d unselected raw "
                        "numeric columns (%d raw <= cap %d) so zero-marginal synergy pairs "
                        "(a*d / sign products / log*sin) get joint-MI screened.",
                        len(_added), _n_raw, synergy_cap,
                    )
        elif 0 < _n_raw <= synergy_cap and _sweep_cost > synergy_max_sweep_cost and verbose:
            logger.info(
                "MRMR FE synergy bootstrap: %d raw numeric cols <= cap %d but sweep cost n*p^2=%.2g exceeds "
                "fe_synergy_max_sweep_cost=%.2g (O(p^2*n) blow-up on large n); skipping the all-pairs sweep. "
                "Raise fe_synergy_max_sweep_cost to force it.",
                _n_raw, synergy_cap, float(_sweep_cost), synergy_max_sweep_cost,
            )
        elif _n_raw > synergy_cap and verbose:
            logger.info(
                "MRMR FE synergy bootstrap: %d raw numeric columns > cap %d; skipping the "
                "all-pairs synergy sweep (keeping the selected-only pool). Raise "
                "fe_synergy_screen_max_features to enable it on this frame.",
                _n_raw, synergy_cap,
            )
    return numeric_vars_to_consider, synergy_added_idx


def compute_pair_maxt_floor(
    self,
    *,
    numeric_vars_to_consider,
    n_pairs: int,
    data,
    nbins,
    classes_y,
    freqs_y,
    verbose,
) -> float:
    """Order-2 Westfall-Young permutation-null joint-MI floor over the candidate-pair pool.

    The pair-gating loop ranks O(p^2) candidate pairs by JOINT MI(x_i, x_j; y); at high p the MAX joint MI over PURE-NOISE pairs is a positive order statistic that grows with the
    pool size -- the same best-of-p selection bias the order-1 screening floor rejects, now at order 2. The per-pair prevalence gates are PER-PAIR; they do NOT account for the
    max-over-pool selection, so a wide noise matrix still surfaces "synergistic-looking" noise pairs. Compute the floor ONCE over the WHOLE candidate pool: shuffle the discretised
    target K times, take the per-shuffle MAX joint MI via the SAME batched plug-in estimator the screen scores ``pair_mi`` with, floor at the q-th quantile. SELF-GATING: below
    ``fe_pair_maxt_min_pairs`` candidate pairs the floor is 0.0 (no-op => byte-identical narrow pools). ``fe_pair_maxt_null_permutations=0`` disables.
    """
    _pair_maxt_floor = 0.0
    _pair_maxt_perms = int(getattr(self, "fe_pair_maxt_null_permutations", 25) or 0)
    if _pair_maxt_perms > 0 and len(numeric_vars_to_consider) >= 2 and n_pairs >= int(getattr(self, "fe_pair_maxt_min_pairs", 30)):
        try:
            from ._permutation_null import pooled_pair_permutation_null_joint_mi_floor

            _maxt_pairs = list(combinations(numeric_vars_to_consider, 2))
            _maxt_pa = np.fromiter((p[0] for p in _maxt_pairs), dtype=np.int64, count=len(_maxt_pairs))
            _maxt_pb = np.fromiter((p[1] for p in _maxt_pairs), dtype=np.int64, count=len(_maxt_pairs))
            _pair_maxt_floor = pooled_pair_permutation_null_joint_mi_floor(
                factors_data=data,
                nbins=nbins,
                pair_a=_maxt_pa,
                pair_b=_maxt_pb,
                classes_y=classes_y,
                freqs_y=freqs_y,
                n_permutations=_pair_maxt_perms,
                quantile=float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                random_seed=getattr(self, "random_seed", None),
            )
            if _pair_maxt_floor > 0.0 and verbose >= 1:
                logger.info(
                    "MRMR FE: order-2 maxT permutation-null joint-MI floor=%.5f over %d candidate "
                    "pairs (q=%.2f, K=%d) - rejects best-of-p chance-max noise pairs.",
                    _pair_maxt_floor, n_pairs, float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                    _pair_maxt_perms,
                )
        except Exception:
            logger.warning(
                "MRMR FE: order-2 maxT permutation-null floor failed; continuing without it.",
                exc_info=True,
            )
            _pair_maxt_floor = 0.0
    return _pair_maxt_floor


def run_cluster_aggregate_emission(
    self,
    *,
    data,
    cols,
    nbins,
    X,
    target_indices,
    categorical_vars,
    cached_MIs,
    engineered_recipes,
    selected_vars,
    n_recommended_features: int,
    num_fs_steps: int,
    _is_polars_input: bool,
    verbose,
):
    """Opt-in clustered-feature aggregation: denoise correlated "reflection" clusters into aggregate columns.

    Runs once on the first FE step (clusters only raw ``feature_names_in_`` columns, which don't change across FE steps). Guarded so a failure never aborts fit. Returns the updated
    ``(data, cols, nbins, X, selected_vars, n_recommended_features)`` tuple; on success also appends to ``self._cluster_aggregate_removals_`` and ``self.cluster_aggregate_``.
    """
    if getattr(self, "cluster_aggregate_enable", False) and num_fs_steps == 0:
        try:
            from ._cluster_aggregate import run_cluster_aggregate_step

            data, cols, nbins, X, _ca_added, _ca_removed, _ca_indices, _ca_summary = run_cluster_aggregate_step(
                data=data, cols=cols, nbins=nbins, X=X, target_indices=target_indices,
                feature_names_in_=list(self.feature_names_in_), categorical_idx=categorical_vars,
                cached_MIs=cached_MIs, engineered_recipes=engineered_recipes,
                quantization_nbins=self.quantization_nbins, quantization_method=self.quantization_method,
                quantization_dtype=self.quantization_dtype,
                methods=tuple(self.cluster_aggregate_methods),
                mi_prevalence=self.cluster_aggregate_mi_prevalence,
                min_member_relevance=self.cluster_aggregate_min_member_relevance,
                min_cluster_size=self.cluster_aggregate_min_cluster_size,
                max_cluster_size=self.cluster_aggregate_max_cluster_size,
                corr_threshold=self.cluster_aggregate_corr_threshold,
                homogeneity_tau=self.cluster_aggregate_homogeneity_tau,
                max_candidates=self.cluster_aggregate_max_candidates,
                mode=self.cluster_aggregate_mode, is_polars_input=_is_polars_input,
                verbose=verbose, dtype=self.dtype,
            )
            n_recommended_features += int(_ca_added)
            if _ca_indices:
                # The aggregate already passed the supervised MI gate (beats best member), so select it
                # directly -- don't rely on a re-screen (with the default fe_max_steps=1 the loop breaks
                # before re-screening). Remap routes the engineered name into _engineered_recipes_.
                _sv = list(selected_vars) if not isinstance(selected_vars, list) else selected_vars
                selected_vars = _sv + [i for i in _ca_indices if i not in _sv]
            if _ca_removed:  # replace mode: consumed in _fit_impl before the cols->original remap
                self._cluster_aggregate_removals_ = list(getattr(self, "_cluster_aggregate_removals_", [])) + list(_ca_removed)
            if _ca_summary:
                # Fitted summary -> surfaced into meta_info["feature_selection_report"]["cluster_aggregate"].
                self.cluster_aggregate_ = list(getattr(self, "cluster_aggregate_", []) or []) + _ca_summary
                logger.info(
                    "MRMR cluster_aggregate (%s): built %d denoised aggregate(s) from correlated clusters: %s",
                    self.cluster_aggregate_mode, _ca_added,
                    [f"{r['name']} (method={r['method']}, k={len(r['members'])}, +MI {r['mi_gain']:.4f})" for r in _ca_summary],
                )
        except Exception:
            logger.warning("cluster_aggregate step failed; continuing without it.", exc_info=True)
    return data, cols, nbins, X, selected_vars, n_recommended_features


def log_fe_summary(
    *,
    prospective_pairs,
    prospective_additions,
    n_recommended_features: int,
    fe_min_pair_mi_prevalence,
    fe_min_engineered_mi_prevalence,
    fe_min_nonzero_confidence,
    fe_good_to_best_feature_mi_threshold,
    verbose,
) -> None:
    """Log the per-step FE summary (pairs considered / with additions / total engineered + gate thresholds).

    Surfaces WHY FE added 0 features when the operator configured it explicitly: a prod log showed 88 min of Hermite Optuna yielding 0 engineered cols with no visible explanation.
    The summary reports n_pairs_considered (pairs screened), n_pairs_with_additions (pairs that produced ANY recipe), and n_engineered_features (recipes that survived all gates); at
    ``verbose >= 1`` with 0 additions it also names the likely too-tight knob (often ``fe_min_engineered_mi_prevalence``). Pure logging, no state mutation.
    """
    try:
        _n_pairs_considered = int(len(prospective_pairs))
    except Exception:
        _n_pairs_considered = -1
    try:
        _n_pairs_with_additions = sum(
            1 for v in prospective_additions.values()
            if v[0]  # this_pair_features non-empty
        )
    except Exception:
        _n_pairs_with_additions = -1
    if verbose >= 1:
        logger.info(
            "FE summary: %d pair(s) considered, %d produced engineered cols, "
            "n_total_engineered=%d. Gate thresholds: "
            "fe_min_pair_mi_prevalence=%.3f, "
            "fe_min_engineered_mi_prevalence=%.3f, "
            "fe_min_nonzero_confidence=%.3f, "
            "fe_good_to_best_feature_mi_threshold=%.3f.",
            _n_pairs_considered, _n_pairs_with_additions,
            n_recommended_features,
            float(fe_min_pair_mi_prevalence),
            float(fe_min_engineered_mi_prevalence),
            float(fe_min_nonzero_confidence),
            float(fe_good_to_best_feature_mi_threshold),
        )
        if n_recommended_features == 0 and _n_pairs_considered > 0:
            logger.warning(
                "FE produced 0 engineered features despite %d pair(s) "
                "passing the pair-MI gate. Likely cause: the "
                "fe_min_engineered_mi_prevalence=%.3f threshold is "
                "tight relative to the pair-level MI. Try lowering "
                "to 0.90 (5%% under the default) or set "
                "fe_min_pair_mi_prevalence=1.02 to widen the pool.",
                _n_pairs_considered,
                float(fe_min_engineered_mi_prevalence),
            )
