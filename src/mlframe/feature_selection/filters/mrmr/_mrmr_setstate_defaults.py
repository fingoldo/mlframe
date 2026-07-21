"""Legacy-pickle default-injection roster for ``MRMR.__setstate__`` (carved verbatim from ``_mrmr_class.py``).

``__setstate__`` builds a fresh ``defaults`` dict on every call (so its mutable values --
empty lists / dicts -- are never shared across unpickled instances) by deep-copying this
template. The values are byte-identical to the former method-local literal; the D5 no-drift
overlay (re-sourcing every shared ctor-param key from ``_ctor_defaults()`` except the
documented ``_SETSTATE_LEGACY_OVERRIDES``) still runs in ``__setstate__`` against the copy.
This module holds only literal data -- no class refs -- so it is a safe top-level import.
"""
from __future__ import annotations

import copy

# Verbatim legacy-injection roster. ``__setstate__`` deep-copies this so each unpickled
# instance gets its own mutable default objects (empty lists / tuples are fine to share,
# but lists must not be aliased across instances).
_SETSTATE_LEGACY_DEFAULTS = {
    "max_confirmation_cand_nbins": 50,  # legacy default
    "fe_fallback_to_all": True,  # legacy default
    "_engineered_features_": [],
    # Recipes-based replay so transform() can recompute engineered features on test data. Old pickles
    # have no recipes (their engineered cols were never replayable); empty list reproduces the legacy
    # "engineered cols dropped from transform output" behaviour bit-exact.
    "_engineered_recipes_": [],
    # Cat-FE: ``None`` means disabled or never ran; injecting ``None`` makes getattr(...) no-op.
    "cat_fe_config": None,
    "_cat_fe_state_": None,
    "_cat_fe_cache_": None,  # streaming cache; None on legacy pickles
    "strict_groups": False,  # legacy pickles default to warn-only behaviour
    # Renamed from skip_retraining_on_same_shape (misnomer; cache keys on content). Legacy pickles carry only the old attr; inject the new one so _fit_impl's getattr resolves.
    "skip_retraining_on_same_content": True,
    # Identity-cache y-correlation gate (added later); legacy pickles default to 0.0 (gate off = legacy any-X-fingerprint short-circuit) to preserve their behaviour bit-for-bit.
    "mrmr_identity_cache_ycorr_threshold": 0.0,
    # Friend-graph post-analysis. Legacy pickles refit with the
    # current defaults; ``friend_graph_`` itself is a fitted attribute, not seeded here.
    "build_friend_graph": False,
    "friend_graph_prune": False,
    "friend_graph_max_nodes": 200,
    "friend_graph_mi_eps": 1e-6,
    "friend_graph_edge_significance": 3.0,
    "friend_graph_garbage_min_degree": 3,
    "friend_graph_unique_ratio": 1.0,
    "friend_graph_unique_max_degree": 1,
    # Clustered-feature aggregation. ENABLED by default
    # (cluster_aggregate_enable=True); legacy pickles refit with these
    # defaults so an attribute-less pickle behaves like a fresh MRMR.
    # Mode was "augment" here but the constructor default is "replace"
    # (the deliberate fix for the duplicate-vote effect) -- legacy
    # pickles must refit to the corrected behaviour, not the
    # superseded one.
    "cluster_aggregate_enable": True,
    "cluster_aggregate_mode": "replace",
    "cluster_aggregate_methods": ("mean_z",),
    "cluster_aggregate_mi_prevalence": 1.0,
    "cluster_aggregate_min_member_relevance": 0.0,
    "cluster_aggregate_min_cluster_size": 3,
    "cluster_aggregate_max_cluster_size": 12,
    "cluster_aggregate_corr_threshold": 0.6,
    "cluster_aggregate_homogeneity_tau": 0.6,
    "cluster_aggregate_max_candidates": 200,
    # hybrid orthogonal-poly FE auto-wire.
    # Defaults preserve legacy behaviour: master switch OFF, so old
    # pickles unpickle to "hybrid FE disabled".
    # per-operand pre-warp for the unary/binary pair search.
    # OFF by default; legacy pickles unpickle to "pre-warp disabled".
    "fe_pair_prewarp_enable": False,
    "fe_pair_prewarp_basis": "chebyshev",
    "fe_pair_prewarp_max_degree": 4,
    "fe_pair_prewarp_uplift_threshold": 1.20,
    # per-operand median gate for the unary/binary pair
    # search. OFF by default; legacy pickles unpickle to "gate disabled".
    "fe_gate_med_enable": False,
    "fe_hybrid_orth_enable": False,
    "fe_hybrid_orth_degrees": (2, 3),
    "fe_hybrid_orth_basis": "auto",
    "fe_hybrid_orth_top_k": 5,
    "fe_hybrid_orth_pair_enable": True,
    "fe_hybrid_orth_pair_max_degree": 2,
    # triplet cross-basis FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_triplet_enable": False,
    "fe_hybrid_orth_triplet_max_degree": 1,
    "fe_hybrid_orth_triplet_seed_k": 4,
    "fe_hybrid_orth_triplet_top_count": 2,
    # quadruplet (4-way) cross-basis FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_quadruplet_enable": False,
    "fe_hybrid_orth_quadruplet_max_degree": 1,
    "fe_hybrid_orth_quadruplet_seed_k": 4,
    "fe_hybrid_orth_quadruplet_top_count": 2,
    # adaptive-arity cross-basis FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_adaptive_arity_enable": False,
    "fe_hybrid_orth_adaptive_arity_max_arity": 3,
    "fe_hybrid_orth_adaptive_arity_max_degree": 1,
    "fe_hybrid_orth_adaptive_arity_seed_k": 4,
    "fe_hybrid_orth_adaptive_arity_top_count": 3,
    # semi-supervised basis-preprocess fitting.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_semi_supervised_enable": False,
    # Lasso-based pre-selection defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_lasso_enable": False,
    "fe_hybrid_orth_lasso_alpha": 0.01,
    # Elastic Net (L1 + L2) pre-selection defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_elasticnet_enable": False,
    "fe_hybrid_orth_elasticnet_alpha": 0.01,
    "fe_hybrid_orth_elasticnet_l1_ratio": 0.5,
    # adaptive per-column degree defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_adaptive_degree_enable": False,
    "fe_hybrid_orth_adaptive_degree_range": (1, 2, 3, 4, 5, 6),
    "fe_hybrid_orth_adaptive_degree_min_uplift": 1.05,
    # conditional basis routing FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_conditional_routing_enable": False,
    "fe_hybrid_orth_conditional_routing_top_k": 5,
    "fe_hybrid_orth_conditional_routing_min_uplift": 1.10,
    "fe_hybrid_orth_conditional_routing_degrees": (2, 3),
    # diff-basis FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_diff_basis_enable": False,
    "fe_hybrid_orth_diff_basis_corr_threshold": 0.7,
    "fe_hybrid_orth_diff_basis_degrees": (1, 2, 3),
    "fe_hybrid_orth_diff_basis_top_k": 3,
    # per-cluster shared-basis FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_cluster_basis_enable": False,
    "fe_hybrid_orth_cluster_basis_aggregator": "mean_z",
    "fe_hybrid_orth_cluster_basis_degrees": (2, 3),
    "fe_hybrid_orth_cluster_basis_top_k": 3,
    # bootstrap-stable MI ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_bootstrap_enable": False,
    "fe_hybrid_orth_bootstrap_n_boot": 10,
    "fe_hybrid_orth_bootstrap_sample_fraction": 0.8,
    # three-gate + K-fold OOF MI defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_three_gate_enable": False,
    "fe_hybrid_orth_three_gate_n_folds": 5,
    "fe_hybrid_orth_three_gate_cmi_min": 0.001,
    # KSG / k-NN MI ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_ksg_enable": False,
    "fe_hybrid_orth_ksg_n_neighbors": 3,
    "fe_hybrid_orth_ksg_min_uplift": 0.95,
    "fe_hybrid_orth_ksg_min_abs_mi_frac": 0.05,
    # copula-MI ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_copula_enable": False,
    "fe_hybrid_orth_copula_n_bins": 20,
    # distance-correlation ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_dcor_enable": False,
    "fe_hybrid_orth_dcor_n_sample": 500,
    # HSIC ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_hsic_enable": False,
    "fe_hybrid_orth_hsic_kernel": "rbf",
    "fe_hybrid_orth_hsic_n_sample": 500,
    # JMIM (Bennasar 2015) defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_jmim_enable": False,
    "fe_hybrid_orth_jmim_n_bins": 10,
    # TC (Watanabe 1960) ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_tc_enable": False,
    "fe_hybrid_orth_tc_n_bins": 10,
    # CMIM (Fleuret 2004) ranking defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_cmim_enable": False,
    "fe_hybrid_orth_cmim_n_bins": 10,
    # per-column scorer auto-selection defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_auto_scorer_enable": False,
    "fe_hybrid_orth_auto_scorer_n_boot": 5,
    # ensemble rank-fusion defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_ensemble_enable": False,
    "fe_hybrid_orth_ensemble_aggregator": "mean_rank",
    "fe_hybrid_orth_ensemble_scorers": (
        "plug_in", "ksg", "copula", "dcor", "hsic",
    ),
    # meta-scorer auto-selection defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_hybrid_orth_meta_enable": False,
    "fe_hybrid_orth_meta_force_scorer": None,
    # extra-basis (spline / fourier) defaults.
    "fe_hybrid_orth_extra_bases": (),
    "fe_hybrid_orth_fourier_freqs": (1.0, 2.0),
    "fe_hybrid_orth_spline_knots": 5,
    # Fitted attribute (list of engineered names from hybrid stage);
    # legacy pickles default to empty list.
    "hybrid_orth_features_": [],
    # MI-greedy FE constructor. Defaults
    # preserve legacy behaviour: master switch OFF.
    "fe_mi_greedy_enable": False,
    "fe_mi_greedy_top_k": 5,
    "fe_mi_greedy_seed_cols_count": 5,
    "fe_mi_greedy_include_unary": True,
    "fe_mi_greedy_include_binary": True,
    "mi_greedy_features_": [],
    # CMI-greedy FE constructor. Defaults
    # preserve legacy behaviour: master switch OFF.
    "fe_mi_greedy_cmi_enable": False,
    "fe_mi_greedy_cmi_top_k": 5,
    "fe_mi_greedy_cmi_seed_cols_count": 4,
    "fe_mi_greedy_cmi_min_gain": 0.005,
    # K-fold target-encoding FE defaults.
    # Master switch OFF preserves legacy pickle byte-equivalence.
    "fe_kfold_te_enable": False,
    "fe_kfold_te_cols": (),
    "fe_kfold_te_folds": 5,
    "fe_kfold_te_smoothing": 10.0,
    "kfold_te_features_": [],
    # count / frequency / cat x num residual.
    # Master switches OFF preserve legacy pickle byte-equivalence.
    "fe_count_encoding_enable": False,
    "fe_count_encoding_cols": (),
    "fe_frequency_encoding_enable": False,
    "fe_frequency_encoding_cols": (),
    "fe_cat_num_interaction_enable": False,
    "fe_cat_num_interaction_cat_cols": (),
    "fe_cat_num_interaction_num_cols": (),
    "fe_cat_num_interaction_folds": 5,
    "fe_cat_num_interaction_smoothing": 10.0,
    "count_encoding_features_": [],
    "frequency_encoding_features_": [],
    "cat_num_interaction_features_": [],
    # two-tier IT gates on the recipe-emitting
    # FE mechanisms. Master switches OFF preserve legacy pickle
    # byte-equivalence.
    "fe_local_mi_gate": True,  # default-flip (corrective gate, see __init__)
    "fe_local_mi_gate_top_k": 20,
    "fe_unified_second_pass_gate": False,  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
    "fe_unified_second_pass_max_keep": None,  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
    "fe_unified_second_pass_min_gain": 0.005,  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
    # partial_fit / streaming refit.
    # Legacy pickles default OFF (decay 0, threshold 100, no window);
    # fitted-state buffers default to None until partial_fit is called.
    "partial_fit_decay": 0.0,
    "partial_fit_min_recompute": 100,
    "partial_fit_window": None,
    "_partial_fit_X_buffer_": None,
    "_partial_fit_y_buffer_": None,
    "_partial_fit_n_seen_": 0,
    "_partial_fit_n_since_refit_": 0,
    # Per-batch row counts backing the decay-weight schedule. Sibling buffers above are in the roster but
    # this one was omitted, so a pickled-then-resumed partial_fit instance fell back to a single fictional
    # batch (getattr default) -- collapsing multi-batch history and zeroing recency decay. Empty = no history.
    "_partial_fit_batch_sizes_": [],
    # FE provenance tracking.
    # Legacy pickles default to ``None`` and the empty predictor log;
    # the next fit() repopulates from the live greedy run.
    "fe_provenance_": None,
    # Per-gate FE REJECTION LEDGER (the rejection side of fe_provenance_):
    # a DataFrame (one row per rejected FE candidate-per-gate) + its raw record
    # list. Legacy pickles default to None / [] and the next fit() repopulates.
    "fe_rejection_ledger_": None,
    "_fe_rejection_records_": [],
    "_predictors_log_": (),
    # Produced-recipes audit ledger: every EngineeredRecipe the FE stages emitted this fit (pre-screen). fe_provenance_ reads it so the audit / pickle-replay paths recover which mechanism
    # produced each engineered column even when the greedy CMI screen dropped it. Legacy pickles default to [] and the next fit() repopulates from the live FE run.
    "_produced_recipes_": [],
    # fe_auto "1-knob" mode. Pre-L99 pickles
    # default to False -> byte-identical legacy path on reload.
    "fe_auto": False,
    # three new recipe-based FE families.
    # Master switches OFF preserve legacy pickle byte-equivalence;
    # fitted-attr lists default empty until the next fit repopulates.
    "fe_rare_category_enable": False,
    "fe_rare_category_cols": (),
    "fe_rare_category_threshold": 0.01,
    "fe_rare_category_top_k": 10,
    "fe_conditional_residual_enable": False,
    "fe_conditional_residual_cols": (),
    "fe_conditional_residual_n_bins": 10,
    "fe_conditional_residual_top_k": 10,
    "fe_conditional_residual_max_pair_cols": 6,
    # Family D conditional dispersion. Pre-#12 pickles default OFF so the
    # legacy reload path is byte-identical (the live default is ON for new
    # fits via __init__); the fitted-attr list defaults empty.
    "fe_conditional_dispersion_enable": False,
    "fe_conditional_dispersion_cols": (),
    "fe_conditional_dispersion_n_bins": 10,
    "fe_conditional_dispersion_top_k": 10,
    "fe_conditional_dispersion_max_pair_cols": 6,
    # Conditional quantile-rank (mrmr_audit_2026-07-20 fe_expansion.md). Default OFF in both
    # legacy pickles and the live ctor -- not yet validated against the existing regression/
    # biz_value/fuzz-combo suite the way the sibling conditional_dispersion family was before its
    # own default flipped to ON.
    "fe_conditional_quantile_rank_enable": False,
    "fe_conditional_quantile_rank_cols": (),
    "fe_conditional_quantile_rank_n_bins": 10,
    "fe_conditional_quantile_rank_top_k": 10,
    "fe_conditional_quantile_rank_max_pair_cols": 6,
    # Haar wavelet basis (backlog #13). Pre-#13 pickles default OFF so the
    # legacy reload path is byte-identical (the live default is ON for new
    # fits via __init__); the fitted-attr list defaults empty.
    "fe_wavelet_enable": False,
    "fe_wavelet_cols": (),
    "fe_wavelet_max_scale": 3,
    "fe_wavelet_max_legs": 6,
    "fe_wavelet_top_k": 8,
    "wavelet_features_": [],
    "fe_rankgauss_enable": False,
    "fe_rankgauss_cols": (),
    "fe_rankgauss_top_k": 10,
    "rare_category_features_": [],
    "conditional_residual_features_": [],
    "conditional_dispersion_features_": [],
    "rankgauss_features_": [],
    # ADAPTIVE-FREQUENCY Fourier. Pre-adaptive pickles
    # default to the on/0.15 contract for re-fits; the fitted-attr
    # list defaults empty (no adaptive features replayed on reload
    # until the next fit repopulates it).
    "fe_univariate_fourier_adaptive": True,
    "fe_univariate_fourier_adaptive_min_val_corr": 0.15,
    # ADAPTIVE-CHIRP Fourier. Pre-chirp pickles default to
    # the on/0.15 contract for re-fits (the chirp legs share the
    # adaptive-feature capture/protection list above).
    "fe_univariate_fourier_chirp": True,
    "fe_univariate_fourier_chirp_min_val_corr": 0.15,
    "_adaptive_fourier_features_": [],
    # MILLER-MADOW DEBIAS of the joint-prevalence ratio gate (2026-06-09,
    # backlog #1 + #4). bench-rejected as default (admits cross-mix noise on
    # weak F2); kept OPT-IN. Pre-MM pickles default to OFF for re-fits.
    "fe_mm_debias_prevalence": False,
    # SURROGATE-GBM SPLIT-CO-OCCURRENCE SEEDER (#6) + ORDER-3 maxT floor (#7),
    # 2026-06-09. Seeder is OPT-IN; the order-3 floor mirrors the order-2 knobs.
    # Pre-seeder pickles default to the same safe values for re-fits.
    # PREVALENCE-FAILED SYNERGY RESCUE (2026-06-12). Live default ON; pre-fix
    # pickles default ON too (the rescue only adds a leak-safe second-chance path
    # behind the full admission gates -- it never changes an already-admitted
    # column's values, so replay of a pre-fix recipe is unaffected).
    "fe_synergy_prevalence_rescue_enable": True,
    # TAIL-CONCENTRATED USABILITY ADMISSION (2026-07-02). Live default ON; pre-fix pickles default ON too
    # (it only credits a raw-operand |corr(continuous y)| signal for rank-MI-under-ranked/rejected pairs behind
    # the full downstream FE gates -- it never changes an already-admitted column's values, so replay of a
    # pre-fix recipe is unaffected).
    "fe_pair_usability_admission_enable": True,
    "fe_pair_usability_admission_min_corr": 0.6,
    "fe_pair_usability_admission_pairness_margin": 1.05,
    "fe_pair_usability_admission_rank_frac": 0.7,
    "fe_raw_tail_subsume_min_corr": 0.85,
    "fe_pair_usability_prescan_max_pairs": 256,
    # ESCALATION FEATURES TERMINAL in feed-forward (2026-06-12). Pre-fix pickles
    # default OFF too (matches the live default; escalation features were rare and
    # this only restricts NEW composite seeding, never replay of an existing recipe).
    "fe_escalation_feedforward_enable": False,
    "fe_gbm_seeder_enable": False,
    "fe_gbm_seeder_min_features": 30,
    "fe_gbm_seeder_top_k_pairs": 12,
    "fe_gbm_seeder_top_k_triples": 8,
    "fe_gbm_seeder_n_estimators": 300,
    "fe_gbm_seeder_max_depth": 4,
    "fe_gbm_seeder_self_gate_margin": 0.0,
    "fe_gbm_seeder_self_gate_reps": 5,
    "fe_gbm_seeder_self_gate_min_z": 2.0,
    # GRADIENT-INTERACTION (MIXED SECOND PARTIALS) SEEDER (#21), 2026-06-10. OPT-IN;
    # pre-feature pickles default to OFF for re-fits (byte-identical legacy path).
    "fe_gradient_interaction_enable": False,
    "fe_triple_maxt_null_permutations": 25,
    "fe_triple_maxt_null_quantile": 0.95,
    "fe_triple_maxt_min_triples": 4,
    # SUFFICIENT-SUMMARY EARLY-STOP (backlog #22, 2026-06-10). DEFAULT-ON; pre-feature
    # pickles default to the same on/0.25 contract for re-fits (the early-stop never
    # changes the final selection, so an old pickle's behaviour is unchanged either way).
    "fe_sufficient_summary_early_stop": True,
    "fe_sufficient_summary_residual_frac": 0.25,
    "fe_sufficient_summary_maxt_permutations": 25,
    "fe_sufficient_summary_maxt_quantile": 0.95,
    "fe_sufficient_summary_ridge_alpha": 1e-3,
    # CMI-redundancy gate cost guard (2026-06-11). Old pickles default to the
    # same 64-candidate cap; it only fires on pools WIDER than the cap (deep-tail
    # low-marginal-MI redundant remaps), so a re-fit of any pickle whose pool was
    # already <= 64 candidates is byte-identical.
    "fe_engineered_cmi_max_candidates": 64,
    # CMI-redundancy gate strong-significance escape (2026-06-11). Old pickles
    # default to the same 3.0 margin (the gate's _step_core read already falls
    # back to 3.0 via getattr; pinned here for an explicit, pickle-stable default).
    # The escape only LOOSENS leg 2 for a candidate that already clears its floor
    # 3x, so a re-fit of any pickle whose pool had no such candidate is byte-identical.
    "fe_engineered_cmi_significance_escape_margin": 3.0,
    # EMBEDDING / FREE-TEXT PASSTHROUGH (default ON). Old pickles default to the same on contract; a re-fit of any pickle whose X had no non-scalar columns
    # is byte-identical (the detector finds nothing and never narrows the frame).
    "embedding_passthrough": True,
    "embedding_passthrough_detect_embeddings": True,
    "embedding_passthrough_detect_text": True,
    # Fitted-attribute mirror: an unpickled pre-feature fit has no passthrough roster; default empty so transform's re-attach loop is a no-op.
    "_passthrough_features_": [],
}


def build_setstate_defaults() -> dict:
    """Return a fresh deep copy of the legacy-injection roster.

    ``__setstate__`` mutates this dict (the D5 ctor-default overlay writes shared keys)
    and seeds it into ``state``; returning a per-call deep copy keeps the module-level
    template pristine and guarantees no two unpickled instances alias a mutable default.
    """
    return copy.deepcopy(_SETSTATE_LEGACY_DEFAULTS)
