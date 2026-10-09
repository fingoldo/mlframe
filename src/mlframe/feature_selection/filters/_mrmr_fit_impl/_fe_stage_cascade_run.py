"""Run the four FE stage cascades of one fit under the step-input contract.

Every cascade function reads the SAME frame (the base features the step started with) and returns it plus its own new columns; the new columns are folded together in the fixed cascade order, so no stage consumes another stage's output.

bench-attempt-rejected: running early_b, mid_a and mid_b in threads (each on a copy of the estimator, outputs folded in the fixed order) was measured with
_benchmarks/bench_fe_stage_threads.py at 300k rows on the single-GPU dev box. The three stages' wall rose from 0.7 s each to 1.0-1.7 s each (they queue on the same device), the cascade
went from 2.92 s to 2.5 s and the whole fit from 8.7 s to 8.9 s (no gain). The stages also share list attributes of the estimator that are appended in stage order (hybrid_orth_features_),
which a threaded run would have to merge back deterministically. Revisit on a multi-GPU host or once the stages are CPU-bound.
"""

from __future__ import annotations

from ._fe_stage_cascade_early_a import _fe_stage_cascade_early_a
from ._fe_stage_cascade_early_b import _fe_stage_cascade_early_b
from ._fe_stage_cascade_mid_a import _fe_stage_cascade_mid_a
from ._fe_stage_cascade_mid_b import _fe_stage_cascade_mid_b
from ._fe_stage_merge import _fe_merge_new_columns


def _run_fe_stage_cascades(self, X, y, verbose, fe_max_steps, _y_np, _fe_family_on, _fe_budget_ok, _fit_entry_nan_mask, recipes):
    """Run early_a, early_b, mid_a and mid_b on the same step-input frame; return the merged frame and the raw input columns recorded before FE."""
    _X_step_input = X
    X, _raw_input_cols_pre_fe, recipes.hinge_deferred_values, recipes.hinge_deferred = _fe_stage_cascade_early_a(
        self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok,
        _hybrid_orth_pre_recipes=recipes.hybrid_orth, _mi_greedy_pre_recipes=recipes.mi_greedy,
    )
    X = _fe_merge_new_columns(X, _fe_stage_cascade_early_b(
        self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fit_entry_nan_mask=_fit_entry_nan_mask, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _kfold_te_pre_recipes=recipes.kfold_te, _binned_agg_pre_recipes=recipes.binned_agg,
        _count_enc_pre_recipes=recipes.count_enc, _freq_enc_pre_recipes=recipes.freq_enc,
        _cat_num_pre_recipes=recipes.cat_num,
        _miss_ind_pre_recipes=recipes.miss_ind, _miss_cnt_pre_recipes=recipes.miss_cnt,
        _miss_pat_pre_recipes=recipes.miss_pat,
        _ratio_pre_recipes=recipes.ratio, _log_ratio_pre_recipes=recipes.log_ratio,
        _grouped_delta_pre_recipes=recipes.grouped_delta, _lagged_diff_pre_recipes=recipes.lagged_diff,
    ), _X_step_input)
    X = _fe_merge_new_columns(X, _fe_stage_cascade_mid_a(
        self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _cat_pair_pre_recipes=recipes.cat_pair, _cat_triple_pre_recipes=recipes.cat_triple,
        _composite_group_agg_pre_recipes=recipes.composite_group_agg,
        _conditional_gate_pre_recipes=recipes.conditional_gate,
        _grouped_agg_pre_recipes=recipes.grouped_agg,
        _grouped_quantile_pre_recipes=recipes.grouped_quantile,
        _integer_lattice_pre_recipes=recipes.integer_lattice,
        _modular_pre_recipes=recipes.modular,
        _numeric_decompose_pre_recipes=recipes.numeric_decompose,
        _pairwise_modular_pre_recipes=recipes.pairwise_modular,
        _row_argmax_pre_recipes=recipes.row_argmax,
    ), _X_step_input)
    X = _fe_merge_new_columns(X, _fe_stage_cascade_mid_b(
        self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _group_distance_pre_recipes=recipes.group_distance,
        _rare_category_pre_recipes=recipes.rare_category,
        _conditional_residual_pre_recipes=recipes.conditional_residual,
        _conditional_dispersion_pre_recipes=recipes.conditional_dispersion,
        _conditional_quantile_rank_pre_recipes=recipes.conditional_quantile_rank,
        _ordinal_pattern_pre_recipes=recipes.ordinal_pattern,
        _random_fourier_pre_recipes=recipes.random_fourier,
        _sir_direction_pre_recipes=recipes.sir_direction,
        _lof_pre_recipes=recipes.lof,
        _mahalanobis_density_pre_recipes=recipes.mahalanobis_density,
        _wavelet_pre_recipes=recipes.wavelet,
        _rankgauss_pre_recipes=recipes.rankgauss,
    ), _X_step_input)
    return X, _raw_input_cols_pre_fe
