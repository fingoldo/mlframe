"""Run the four FE stage cascades of one fit under the step-input contract.

Every cascade function reads the SAME frame (the base features the step started with) and returns it plus its own new columns; the new columns are folded together in the fixed cascade order, so no stage consumes another stage's output.

bench-attempt-rejected: running early_b, mid_a and mid_b in threads (each on a copy of the estimator, outputs folded in the fixed order) was measured with
_benchmarks/bench_fe_stage_threads.py at 300k rows on the single-GPU dev box. The three stages' wall rose from 0.7 s each to 1.0-1.7 s each (they queue on the same device), the cascade
went from 2.92 s to 2.5 s and the whole fit from 8.7 s to 8.9 s (no gain). The stages also share list attributes of the estimator that are appended in stage order (hybrid_orth_features_),
which a threaded run would have to merge back deterministically. Revisit on a multi-GPU host or once the stages are CPU-bound.
Revisited 2026-11-03 at 100000 x 100 with every family (after that day's GPU and CPU changes): concurrent 118.0 / 124.4 / 143.1 s against sequential
129.4 / 171.6 s, same selection, no device errors; the concurrent cascade (``_fe_stage_concurrency``) is now the default. Re-checked under host memory
pressure (5 GB free virtual memory): back to back in one process, sequential 125.7 s, concurrent 111.9 s, same selection.
"""

from __future__ import annotations

import os

from ._fe_stage_cascade_early_a import _fe_stage_cascade_early_a
from ._fe_stage_cascade_early_b import _fe_stage_cascade_early_b
from ._fe_stage_cascade_mid_a import _fe_stage_cascade_mid_a
from ._fe_stage_cascade_mid_b import _fe_stage_cascade_mid_b
from ._fe_stage_merge import _fe_merge_new_columns
from ._fe_stage_concurrency import STAGE_WEIGHTS_WITH_EARLY_A, concurrency_enabled, run_stages_concurrently
from .._fe_dominant_candidate import dominant_candidate
from .._fe_frame_ops import fe_drop_columns


def _run_four_stages_together(self, stages, after_early_a, _y_np, _X_step_input, close_lap):
    """The opt-in four-cascade run (``MLFRAME_FE_STAGE_CONCURRENCY_EARLY_A=1``): ``(frame, raw input columns)``, or ``None`` when the concurrent run did not happen
    (the caller then runs early_a first)."""
    # all four cascades are independent (same step input, disjoint recipes): run them together, each on a copy of the estimator in a device slot of its own, and fold
    # in the fixed order. A dominant candidate from early_a stops a sequential run after it, so only early_a's state is kept then.
    # Opt-in: at 30000 x 40 it gives the same selection and the same wall as early_a first (44.4 vs 44.5 s warm); at 100000 x 100 it could not be measured on the
    # dev box (four stages' buffers ran the host out of pinned memory), so the default keeps early_a ahead of the three concurrent cascades.
    dominant: dict = {}

    def _keep(outcomes: list) -> int:
        """1 when early_a's frame already has a dominant candidate (the sequential path returns there), else every stage."""
        X_a, _raw = after_early_a(outcomes[0])
        dominant["X"], dominant["raw"] = X_a, _raw
        return 1 if dominant_candidate(X_a, _y_np) is not None else 4

    outs = run_stages_concurrently(self, stages, STAGE_WEIGHTS_WITH_EARLY_A, merge_count=_keep, n_rows=len(_X_step_input))
    if outs is not None:
        X, raw = dominant["X"], dominant["raw"]
        close_lap()
        for out in outs[1:]:
            if out is not None:
                X = _fe_merge_new_columns(X, out, _X_step_input)
        close_lap()
        return X, raw
    return None


def _run_fe_stage_cascades(self, X, y, verbose, fe_max_steps, _y_np, _fe_family_on, _fe_budget_ok, _fit_entry_nan_mask, recipes):
    """Run early_a, early_b, mid_a and mid_b on the same step-input frame; return the merged frame and the raw input columns recorded before FE."""
    # A constant or all-NaN input column carries no signal, and as an operand it only dilutes the candidate pools (the orth triplet seeds, the pair budget): the stages read the step input without
    # them, and the frame that continues keeps every column, so the support indices and the recorded raw columns are unchanged.
    # A lap ends with its stage; an unset hook (tests passing a plain callable) is skipped.
    _close_lap = getattr(_fe_family_on, "close_lap", lambda: None)
    _X_full = X
    _dead = [c for c, reason in (getattr(self, "degenerate_columns_", None) or {}).items() if reason in ("constant", "all_nan")]
    _X_step_input = fe_drop_columns(_X_full, _dead)
    # The other cascades take the raw input columns early_a records (``list(X.columns)`` of the step input), so they are known before it runs and early_a can run beside them.
    # With dropped constant / all-NaN columns the sequential path replaces the recorded columns by the full frame's (``_after_early_a``): the same value here.
    _src = _X_full if _X_step_input is not _X_full else _X_step_input
    _raw_input_cols_pre_fe = list(_src.columns) if hasattr(_src, "columns") else []

    def _stage_early_a(stage_self):
        """Run the early_a cascade on the given estimator (the real one, or a private copy when the cascades run together)."""
        return _fe_stage_cascade_early_a(
            stage_self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
            _fe_budget_ok=_fe_budget_ok,
            _hybrid_orth_pre_recipes=recipes.hybrid_orth, _mi_greedy_pre_recipes=recipes.mi_greedy,
        )

    def _after_early_a(out_a):
        """The frame after early_a and the recorded raw columns, as the sequential path forms them."""
        X_a, raw_a, recipes.hinge_deferred_values, recipes.hinge_deferred = out_a
        if _X_step_input is not _X_full:
            X_a = _fe_merge_new_columns(_X_full, X_a, _X_step_input)
            raw_a = list(_X_full.columns) if hasattr(_X_full, "columns") else raw_a
        return X_a, raw_a

    def _stage_early_b(stage_self):
        """Run the early_b cascade on the given estimator (the real one, or a private copy when the cascades run together)."""
        return _fe_stage_cascade_early_b(
            stage_self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
            _fit_entry_nan_mask=_fit_entry_nan_mask, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
            _kfold_te_pre_recipes=recipes.kfold_te, _binned_agg_pre_recipes=recipes.binned_agg,
            _count_enc_pre_recipes=recipes.count_enc, _freq_enc_pre_recipes=recipes.freq_enc,
            _cat_num_pre_recipes=recipes.cat_num,
            _miss_ind_pre_recipes=recipes.miss_ind, _miss_cnt_pre_recipes=recipes.miss_cnt,
            _miss_pat_pre_recipes=recipes.miss_pat,
            _ratio_pre_recipes=recipes.ratio, _log_ratio_pre_recipes=recipes.log_ratio,
            _grouped_delta_pre_recipes=recipes.grouped_delta, _lagged_diff_pre_recipes=recipes.lagged_diff,
        )

    def _stage_mid_a(stage_self):
        """Run the mid_a cascade on the given estimator (the real one, or a private copy when the cascades run together)."""
        return _fe_stage_cascade_mid_a(
            stage_self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
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
        )

    def _stage_mid_b(stage_self):
        """Run the mid_b cascade on the given estimator (the real one, or a private copy when the cascades run together)."""
        return _fe_stage_cascade_mid_b(
            stage_self, X=_X_step_input, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
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
            _offset_product_pre_recipes=recipes.offset_product,
            _oof_warp_pre_recipes=recipes.oof_warp,
            _row_stat_pre_recipes=recipes.row_stat,
            _wavelet_pre_recipes=recipes.wavelet,
            _rankgauss_pre_recipes=recipes.rankgauss,
        )

    if concurrency_enabled() and os.environ.get("MLFRAME_FE_STAGE_CONCURRENCY_EARLY_A", "0").strip() == "1":
        together = _run_four_stages_together(self, [_stage_early_a, _stage_early_b, _stage_mid_a, _stage_mid_b], _after_early_a, _y_np, _X_step_input, _close_lap)
        if together is not None:
            return together
    X, _raw_input_cols_pre_fe = _after_early_a(_stage_early_a(self))
    _close_lap()
    # One column that already carries nearly all of y leaves nothing for the remaining families to add.
    if dominant_candidate(X, _y_np) is not None:
        return X, _raw_input_cols_pre_fe
    if concurrency_enabled():
        # no device plan for four stages: the three later cascades together, as before
        outs = run_stages_concurrently(self, [_stage_early_b, _stage_mid_a, _stage_mid_b], n_rows=len(_X_step_input))
        if outs is not None:
            for out in outs:
                X = _fe_merge_new_columns(X, out, _X_step_input)
            _close_lap()
            return X, _raw_input_cols_pre_fe
    X = _fe_merge_new_columns(X, _stage_early_b(self), _X_step_input)
    _close_lap()
    X = _fe_merge_new_columns(X, _stage_mid_a(self), _X_step_input)
    _close_lap()
    X = _fe_merge_new_columns(X, _stage_mid_b(self), _X_step_input)
    _close_lap()
    return X, _raw_input_cols_pre_fe
