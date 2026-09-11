"""No src/mlframe file exceeds 1000 lines, and no grandfathered one grows.

The check is `py_ci_shared.loc_budget`. `_loc_over_1k_baseline.json` records every file already over the
limit at its current size: such a file may grow by the shared slack and no further, a file that shrinks past
the slack must have its ceiling lowered, and one that drops under the limit or disappears must leave the
baseline. A plain exempt set could say none of that -- an exempt file could grow without bound. New code that
would push a module past the limit belongs in a sibling module from the start (CLAUDE.md: "Monolith split").

Refresh after a real carve with `pytest tests/test_meta/test_no_file_over_1k_loc.py --refresh-loc-budget-baseline`.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.loc_budget import assert_no_new_oversized_file

from ._scan_guard import assert_scanned_enough

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "_loc_over_1k_baseline.json"

# Why each file in the baseline is still over the limit, and what carving it would take.
# FIXME(carve-wave-next): filters/mrmr/_mrmr_class.py at ~4.76k LOC -- the irreducible
# ``MRMR`` estimator class body after the mrmr subpackage split (class moved verbatim;
# the package ``__init__.py`` facade re-exports it + runs the method bindings).
# Carved 2026-06-22: the Gate-A SIS front-screen ``_apply_sis_screen`` method (~76 LOC)
# lifted verbatim into filters/_mrmr_sis_apply.py and bound the same way as ``_fit_impl`` /
# ``_run_fe_step`` (4836 -> 4760 LOC; still exempt).
# Carved 2026-06-22 (constants drain): the ``_VALID_*`` / ``_DEMOTED_*`` ctor-param
# validation allow-lists -> filters/mrmr/_mrmr_param_constants.py (re-bound onto the class
# so ``self._VALID_*`` stays byte-identical), and the ~354-line ``__setstate__`` legacy
# default-injection dict literal -> filters/mrmr/_mrmr_setstate_defaults.py (re-imported via
# ``build_setstate_defaults()`` which deep-copies per call; the D5 no-drift ctor overlay runs
# unchanged on the copy). 4794 -> ~4404 LOC; still exempt. Remaining carve candidates if it
# must shrink further: the FE-flag plumbing block + the giant ``__init__`` attribute plumbing;
# the validate/transform/fit/fe-step/partial-fit/provenance method bodies already live in
# sibling modules.
# Carved 2026-07-02 (mixin split): ~26 non-central methods moved verbatim into three sibling mixins the
# class inherits -- _mrmr_class_config.py (subsample/fast-search profiles, scorer/enabled-FE recommend,
# seed/prefix/dtype/cv-kwargs/ctor-defaults, clear_fit_cache), _mrmr_class_transform.py (get_support,
# transform, get_feature_names_out, transform_usability, discovered_structure_, usability-union,
# __sklearn_is_fitted__), _mrmr_class_fit_helpers.py (_stability_outer_fit, _fit_multioutput,
# _fit_identity_shortcut, _maybe_resample_for_sample_weight, _print_fit_summary, export_artifacts,
# __setstate__). 4497 -> 3544 LOC; still exempt. The irreducible residual is __init__'s ~2080-line
# parameter docstring + fit; further LOC drop needs relocating that docstring, not more logic carving.
# - src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py
# FIXME(carve-wave-next): filters/_mrmr_fit_impl/_fit_impl_core.py -- the irreducible
# single-function body of ``_fit_impl`` (bound onto ``MRMR``) after the _mrmr_fit_impl
# subpackage split. The four small free helpers (``_orth_fe_numeric_cols`` /
# ``_dispatch_default_scorer`` / ``_mrmr_instance_state_size_bytes`` /
# ``_mrmr_cache_bytes_total``) live in the sibling ``_helpers.py``; only the one giant
# fit-orchestration function remains over budget (mirrors ``_step_core.py`` /
# ``_pairs_core.py``). 2026-06-22 (Tier E partial): the empty-RAW-support fallback rescue
# (the ``else`` branch of the post-selection raw-support reconciliation) was carved verbatim
# into ``_finalise._finalise_empty_support_fallback(self, n_engineered_out, cols, data, nbins,
# target_indices)`` -- parent shrank ~9.8k -> ~9.5k LOC (still over budget). 2026-06-23
# (Tier E partial): the Layer 92 temporal leak-safe grouped-aggregation FE stage (the
# self-contained ``if fe_temporal_agg_enable:`` block) was carved verbatim into
# ``_fe_stage_temporal_agg._fe_stage_temporal_agg(self, X, _y_np, verbose,
# _temporal_agg_pre_recipes) -> X`` -- threads self + the two fit-body locals + the recipes
# dict explicitly, mutates self/recipes in place, returns the (possibly replaced) ``X``;
# parent shrank ~9.5k -> ~9.4k LOC (still over budget). Remaining carve candidates if it must
# shrink further: other single ``fe_<X>_enable`` FE-stage blocks, or the FE/RFECV post-pass.
# 2026-07-02: assessed for a BULK automated split and rejected as unsafe. ``_fit_impl`` is one
# control-flow-entangled function -- return / continue / break / try-except span would-be block
# boundaries, so a mechanical whole-function carve changes semantics. Only self-contained FE-stage
# blocks (compute-and-assign, no early exit, explicit local threading) are verbatim-extractable, and
# those are drained ONE per wave as the entries above show. Left exempt BY DESIGN, drained incrementally
# -- not a pending bulk-split debt.
# 2026-08-15 (Tier F wave): Layers 23/26/56/60 (hybrid orth + hinge/tri-product basis, generic
# MI-greedy, CMI-greedy) carved into ``_fe_stage_cascade_early_a.py``; Layers 33/34/37/38 (k-fold TE,
# count/frequency/cat-num encoding, missingness-aware FE, ratio/grouped-delta/lagged-diff) carved
# into ``_fe_stage_cascade_early_b.py``. Both take the full fit-body local set each family reads
# (~30 ``_*_pre_recipes`` dicts hoisted to the caller and threaded by reference -- a dict mutated
# in place propagates without a return, but three genuinely-REASSIGNED locals
# (``_raw_input_cols_pre_fe``, ``_hinge_deferred_values``, ``_hinge_deferred_recipes``) do not and
# are threaded back out explicitly; verified via a systematic reassignment-vs-mutation grep per
# name, not by inspection). Caught and fixed three latent bugs before this landed: a relative-import
# depth miss, a `del` cleanup block referencing now-sibling-local temp-frame names (ruff F821), and
# (the interesting one) ~30 pre-registered recipe dicts that were declared once, deep inside the
# extracted range, and silently needed by code far downstream -- a class of bug invisible to
# import-only checks, only caught by an actual forced-execution trigger test with every family flag
# enabled plus a caplog assertion (a bare "does it run" smoke test would NOT have caught it, since
# every family swallows its own exceptions into a warning log). Parent shrank ~5.5k -> ~4.2k LOC;
# still over budget. Remaining carve candidates: Layers 87-104 (grouped-stat aggregators, cat x cat
# synergy crosses, periodic/modular decomposition, and the Layer 104 recipe-based families) --
# deferred to a future wave given the same entanglement risk this wave's three bugs demonstrate.
# - src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py
# (de-exempted 2026-06-22: per-candidate scoring block carved to _step_score.py
# [+ the per-pair rank loop to _step_pairs_rank.py, the batch pair-MI/maxT-floor stage
# to _step_pairmi.py, and the operand-pool construction to _step_pool.py];
# _step_core.py is now under the 1k ceiling.)
# (de-exempted 2026-06-22: extreme-AR gate + per-model post-train tail (uncertainty-eval +
# composite y-scale emit + RAM reclaim) + selector-sticky-attrs helper carved to
# _phase_train_one_target_post.py; _phase_train_one_target_body.py now under the 1k ceiling.)
# (de-exempted 2026-06-22: per-pair scoring block carved to _pairs_score.py; the admitted-pair
# emission tail to _pairs_emit.py; the prewarp/gate-med + operand-table setup to _pairs_setup.py.
# _pairs_core.py is now under the 1k ceiling.)
# (de-exempted 2026-06-22: RecurrentDataset + collate carved to recurrent_dataset_helpers.py)
# FIXME(carve-wave-next): filters/_screen_predictors.py -- the irreducible single-function
# body of ``screen_predictors`` (one sequential orchestration: input validation, RNG
# snapshot/restore try/finally, the candidate-generate -> confirm -> select greedy loop with
# the inline Miller-Madow / maxT-floor / DCD-swap blocks). The two small free helpers
# (``_short_name`` / ``_pool_warmup_noop``) plus the confirmation math (``confirm_one_predictor``
# in ``_confirm_predictor.py``) and the prescreen (``_screen_predictors_prescreen.py``) already
# live in siblings; only the one giant orchestration function remains over budget (mirrors
# ``_step_core.py`` / ``_pairs_core.py``).
# (de-exempted 2026-06-22: inline DCD discover/swap block carved out of the select loop into
# _screen_dcd_swap.py; _screen_predictors.py now ~934 LOC, under the 1k ceiling)
# (de-exempted 2026-06-22: prefilter holdout/clustering block carved to _shap_proxied_fit_prefilter.py)
# FIXME(carve-wave-next): training/core/_phase_composite_post_xt_ensemble/__init__.py -- the
# irreducible single-function body of ``_build_cross_target_ensemble_for_target`` (the
# CT_ENSEMBLE builder lifted out of the per-target training loop). Its three nested closures
# (``_get_train_pred`` / ``_compute_train_rmse_proxy`` / ``_drop_unscored_from_pool``) capture
# the build-local prediction cache + the candidate pool + ~20 frame/index locals, so they are
# not cleanly liftable to module scope. Carve candidate if it must shrink: extract the honest-
# OOF split + per-candidate scoring block into a ``_post_xt_score.py`` helper taking the pool +
# frames explicitly, leaving the assembly/mutate-in-place tail in the parent.
# Assessed 2026-06-22: NOT safely carvable -- the OOF/scoring block mutably REBINDS the candidate
# pool (``_components``/``_component_names``/``_component_specs`` in the external_val pre-screen)
# that the ``_compute_train_rmse_proxy``/``_get_train_pred`` closures close over AND that the
# post-block proxy-fallback re-reads; ``_get_train_pred`` is also re-called in the assembly tail.
# Threading this out would require passing the 3 closures in + returning ~10 rebound locals,
# reproducing the whole local env as an arg list -- an unvalidated training-behavior risk for no
# real decoupling. Left exempt by design.
# - src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py
# FIXME(carve-wave-next): training/io.py at ~1.02k LOC -- crossed the ceiling via the perf-loop save/load work
# (asizeof precheck + sha256 reopen + lib-version memoisation). Carve candidate: the ~380-line
# ``save_mlframe_model`` body (atomic-write + sidecar + version-stamp orchestration) lifts cleanly into a
# ``_io_save.py`` sibling re-exported from io.py; ``load_mlframe_model`` + the ``_SafeUnpickler`` stay in the parent.
# (de-exempted 2026-06-22: save_mlframe_model carved to _io_save.py)
# (de-exempted 2026-06-22: radix-select/residency block carved to _gpu_resident_select.py [+ the
# prewarp/orth-basis + grand-fusion block to _gpu_resident_basis.py]; _gpu_resident_fe.py now under 1k)
# (de-exempted 2026-06-22: kernels carved to _batch_mi_noise_gate_kernels.py)
# (de-exempted 2026-06-22: candidate-evaluation driver carved to _evaluation_driver.py)
# (de-exempted 2026-06-22: _orthogonal_univariate_fe/__init__.py carved to ~860 LOC via _orth_dedup.py)
# FIXME(carve-wave-next): filters/_feature_engineering_pairs/_pairs_score.py -- single irreducible
# ``score_pair_combos``-family function body (1 top-level def spanning the whole file) after the
# _feature_engineering_pairs subpackage split; same shape as ``_step_core.py``/``_pairs_core.py``. Not
# cleanly carvable without threading the per-pair scoring closure's ~dozen locals out as an arg list.
# - src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_score.py
# FIXME(carve-wave-next): filters/_mrmr_fe_step/_step_score.py -- single irreducible per-candidate
# CMI-scoring function body (1 top-level def spanning the whole file) after the _mrmr_fe_step subpackage
# split. Mirrors ``_pairs_score.py``; the surrounding pool/pair-rank/pair-MI stages already live in siblings.
# - src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_score.py
# FIXME(carve-wave-next): filters/_feature_engineering_pairs/_pairs_core.py -- the irreducible per-pair
# orchestration body (biggest def ~934 LOC) after the _feature_engineering_pairs split; the scoring /
# emit / setup blocks already carved to sibling modules, only the orchestration loop remains over budget.
# - src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py
# FIXME(carve-wave-next): the four GPU-resident FE / discretization / CMI-FE modules below are the
# in-flight born-on-device perf-replatform work (committed within the last day, mid perf-loop). They are
# carvable (multiple top-level defs) but actively churning under that effort; carving them here would
# collide with the in-flight rewrite. Carve to be folded into the next FE perf-loop wave that owns them.
# (de-exempted: _gpu_resident_select carved -- fused-binning/discretize block -> _gpu_resident_discretize.py,
# materialise/operand-table/host-fast-path block -> _gpu_resident_materialise.py; parent now <1k)
# (de-exempted: _fe_batched_mi carved -- batched CMI count/entropy kernel infra -> _fe_batched_mi_cmi.py;
# parent now <1k, sibling <1k, parent re-exports all public names.)
# (de-exempted 2026-09-08: _gpu_resident_basis is at 868 LOC and no longer needs the budget.)
# - src/mlframe/feature_selection/filters/_gpu_resident_fe.py
# - src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py
# - src/mlframe/feature_selection/filters/discretization/__init__.py
# FIXME(carve-wave-next): training/_trainer_train_and_evaluate.py at 1001 LOC -- the single
# ``train_and_evaluate_model`` orchestration function (per-model fit / eval-set setup / fallback /
# posthoc-calibration / OOF+calib outputs / split-metrics / report assembly) after the split-metrics
# emitters were carved to _trainer_train_and_evaluate_helpers.py. The residual is one control-flow-
# entangled function (early returns, try/finally RAM reclaim, per-backend branches) whose only cleanly
# verbatim-extractable blocks are already in the helpers sibling; a mechanical whole-function carve would
# cross return/finally boundaries and change semantics (same class as _fit_impl_core / _pairs_core).
# Drain the next self-contained compute-and-assign block when one surfaces.
# - src/mlframe/training/_trainer_train_and_evaluate.py
# (de-exempted 2026-09-08: transforms/nonlinear.py is at 624 LOC and no longer needs the budget.)
# FIXME(carve-wave-next): filters/_mrmr_fe_step/_step_core.py -- the residual body of
# ``_run_fe_step_impl`` after the operand-pool / pair-MI-floor / pair-rank / candidate-scoring
# stages were already carved to _step_pool.py / _step_pairmi.py / _step_pairs_rank.py /
# _step_score.py (same carve wave as _pairs_core.py / _fit_impl_core.py). What remains is one
# long sequential orchestration function threading ~40 fit-scoped locals (data/cols/nbins/X,
# the prewarp/gate-med spec accumulators, the polynom-pair injection indices, the serial-vs-
# joblib dispatch branch) through non-early-exit control flow with a `try/except` RNG-safe
# subsample resolution and two structurally-different call shapes (serial single dict vs.
# joblib per-chunk merge-with-reserved-key). Assessed 2026-07-18: the joblib branch (~140 LOC)
# is the only block that reads as visually separable, but it still closes over ~25 of those same
# locals (X, classes_y, cols, original_cols, numeric_vars_to_consider, every fe_pair_prewarp_*
# knob, the shared subsample index) that the serial branch right above it also needs -- lifting
# it out would mean threading the same ~25-argument list the sibling carves already show is the
# ceiling of what stays a safe verbatim move, for a block that is two-thirds a literal copy of
# the arguments to `check_prospective_fe_pairs` already visible in the serial branch. Left exempt
# BY DESIGN, same class as `_pairs_core.py` / `_fit_impl_core.py` above.
# - src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_core.py


def test_no_mlframe_file_exceeds_1k_loc() -> None:
    """No src/mlframe file is over the limit unless the baseline holds it, and none outgrows its ceiling."""
    files = sorted((REPO_ROOT / "src" / "mlframe").rglob("*.py"))
    assert_scanned_enough(len(files), "src/mlframe")
    assert_no_new_oversized_file(files, REPO_ROOT, BASELINE)
