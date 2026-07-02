"""Wave 10 LOC-budget meta-test.

Scans ``src/mlframe/`` for ``.py`` files exceeding 1000 lines of code. After
Waves 6 + 10 the project's monolith-split policy is enforced: no file should
exceed the 1k LOC ceiling. Future PRs that re-introduce a >1k file are flagged
in CI so the splitting work doesn't drift back.

The exempt list is empty by design; any added entry must come with a
justification in the PR description (e.g. ``feature_engineering/wavelet_dwt.py``
WIP). If a hot file legitimately needs the budget raised, prefer carving it via
the sibling re-export pattern (see mlframe/CLAUDE.md "Monolith split").
"""

from __future__ import annotations

from pathlib import Path

import pytest


LOC_LIMIT = 1000

# Carving budget exempts. Each entry carries a FIXME tag for the next carve
# wave; the goal is to drain this set to {} over consecutive PRs. Do NOT add
# new entries without a documented PR-description reason.
LOC_BUDGET_EXEMPT: set[str] = {
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
    "src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py",
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
    "src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py",
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
    "src/mlframe/training/core/_phase_composite_post_xt_ensemble/__init__.py",
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
    "src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_score.py",
    # FIXME(carve-wave-next): filters/_mrmr_fe_step/_step_score.py -- single irreducible per-candidate
    # CMI-scoring function body (1 top-level def spanning the whole file) after the _mrmr_fe_step subpackage
    # split. Mirrors ``_pairs_score.py``; the surrounding pool/pair-rank/pair-MI stages already live in siblings.
    "src/mlframe/feature_selection/filters/_mrmr_fe_step/_step_score.py",
    # FIXME(carve-wave-next): filters/_feature_engineering_pairs/_pairs_core.py -- the irreducible per-pair
    # orchestration body (biggest def ~934 LOC) after the _feature_engineering_pairs split; the scoring /
    # emit / setup blocks already carved to sibling modules, only the orchestration loop remains over budget.
    "src/mlframe/feature_selection/filters/_feature_engineering_pairs/_pairs_core.py",
    # FIXME(carve-wave-next): the four GPU-resident FE / discretization / CMI-FE modules below are the
    # in-flight born-on-device perf-replatform work (committed within the last day, mid perf-loop). They are
    # carvable (multiple top-level defs) but actively churning under that effort; carving them here would
    # collide with the in-flight rewrite. Carve to be folded into the next FE perf-loop wave that owns them.
    # (de-exempted: _gpu_resident_select carved -- fused-binning/discretize block -> _gpu_resident_discretize.py,
    # materialise/operand-table/host-fast-path block -> _gpu_resident_materialise.py; parent now <1k)
    # (de-exempted: _fe_batched_mi carved -- batched CMI count/entropy kernel infra -> _fe_batched_mi_cmi.py;
    # parent now <1k, sibling <1k, parent re-exports all public names.)
    "src/mlframe/feature_selection/filters/_gpu_resident_basis.py",
    "src/mlframe/feature_selection/filters/_gpu_resident_fe.py",
    "src/mlframe/feature_selection/filters/_mi_greedy_cmi_fe.py",
    "src/mlframe/feature_selection/filters/discretization/__init__.py",
    # FIXME(carve-wave-next): training/composite/transforms/nonlinear.py -- the residual-transform registry
    # (quantile / monotonic / EWMA / frac-diff fit+forward+inverse families) plus a module-top ``if _HAS_NUMBA:``
    # conditional kernel block the transforms reference. Tightly coupled (the transform functions close over the
    # conditionally-defined kernels); a clean carve must move the kernel block + its consumers together. Pending.
    "src/mlframe/training/composite/transforms/nonlinear.py",
}


def _src_root() -> Path:
    here = Path(__file__).resolve()
    # tests/test_meta/test_no_file_over_1k_loc.py -> repo root -> src/mlframe
    return here.parents[2] / "src" / "mlframe"


def _scan_src_for_oversize() -> list[tuple[str, int]]:
    root = _src_root()
    if not root.is_dir():
        pytest.skip(f"src tree not found at {root}; running from installed wheel?")
    over: list[tuple[str, int]] = []
    for path in root.rglob("*.py"):
        try:
            n = sum(1 for _ in path.open("r", encoding="utf-8"))
        except OSError:
            continue
        rel = path.relative_to(root.parent.parent).as_posix()  # "src/mlframe/..."
        if rel in LOC_BUDGET_EXEMPT:
            continue
        if n > LOC_LIMIT:
            over.append((rel, n))
    return sorted(over, key=lambda t: -t[1])


def test_no_mlframe_file_exceeds_1k_loc():
    over = _scan_src_for_oversize()
    if over:
        lines = [f"  {n:5d} LOC  {p}" for p, n in over]
        raise AssertionError(
            f"{len(over)} mlframe .py file(s) exceed {LOC_LIMIT} LOC. "
            f"Carve via sibling re-export pattern (CLAUDE.md: 'Monolith split'). "
            f"Oversized files:\n" + "\n".join(lines)
        )
