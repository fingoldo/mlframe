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
    # FIXME(carve-wave-next): _shap_proxy_revalidate.py at ~1.4k LOC carries
    # the trust-guard + topK-ablation + honest-revalidation sub-bodies; sensible
    # split is to lift the honest/ablation block to ``_shap_proxy_revalidate_honest.py``
    # behind a sibling-re-export. Tracked as the largest remaining monolith.
    "src/mlframe/feature_selection/_shap_proxy_revalidate.py",
    # FIXME(carve-wave-next): shap_proxied_fs.py at ~1.03k LOC; the fit body
    # is the obvious candidate for ``_shap_proxied_fs_fit.py``.
    "src/mlframe/feature_selection/shap_proxied_fs.py",
    # FIXME(carve-wave-next): filters/discretization.py at ~1.1k LOC after
    # the wrappers-iter rewrite added KBD / chi-merge / monotone-PAV branches;
    # split to ``_discretization_pav.py`` sibling.
    "src/mlframe/feature_selection/filters/discretization.py",
    # FIXME(carve-wave-next): filters/mrmr.py at ~1.03k LOC after the
    # in-flight feature_selection wrappers iteration grew the screening
    # body; the validate/transform side is already carved (sibling
    # ``_mrmr_validate_transform.py``). The remaining surface candidate is
    # to lift the predictor-screening loop into ``_mrmr_screening_loop.py``.
    "src/mlframe/feature_selection/filters/mrmr.py",
    # FIXME(carve-wave-next): filters/_mrmr_fit_impl.py at ~1.1k LOC after
    # the Wave 9.1 DCD + fallback hardening grew the post-screening section.
    # Carve candidates: the empty-support fallback block + the FE/RFECV
    # post-pass into ``_mrmr_fit_impl_finalise.py``.
    "src/mlframe/feature_selection/filters/_mrmr_fit_impl.py",
    # FIXME(carve-wave-next): training/neural/base.py at ~1.16k LOC after the
    # recent MLP-iter-3 (random_state, regression-metric rename, sklearn-
    # canonical label encoding, all-zero-sample-weight warning) + CUDA-probe
    # wiring. Sibling carve already covers logging / callbacks / tensor helpers
    # / sklearn-params; the remaining facade is the ``PytorchLightningEstimator``
    # class itself. Reasonable next splits: fit / predict body to siblings.
    "src/mlframe/training/neural/base.py",
    # FIXME(carve-wave-next): filters/engineered_recipes.py at ~1.69k LOC --
    # the recipe-replay dispatch grew with the cluster-aggregate + hermite-pair
    # + factorize / target-encoding branches. Sensible carve: lift each recipe
    # kind's ``_apply_*`` body into ``_engineered_recipes_<kind>.py`` siblings,
    # keep the dispatch table + dataclass in the parent.
    "src/mlframe/feature_selection/filters/engineered_recipes.py",
    # FIXME(carve-wave-next): filters/_orthogonal_univariate_fe.py at ~1.49k
    # LOC. The Hermite / Chebyshev / Legendre / Laguerre per-basis optimisers
    # share scaffolding; sibling carve into ``_orthogonal_univariate_fe_<basis>.py``
    # mirrors the polynom-pair carve already shipped.
    "src/mlframe/feature_selection/filters/_orthogonal_univariate_fe.py",
    # FIXME(carve-wave-next): training/core/_phase_train_one_target_body.py
    # at ~1.02k LOC after the recurrent-ensemble integration + composite-
    # discovery wiring. Sibling carve candidates: the recurrent rerun block
    # and the composite-post tail into per-phase helpers.
    "src/mlframe/training/core/_phase_train_one_target_body.py",
    # FIXME(carve-wave-next): filters/_dynamic_cluster_discovery.py at
    # ~1.7k LOC -- Wave 9.1 DCD's core swap / propagate / accept machinery.
    # Sibling carve candidates: swap-matrix extension into
    # ``_dynamic_cluster_discovery_swap.py``, anchor->PC1 / kernel-tuning
    # tau into ``_dynamic_cluster_discovery_anchor.py``.
    "src/mlframe/feature_selection/filters/_dynamic_cluster_discovery.py",
    # FIXME(carve-wave-next): training/core/_phase_composite_post_xt_ensemble.py
    # at ~1.17k LOC after the cross-target ensemble + ``compute_valset_metrics``
    # gating + reporting branch landed. Reasonable next split: the report-emit
    # loop into ``_phase_composite_post_xt_ensemble_report.py``.
    "src/mlframe/training/core/_phase_composite_post_xt_ensemble.py",
    # FIXME(carve-wave-next): _flat_torch_module.py at ~1.1k LOC after the F-37
    # BoundedTanh, F-38 CUDA-graph predict cache, F-39 torch.compile predict,
    # F-40 low-level CUDAGraph() rewrite, F-58 first-batch fix, and F-61
    # pickle hooks. Sensible carve: lift the CUDA-graph + compile predict
    # helpers (~250 LOC, methods ``_maybe_cuda_graph_forward`` +
    # ``_maybe_compile_predict_forward`` + the F-58 sync) into
    # ``_flat_torch_module_predict_accel.py`` as plain functions taking
    # ``self`` -- parent re-attaches them in __init_subclass__-style binding.
    "src/mlframe/training/neural/_flat_torch_module.py",
    # FIXME(carve-wave-next): filters/hermite_fe.py at ~1.41k LOC after the
    # outlier-robust univariate-basis FE axis landed alongside the 4-basis x
    # 4-backend polyeval dispatcher. Sensible carve: lift the per-basis optimiser
    # bodies into ``hermite_fe_<basis>.py`` siblings, keep the dispatcher + the
    # ``optimise_*`` public facade in the parent.
    "src/mlframe/feature_selection/filters/hermite_fe.py",
    # FIXME(carve-wave-next): filters/_feature_engineering_pairs.py at ~1.32k LOC
    # after the batched per-candidate quantile-discretization perf rewrite grew
    # the pair-search body. Carve candidate: the candidate-scoring loop into
    # ``_feature_engineering_pairs_score.py``.
    "src/mlframe/feature_selection/filters/_feature_engineering_pairs.py",
    # FIXME(carve-wave-next): filters/_mrmr_fe_step.py at ~1.09k LOC after the
    # empirical-null (Fix-B) reconciliation grew the per-step accept/gate body.
    # Carve candidate: the null-calibration + accept-decision block into
    # ``_mrmr_fe_step_null.py``.
    "src/mlframe/feature_selection/filters/_mrmr_fe_step.py",
    # FIXME(carve-wave-next): training/_composite_discovery_fit.py at ~1.07k LOC
    # after the gc.collect / commit-charge logging additions. Carve candidate:
    # the discovery fit-loop body into ``_composite_discovery_fit_loop.py``.
    "src/mlframe/training/_composite_discovery_fit.py",
    # FIXME(carve-wave-next): models/_ensembling_base.py at ~1.06k LOC after the
    # HW-calibrated numpy-vs-numba dispatch for 2-D per-member MAE/std landed.
    # Carve candidate: the per-member metric kernels + dispatcher into
    # ``_ensembling_member_metrics.py``.
    "src/mlframe/models/_ensembling_base.py",
    # FIXME(carve-wave-next): training/neural/recurrent.py at ~1.01k LOC after
    # the F-44 bf16-mixed auto-promote + F-46 fused-AdamW + F-47 cuDNN
    # persistent-RNN + F-48 nested-tensor + F-51 share_memory_() + F-53
    # lengths.cpu() non-sync sequence landed. Sensible carve: lift the
    # ``RecurrentDataset`` + collate helpers into
    # ``recurrent_dataset_helpers.py`` sibling; keep the LightningModule
    # in the parent facade.
    "src/mlframe/training/neural/recurrent.py",
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
