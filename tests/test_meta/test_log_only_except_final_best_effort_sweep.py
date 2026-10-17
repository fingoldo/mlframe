"""log_only_except wave 7 (final non-MRMR wave): closes the remaining 23 findings across
reporting/renderers/_kaleido.py (2, best-effort chart save), training/_training_loop.py (2,
best-effort multilabel-stack / post-hoc calibration fallback), training/baselines/
_dummy_metrics_pick_plot.py (1), training/composite/cache_store.py (2, cache-miss/lock-release
fallback), training/composite/discovery/_screening_tiny.py (2), training/composite/highlevel.py
(1), training/core/_main_train_suite_target_distribution.py (1), training/core/
_predict_main_from_models.py (2), training/evaluation.py (1), training/feature_handling/
locking.py (1), training/io.py (1, escalated via the returned None sentinel), training/
reporting/_reporting.py (1), training/targets/regression_residual_audit.py (1), training/
neural/base/_cuda_fallback.py (1), training/neural/_flat_torch_module/__init__.py (2),
training/neural/_flat_torch_module/_flat_torch_loss.py (1), training/neural/_flat_torch_module/
_flat_torch_predict_accel.py (1) -- all genuinely non-fatal graceful-degradation sites marked
with the scanner's recognized "best-effort" rationale comment. This closes the non-MRMR
log_only_except backlog to 0 (confirmed via a live pyutilz.dev.code_audit.silent_escalation
re-scan, not just baseline-count arithmetic).
"""

from __future__ import annotations

from tests.test_meta._comment_markers import comment_text


def test_kaleido_chart_save_sites_marked_best_effort():
    """Both chart-save except handlers carry a marker."""
    import mlframe.reporting.renderers._kaleido as kaleido

    src = comment_text(kaleido)
    assert src.count("# best-effort:") == 2


def test_training_loop_sites_marked_best_effort():
    """The multilabel-stack and post-hoc-calibration except handlers carry a marker."""
    import mlframe.training._training_loop as tl
    import mlframe.training._training_loop_cb_utils as tl_cb_utils
    import mlframe.training._training_loop_fallback_helpers as tl_fallback

    # the facade's fallback body is split into helper modules; the markers live next to their handlers
    src = comment_text(tl, tl_cb_utils, tl_fallback)
    assert src.count("# best-effort:") == 2


def test_dummy_metrics_pick_plot_site_marked_best_effort():
    """The baseline-overlay plot save except handler carries a marker."""
    import mlframe.training.baselines._dummy_metrics_pick_plot as dmp

    src = comment_text(dmp)
    assert "# best-effort:" in src


def test_cache_store_sites_marked_best_effort():
    """Both cache-miss/lock-release except handlers carry a marker."""
    import mlframe.training.composite.cache_store as cache_store

    src = comment_text(cache_store)
    assert src.count("# best-effort:") == 2


def test_screening_tiny_sites_marked_best_effort():
    """Both n_jobs-cap/CV-fold except handlers carry a marker."""
    import mlframe.training.composite.discovery._screening_tiny as st
    import mlframe.training.composite.discovery._screening_tiny_shared as st_shared

    # the n_jobs-cap handler moved into the shared helper module; the markers live next to their handlers in either
    src = comment_text(st, st_shared)
    assert src.count("best-effort:") == 2


def test_highlevel_conformal_calibration_site_marked_best_effort():
    """The conformal-calibration except handler carries a marker."""
    import mlframe.training.composite.highlevel as highlevel

    src = comment_text(highlevel)
    assert "best-effort:" in src


def test_main_train_suite_target_distribution_site_marked_best_effort():
    """The mini-HPT auto-drop except handler carries a marker."""
    import mlframe.training.core._main_train_suite_target_distribution as tdist

    src = comment_text(tdist)
    assert "# best-effort:" in src


def test_predict_main_from_models_sites_marked_best_effort():
    """Both back-merge/quantile-crossing except handlers carry a marker."""
    import mlframe.training.core._predict_main_from_models as predict_main

    src = comment_text(predict_main)
    assert src.count("# best-effort:") == 2


def test_evaluation_calibration_policy_site_marked_best_effort():
    """The calibration-policy auto-pick except handler carries a marker."""
    import mlframe.training.evaluation as evaluation

    src = comment_text(evaluation)
    assert "# best-effort:" in src


def test_locking_release_site_marked_best_effort():
    """The filelock release except handler carries a marker."""
    import mlframe.training.feature_handling.locking as locking

    src = comment_text(locking)
    assert "# best-effort:" in src


def test_io_load_model_site_marked_best_effort():
    """The model-load except handler (returns None on failure) carries a marker."""
    import mlframe.training.io as io

    src = comment_text(io)
    assert "# best-effort: escalated via the returned None sentinel below" in src


def test_reporting_binary_decile_table_site_marked_best_effort():
    """The binary-decile-table except handler carries a marker."""
    import mlframe.training.reporting._reporting_common as reporting_common

    # the decile-table stamping helper lives in _reporting_common; its handler carries the marker
    src = comment_text(reporting_common)
    assert "# best-effort:" in src


def test_regression_residual_audit_site_marked_best_effort():
    """The chart-retry residual-audit except handler carries a marker."""
    import mlframe.training.targets.regression_residual_audit as rra

    src = comment_text(rra)
    assert "# best-effort:" in src


def test_cuda_fallback_move_to_cpu_site_marked_best_effort():
    """The GPU-to-CPU move except handler carries a marker."""
    import mlframe.training.neural.base._cuda_fallback as cuda_fallback

    src = comment_text(cuda_fallback)
    assert "# best-effort:" in src


def test_flat_torch_module_sites_marked_best_effort():
    """Both optimizer-restore/checkpoint-load except handlers carry a marker."""
    import mlframe.training.neural._flat_torch_module as ftm

    src = comment_text(ftm)
    assert src.count("# best-effort:") == 2


def test_flat_torch_loss_site_marked_best_effort():
    """The per-step metric-compute except handler carries a marker."""
    import mlframe.training.neural._flat_torch_module._flat_torch_loss as ftl

    src = comment_text(ftl)
    assert "# best-effort:" in src


def test_flat_torch_predict_accel_site_marked_best_effort():
    """The torch.compile except handler carries a marker."""
    import mlframe.training.neural._flat_torch_module._flat_torch_predict_accel as ftpa

    src = comment_text(ftpa)
    assert "# best-effort:" in src


def test_non_mrmr_log_only_except_backlog_is_zero():
    """A live scanner re-scan confirms the non-MRMR log_only_except backlog is fully closed."""
    from pathlib import Path

    from pyutilz.dev.code_audit import silent_escalation

    findings = silent_escalation.scan_log_only_except(Path("src/mlframe"))

    def _is_mrmr(f) -> bool:
        """True if a finding's file is under the excluded MRMR subtree."""
        p = f.file.replace("\\", "/")
        return p.startswith("feature_selection/filters/") or p.startswith("feature_selection/shap_proxied_fs/")

    non_mrmr = [f for f in findings if not _is_mrmr(f)]
    assert non_mrmr == [], f"non-MRMR log_only_except findings remain: {non_mrmr}"
