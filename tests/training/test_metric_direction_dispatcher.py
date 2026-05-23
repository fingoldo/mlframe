"""Wave-20 sensors: central metric_name_higher_is_better dispatcher +
migration of 6 production sites from ad-hoc whitelists.

Wave 20 audit found 6 production sites where ad-hoc substring / tuple
whitelists determined metric direction (minimize vs maximize):

#1 P0 _callbacks.py:73 derive_mode - drives ACTUAL training early-stop;
   `endswith("e") -> "min"` fallback misclassified custom metric names
   (gini, kappa, mcc, r2, pr_auc, accuracy_score) and silently trained
   the WORST iteration.
#2 P0 dummy_baselines.py:1807 _pick_strongest - whitelist excluded
   val_AUC, val_F1, val_accuracy, val_R2 -> picked WORST baseline as
   strongest via idxmin.
#3 P0 dummy_baselines.py:2102 paired-bootstrap runner-up - same
   whitelist as #2, inherited error -> p_strongest_beats inverted.
#4 P0 dummy_baselines.py:2802 lift-pct verdict gate - substring whitelist
   missed MAPE/MSE/ICE/brier/KL/perplexity -> wrong-sign lift, wrong
   TASK_NON_TRIVIAL verdict.
#5 P1 dummy_baselines.py:606 lift-pct reporter - tuple whitelist that
   disagreed with #2/#3 (included AUC; #2/#3 excluded it). Drift
   evidence of the bug class.
#6 P0 _phase_composite_post.py:509 best-model verdict - substring
   whitelist same shape as #4.

The fix is a single new helper ``metric_name_higher_is_better(name)`` in
metrics_registry that strips val_/test_/oof_ prefixes and @k cutoffs,
looks up against a comprehensive built-in table (higher: AUC, F1,
accuracy, R2, AP, NDCG, MAP, MRR, gini, kappa, mcc, pr_auc, precision,
recall, balanced_accuracy, ...; lower: RMSE, MAE, MSE, MAPE, log_loss,
brier, ICE, ECE, KL, perplexity, pinball, hamming, ...), and falls back
to the per-target registry. Returns None for genuinely unknown metrics
so callers can default explicitly (instead of silently picking wrong).

All 6 sites now call this single dispatcher.
"""
from __future__ import annotations

import logging

import pytest


# ---- direction lookup ---------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    # Higher-is-better classification + ranking
    ("val_AUC", True), ("val_roc_auc", True), ("val_pr_auc", True),
    ("val_F1", True), ("val_f1_macro", True), ("val_accuracy", True),
    ("val_R2", True), ("val_explained_variance", True),
    ("val_AP", True), ("val_average_precision", True),
    ("val_NDCG@10", True), ("val_NDCG@5", True), ("val_MAP@10", True),
    ("val_MRR", True), ("val_recall_at_k", True),
    ("val_gini", True), ("val_kappa", True), ("val_mcc", True),
    ("val_balanced_accuracy", True), ("val_jaccard", True),
    ("val_precision", True), ("val_recall", True),
    # Lower-is-better losses + calibration
    ("val_RMSE", False), ("val_MAE", False), ("val_MSE", False),
    ("val_MAPE", False), ("val_smape", False),
    ("val_log_loss", False), ("val_logloss", False),
    ("val_brier", False), ("val_brier_score", False),
    # Multi-class / multi-label aggregation variants. Without these, the
    # canonicalised lookup misses the parent ("log_loss" != "log_loss_macro")
    # and _pick_strongest warns then silently defaults to minimize.
    # _dummy_metrics_pick_plot.py always emits the *_macro / *_micro
    # variants for multilabel / multiclass dummy baseline tables.
    ("val_log_loss_macro", False), ("val_log_loss_micro", False),
    ("test_logloss_weighted", False), ("val_brier_macro", False),
    ("val_brier_score_micro", False), ("val_cross_entropy_macro", False),
    ("val_ICE", False), ("val_integral_error", False),
    ("val_ECE", False), ("val_KL", False), ("val_perplexity", False),
    ("val_pinball", False), ("val_hamming_loss", False),
    # @k cutoffs strip correctly
    ("val_NDCG@20", True), ("val_precision_at_k@5", True),
    # Mixed prefix
    ("test_AUC", True), ("oof_log_loss", False), ("train_R2", True),
])
def test_metric_name_higher_is_better_known(name, expected):
    from mlframe.training.metrics_registry import metric_name_higher_is_better
    assert metric_name_higher_is_better(name) is expected, (
        f"direction for {name!r} should be higher_is_better={expected}; "
        f"got {metric_name_higher_is_better(name)}"
    )


def test_metric_name_higher_is_better_unknown_returns_none():
    """Genuinely unknown metric MUST return None (not a silent default)
    so the caller is forced to decide whether to raise / warn / default."""
    from mlframe.training.metrics_registry import metric_name_higher_is_better
    assert metric_name_higher_is_better("val_completely_made_up_xyz") is None
    assert metric_name_higher_is_better("") is None
    assert metric_name_higher_is_better("   ") is None


def test_metric_name_higher_is_better_handles_non_string():
    from mlframe.training.metrics_registry import metric_name_higher_is_better
    assert metric_name_higher_is_better(None) is None
    assert metric_name_higher_is_better(123) is None  # type: ignore[arg-type]


# ---- derive_mode migration ---------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("val_AUC", "max"), ("val_RMSE", "min"), ("val_F1", "max"),
    ("val_accuracy", "max"), ("val_R2", "max"), ("val_NDCG@10", "max"),
    ("val_pinball", "min"), ("val_ICE", "min"), ("val_brier", "min"),
    # Wave-20 P0 cases: pre-fix derive_mode classified these as "min" via
    # the endswith("e") fallback. Post-fix: registry dispatcher returns
    # correct direction.
    ("val_gini", "max"), ("val_kappa", "max"), ("val_pr_auc", "max"),
    ("val_mcc", "max"), ("val_balanced_accuracy", "max"),
    ("val_MAPE", "min"), ("val_MSE", "min"), ("val_perplexity", "min"),
    ("val_KL", "min"),
])
def test_callbacks_derive_mode_correct_direction(name, expected):
    from mlframe.training._callbacks import UniversalCallback

    class _MockCallback(UniversalCallback):
        def __init__(self):
            self.verbose = 0
            self.metric_history = {}

    cb = _MockCallback()
    actual = cb.derive_mode(name)
    assert actual == expected, (
        f"Wave 20 P0 regression: derive_mode({name!r}) returned {actual!r}; "
        f"expected {expected!r}. The endswith('e') fallback (and other "
        f"substring heuristics) is the wave-20 bug class."
    )


def test_derive_mode_unknown_warns_and_defaults_min(caplog):
    """Genuinely unknown metric must WARN loudly + return 'min'. The
    WARN is the loud-fail surface: pre-fix the unknown silently became
    'min' with no log signal."""
    from mlframe.training._callbacks import UniversalCallback

    class _MockCallback(UniversalCallback):
        def __init__(self):
            self.verbose = 0
            self.metric_history = {}

    cb = _MockCallback()
    with caplog.at_level(logging.WARNING, logger="mlframe.training._callbacks"):
        out = cb.derive_mode("totally_made_up_metric")
    assert out == "min"
    assert any(
        "cannot determine optimization direction" in r.message
        for r in caplog.records
    )


# ---- migration source-level guards ------------------------------------


def test_dummy_baselines_uses_metric_dispatcher():
    """All 3+ dummy_baselines sites import the dispatcher.

    ``dummy_baselines.py`` was split into themed siblings
    (``_dummy_bootstrap``, ``_dummy_metrics_pick_plot``,
    ``_dummy_report_type``, ``_dummy_summary_format``, ...). The
    dispatcher imports moved with the call sites; this guard now
    concatenates the parent + every sibling so the >=3 occurrence count
    survives the split.
    """
    import pathlib
    import mlframe as _mlframe
    root = pathlib.Path(_mlframe.__file__).resolve().parent / "training"
    src_parts = []
    for rel in (
        "dummy_baselines.py",
        "_dummy_bootstrap.py",
        "_dummy_metrics_pick_plot.py",
        "_dummy_report_type.py",
        "_dummy_summary_format.py",
        "_dummy_compute_helpers.py",
        "_dummy_timeseries.py",
        "_dummy_numba_kernels.py",
    ):
        p = root / rel
        if p.exists():
            src_parts.append(p.read_text(encoding="utf-8"))
    src = "\n".join(src_parts)
    # The pre-fix tuple/substring shapes MUST be gone:
    assert 'primary_metric not in ("val_NDCG@10",' not in src, (
        "Wave 20 P0 regression: _pick_strongest reverted to whitelist"
    )
    assert '"RMSE" in primary_metric\n                or "MAE" in primary_metric' not in src, (
        "Wave 20 P0 regression: lift-pct verdict reverted to substring whitelist"
    )
    # Post-fix marker (the dispatcher must be called at multiple sites):
    occurrences = src.count("metric_name_higher_is_better as _mhb")
    assert occurrences >= 3, (
        f"expected metric_name_higher_is_better imported in >=3 sites; "
        f"got {occurrences} (sites: _pick_strongest, paired-bootstrap, "
        f"lift-pct reporter, lift-pct verdict)"
    )


def test_composite_post_uses_metric_dispatcher():
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "_phase_composite_post.py"
    ).read_text(encoding="utf-8")
    assert "metric_name_higher_is_better as _mhb" in src, (
        "Wave 20 P0 regression: _phase_composite_post no longer uses the "
        "dispatcher"
    )
    # Pre-fix substring whitelist must be gone (the specific shape):
    assert '"RMSE" in _metric_name or "MAE" in _metric_name' not in src


def test_callbacks_uses_metric_dispatcher():
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "_callbacks.py"
    ).read_text(encoding="utf-8")
    assert "metric_name_higher_is_better" in src
    # Pre-fix endswith("e") -> "min" fallback ladder MUST be gone:
    assert 'elif name.endswith("e"):\n            return "min"' not in src, (
        "Wave 20 P0 regression: derive_mode reverted to endswith('e') "
        "heuristic which silently misclassified custom metric names "
        "and trained the WORST iteration."
    )
