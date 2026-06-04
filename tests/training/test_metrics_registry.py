"""Unit tests for mlframe.training.metrics_registry.

Covers public surface: ``register_metric``, ``unregister_metric``, ``iter_extra_metrics``,
``list_registered``, ``list_registered_specs``, ``get_metric_direction``,
``metric_name_higher_is_better``, ``MetricSpec``.

Uses an autouse fixture that snapshot+restores the registry per test (per CLAUDE.md
test-pollution rules + memory feedback_no_module_reload_without_snapshot) so the
built-in multilabel registrations remain visible to other tests after this suite runs.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training import metrics_registry as mr
from mlframe.training.configs import TargetTypes


@pytest.fixture(autouse=True)
def _snapshot_registry():
    """Snapshot the registry before each test and restore after, so adding a
    custom metric never leaks across tests."""
    snapshot = {tt: dict(specs) for tt, specs in mr._REGISTRY.items()}
    yield
    mr._REGISTRY.clear()
    mr._REGISTRY.update(snapshot)


def _dummy_metric(y_true, probs_NK, preds_NK):
    return 0.42


def _raising_metric(y_true, probs_NK, preds_NK):
    raise ValueError("intentional metric failure for tests")


def test_register_and_list_registered():
    mr.register_metric(TargetTypes.REGRESSION, "test_dummy", _dummy_metric)
    names = mr.list_registered(TargetTypes.REGRESSION)
    assert "test_dummy" in names, f"newly registered metric must appear in list_registered; got {names!r}"


def test_register_idempotent_overwrites():
    """Module docstring contract: re-registering the same name overwrites."""
    mr.register_metric(TargetTypes.REGRESSION, "test_metric", _dummy_metric)

    def _replacement(y_true, probs_NK, preds_NK):
        return 1.0

    mr.register_metric(TargetTypes.REGRESSION, "test_metric", _replacement, higher_is_better=False)
    specs = mr.list_registered_specs(TargetTypes.REGRESSION)
    assert specs["test_metric"].fn is _replacement, "second register_metric must overwrite the first"
    assert specs["test_metric"].higher_is_better is False, "second register_metric must overwrite direction too"


def test_unregister_removes_metric():
    mr.register_metric(TargetTypes.REGRESSION, "todelete", _dummy_metric)
    assert "todelete" in mr.list_registered(TargetTypes.REGRESSION)
    mr.unregister_metric(TargetTypes.REGRESSION, "todelete")
    assert "todelete" not in mr.list_registered(TargetTypes.REGRESSION), \
        "unregister_metric must remove the named entry"


def test_unregister_missing_is_noop():
    """Module docstring: ``no-op if not registered``."""
    mr.unregister_metric(TargetTypes.REGRESSION, "this_was_never_registered")  # must not raise


def test_unregister_unknown_target_type_is_noop():
    """The target_type may have nothing registered at all."""
    # Build a fresh target type space (one we know has no entries)
    mr.unregister_metric(TargetTypes.LEARNING_TO_RANK, "anything")  # must not raise


def test_iter_extra_metrics_yields_registered():
    mr.register_metric(TargetTypes.REGRESSION, "spec1", _dummy_metric)
    y = np.array([0.0, 1.0, 2.0])
    out = dict(mr.iter_extra_metrics(TargetTypes.REGRESSION, y, y, y))
    assert "spec1" in out, "registered metric must surface from iter_extra_metrics"
    assert out["spec1"] == pytest.approx(0.42), \
        f"metric value must equal callable return; got {out['spec1']!r}"


def test_iter_extra_metrics_skips_raising_metric(caplog):
    """The module narrowly catches ValueError / ZeroDivisionError / TypeError /
    FloatingPointError. A raising metric must NOT crash iter_extra_metrics — it
    is omitted from output with a WARNING log row."""
    mr.register_metric(TargetTypes.REGRESSION, "bad_metric", _raising_metric)
    mr.register_metric(TargetTypes.REGRESSION, "good_metric", _dummy_metric)

    y = np.array([0.0, 1.0])
    with caplog.at_level(logging.WARNING, logger="mlframe.training.metrics_registry"):
        out = dict(mr.iter_extra_metrics(TargetTypes.REGRESSION, y, y, y))
    assert "good_metric" in out, "non-failing metric must still yield"
    assert "bad_metric" not in out, "raising metric must be omitted from output"
    assert any("bad_metric" in rec.getMessage() for rec in caplog.records), \
        "expected warning log row naming the failing metric"


def test_iter_extra_metrics_keyboardinterrupt_propagates():
    """Programming bugs (e.g. KeyboardInterrupt, RuntimeError) must NOT be swallowed."""
    def _ki_metric(y_true, probs_NK, preds_NK):
        raise KeyboardInterrupt("user pressed ctrl-c")

    mr.register_metric(TargetTypes.REGRESSION, "ki", _ki_metric)
    y = np.array([0.0])
    with pytest.raises(KeyboardInterrupt):
        list(mr.iter_extra_metrics(TargetTypes.REGRESSION, y, y, y))


def test_get_metric_direction_returns_flag():
    mr.register_metric(TargetTypes.REGRESSION, "high_is_good", _dummy_metric, higher_is_better=True)
    mr.register_metric(TargetTypes.REGRESSION, "low_is_good", _dummy_metric, higher_is_better=False)
    assert mr.get_metric_direction(TargetTypes.REGRESSION, "high_is_good") is True
    assert mr.get_metric_direction(TargetTypes.REGRESSION, "low_is_good") is False


def test_get_metric_direction_unknown_returns_none():
    assert mr.get_metric_direction(TargetTypes.REGRESSION, "never_registered") is None


def test_list_registered_specs_returns_metric_specs():
    mr.register_metric(
        TargetTypes.REGRESSION, "with_desc", _dummy_metric,
        higher_is_better=False, description="lower is better metric for X",
    )
    specs = mr.list_registered_specs(TargetTypes.REGRESSION)
    assert isinstance(specs, dict)
    spec = specs["with_desc"]
    assert isinstance(spec, mr.MetricSpec), "list_registered_specs must yield MetricSpec instances"
    assert spec.description == "lower is better metric for X"
    assert spec.higher_is_better is False


def test_list_registered_empty_returns_list():
    """A target_type with no registrations returns [] (not raise, not None)."""
    rv = mr.list_registered(TargetTypes.LEARNING_TO_RANK)
    assert rv == [], f"expected empty list; got {rv!r}"


def test_metric_spec_is_frozen():
    """MetricSpec is decorated frozen=True; mutating must raise."""
    spec = mr.MetricSpec(fn=_dummy_metric, higher_is_better=True, description="x")
    with pytest.raises((AttributeError, Exception)):
        spec.higher_is_better = False


# ----------------------------------------------------------------------------
# metric_name_higher_is_better — direction lookup (canonicalisation)
# ----------------------------------------------------------------------------


def test_metric_direction_higher_known():
    """Canonical higher-is-better names: AUC, NDCG, accuracy, F1, R2."""
    for name in ("auc", "AUC", "val_AUC", "test_NDCG@10", "ndcg", "r2", "f1_macro", "MAP"):
        assert mr.metric_name_higher_is_better(name) is True, \
            f"{name!r} must be classified as higher-is-better"


def test_metric_direction_lower_known():
    for name in ("rmse", "MAE", "test_RMSE", "log_loss", "brier", "hamming_loss", "pinball"):
        assert mr.metric_name_higher_is_better(name) is False, \
            f"{name!r} must be classified as lower-is-better"


def test_metric_direction_unknown_returns_none():
    """No anti-pattern endswith('e') fallback — unknown returns None so caller decides."""
    assert mr.metric_name_higher_is_better("completely_made_up_metric_xyz") is None


def test_metric_direction_prefix_stripped():
    """val_/test_/oof_/train_/holdout_ prefixes must be canonicalised away."""
    assert mr.metric_name_higher_is_better("val_AUC") is True
    assert mr.metric_name_higher_is_better("oof_RMSE") is False
    assert mr.metric_name_higher_is_better("train_F1") is True
    assert mr.metric_name_higher_is_better("holdout_log_loss") is False


def test_metric_direction_at_k_suffix_stripped():
    """@k rank-cutoff suffix must be removed before lookup."""
    assert mr.metric_name_higher_is_better("NDCG@10") is True
    assert mr.metric_name_higher_is_better("val_NDCG@100") is True
    assert mr.metric_name_higher_is_better("precision_at_k") is True


def test_metric_direction_non_string_returns_none():
    """Non-string input returns None (no AttributeError)."""
    assert mr.metric_name_higher_is_better(42) is None
    assert mr.metric_name_higher_is_better(None) is None
    assert mr.metric_name_higher_is_better("") is None


def test_metric_direction_falls_back_to_registry():
    """When name is neither in higher nor lower set, the function scans the registry."""
    mr.register_metric(
        TargetTypes.REGRESSION, "custom_thing", _dummy_metric, higher_is_better=False,
    )
    # Not in either built-in set, but is in registry — must return its direction.
    assert mr.metric_name_higher_is_better("custom_thing") is False


# ----------------------------------------------------------------------------
# Built-in multilabel registrations — biz_value-style assertion
# ----------------------------------------------------------------------------


def test_builtin_multilabel_metrics_registered_at_import():
    """Built-in registrations must land at import time per module docstring."""
    names = set(mr.list_registered(TargetTypes.MULTILABEL_CLASSIFICATION))
    expected = {"hamming_loss", "subset_accuracy", "jaccard_samples"}
    missing = expected - names
    assert not missing, f"expected built-in multilabel metrics missing: {missing}; got {names}"


def test_builtin_hamming_loss_direction_is_lower():
    direction = mr.get_metric_direction(TargetTypes.MULTILABEL_CLASSIFICATION, "hamming_loss")
    assert direction is False, "hamming_loss is a loss (lower-is-better)"


def test_builtin_subset_accuracy_direction_is_higher():
    direction = mr.get_metric_direction(TargetTypes.MULTILABEL_CLASSIFICATION, "subset_accuracy")
    assert direction is True, "subset_accuracy is an accuracy (higher-is-better)"


def test_builtin_metrics_yield_finite_values():
    """biz_value: the built-in multilabel metrics actually run on a tiny synthetic
    and produce finite values — not a crash, not NaN."""
    y_true = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=np.int32)
    preds = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
    # iter_extra_metrics passes probs_NK as the 2nd arg; the builtins ignore it.
    out = dict(mr.iter_extra_metrics(TargetTypes.MULTILABEL_CLASSIFICATION, y_true, preds, preds))
    for name in ("hamming_loss", "subset_accuracy", "jaccard_samples"):
        assert name in out, f"built-in {name} must surface"
        assert np.isfinite(out[name]), f"built-in {name} must yield a finite value; got {out[name]!r}"


def test_mtr_metrics_recover_flattened_preds(caplog):
    """Regression: multi_target_regression metrics must compute when preds arrive
    C-order-flattened to (N*K,) alongside a (N, K) y_true.

    The reporter sometimes passes a raveled (N*K,) preds vector; pre-fix _coerce_nk
    hit the (N,)->(N,1) fallback -> (N*K, 1), so sklearn raised "inconsistent samples
    [N, N*K]" and every MTR metric was silently omitted from the report. _coerce_nk
    now recovers (N, K) from the size-matched flat vector (inverse of the C-order
    ravel), so the metrics compute and equal the values of the (N, K) path.
    """
    from sklearn.metrics import mean_squared_error

    rng = np.random.default_rng(0)
    N, K = 200, 3
    y_true = rng.normal(size=(N, K))
    preds_nk = y_true + rng.normal(scale=0.5, size=(N, K))
    expected_rmse_macro = float(np.sqrt(mean_squared_error(y_true, preds_nk, multioutput="raw_values")).mean())

    with caplog.at_level(logging.WARNING):
        out = dict(mr.iter_extra_metrics(
            TargetTypes.MULTI_TARGET_REGRESSION, y_true, None, preds_nk.ravel(),
        ))

    # No "omitted from report" warning -- the metrics ran rather than being skipped.
    assert not any("omitted from report" in r.message for r in caplog.records), (
        "MTR metrics must not be omitted when preds arrive flattened"
    )
    for name in ("rmse_macro", "rmse_max", "mae_macro", "r2_macro"):
        assert name in out, f"MTR metric {name} must surface from a flattened-preds call"
        assert np.isfinite(out[name]), f"{name} must be finite; got {out[name]!r}"
    # Value matches the (N, K) computation -> the per-row pairing was recovered correctly.
    assert abs(out["rmse_macro"] - expected_rmse_macro) < 1e-9, (
        f"rmse_macro from flattened preds ({out['rmse_macro']}) must equal the (N,K) value "
        f"({expected_rmse_macro}); a mismatch means the flat->(N,K) recovery mis-paired rows"
    )
