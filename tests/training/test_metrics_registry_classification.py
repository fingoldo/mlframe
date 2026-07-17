"""Built-in scalar classification metrics wired through the registry + probabilistic reporter.

Covers the wiring of ``quadratic_weighted_kappa`` / ``weighted_kappa`` (ordinal agreement,
higher-is-better) and ``exploss`` (exponential proper scoring, lower-is-better) into
``metrics_registry`` for BINARY / MULTICLASS targets, and their appearance in the metrics dict
emitted by ``report_probabilistic_model_perf``.

Autouse snapshot+restore keeps the built-in registrations visible to other tests (per CLAUDE.md
test-pollution rules + memory feedback_no_module_reload_without_snapshot).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training import metrics_registry as mr
from mlframe.training.configs import TargetTypes


@pytest.fixture(autouse=True)
def _snapshot_registry():
    snapshot = {tt: dict(specs) for tt, specs in mr._REGISTRY.items()}
    yield
    mr._REGISTRY.clear()
    mr._REGISTRY.update(snapshot)


# ---- registration presence + direction ----


@pytest.mark.parametrize("tt", [TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION])
def test_kappa_metrics_registered_for_binary_and_multiclass(tt):
    names = set(mr.list_registered(tt))
    assert {"quadratic_weighted_kappa", "weighted_kappa"} <= names, f"kappa metrics missing for {tt}: {names}"


def test_exploss_registered_for_binary_only():
    assert "exploss" in mr.list_registered(TargetTypes.BINARY_CLASSIFICATION)
    # exploss is a single-vector binary proper scoring rule with no canonical multiclass form.
    assert "exploss" not in mr.list_registered(TargetTypes.MULTICLASS_CLASSIFICATION)


def test_registered_directions_correct():
    for tt in (TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION):
        assert mr.get_metric_direction(tt, "quadratic_weighted_kappa") is True
        assert mr.get_metric_direction(tt, "weighted_kappa") is True
    assert mr.get_metric_direction(TargetTypes.BINARY_CLASSIFICATION, "exploss") is False


def test_direction_name_table_correct():
    assert mr.metric_name_higher_is_better("quadratic_weighted_kappa") is True
    assert mr.metric_name_higher_is_better("weighted_kappa") is True
    assert mr.metric_name_higher_is_better("qwk") is True
    assert mr.metric_name_higher_is_better("exploss") is False


# ---- iter_extra_metrics computes finite values ----


def test_iter_extra_metrics_binary_finite():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    p1 = np.clip(0.25 + 0.5 * y + rng.normal(scale=0.15, size=n), 0.01, 0.99)
    probs = np.column_stack([1 - p1, p1])
    preds = (p1 >= 0.5).astype(np.int64)
    out = dict(mr.iter_extra_metrics(TargetTypes.BINARY_CLASSIFICATION, y, probs, preds))
    for name in ("quadratic_weighted_kappa", "weighted_kappa", "exploss"):
        assert name in out, f"{name} must surface for binary"
        assert np.isfinite(out[name]), f"{name} must be finite; got {out[name]!r}"


def test_iter_extra_metrics_multiclass_finite():
    rng = np.random.default_rng(1)
    n, k = 300, 4
    y = rng.integers(0, k, size=n)
    logits = np.zeros((n, k))
    logits[np.arange(n), y] = 1.5  # signal so preds correlate with y
    logits += rng.normal(scale=1.0, size=(n, k))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    out = dict(mr.iter_extra_metrics(TargetTypes.MULTICLASS_CLASSIFICATION, y, probs, preds))
    for name in ("quadratic_weighted_kappa", "weighted_kappa"):
        assert name in out, f"{name} must surface for multiclass"
        assert np.isfinite(out[name]), f"{name} must be finite; got {out[name]!r}"


def test_qwk_perfect_agreement_is_one():
    """biz_value: on perfectly ordered predictions QWK == 1.0, and a systematic +1 shift drops it."""
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    probs = np.eye(4)[y]  # one-hot => argmax==y
    out_perfect = dict(mr.iter_extra_metrics(TargetTypes.MULTICLASS_CLASSIFICATION, y, probs, y))
    assert out_perfect["quadratic_weighted_kappa"] == pytest.approx(1.0)
    shifted = np.clip(y + 1, 0, 3)
    out_shift = dict(mr.iter_extra_metrics(TargetTypes.MULTICLASS_CLASSIFICATION, y, probs, shifted))
    assert out_shift["quadratic_weighted_kappa"] < out_perfect["quadratic_weighted_kappa"]


def test_exploss_rewards_confident_correct_scores():
    """biz_value: a well-calibrated confident model has lower exploss than a coin-flip model."""
    rng = np.random.default_rng(2)
    n = 400
    y = rng.integers(0, 2, size=n)
    p_good = np.where(y == 1, 0.9, 0.1)
    p_bad = np.full(n, 0.5)
    good = dict(
        mr.iter_extra_metrics(
            TargetTypes.BINARY_CLASSIFICATION,
            y,
            np.column_stack([1 - p_good, p_good]),
            (p_good >= 0.5).astype(int),
        )
    )
    bad = dict(
        mr.iter_extra_metrics(
            TargetTypes.BINARY_CLASSIFICATION,
            y,
            np.column_stack([1 - p_bad, p_bad]),
            (p_bad >= 0.5).astype(int),
        )
    )
    assert good["exploss"] < bad["exploss"], "confident-correct scores must beat coin-flip on exploss (lower better)"


def test_kappa_handles_non_zero_indexed_labels():
    """Labels like {1, 2, 3} must not crash or mis-code (joint searchsorted encoding)."""
    y = np.array([1, 2, 3, 1, 2, 3, 1, 3], dtype=np.int64)
    probs = np.eye(3)[y - 1]
    out = dict(mr.iter_extra_metrics(TargetTypes.MULTICLASS_CLASSIFICATION, y, probs, y))
    assert out["quadratic_weighted_kappa"] == pytest.approx(1.0)


# ---- end-to-end through the actual reporting path ----


def _fit_reference_model(n, k, seed):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    logits = X @ rng.standard_normal((5, k))
    y = logits.argmax(axis=1)
    model = LogisticRegression(max_iter=200).fit(X, y)
    return X, y, model


def test_reporting_path_binary_lands_scalars():
    from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf

    X, y, model = _fit_reference_model(400, 2, seed=10)
    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y,
        columns=[f"f{i}" for i in range(5)],
        model_name="lr",
        model=model,
        probs=model.predict_proba(X),
        preds=model.predict(X),
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    for name in ("quadratic_weighted_kappa", "weighted_kappa", "exploss"):
        assert name in metrics, f"{name} must land in the binary report metrics dict; keys={sorted(map(str, metrics))}"
        assert np.isfinite(metrics[name]), f"{name} must be finite; got {metrics[name]!r}"


def test_reporting_path_multiclass_lands_kappa_scalars():
    from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf

    X, y, model = _fit_reference_model(500, 4, seed=11)
    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y,
        columns=[f"f{i}" for i in range(5)],
        model_name="lr",
        model=model,
        probs=model.predict_proba(X),
        preds=model.predict(X),
        plot_file="",
        show_perf_chart=False,
        print_report=False,
        metrics=metrics,
    )
    for name in ("quadratic_weighted_kappa", "weighted_kappa"):
        assert name in metrics, f"{name} must land in the multiclass report metrics dict"
        assert np.isfinite(metrics[name]), f"{name} must be finite; got {metrics[name]!r}"
