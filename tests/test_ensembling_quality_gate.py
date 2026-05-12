"""Unit tests for ``compute_member_quality_gate`` (2026-05-10).

Background: the gate was extracted out of ``ensemble_probabilistic_predictions``
to be invoked ONCE at the ``score_ensemble`` level (before the flavor loop)
instead of N-times per (flavor, split). Tests pin the contract:

* outlier members get excluded by ``max_mae_relative`` AND/OR ``max_mae``;
* the >2-member precondition is honored (2-member ensembles return no-op);
* a too-tight filter (would exclude EVERY member) falls back to the
  original list rather than producing a degenerate empty ensemble;
* stats dict carries ``median_mae`` / ``median_std`` /
  ``rel_*_threshold`` / ``per_member_*`` for the caller's log line.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.ensembling import compute_member_quality_gate


def _make_members(n_members: int, n_rows: int = 100, *, outlier_idx: int = -1,
                  outlier_noise_mult: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    ground = rng.random(n_rows)
    members = []
    for i in range(n_members):
        noise_mult = outlier_noise_mult if i == outlier_idx else 1.0
        members.append(ground + rng.standard_normal(n_rows) * 0.05 * noise_mult)
    return members


def test_excludes_clear_outlier_relative():
    members = _make_members(4, outlier_idx=3, outlier_noise_mult=100.0)
    kept, excluded, stats = compute_member_quality_gate(
        members, max_mae_relative=2.5,
    )
    assert kept == [0, 1, 2]
    assert len(excluded) == 1 and excluded[0][0] == 3
    assert "rel(mae" in excluded[0][1]
    assert stats["median_mae"] > 0
    assert stats["rel_mae_threshold"] == pytest.approx(2.5 * stats["median_mae"])


def test_two_member_is_noop():
    """Filter only fires for 3+ members; with exactly 2 the median-vs-self
    distance is degenerate so we deliberately skip the gate."""
    members = _make_members(2)
    kept, excluded, stats = compute_member_quality_gate(
        members, max_mae_relative=2.5,
    )
    assert kept == [0, 1]
    assert excluded == []
    assert stats == {}


def test_one_member_is_noop():
    members = _make_members(1)
    kept, excluded, stats = compute_member_quality_gate(members)
    assert kept == [0]
    assert excluded == [] and stats == {}


def test_zero_members():
    kept, excluded, stats = compute_member_quality_gate([])
    assert kept == [] and excluded == [] and stats == {}


def test_too_tight_filter_falls_back_to_full_list():
    """When the active threshold(s) would exclude EVERY member, return
    the original list verbatim AND mark ``filter_too_restrictive=True``
    in the stats dict so the caller can warn."""
    members = _make_members(4)
    kept, excluded, stats = compute_member_quality_gate(
        members,
        max_mae=1e-9,  # impossible absolute threshold
    )
    assert kept == [0, 1, 2, 3]
    assert excluded == []
    assert stats.get("filter_too_restrictive") is True


def test_absolute_threshold_excludes():
    """Absolute thresholds (``max_mae`` / ``max_std``) trigger
    independently of the relative ones."""
    members = _make_members(4, outlier_idx=2, outlier_noise_mult=50.0)
    # Disable relative; use a permissive absolute that still kicks the outlier.
    kept, excluded, stats = compute_member_quality_gate(
        members,
        max_mae=0.5, max_std=0.5,
        max_mae_relative=0.0, max_std_relative=0.0,
    )
    # Outlier (member 2) should be the only excluded
    assert kept == [0, 1, 3]
    assert len(excluded) == 1 and excluded[0][0] == 2
    assert "abs(mae" in excluded[0][1]


def test_uniform_members_keeps_all():
    """All members near-identical: nothing should be excluded."""
    members = _make_members(5, outlier_noise_mult=1.0)  # no outlier
    kept, excluded, stats = compute_member_quality_gate(
        members, max_mae_relative=2.5,
    )
    assert kept == [0, 1, 2, 3, 4]
    assert excluded == []


def test_score_ensemble_label_drops_excluded_member(caplog):
    """User-facing: after the member quality gate fires, the
    ``ensemble_name`` used for downstream model_name_prefix / report
    titles must be rebuilt from SURVIVING members. The historical
    behaviour kept the caller's stamped label (e.g. ``[cb+xgb+lgb+linear]``)
    advertising the dropped member, which made the report misleading.

    Setup: 4 members, one clear outlier; assert the gate's per-flavor
    log line still mentions the outlier (it's the source of truth
    about WHO was dropped) but the ensemble model_name_prefix /
    flavor result names show only the survivors.
    """
    import logging
    from types import SimpleNamespace
    from mlframe.ensembling import score_ensemble

    rng = np.random.default_rng(0)
    n_rows = 200
    ground = rng.random(n_rows)
    members = []
    member_names = ["cb", "xgb", "lgb", "linear"]
    for i, name in enumerate(member_names):
        if i == 3:  # outlier
            preds = ground + rng.standard_normal(n_rows) * 5.0
        else:
            preds = ground + rng.standard_normal(n_rows) * 0.05
        members.append(SimpleNamespace(
            model=None, model_name=name,
            val_preds=preds, test_preds=preds, train_preds=preds,
            val_probs=None, test_probs=None, train_probs=None,
            columns=[], pre_pipeline=None,
        ))

    with caplog.at_level(logging.INFO, logger="mlframe.ensembling"):
        score_ensemble(
            models_and_predictions=members,
            ensemble_name="[cb+xgb+lgb+linear] ",
            target=ground, train_target=ground, val_target=ground, test_target=ground,
            train_idx=np.arange(n_rows), val_idx=np.arange(n_rows),
            test_idx=np.arange(n_rows),
            df=None, verbose=True,
            max_mae_relative=2.5,
            ensembling_methods=("arithm",),
        )

    # The gate log line should mention the dropped outlier so user can
    # diagnose the exclusion.
    gate_lines = [r.getMessage() for r in caplog.records
                  if "member quality gate" in r.getMessage()]
    assert any("linear" in m for m in gate_lines), (
        f"expected gate line to flag 'linear' as excluded; got: {gate_lines}"
    )

    # ANY downstream prefix line referencing the ensemble must NOT
    # include the dropped 'linear' member in the [...] label.
    # The model_name_prefix builds out via "Ens... {ensemble_name}"
    # so we look for matching log lines.
    ens_prefix_lines = [
        r.getMessage() for r in caplog.records
        if "[cb+xgb+lgb+linear]" in r.getMessage()
        and "quality gate" not in r.getMessage()
    ]
    assert not ens_prefix_lines, (
        f"ensemble name should drop 'linear' after gate; still saw: "
        f"{ens_prefix_lines}"
    )


def test_multioutput_predictions_supported():
    """Members can be (N, K) 2-D predictions; gate aggregates per-column
    distance into per-member MAE / STD scalars."""
    rng = np.random.default_rng(0)
    ground = rng.random((50, 3))
    good = [ground + rng.standard_normal((50, 3)) * 0.05 for _ in range(3)]
    bad = ground + rng.standard_normal((50, 3)) * 5.0
    members = good + [bad]
    kept, excluded, stats = compute_member_quality_gate(
        members, max_mae_relative=2.5,
    )
    assert kept == [0, 1, 2]
    assert len(excluded) == 1 and excluded[0][0] == 3


def test_compute_split_metrics_skips_model_none_without_predictions():
    """``model=None`` plus no precomputed split predictions is not a valid prediction request; metrics must skip instead of calling ``None.predict``."""
    import pandas as pd
    from mlframe.training.trainer import _compute_split_metrics

    metrics = {}
    preds, probs, columns = _compute_split_metrics(
        split_name="val",
        df=pd.DataFrame({"x": [0.0, 1.0, 2.0]}),
        target=np.array([0.0, 1.0, 2.0]),
        idx=np.arange(3),
        model=None,
        model_type_name="NoneType",
        model_name="EnsARITHM",
        metrics_dict=metrics,
        preds=None,
        probs=None,
        print_report=False,
        show_perf_chart=False,
        show_fi=False,
    )

    assert preds is None
    assert probs is None
    assert columns == ["x"]
    assert metrics == {}


def test_ensemble_scoring_respects_reporting_metric_switches(monkeypatch):
    """Ensemble pseudo-models must inherit ReportingConfig compute_* flags
    instead of forcing val/test metrics back on."""
    from types import SimpleNamespace

    import mlframe.training as training_mod
    from mlframe.ensembling import _process_single_ensemble_method

    captured = []

    def _fake_train_and_evaluate_model(*, model, data, control, metrics, reporting,
                                       naming, output, confidence, predictions):
        captured.append(
            (
                control.compute_trainset_metrics,
                control.compute_valset_metrics,
                control.compute_testset_metrics,
            )
        )
        return SimpleNamespace(model=model, model_name=naming.model_name_prefix)

    monkeypatch.setattr(
        training_mod,
        "train_and_evaluate_model",
        _fake_train_and_evaluate_model,
    )

    n_rows = 8
    base = np.linspace(0.0, 1.0, n_rows)
    members = [
        SimpleNamespace(
            train_preds=base + i * 0.01,
            val_preds=base + i * 0.01,
            test_preds=base + i * 0.01,
            train_probs=None,
            val_probs=None,
            test_probs=None,
        )
        for i in range(3)
    ]

    _process_single_ensemble_method(
        ensemble_method="arithm",
        level_models_and_predictions=members,
        is_regression=True,
        ensembling_level=0,
        ensemble_name="[a+b+c]",
        target=None,
        train_idx=np.arange(n_rows),
        test_idx=np.arange(n_rows),
        val_idx=np.arange(n_rows),
        train_target=base,
        test_target=base,
        val_target=base,
        target_label_encoder=None,
        max_mae=0.0,
        max_std=0.0,
        max_mae_relative=0.0,
        max_std_relative=0.0,
        ensure_prob_limits=True,
        nbins=10,
        uncertainty_quantile=0.0,
        normalize_stds_by_mean_preds=False,
        custom_ice_metric=None,
        custom_rice_metric=None,
        subgroups=None,
        n_features=1,
        verbose=False,
        kwargs={
            "compute_trainset_metrics": False,
            "compute_valset_metrics": False,
            "compute_testset_metrics": True,
        },
    )

    assert captured == [(False, False, True)]
