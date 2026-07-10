"""biz_value test for ``votenrank.correlation_diversity_ablation.diversity_ablation_report``.

The win (1st_mechanisms-of-action-moa-prediction.md): a low-correlation, lower-individual-accuracy model
(different architecture, different feature view) can genuinely improve a blend despite ranking below the
top models on individual score -- a naive "keep only the top-K by individual score" filter would wrongly
drop it. This test constructs exactly that scenario: two highly-correlated strong models (near-duplicates of
each other) plus one weaker-individually but genuinely-diverse model, and confirms the ablation report both
(a) flags the diverse model as low-correlation-but-lower-accuracy and (b) measures a REAL blend improvement
from including it.
"""
from __future__ import annotations

import numpy as np

from mlframe.votenrank.correlation_diversity_ablation import diversity_ablation_report


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _make_diverse_ensemble_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x_shared = rng.normal(size=n)
    x_diverse = rng.normal(size=n)
    y_true = 2.0 * x_shared + 1.0 * x_diverse

    # Two strong, near-duplicate models: both see x_shared well, both miss the x_diverse component -- highly
    # correlated with each other (same errors), individually strong-looking on a metric that undersells the
    # blind spot they share.
    model_a = 2.0 * x_shared + 0.1 * rng.standard_normal(n)
    model_b = 2.0 * x_shared + 0.1 * rng.standard_normal(n)

    # A genuinely diverse model: WORSE individually (barely sees x_shared, more noise overall) but captures
    # the x_diverse component the other two entirely miss -- low correlation with A/B by construction, and a
    # real blend contribution despite ranking last on individual RMSE.
    model_c = 0.3 * x_shared + 1.0 * x_diverse + 1.2 * rng.standard_normal(n)

    return y_true, {"model_a": model_a, "model_b": model_b, "model_c": model_c}


def test_biz_val_diversity_ablation_flags_and_confirms_real_blend_improvement():
    y_true, oof_preds = _make_diverse_ensemble_dataset(n=2000, seed=0)
    individual_scores = {name: -_rmse(y_true, pred) for name, pred in oof_preds.items()}  # higher (less negative) is better

    report = diversity_ablation_report(oof_preds, individual_scores, y_true, _rmse, correlation_threshold=0.9, higher_score_is_better=True)

    flagged_names = {entry["model"] for entry in report}
    assert "model_c" in flagged_names, f"expected model_c (diverse, lower individual score) to be flagged, got {flagged_names}"

    model_c_entry = next(e for e in report if e["model"] == "model_c")
    assert model_c_entry["ablation_improvement"] > 0, f"expected including model_c to genuinely improve the blend, got ablation_improvement={model_c_entry['ablation_improvement']:.4f}"
    assert model_c_entry["max_correlation"] < 0.9


def test_diversity_ablation_no_flags_when_all_models_are_near_duplicates():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=500)
    oof_preds = {f"m{i}": y_true + 0.1 * rng.standard_normal(500) for i in range(4)}  # all near-identical
    individual_scores = {name: -_rmse(y_true, pred) for name, pred in oof_preds.items()}

    report = diversity_ablation_report(oof_preds, individual_scores, y_true, _rmse, correlation_threshold=0.9, higher_score_is_better=True)
    assert report == [], f"expected no flags when all models are highly correlated near-duplicates, got {report}"


def test_diversity_ablation_single_model_returns_empty():
    y_true = np.array([1.0, 2.0, 3.0])
    report = diversity_ablation_report({"only": y_true}, {"only": 1.0}, y_true, _rmse)
    assert report == []
