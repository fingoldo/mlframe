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
    assert (
        model_c_entry["ablation_improvement"] > 0
    ), f"expected including model_c to genuinely improve the blend, got ablation_improvement={model_c_entry['ablation_improvement']:.4f}"
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


def _make_redundant_trio_dataset(n: int, seed: int):
    """Three candidates each pairwise-diverse from ``best`` (and from each other, below the correlation
    threshold), but jointly redundant: ``c3`` is built as a near-recombination of ``c1``'s and ``c2``'s own
    realized values, so it captures no directional information beyond what ``c1``+``c2`` (or ``c3`` alone)
    already spans. A naive strategy that includes every pairwise-flagged candidate would add all three; the
    genuinely-informative blend only needs ONE of them (whichever the greedy search finds captures the joint
    signal best) on top of ``best``.
    """
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=n)
    u1 = rng.normal(size=n)
    u2 = rng.normal(size=n)
    y_true = shared + u1 + u2

    best = shared + 0.2 * rng.standard_normal(n)

    noise = 1.9
    n1 = noise * rng.standard_normal(n)
    n2 = noise * rng.standard_normal(n)
    c1 = shared + u1 + n1
    c2 = shared + u2 + n2
    c3 = 0.5 * c1 + 0.5 * c2 + 0.35 * rng.standard_normal(n)  # near-recombination of c1's and c2's OWN values

    return y_true, {"best": best, "c1": c1, "c2": c2, "c3": c3}


def test_biz_val_diversity_ablation_greedy_search_avoids_redundant_trio():
    y_true, oof_preds = _make_redundant_trio_dataset(n=4000, seed=0)
    individual_scores = {name: -_rmse(y_true, pred) for name, pred in oof_preds.items()}

    # Default (opt-out) call must stay bit-identical to the pairwise-only report -- no greedy keys added.
    plain_report = diversity_ablation_report(oof_preds, individual_scores, y_true, _rmse, correlation_threshold=0.85, higher_score_is_better=True)
    assert {e["model"] for e in plain_report} == {"c1", "c2", "c3"}, "expected all three trio members individually pairwise-flagged"
    for entry in plain_report:
        assert "greedy_selected" not in entry and "greedy_step" not in entry

    greedy_report = diversity_ablation_report(
        oof_preds, individual_scores, y_true, _rmse, correlation_threshold=0.85, higher_score_is_better=True, use_greedy_search=True, greedy_tolerance=0.0
    )
    # The non-greedy fields must be untouched by opting in.
    for plain_entry, greedy_entry in zip(plain_report, greedy_report):
        for key in plain_entry:
            assert greedy_entry[key] == plain_entry[key]

    selected = {e["model"] for e in greedy_report if e["greedy_selected"]}
    assert len(selected) == 1, f"expected the greedy search to avoid over-including the redundant trio (only 1 of 3 needed), got {selected}"

    def _blend_rmse(names):
        return _rmse(y_true, np.mean([oof_preds[n] for n in names], axis=0))

    naive_all_loss = _blend_rmse(["best", "c1", "c2", "c3"])  # naive: include every pairwise-flagged candidate
    greedy_loss = _blend_rmse(["best", *list(selected)])

    # Measured: naive_all_loss=1.3325, greedy_loss=1.2640, improvement=0.0685 (seed=0, n=4000) -- threshold set
    # ~12% below the measured improvement.
    assert greedy_loss < naive_all_loss - 0.06, (
        f"expected the greedy search's smaller selection to beat blindly including the whole redundant trio "
        f"by >=0.06 RMSE, got naive_all={naive_all_loss:.4f} greedy={greedy_loss:.4f}"
    )
