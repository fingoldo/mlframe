"""biz_value test for ``calibration.compute_oof_confidence`` / ``apply_confidence_shrinkage``.

The win: in a multi-output ranking scenario (e.g. per-product recommendation scores), outputs with genuinely
NO discriminative power can still occasionally emit spuriously high raw scores by chance, competing on equal
footing with genuinely-informative outputs and hurting top-K accuracy. Shrinking low-OOF-confidence outputs
toward a neutral value before ranking should measurably improve top-1 accuracy.
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration.confidence_shrinkage import apply_confidence_shrinkage, compute_oof_confidence


def _top1_accuracy(pred_dict: dict, true_relevant: np.ndarray) -> float:
    """Helper that top1 accuracy."""
    names = list(pred_dict.keys())
    matrix = np.column_stack([pred_dict[n] for n in names])
    top1_idx = np.argmax(matrix, axis=1)
    top1_names = np.array(names)[top1_idx]
    return float((top1_names == true_relevant).mean())


def test_biz_val_confidence_shrinkage_improves_top1_ranking_accuracy():
    """Confidence shrinkage improves top1 ranking accuracy."""
    rng = np.random.default_rng(0)
    n_users = 2000
    true_relevant = rng.choice(["A", "B", "C"], size=n_users)

    preds, labels = {}, {}
    for name in ["A", "B", "C"]:
        label = (true_relevant == name).astype(int)
        preds[name] = np.where(label == 1, rng.uniform(0.6, 0.9, n_users), rng.uniform(0.05, 0.2, n_users))
        labels[name] = label
    for name in ["D", "E"]:
        labels[name] = np.zeros(n_users, dtype=int)  # never relevant -- genuinely zero discriminative power
        preds[name] = rng.uniform(0.0, 0.95, n_users)  # but noisy raw scores occasionally spike high

    confidences = {name: compute_oof_confidence(preds[name], labels[name]) for name in preds}
    assert confidences["D"] == 1.0 and confidences["E"] == 1.0
    assert confidences["A"] > 2.0 and confidences["B"] > 2.0 and confidences["C"] > 2.0

    shrunk = apply_confidence_shrinkage(preds, confidences, neutral_value=0.1, min_confidence=1.0)

    raw_accuracy = _top1_accuracy(preds, true_relevant)
    shrunk_accuracy = _top1_accuracy(shrunk, true_relevant)

    assert shrunk_accuracy > raw_accuracy + 0.2, (
        f"confidence shrinkage should measurably improve top-1 ranking accuracy: shrunk={shrunk_accuracy:.4f} raw={raw_accuracy:.4f}"
    )


def test_compute_oof_confidence_no_positives_returns_neutral():
    """Compute oof confidence no positives returns neutral."""
    assert compute_oof_confidence(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])) == 1.0


def test_apply_confidence_shrinkage_zero_span_uses_hard_cutoff():
    """Apply confidence shrinkage zero span uses hard cutoff."""
    preds = {"a": np.array([0.8, 0.9]), "b": np.array([0.1, 0.2])}
    confidences = {"a": 3.0, "b": 3.0}
    shrunk = apply_confidence_shrinkage(preds, confidences, neutral_value=0.5, min_confidence=1.0)
    # equal confidence at the max -> both get full weight (no shrinkage).
    assert np.allclose(shrunk["a"], preds["a"])
    assert np.allclose(shrunk["b"], preds["b"])


def test_biz_val_confidence_shrinkage_per_segment_beats_global_when_reliability_varies():
    """A single output whose discriminative power genuinely differs by subpopulation (e.g. one geo where the
    model works well, one where it's near-random). A global scalar confidence averages the two segments
    together: it either under-shrinks the unreliable segment (letting its noise pollute ranking/decisions)
    or over-shrinks the reliable one (destroying real signal) -- there is no single global ratio that is
    correct for both. Per-segment confidence should shrink each subpopulation by its own true reliability,
    producing predictions that correlate much better with the true labels (segment-wise) than the
    global-scalar shrinkage does.
    """
    rng = np.random.default_rng(1)
    n_per_segment = 3000
    neutral = 0.3  # overall base rate

    # segment "reliable": predictions genuinely track the label.
    label_rel = rng.integers(0, 2, n_per_segment)
    pred_rel = np.where(label_rel == 1, rng.uniform(0.65, 0.95, n_per_segment), rng.uniform(0.05, 0.25, n_per_segment))

    # segment "noisy": model has zero real discriminative power there, but still emits varied scores.
    label_noisy = rng.integers(0, 2, n_per_segment)
    pred_noisy = rng.uniform(0.0, 1.0, n_per_segment)

    segment_ids = np.array(["reliable"] * n_per_segment + ["noisy"] * n_per_segment)
    oof_pred = np.concatenate([pred_rel, pred_noisy])
    oof_label = np.concatenate([label_rel, label_noisy])

    preds = {"out": oof_pred}
    labels = {"out": oof_label}

    # min_confidence set above the noisy segment's small sampling-noise wobble around 1.0 (its true ratio), so
    # the noisy segment gets hard-zero weight (fully neutralized) rather than a near-zero residual weight.
    # max_confidence is fixed explicitly (rather than left to default to "the max confidence observed"), since
    # with only a single output that default would trivially equal whatever confidence is passed in and give
    # every call full weight regardless of shrinkage logic -- an external anchor (e.g. a known best-output
    # confidence from a wider multi-output system) is the realistic setting this feature targets anyway.
    min_confidence = 1.1
    max_confidence = 3.0

    # global (pre-extension) path: one scalar confidence blending both segments together.
    global_confidence = compute_oof_confidence(preds["out"], labels["out"])
    assert isinstance(global_confidence, float)
    global_shrunk = apply_confidence_shrinkage(
        preds, {"out": global_confidence}, neutral_value=neutral, min_confidence=min_confidence, max_confidence=max_confidence
    )["out"]

    # per-segment (new, opt-in) path: one confidence ratio per segment.
    segment_confidence = compute_oof_confidence(preds["out"], labels["out"], segment_ids=segment_ids)
    assert isinstance(segment_confidence, dict)
    assert segment_confidence["reliable"] > 1.5  # genuinely discriminative
    assert abs(segment_confidence["noisy"] - 1.0) < 0.1  # genuinely no signal (up to sampling noise)
    # the pooled global ratio sits strictly between the two segments' true ratios -- neither segment's own
    # reliability is visible from it, which is exactly the failure mode this extension targets.
    assert segment_confidence["noisy"] < global_confidence < segment_confidence["reliable"]

    per_segment_shrunk = apply_confidence_shrinkage(
        preds,
        {"out": segment_confidence},
        neutral_value=neutral,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        segments={"out": segment_ids},
    )["out"]

    def _noisy_segment_variance(shrunk: np.ndarray) -> float:
        """Helper that noisy segment variance."""
        noisy_mask = segment_ids == "noisy"
        return float(np.var(shrunk[noisy_mask]))

    def _reliable_segment_mse(shrunk: np.ndarray) -> float:
        # Brier-style calibration error against the true 0/1 label -- unlike Pearson correlation (invariant
        # to any positive-weight affine shrink), this DOES penalize partial shrinkage: pulling a genuinely
        # informative prediction toward the neutral value moves it away from the true label and raises MSE.
        """Helper that reliable segment mse."""
        reliable_mask = segment_ids == "reliable"
        return float(np.mean((shrunk[reliable_mask] - oof_label[reliable_mask]) ** 2))

    # the global scalar is dragged down by the noisy segment, partially shrinking away the reliable segment's
    # real signal (its shrunk predictions drift toward the neutral value and away from the true labels,
    # relative to per-segment shrinkage, which leaves the reliable segment's full-confidence predictions
    # untouched since its own per-segment confidence clears ``min_confidence`` on its own merits).
    global_reliable_mse = _reliable_segment_mse(global_shrunk)
    per_segment_reliable_mse = _reliable_segment_mse(per_segment_shrunk)
    assert per_segment_reliable_mse < global_reliable_mse - 0.01, (
        f"per-segment shrinkage should preserve the reliable segment's signal (lower MSE vs true label) better than a global scalar: "
        f"per_segment_mse={per_segment_reliable_mse:.4f} global_mse={global_reliable_mse:.4f}"
    )

    # the noisy segment should be fully collapsed to the neutral value under per-segment shrinkage (variance
    # ~0), whereas the global scalar (a blend > min_confidence) leaves some of the noisy segment's spurious
    # spread intact.
    global_noisy_var = _noisy_segment_variance(global_shrunk)
    per_segment_noisy_var = _noisy_segment_variance(per_segment_shrunk)
    assert per_segment_noisy_var < 1e-12
    assert global_noisy_var > per_segment_noisy_var + 1e-4, (
        f"per-segment shrinkage should fully neutralize the genuinely-noisy segment: "
        f"per_segment_var={per_segment_noisy_var:.6f} global_var={global_noisy_var:.6f}"
    )


def test_apply_confidence_shrinkage_segments_omitted_is_bit_identical_to_pre_extension():
    """Apply confidence shrinkage segments omitted is bit identical to pre extension."""
    preds = {"a": np.array([0.1, 0.5, 0.9, 0.3])}
    confidences = {"a": 2.5}
    with_default = apply_confidence_shrinkage(preds, confidences, neutral_value=0.4, min_confidence=1.0, max_confidence=3.0)
    without_new_param = apply_confidence_shrinkage(preds, confidences, neutral_value=0.4, min_confidence=1.0, max_confidence=3.0, segments=None)
    assert np.array_equal(with_default["a"], without_new_param["a"])
