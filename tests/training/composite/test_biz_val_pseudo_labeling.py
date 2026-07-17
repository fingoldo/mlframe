"""biz_value test for ``training.composite.PseudoLabelingLoop``.

The win: with a very small labeled set (a decision tree overfits badly at n=35) and a much larger unlabeled
pool from the SAME distribution, leakage-safe fold-ensemble pseudo-labeling with confidence filtering should
recover a modest but real generalization improvement over training on the labeled data alone -- the
realistic, literature-consistent magnitude of semi-supervised self-training gains (this is NOT a dramatic
win like some other techniques; pseudo-labeling gains are small and noisy per-trial, which is why this test
averages over 10 seeds rather than asserting a single-trial threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from mlframe.training.composite import PseudoLabelingLoop


def _make_dataset(n: int, seed: int, d: int = 6):
    """Make dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:3] = [1.5, -1.0, 0.5]
    logit = X @ w
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(float)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y


def test_biz_val_pseudo_labeling_loop_beats_labeled_only_baseline_mean_auc():
    """Biz val pseudo labeling loop beats labeled only baseline mean auc."""
    aucs_base = []
    aucs_pl = []
    for seed in range(10):
        X_labeled, y_labeled = _make_dataset(35, 1000 + seed)
        X_unlabeled, _ = _make_dataset(3000, 2000 + seed)
        X_test, y_test = _make_dataset(3000, 3000 + seed)

        baseline = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_labeled, y_labeled)
        aucs_base.append(roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1]))

        loop = PseudoLabelingLoop(
            estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            task="classification",
            n_rounds=2,
            n_splits=5,
            confidence_threshold=0.8,
            pseudo_label_weight=0.4,
            random_state=0,
        )
        loop.fit(X_labeled, y_labeled, X_unlabeled)
        aucs_pl.append(roc_auc_score(y_test, loop.predict(X_test)))

    mean_base = float(np.mean(aucs_base))
    mean_pl = float(np.mean(aucs_pl))
    improvement = mean_pl - mean_base
    assert improvement > 0.008, f"expected >0.008 mean AUC improvement across 10 seeds, got {improvement:.4f} (base={mean_base:.4f}, pl={mean_pl:.4f})"


def test_pseudo_labeling_loop_confidence_filtering_rejects_low_confidence_rows():
    """Pseudo labeling loop confidence filtering rejects low confidence rows."""
    X_labeled, y_labeled = _make_dataset(40, 1)
    X_unlabeled, _ = _make_dataset(500, 2)
    loop = PseudoLabelingLoop(
        estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
        task="classification",
        n_rounds=1,
        n_splits=5,
        confidence_threshold=0.9,
        random_state=0,
    )
    loop.fit(X_labeled, y_labeled, X_unlabeled)
    accepted, _, _ = loop.pseudo_labels_history_[0]
    assert 0 < accepted.sum() < len(accepted), "expected confidence filtering to accept SOME but not ALL unlabeled rows at a strict threshold"


def _make_imbalanced_dataset(n: int, seed: int, d: int = 6, pos_rate: float = 0.1):
    """Class-imbalanced variant of ``_make_dataset``: the logit is shifted so the minority (positive) class
    occurs at roughly ``pos_rate``, which is where a STATIC confidence threshold accumulates confirmation-bias
    errors fastest -- an early, noisy fold-ensemble already over-trusts majority-class rows at a loose
    threshold, and every later round reinforces that same bias because the bar never tightens."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:3] = [1.8, -1.3, 0.7]
    bias = np.log(pos_rate / (1 - pos_rate))
    p = 1.0 / (1.0 + np.exp(-(X @ w + bias)))
    y = (rng.random(n) < p).astype(float)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y


def test_biz_val_pseudo_labeling_loop_annealed_threshold_beats_static_on_imbalanced_confirmation_bias():
    """Static threshold: a single loose ``confidence_threshold=0.5`` reused across all 6 rounds lets each
    round's own pseudo-labeling mistakes get reinforced by the next. Annealed threshold: the SAME round-0
    threshold (0.5) tightens linearly to 0.9 by the last round, so later rounds -- built on the pool the
    earlier rounds already contaminated -- demand much higher confidence, curbing confirmation bias."""
    aucs_static = []
    aucs_anneal = []
    for seed in range(20):
        X_labeled, y_labeled = _make_imbalanced_dataset(40, 1000 + seed)
        X_unlabeled, _ = _make_imbalanced_dataset(4000, 2000 + seed)
        X_test, y_test = _make_imbalanced_dataset(3000, 3000 + seed)

        static = PseudoLabelingLoop(
            estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            task="classification",
            n_rounds=6,
            n_splits=5,
            confidence_threshold=0.5,
            pseudo_label_weight=0.6,
            random_state=0,
        )
        static.fit(X_labeled, y_labeled, X_unlabeled)
        aucs_static.append(roc_auc_score(y_test, static.predict(X_test)))

        anneal = PseudoLabelingLoop(
            estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
            task="classification",
            n_rounds=6,
            n_splits=5,
            confidence_threshold=0.5,
            pseudo_label_weight=0.6,
            random_state=0,
            threshold_anneal="linear",
            threshold_final=0.9,
        )
        anneal.fit(X_labeled, y_labeled, X_unlabeled)
        aucs_anneal.append(roc_auc_score(y_test, anneal.predict(X_test)))

    mean_static = float(np.mean(aucs_static))
    mean_anneal = float(np.mean(aucs_anneal))
    improvement = mean_anneal - mean_static
    assert improvement > 0.0022, (
        f"expected >0.0022 mean AUC improvement from annealed vs static threshold across 20 seeds, "
        f"got {improvement:.4f} (static={mean_static:.4f}, anneal={mean_anneal:.4f})"
    )


def test_pseudo_labeling_loop_class_thresholds_override_scalar_threshold():
    """Per-class threshold: setting the minority class to a stricter bar than the scalar ``confidence_threshold``
    must accept fewer minority-predicted rows than the scalar-only run at the same round."""
    X_labeled, y_labeled = _make_imbalanced_dataset(60, 1, pos_rate=0.15)
    X_unlabeled, _ = _make_imbalanced_dataset(800, 2, pos_rate=0.15)

    scalar = PseudoLabelingLoop(
        estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
        task="classification",
        n_rounds=1,
        n_splits=5,
        confidence_threshold=0.3,
        random_state=0,
    )
    scalar.fit(X_labeled, y_labeled, X_unlabeled)
    scalar_accept, scalar_mean, _ = scalar.pseudo_labels_history_[0]
    scalar_class1_accepted = int((scalar_accept & (scalar_mean >= 0.5)).sum())

    per_class = PseudoLabelingLoop(
        estimator_factory=lambda: DecisionTreeClassifier(max_depth=4, random_state=0),
        task="classification",
        n_rounds=1,
        n_splits=5,
        confidence_threshold=0.3,
        random_state=0,
        class_thresholds={1: 0.95},
    )
    per_class.fit(X_labeled, y_labeled, X_unlabeled)
    per_class_accept, per_class_mean, _ = per_class.pseudo_labels_history_[0]
    per_class_class1_accepted = int((per_class_accept & (per_class_mean >= 0.5)).sum())

    assert per_class_class1_accepted < scalar_class1_accepted, "stricter per-class threshold should accept fewer minority-predicted rows"


def test_pseudo_labeling_loop_default_static_threshold_bit_identical_without_new_params():
    """Opt-in guard: omitting ``threshold_anneal``/``threshold_final``/``class_thresholds`` must reproduce the
    exact pre-extension static-threshold behavior bit-for-bit."""
    X_labeled, y_labeled = _make_dataset(35, 1)
    X_unlabeled, _ = _make_dataset(300, 2)

    def _factory():
        """Factory."""
        return DecisionTreeClassifier(max_depth=4, random_state=0)

    old_style = PseudoLabelingLoop(
        estimator_factory=_factory, task="classification", n_rounds=3, n_splits=5, confidence_threshold=0.6, pseudo_label_weight=0.4, random_state=0
    )
    old_style.fit(X_labeled, y_labeled, X_unlabeled)

    new_defaults = PseudoLabelingLoop(
        estimator_factory=_factory,
        task="classification",
        n_rounds=3,
        n_splits=5,
        confidence_threshold=0.6,
        pseudo_label_weight=0.4,
        random_state=0,
        threshold_anneal=None,
        threshold_final=None,
        class_thresholds=None,
    )
    new_defaults.fit(X_labeled, y_labeled, X_unlabeled)

    old_pred = old_style.predict(X_unlabeled)
    new_pred = new_defaults.predict(X_unlabeled)
    assert np.array_equal(old_pred, new_pred), "default static-threshold path must be bit-identical to the pre-extension implementation"
    for (a1, m1, c1), (a2, m2, c2) in zip(old_style.pseudo_labels_history_, new_defaults.pseudo_labels_history_):
        assert np.array_equal(a1, a2) and np.array_equal(m1, m2) and np.array_equal(c1, c2)


def test_pseudo_labeling_loop_regression_task_end_to_end():
    """Pseudo labeling loop regression task end to end."""
    rng = np.random.default_rng(0)
    X_labeled = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    y_labeled = X_labeled["a"].to_numpy() * 2 + rng.normal(scale=0.3, size=30)
    X_unlabeled = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])

    from sklearn.linear_model import LinearRegression

    loop = PseudoLabelingLoop(estimator_factory=lambda: LinearRegression(), task="regression", n_rounds=1, n_splits=3, confidence_threshold=1.0, random_state=0)
    loop.fit(X_labeled, y_labeled, X_unlabeled)
    pred = loop.predict(X_labeled)
    assert pred.shape == (30,)
    assert np.isfinite(pred).all()
