"""``adversarial_stochastic_blend``: Monte-Carlo blend-weight search resampled by train/test-likeness.

Source: 2nd_home-credit-default-risk.md's "Adversarial Stochastic Blending" -- sample training rows with
adversarial-validation train/test-likeness as sampling weight, optimize blend weights on that sample, iterate
~350 times and average converged weights (Monte-Carlo). Useful whenever train/test distribution drift is
present: an ordinary blend-weight fit on the full train set optimizes for the TRAIN distribution, which may
not match what the model actually sees at serving/test time; repeatedly refitting on subsamples that
over-represent test-like rows and averaging the results converges toward weights that generalize better to
the actual serving distribution than any single fit on the raw train rows.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from mlframe.votenrank.constrained_weight_blend import constrained_weight_blend


def compute_test_likeness(
    X_train: np.ndarray,
    X_test: np.ndarray,
    classifier_factory: Optional[Callable[[], object]] = None,
    cv: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """OOF P(row is from the test distribution) for every TRAIN row, via a train-vs-test discriminator.

    Parameters
    ----------
    X_train, X_test
        Feature matrices (same columns) for the two distributions being distinguished.
    classifier_factory
        Callable returning a fresh sklearn-compatible classifier; defaults to a small
        ``RandomForestClassifier``.
    cv
        Number of folds for the out-of-fold probability estimate (avoids each train row seeing its own label
        during the discriminator's training).

    Returns
    -------
    np.ndarray
        ``(n_train,)`` OOF probabilities, one per ``X_train`` row -- HIGH means that row looks like it came
        from the test distribution (a genuine adversarial-validation "test-likeness" score).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    if classifier_factory is None:
        classifier_factory = lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)  # noqa: E731

    n_train = len(X_train)
    X_union = np.concatenate([np.asarray(X_train), np.asarray(X_test)], axis=0)
    source_label = np.concatenate([np.zeros(n_train, dtype=int), np.ones(len(X_test), dtype=int)])

    clf = classifier_factory()
    oof_proba = cross_val_predict(clf, X_union, source_label, cv=cv, method="predict_proba")[:, 1]
    return np.asarray(oof_proba[:n_train])


def adversarial_stochastic_blend(
    oof_preds: Sequence[np.ndarray],
    y_true: np.ndarray,
    test_likeness: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 350,
    sample_frac: float = 0.7,
    n_restarts: int = 2,
    random_state: int = 0,
) -> dict:
    """Monte-Carlo blend-weight search: repeatedly subsample training rows weighted by ``test_likeness``,
    fit blend weights on each resample via ``constrained_weight_blend``, average the converged weights.

    Parameters
    ----------
    oof_preds
        Sequence of ``(n_samples,)`` OOF prediction arrays, one per candidate model.
    y_true
        ``(n_samples,)`` ground truth.
    test_likeness
        ``(n_samples,)`` per-row P(row looks like test), e.g. from ``compute_test_likeness`` -- rows with
        higher test-likeness are sampled more often.
    loss_fn
        ``loss_fn(y_true, y_pred) -> float``, LOWER is better.
    n_iterations
        Number of Monte-Carlo resample-and-refit iterations (the source used ~350).
    sample_frac
        Fraction of rows drawn (with replacement, weighted by ``test_likeness``) per iteration.
    n_restarts
        SLSQP restart count passed to each iteration's ``constrained_weight_blend`` call (kept low since
        this runs ``n_iterations`` times).
    random_state
        Seed.

    Returns
    -------
    dict
        ``weights`` ``(n_models,)`` (averaged across iterations, non-negative, sums to 1), ``ensemble_pred``
        ``(n_samples,)`` (the full-data blend under the averaged weights), ``loss`` (achieved on full data),
        ``weight_std`` ``(n_models,)`` (per-model std of the weight across iterations -- HIGH means the
        optimal weight for that model is unstable/sensitive to which rows get sampled).
    """
    preds = np.stack([np.asarray(p, dtype=np.float64) for p in oof_preds], axis=0)
    n_models, n_samples = preds.shape
    y = np.asarray(y_true)
    weights_arr = np.asarray(test_likeness, dtype=np.float64)
    probs = weights_arr / weights_arr.sum() if weights_arr.sum() > 0 else np.full(n_samples, 1.0 / n_samples)

    rng = np.random.default_rng(random_state)
    n_sample_rows = max(2, int(sample_frac * n_samples))
    collected_weights = np.zeros((n_iterations, n_models), dtype=np.float64)

    for it in range(n_iterations):
        idx = rng.choice(n_samples, size=n_sample_rows, replace=True, p=probs)
        sub_preds = [preds[m, idx] for m in range(n_models)]
        sub_y = y[idx]
        result = constrained_weight_blend(sub_preds, sub_y, loss_fn, n_restarts=n_restarts, random_state=int(rng.integers(0, 2**31 - 1)))
        collected_weights[it] = result["weights"]

    avg_weights = collected_weights.mean(axis=0)
    avg_weights = avg_weights / avg_weights.sum() if avg_weights.sum() > 0 else np.full(n_models, 1.0 / n_models)
    weight_std = collected_weights.std(axis=0)

    ensemble_pred = np.tensordot(avg_weights, preds, axes=(0, 0))
    loss = float(loss_fn(y, ensemble_pred))

    return {"weights": avg_weights, "ensemble_pred": ensemble_pred, "loss": loss, "weight_std": weight_std}


__all__ = ["compute_test_likeness", "adversarial_stochastic_blend"]
