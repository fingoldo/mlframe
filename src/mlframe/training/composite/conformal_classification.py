"""Split-conformal prediction SETS for ``CompositeClassificationEstimator``.

The classification wrapper already emits calibrated-ish ``predict_proba``; this
adds a distribution-free, finite-sample-valid coverage guarantee on the LABEL
SET. Given a held-out calibration set (rows the inner never trained on -- the
suite val split, or an OOF fold), we compute a conformity threshold from the
calibration softmax scores and, at predict time, return per row the SET of
labels whose score clears the threshold. Under exchangeability of calibration
and test rows the true label is in the returned set with marginal probability
``>= 1 - alpha`` -- for ANY underlying classifier, binary or multiclass.

Two nonconformity scores are supported (``score="lac"`` default, ``"aps"``):

LAC / LABEL (Least-Ambiguous set-valued Classifier, a.k.a. the softmax / HPS
score): the nonconformity of a (row, label) pair is ``1 - p_hat(label | x)``.
The calibration scores are the true-label nonconformities ``1 - p_hat(y_i | x_i)``;
the threshold ``q`` is their finite-sample ``(1-alpha)`` quantile. A test label
``k`` is INCLUDED when ``1 - p_hat(k | x) <= q``, i.e. ``p_hat(k | x) >= 1 - q``.
LAC gives the SMALLEST average set size for a given coverage but its per-class
coverage is uneven.

APS (Adaptive Prediction Sets, Romano-Sesia-Candes 2020): the score for a label
is the cumulative softmax mass of all labels ranked at least as likely, so it
accumulates probability from the top down until the true label is reached. APS
sets are larger but more adaptive (conditional coverage closer to nominal). The
threshold is the finite-sample ``(1-alpha)`` quantile of the calibration true-
label cumulative scores; a test label is included while its cumulative score
``<= q``.

Design choices mirror the rest of the package:
- The threshold(s) are stored per-alpha in ``self._conformal_set_q_`` -- a plain
  dict of floats (one nested dict per score type), so ``sklearn.clone`` / pickle
  stay clean and the wrapper carries no captured frames.
- Calibration consumes the wrapper's own ``predict_proba`` (the full base-margin
  + residual path), so the scores are on the model the user actually deploys.
- Tiny-n / hard-data contract: when the rank ``ceil((n+1)(1-alpha))`` exceeds
  ``n`` the threshold is ``+inf`` -- every label clears it, so the set is the
  full label space (valid but uninformative) rather than a too-tight set that
  silently under-covers.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_set_threshold(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample ``(1-alpha)`` quantile of the calibration nonconformity scores.

    Uses the conservative 1-indexed rank ``ceil((n+1)(1-alpha))`` -- the smallest
    calibration score that guarantees marginal coverage ``>= 1-alpha``. Returns
    ``+inf`` when that rank exceeds ``n`` (too few calibration points to certify
    the level), so every label is admitted and the set is the full label space:
    valid-but-uninformative rather than silently under-covering.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"conformal alpha must be in (0, 1), got {alpha!r}")
    if n == 0:
        return float("inf")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(s)[rank - 1])


def _lac_true_label_scores(proba: np.ndarray, y_enc: np.ndarray) -> np.ndarray:
    """LAC calibration scores: ``1 - p_hat(true label)`` per calibration row."""
    rows = np.arange(proba.shape[0])
    return np.asarray(1.0 - proba[rows, y_enc])


def _aps_true_label_scores(proba: np.ndarray, y_enc: np.ndarray) -> np.ndarray:
    """APS calibration scores: cumulative softmax mass down to the true label.

    For each row, labels are ranked most- to least-likely; the score is the sum
    of probabilities of every label ranked at least as likely as the true one,
    INCLUDING the true label itself (the non-randomised APS score).
    """
    n, k = proba.shape
    order = np.argsort(-proba, axis=1, kind="stable")  # most -> least likely
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    # rank position of the true label within each row's ordering.
    rank_of_true = (order == y_enc[:, None]).argmax(axis=1)
    return np.asarray(cum[np.arange(n), rank_of_true])


def _proba_matrix(self, X) -> np.ndarray:
    """Wrapper ``predict_proba`` coerced to a finite ``(n, K)`` float64 matrix."""
    proba = np.asarray(self.predict_proba(X), dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    # Guard against tiny negative / >1 drift from the softmax/sigmoid path.
    return np.clip(proba, 0.0, 1.0)


def calibrate_conformal_set(self, X_cal, y_cal, alpha=0.1, score: str = "lac"):
    """Fit the conformal-set threshold from a held-out calibration set.

    ``X_cal`` / ``y_cal`` MUST be rows the inner estimator did NOT train on (the
    suite val split, or an OOF fold) -- conformal validity rests on the
    calibration rows being exchangeable with the test rows, which in-sample rows
    are not. ``score`` selects the nonconformity score (``"lac"`` default, the
    least-ambiguous softmax score giving the smallest sets; ``"aps"`` for the
    adaptive cumulative score with better conditional coverage).

    Stores ``self._conformal_set_q_[score][round(alpha, 6)]`` and returns
    ``self`` (sklearn-style). ``alpha`` may be a scalar or an iterable of levels;
    each is calibrated and cached so :func:`predict_set` can serve any
    pre-calibrated level cheaply.
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError

        raise NotFittedError("CompositeClassificationEstimator.calibrate_conformal_set called before fit.")
    score = str(score).lower()
    if score not in ("lac", "aps"):
        raise ValueError(f"score must be 'lac' or 'aps', got {score!r}")
    proba = _proba_matrix(self, X_cal)
    y_true = np.asarray(y_cal).reshape(-1)
    # Label-encode against the fitted classes_ (the order predict_proba emits in).
    y_enc = np.searchsorted(self.classes_, y_true)
    if proba.shape[0] != y_enc.shape[0]:
        raise ValueError("calibrate_conformal_set: predict_proba produced " f"{proba.shape[0]} rows but y_cal has {y_enc.shape[0]}")
    if (y_enc < 0).any() or (y_enc >= proba.shape[1]).any() or not np.all(self.classes_[np.clip(y_enc, 0, proba.shape[1] - 1)] == y_true):
        raise ValueError("calibrate_conformal_set: y_cal contains labels unseen at fit; " f"fitted classes_={list(self.classes_)}")
    scores = _lac_true_label_scores(proba, y_enc) if score == "lac" else _aps_true_label_scores(proba, y_enc)
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not hasattr(self, "_conformal_set_q_") or self._conformal_set_q_ is None:
        self._conformal_set_q_ = {}
    table = self._conformal_set_q_.setdefault(score, {})
    for a in alphas:
        table[round(float(a), 6)] = conformal_set_threshold(scores, float(a))
    self._conformal_set_n_cal_ = int(proba.shape[0])
    return self


def _label_in_set_mask(proba: np.ndarray, q: float, score: str) -> np.ndarray:
    """Boolean ``(n, K)`` membership mask under the chosen score and threshold.

    LAC: label ``k`` is in the set iff ``1 - p_k <= q``.
    APS: label ``k`` is in the set iff its top-down cumulative mass ``<= q``;
    the most-likely label is always kept (its cumulative mass is its own prob,
    and if even that exceeds ``q`` the set would be empty -- APS keeps the top-1
    so the set is never empty, the standard non-randomised convention).
    """
    n, k = proba.shape
    if score == "lac":
        return (1.0 - proba) <= q
    order = np.argsort(-proba, axis=1, kind="stable")
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    keep_sorted = cum <= q
    keep_sorted[:, 0] = True  # never return an empty set; keep the top label.
    mask = np.zeros((n, k), dtype=bool)
    np.put_along_axis(mask, order, keep_sorted, axis=1)
    return mask


def predict_set(self, X, alpha=0.1, score: str = "lac"):
    """Return, per row, the SET of class labels of marginal coverage ``>= 1-alpha``.

    Requires a prior :func:`calibrate_conformal_set` at this ``alpha`` and
    ``score`` (a clear error otherwise -- the threshold cannot be invented from
    train rows without breaking conformal validity). Returns a list of length
    ``n``; each element is a 1-D ``np.ndarray`` of the original class labels
    (drawn from ``self.classes_``) whose score clears the calibrated threshold.

    LAC may in principle return an empty set when no label clears the threshold;
    to preserve the "the true label is somewhere" reading we fall back to the
    top-1 label in that case (still valid -- emptiness only happens where every
    label is improbable, and the marginal guarantee is over the calibration
    quantile, untouched by this convention). APS is never empty by construction.
    """
    score = str(score).lower()
    key = round(float(alpha), 6)
    table = (getattr(self, "_conformal_set_q_", {}) or {}).get(score, {})
    if key not in table:
        raise RuntimeError(
            f"predict_set: no conformal-set threshold calibrated for "
            f"alpha={alpha}, score={score!r}. Call calibrate_conformal_set("
            f"X_cal, y_cal, alpha={alpha}, score={score!r}) on a held-out set "
            f"first (calibrated levels: {sorted(table.keys())})."
        )
    q = table[key]
    proba = _proba_matrix(self, X)
    mask = _label_in_set_mask(proba, q, score)
    if score == "lac":
        # Never return an empty set: fall back to the argmax label.
        empty = ~mask.any(axis=1)
        if empty.any():
            top = proba.argmax(axis=1)
            mask[empty, top[empty]] = True
    classes = self.classes_
    # FUTURE: this builds a per-row RAGGED prediction set (each row's set has a different length), so the output is a
    # Python list of variable-length arrays -- inherently resistant to a single dense vectorized form. The boolean
    # ``mask`` is already computed vectorized; only this final ragged materialization is per-row Python. A flat
    # (row_idx, class_idx) COO via np.nonzero + np.split would move the loop into C but yields the same Python list of
    # arrays and complicates the empty-set / argmax-fallback semantics. Deferred as not worth the readability cost.
    return [classes[mask[i]] for i in range(proba.shape[0])]
