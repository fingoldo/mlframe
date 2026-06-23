"""Inductive Venn-Abers probability calibration for ``CompositeClassificationEstimator`` (binary).

The classification wrapper emits a ``predict_proba`` that is only as calibrated as
its inner learner -- a boosted / over-fit inner is routinely OVER-CONFIDENT (its
0.95 means an empirical 0.80). Conformal *sets* (``conformal_classification``)
certify the label set but say nothing about the scalar probability. Venn-Abers
fills that gap: it is a distribution-free MULTIPROBABILITY predictor that, on a
held-out calibration set, maps the raw binary score to a calibrated probability
INTERVAL ``[p0, p1]`` whose width reflects the calibration uncertainty, with the
guarantee that one of ``p0, p1`` is perfectly calibrated (Vovk-Petej-Fedorova).

Inductive Venn-Abers (IVAP), the construction used here:

For a test score ``s`` we ask "what isotonic fit do we get if this test point were
labelled 0?" and "...if it were labelled 1?". Concretely we fit TWO isotonic
regressions of label-on-score on the calibration set AUGMENTED by ``(s, 0)`` and by
``(s, 1)`` respectively, and read the fitted value AT ``s`` from each:

  p0 = isotonic_fit(cal + (s,0)) evaluated at s   (the lower probability)
  p1 = isotonic_fit(cal + (s,1)) evaluated at s   (the upper probability)

with ``p0 <= p1`` always. The Venn guarantee: whichever of the two matches the test
label's eventual value is calibrated, so the true probability is bracketed by
``[p0, p1]``. The standard scalar collapse used as the calibrated point estimate is

  p = p1 / (1 - p0 + p1)

(the regularised mean that is itself well-calibrated and minimises log-loss among
the obvious collapses). We do NOT refit per test point at predict time: the two
isotonic step functions over the sorted calibration scores fully determine, for ANY
score ``s``, the pair ``(p0, p1)`` via the precomputed lower/upper envelopes -- so
calibration is O(n log n) once and prediction is a vectorised searchsorted.

Design choices mirror the rest of the package:
- State is stored as plain numpy arrays on ``self._venn_abers_`` (sorted unique
  calibration scores + the two envelopes + the positive class id), so
  ``sklearn.clone`` / pickle stay clean and the wrapper captures no frames.
- Calibration consumes the wrapper's own ``predict_proba`` (full base-margin +
  residual path), so the calibrated probability is for the model actually deployed.
- Binary only: Venn-Abers multiprobability is defined for a single score; multiclass
  needs a one-vs-rest extension out of scope here -- we raise a clear error.
- Held-out contract: ``X_cal`` / ``y_cal`` MUST be rows the inner never trained on;
  the guarantee rests on calibration/test exchangeability, broken by in-sample rows.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _isotonic_envelopes(s_sorted: np.ndarray, y_sorted: np.ndarray):
    """Precompute the IVAP lower (p0) and upper (p1) envelopes over the calibration grid.

    ``s_sorted`` / ``y_sorted`` are the calibration scores (ascending) and their 0/1
    labels in the same order. For a test score equal to grid point ``g_i`` the IVAP
    p0 is the isotonic fit at ``g_i`` of the calibration set with an extra ``(g_i, 0)``
    point, and p1 with an extra ``(g_i, 1)``. Both are computed on the UNIQUE-score
    grid (PAV merges ties anyway); prediction interpolates as a right-continuous step.

    Returns ``(grid, p0_grid, p1_grid)`` -- the unique ascending scores and the two
    fitted-probability arrays aligned to it, with ``p0_grid <= p1_grid`` elementwise.
    """
    grid = np.unique(s_sorted)
    p0 = np.empty(grid.shape[0], dtype=np.float64)
    p1 = np.empty(grid.shape[0], dtype=np.float64)
    base_x = s_sorted.astype(np.float64)
    base_y = y_sorted.astype(np.float64)
    for i, g in enumerate(grid):
        gx = float(g)
        x0 = np.append(base_x, gx)
        y0 = np.append(base_y, 0.0)
        iso0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso0.fit(x0, y0)
        p0[i] = float(iso0.predict([gx])[0])
        y1 = np.append(base_y, 1.0)
        iso1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso1.fit(x0, y1)
        p1[i] = float(iso1.predict([gx])[0])
    # p0 <= p1 holds by construction (adding a 1 cannot lower the PAV fit); enforce
    # against any FP drift so the interval is never inverted.
    lo = np.minimum(p0, p1)
    hi = np.maximum(p0, p1)
    return grid, lo, hi


def _binary_pos_scores(self, X) -> np.ndarray:
    """Wrapper ``predict_proba`` reduced to the P(positive class) column, finite & clipped."""
    proba = np.asarray(self.predict_proba(X), dtype=np.float64)
    if proba.ndim == 1:
        pos = proba
    elif proba.shape[1] == 2:
        pos = proba[:, 1]
    else:
        raise ValueError(
            "Venn-Abers calibration is binary-only; predict_proba returned "
            f"{proba.shape[1]} columns. Use a one-vs-rest wrapper for multiclass."
        )
    return np.clip(pos, 0.0, 1.0)


def calibrate_venn_abers(self, X_cal, y_cal):
    """Fit the Inductive Venn-Abers calibrator from a held-out calibration set.

    ``X_cal`` / ``y_cal`` MUST be rows the inner estimator did NOT train on (the suite
    val split, or an OOF fold) -- Venn-Abers validity rests on calibration/test
    exchangeability, which in-sample rows break. Binary targets only.

    Fits the two isotonic envelopes (label-0 and label-1 augmented PAV fits) over the
    calibration positive-class scores, stores them as plain arrays on
    ``self._venn_abers_``, and returns ``self`` (sklearn-style).
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError

        raise NotFittedError("CompositeClassificationEstimator.calibrate_venn_abers called before fit.")
    classes = np.asarray(self.classes_)
    if classes.size != 2:
        raise ValueError(
            f"Venn-Abers calibration is binary-only; fitted classes_={list(classes)} "
            "has cardinality != 2."
        )
    s = _binary_pos_scores(self, X_cal)
    y_true = np.asarray(y_cal).reshape(-1)
    if s.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"calibrate_venn_abers: predict_proba produced {s.shape[0]} rows but y_cal has {y_true.shape[0]}"
        )
    y_enc = np.searchsorted(classes, y_true)
    if (y_enc < 0).any() or (y_enc >= 2).any() or not np.all(classes[np.clip(y_enc, 0, 1)] == y_true):
        raise ValueError(
            f"calibrate_venn_abers: y_cal contains labels unseen at fit; fitted classes_={list(classes)}"
        )
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    y_sorted = y_enc[order].astype(np.float64)
    grid, p0_grid, p1_grid = _isotonic_envelopes(s_sorted, y_sorted)
    self._venn_abers_ = {
        "grid": grid,
        "p0": p0_grid,
        "p1": p1_grid,
        "n_cal": int(s.shape[0]),
        "pos_label": classes[1],
    }
    return self


def _lookup_interval(self, scores: np.ndarray):
    """Map raw positive-class scores to ``(p_low, p_high)`` via the stored envelopes.

    Right-continuous step lookup: each score is snapped to its nearest calibration
    grid point (searchsorted), then read off the precomputed p0 / p1 arrays. Scores
    below / above the calibration range clip to the boundary grid values (the
    ``out_of_bounds="clip"`` isotonic convention), so prediction is total.

    OFF-GRID CAVEAT: a test score strictly BETWEEN two calibration grid points does NOT interpolate --
    ``searchsorted(side="right") - 1`` reads the LEFT (last grid point <= s) envelope value, a piecewise
    -constant step. This is the correct IVAP semantics (the augmented isotonic fit is constant between
    consecutive calibration scores), so it is by design, not an approximation; but it means the calibrated
    probability is locally flat across each inter-grid gap and the resolution is bounded by the calibration
    set's score granularity. A score exactly on a grid point reads that point; a tie at the grid boundary
    inherits the left bin (right-continuous).
    """
    va = getattr(self, "_venn_abers_", None)
    if not va:
        raise RuntimeError(
            "predict_proba_interval / venn-abers predict_proba called before "
            "calibrate_venn_abers(X_cal, y_cal) on a held-out set."
        )
    grid = va["grid"]
    s = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0, 1.0)
    # nearest grid index on the right-continuous step (idx of last grid point <= s).
    idx = np.searchsorted(grid, s, side="right") - 1
    idx = np.clip(idx, 0, grid.shape[0] - 1)
    return va["p0"][idx], va["p1"][idx]


def predict_proba_interval(self, X):
    """Return the Venn-Abers probability INTERVAL ``(p_low, p_high)`` for the positive class.

    Requires a prior :func:`calibrate_venn_abers` (clear error otherwise -- the
    envelopes cannot be invented from train rows without breaking validity). Both
    outputs are length-``n`` arrays with ``p_low <= p_high`` elementwise; the interval
    brackets the calibrated probability and its width reflects calibration uncertainty.
    """
    scores = _binary_pos_scores(self, X)
    return _lookup_interval(self, scores)


def predict_proba_venn_abers(self, X) -> np.ndarray:
    """Venn-Abers calibrated ``predict_proba`` -- an ``(n, 2)`` matrix [P(neg), P(pos)].

    Collapses the multiprobability interval ``[p0, p1]`` to the regularised point
    estimate ``p = p1 / (1 - p0 + p1)`` (the calibrated, log-loss-minimising mean of
    the two Venn probabilities), then returns it as a proper two-column matrix aligned
    to ``self.classes_`` so it is a drop-in for the raw ``predict_proba``.
    """
    p0, p1 = predict_proba_interval(self, X)
    denom = 1.0 - p0 + p1
    # denom in [1, 2] since 0<=p0<=p1<=1, so it is strictly positive; guard FP anyway.
    p = np.where(denom > 0.0, p1 / denom, p1)
    p = np.clip(p, 0.0, 1.0)
    return np.column_stack([1.0 - p, p])
