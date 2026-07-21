"""Shared top-label reliability-diagram binning core.

``CompositeClassificationEstimator.calibration_report`` (classification.py) and
``diagnostics._bin_top_label_calibration`` (diagnostics.py) both bin the predicted CONFIDENCE
(max class probability) into equal-width bins and compare mean confidence vs observed accuracy --
the standard top-label reliability curve + Expected Calibration Error (ECE). Before this module
existed, the two were independent implementations that could silently drift (a fix to one, e.g. a
tie-breaking edge case, would not propagate to the other) despite the diagnostics-side docstring
claiming they "mirror" each other. This module carries the ONE binning implementation; each caller
resolves its own ``classes`` array (the estimator-backed caller uses ``self.classes_``; the
standalone plotter, which has no estimator object, derives a proxy from ``np.unique(y_true)``) and
passes it in.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def top_label_calibration_bins(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    n_bins: int,
) -> Dict[str, Any]:
    """Equal-width top-label reliability binning + ECE.

    Parameters
    ----------
    y_true : true labels, shape (n,).
    proba : predicted class probabilities, shape (n, n_classes).
    classes : label for each column of ``proba`` (``proba[:, k]`` is the probability of ``classes[k]``).
    n_bins : number of equal-width confidence bins in [0, 1].

    Returns
    -------
    dict with ``bin_confidence`` / ``bin_accuracy`` (NaN for empty bins), ``bin_count``, and the
    scalar ``ece`` (count-weighted mean ``|confidence - accuracy|``).
    """
    proba_arr = np.asarray(proba, dtype=np.float64)
    y_arr = np.asarray(y_true).reshape(-1)
    conf = proba_arr.max(axis=1)
    pred = np.asarray(classes)[np.argmax(proba_arr, axis=1)]
    correct = (pred == y_arr).astype(np.float64)

    nb = int(n_bins)
    edges = np.linspace(0.0, 1.0, nb + 1)
    binid = np.clip(np.digitize(conf, edges[1:-1]), 0, nb - 1)
    bin_conf = np.full(nb, np.nan)
    bin_acc = np.full(nb, np.nan)
    bin_cnt = np.zeros(nb, dtype=np.int64)
    ece = 0.0
    m = conf.size
    for b in range(nb):
        sel = binid == b
        c = int(sel.sum())
        bin_cnt[b] = c
        if c:
            bin_conf[b] = float(conf[sel].mean())
            bin_acc[b] = float(correct[sel].mean())
            ece += (c / m) * abs(bin_conf[b] - bin_acc[b])
    return {"bin_confidence": bin_conf, "bin_accuracy": bin_acc, "bin_count": bin_cnt, "ece": float(ece)}
