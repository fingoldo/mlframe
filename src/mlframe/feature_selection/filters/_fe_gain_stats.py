"""Statistics shared by the FE operators that accept a candidate on a held-out MI gain over a baseline.

A fixed margin on the gain cannot follow n, noise or the shape of the target. The bar used instead is the standard error of the PAIRED difference of two plug-in MIs on the same rows: the
plug-in MI is the mean of the per-row pointwise MI ``log p(x, y) / (p(x) p(y))``, so the difference of two features' MIs is the mean of the per-row differences and its standard error is the
spread of those differences over sqrt(n) (delta method). Bin edges for the held-out rows come from the other half of the rows, never from the rows scored.
"""

from __future__ import annotations

import numpy as np

__all__ = ["quantile_codes", "pointwise_mi", "held_out_codes", "paired_gain_se", "plugin_mi_of_codes"]

N_BINS = 10


def quantile_codes(x: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Equal-frequency bin codes of ``x`` in ``[0, n_bins)`` (ties share a bin; non-finite values go to the lowest bin)."""
    x = np.where(np.isfinite(x), x, -np.inf)
    finite = x[np.isfinite(x)]
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]) if finite.size else np.zeros(n_bins - 1)
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def pointwise_mi(codes: np.ndarray, n_codes: int, y_codes: np.ndarray, ky: int) -> np.ndarray:
    """Per-row pointwise mutual information of integer ``codes`` (in ``[0, n_codes)``) and ``y_codes`` (in ``[0, ky)``); its mean is the plug-in MI in nats."""
    joint = np.bincount(codes * ky + y_codes, minlength=n_codes * ky).reshape(n_codes, ky).astype(np.float64)
    n = joint.sum()
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        table = np.log(joint * n / (px @ py))
    return np.where(joint > 0, table, 0.0)[codes, y_codes]


def plugin_mi_of_codes(codes: np.ndarray, n_codes: int, y_codes: np.ndarray, ky: int) -> float:
    """Plug-in MI (nats) of integer ``codes`` and ``y_codes``."""
    return float(pointwise_mi(codes, n_codes, y_codes, ky).mean())


def held_out_codes(feature: np.ndarray, fit_rows: np.ndarray, score_rows: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin codes of ``feature[score_rows]`` using equal-frequency edges taken from ``feature[fit_rows]`` only."""
    edges = np.quantile(feature[fit_rows], np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.searchsorted(edges, feature[score_rows], side="right").astype(np.int64)


def paired_gain_se(cand_codes: np.ndarray, base_codes: np.ndarray, y_codes: np.ndarray, ky: int, n_codes: int = N_BINS) -> float:
    """``(gain, standard error)`` is returned as the standard error alone: the spread of the per-row difference of the candidate's and the baseline's pointwise MIs over sqrt(n). The gain itself is
    ``plugin_mi_of_codes(cand) - plugin_mi_of_codes(base)`` on the same rows."""
    d = pointwise_mi(cand_codes, n_codes, y_codes, ky) - pointwise_mi(base_codes, n_codes, y_codes, ky)
    return float(d.std() / np.sqrt(len(d)))
