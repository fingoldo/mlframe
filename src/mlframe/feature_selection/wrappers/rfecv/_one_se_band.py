"""Tolerance band of the ``one_se_*`` subset-size rules.

``cv_std_perf`` holds the across-FOLD population std of the k fold scores.  The standard error of the mean CV score is that std over sqrt(k), so
a band of one fold-std is sqrt(k) times wider than one standard error and on noise-robust learners swallows the whole N-range.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

ONE_SE_RULES = ("one_se_max", "one_se_min")
LEGACY_FOLDSTD_RULES = ("one_se_max_foldstd", "one_se_min_foldstd")


def split_rule_band(rule: str) -> tuple:
    """``'one_se_max_foldstd'`` -> ``('one_se_max', 'foldstd')``; any other rule -> ``(rule, 'se')``."""
    if rule in LEGACY_FOLDSTD_RULES:
        return rule[: -len("_foldstd")], "foldstd"
    return rule, "se"


def fold_counts(self: Any, checked_nfeatures: "Sequence[int] | np.ndarray") -> np.ndarray:
    """Number of finite fold scores behind each evaluated N; 1 (no shrinkage) where it cannot be determined."""
    counts = np.ones(len(checked_nfeatures), dtype=float)
    pfs = getattr(self, "_per_fold_scores", None) or {}
    cv_results = getattr(self, "cv_results_", None) or {}
    split_cols = [np.asarray(v, dtype=float) for key, v in cv_results.items() if key.startswith("split") and key.endswith("_test_score")]
    for pos, n in enumerate(checked_nfeatures):
        scores = pfs.get(n)
        if scores is not None and len(scores):
            counts[pos] = max(int(np.isfinite(np.asarray(scores, dtype=float)).sum()), 1)
        elif split_cols and all(len(c) == len(checked_nfeatures) for c in split_cols):
            counts[pos] = max(int(sum(np.isfinite(c[pos]) for c in split_cols)), 1)
    return counts


def band_half_width(std: np.ndarray, k: np.ndarray, band: str = "se") -> np.ndarray:
    """Half-width of the tolerance band: ``std / sqrt(k)`` (standard error) or the raw across-fold ``std`` (``band='foldstd'``)."""
    std = np.asarray(std, dtype=float)
    if band == "foldstd":
        return std
    return np.asarray(std / np.sqrt(np.maximum(np.asarray(k, dtype=float), 1.0)), dtype=float)
