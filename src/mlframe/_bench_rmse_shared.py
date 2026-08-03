"""Shared micro-benchmark RMSE helper used by several unrelated ``_benchmarks/`` packages
(training, training/composite/ensemble, models/ensembling): independently duplicated across
those scripts, consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import logging

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-squared error between two same-shaped arrays."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_asarray(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-squared error, coercing both inputs through ``np.asarray`` first."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae_asarray(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error, coercing both inputs through ``np.asarray`` first."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, elementwise."""
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def rss_mb(logger: logging.Logger) -> float:
    """Process resident set size in MB; psutil-optional so the bench runs in minimal envs. NaN (never raises) when psutil is absent or the probe fails."""
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception as exc:
        logger.debug("rss_mb: psutil probe failed: %s", exc)
        return float("nan")


def trimmed_mean(members: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """Symmetric trimmed mean across members: sort each column, drop the lowest/highest ``trim`` fraction, mean the rest."""
    m = members.shape[0]
    k = int(np.floor(trim * m))
    if k == 0:
        return np.asarray(members.mean(axis=0))
    s = np.sort(members, axis=0)
    return np.asarray(s[k : m - k].mean(axis=0))
