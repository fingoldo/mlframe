"""Cluster and moving-block bootstrap for correlated rows.

An i.i.d. row bootstrap treats rows that share a group (or sit next to each other in time) as independent, so the
resampled metric varies less than it does across fresh panels and the interval is too narrow. Resampling whole groups
(cluster bootstrap) or contiguous blocks (moving-block bootstrap) keeps the within-cluster dependence in every resample.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import Any, Optional

import numpy as np

from ._bootstrap_jackknife import _ci_from_samples

logger = logging.getLogger(__name__)


def default_block_length(n: int) -> int:
    """Moving-block length ``ceil(n ** (1/3))``, the standard rate for the mean of a weakly dependent series."""
    return max(2, math.ceil(math.pow(n, 1.0 / 3.0)))


def _group_slices(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(order, offsets)`` such that ``order[offsets[g]:offsets[g+1]]`` are the row indices of group ``g``."""
    _, codes = np.unique(groups, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return order, offsets


def cluster_resample_indices(rng: np.random.Generator, order: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Row indices of one cluster-bootstrap resample: draw as many groups as exist, with replacement, and concatenate their rows."""
    n_groups = offsets.shape[0] - 1
    picks = rng.integers(0, n_groups, size=n_groups)
    return np.concatenate([order[offsets[g] : offsets[g + 1]] for g in picks])


def block_resample_indices(rng: np.random.Generator, n: int, block_length: int) -> np.ndarray:
    """Row indices of one moving-block resample: overlapping blocks of ``block_length`` rows drawn with replacement, trimmed to ``n``."""
    block_length = max(1, min(int(block_length), n))
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block_length)[None, :]).ravel()[:n]


def bootstrap_metrics_clustered(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
    *,
    groups: Optional[np.ndarray] = None,
    block_length: Optional[int] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """Percentile CIs for several metrics under cluster (``groups``) or moving-block (``block_length``) resampling.

    Exactly one of ``groups`` / ``block_length`` selects the scheme; ``groups`` wins when both are given. Returns
    ``{name: {"point", "lo", "hi", "samples", "resampling"}}`` or ``{name: {"error": str}}`` when a metric fails on the
    full sample or on every resample. Rows must be in time order for the block scheme.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    if n < 2 or y_pred.shape[0] != n:
        raise ValueError(f"bootstrap_metrics_clustered: need n>=2 aligned rows; got y_true={n}, y_pred={y_pred.shape[0]}")
    if groups is None and block_length is None:
        raise ValueError("bootstrap_metrics_clustered: pass groups or block_length")
    rng = np.random.default_rng(random_state)
    if groups is not None:
        groups = np.asarray(groups).ravel()
        if groups.shape[0] != n:
            raise ValueError(f"bootstrap_metrics_clustered: groups length {groups.shape[0]} must match y_true length {n}")
        order, offsets = _group_slices(groups)
        if offsets.shape[0] - 1 < 2:
            raise ValueError("bootstrap_metrics_clustered: need at least 2 distinct groups")
        scheme = f"cluster(groups={offsets.shape[0] - 1})"
    else:
        scheme = f"moving_block(length={block_length})"

    results: dict[str, dict[str, Any]] = {}
    points: dict[str, float] = {}
    for name, fn in metric_fns.items():
        try:
            points[name] = float(fn(y_true, y_pred))
        except Exception as exc:  # noqa: PERF203 -- per-metric fault isolation, one bad metric must not sink the others
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    samples: dict[str, list[float]] = {name: [] for name in points}
    for _ in range(n_bootstrap):
        if groups is not None:
            idx = cluster_resample_indices(rng, order, offsets)
        else:
            idx = block_resample_indices(rng, n, int(block_length if block_length is not None and block_length > 0 else default_block_length(n)))
        yt, yp = y_true[idx], y_pred[idx]
        for name in points:
            try:
                v = float(metric_fns[name](yt, yp))
            except Exception as exc:
                logger.debug("clustered bootstrap resample failed for %r: %r", name, exc)
                continue
            if math.isfinite(v):
                samples[name].append(v)
    for name, pt in points.items():
        arr = np.asarray(samples[name], dtype=np.float64)
        if arr.size == 0:
            results[name] = {"error": f"all {n_bootstrap} clustered resamples failed"}
            continue
        lo, hi = _ci_from_samples(arr, pt, alpha, "percentile", None)
        results[name] = {"point": pt, "lo": lo, "hi": hi, "samples": arr, "resampling": scheme}
    return results


def bootstrap_metric_clustered(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    groups: Optional[np.ndarray] = None,
    block_length: Optional[int] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Single-metric form of :func:`bootstrap_metrics_clustered`; returns ``{"point", "lo", "hi", "samples"}`` and raises ``ValueError`` when the metric fails.

    ``block_length=0`` selects :func:`default_block_length`.
    """
    res = bootstrap_metrics_clustered(
        y_true, y_pred, {"m": metric_fn}, groups=groups,
        block_length=block_length,
        n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state,
    )["m"]
    if "error" in res:
        raise ValueError(f"bootstrap_metric_clustered: {res['error']}")
    return {"point": res["point"], "lo": res["lo"], "hi": res["hi"], "samples": res["samples"]}
