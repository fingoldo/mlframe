"""Quantile-regression dummy baseline computation.

Wave 92 (2026-05-21): split out from `_dummy_baseline_compute.py` to keep
that file below the 1k-line threshold. Behaviour preserved bit-for-bit;
the function is re-exported from `_dummy_baseline_compute` so existing
`from ._dummy_baseline_compute import _compute_quantile_baselines`
imports continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _compute_quantile_baselines(
    target_name: str,
    train_y: np.ndarray,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    alphas: Sequence[float],
    config: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Per-alpha empirical-quantile baselines for QUANTILE_REGRESSION.

    Emits, per requested alpha:
      - ``quantile_alpha_{a:.3f}``: constant prediction = empirical alpha-th
        percentile of train_y (clamped to [1e-3, 1-1e-3] for boundary alpha
        for boundary alpha; shape ``(N, K)`` where K=len(alphas).
      - ``median_for_all``: single ``np.median(train_y)`` constant
        broadcast across all alpha (identical to alpha=0.5 row by
        construction; documented in row label).

    Predictions are 2D ``(N, K)``. Pinball loss is computed per alpha
    plus a ``mean_pinball`` aggregate over non-boundary alpha (alpha in
    ``[0.05, 0.95]``).
    """
    val_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    extras: dict[str, Any] = {}
    n_val = 0 if val_y is None else len(val_y)
    n_test = 0 if test_y is None else len(test_y)
    K = len(alphas)
    if K == 0:
        return val_preds, test_preds, extras

    train_median = float(np.median(train_y))
    boundary_log: list[tuple[float, float]] = []  # (orig, clamped)
    n_eff_val: dict[float, int] = {}
    n_eff_test: dict[float, int] = {}

    # Per-alpha: emit one baseline whose prediction is a constant column for
    # that alpha only, broadcast across the K-output shape so the metrics
    # table can compute pinball@alpha uniformly.
    consts_per_alpha: list[float] = []
    for a in alphas:
        clamped_a = float(min(max(a, 1e-3), 1 - 1e-3))
        if clamped_a != a:
            boundary_log.append((float(a), clamped_a))
        c = float(np.quantile(train_y, clamped_a, method="linear"))
        consts_per_alpha.append(c)
        if val_y is not None:
            n_eff_val[a] = int(np.sum(val_y < c))
        if test_y is not None:
            n_eff_test[a] = int(np.sum(test_y < c))

    # Build (N, K) predictions per baseline.
    if K > 0:
        # Per-alpha empirical-quantile baselines: each one is a (N, K)
        # constant matrix where every output uses its own alpha-th percentile.
        for j, a in enumerate(alphas):
            row_const = consts_per_alpha[j]
            # The j-th baseline emits the j-th constant for ALL alphas
            # (interpretation: "use this alpha-th percentile to predict every
            # quantile" -- degenerate but informative as a reference).
            label = f"quantile_alpha_{a:.3f}"
            if a == 0.5:
                label = f"quantile_alpha_{a:.3f} (=median by construction)"
            val_preds[label] = np.full((n_val, K), row_const)
            test_preds[label] = np.full((n_test, K), row_const)

        # median_for_all: single np.median(train_y) across all alpha.
        val_preds["median_for_all"] = np.full((n_val, K), train_median)
        test_preds["median_for_all"] = np.full((n_test, K), train_median)

        # multi_quantile_empirical: predicts the j-th alpha-th percentile in
        # the j-th column -- the "right" multi-quantile constant baseline.
        # This is actually what most quantile-loss models should beat.
        consts_arr = np.asarray(consts_per_alpha, dtype=np.float64)
        val_preds["multi_quantile_empirical"] = np.broadcast_to(
            consts_arr, (n_val, K)
        ).copy()
        test_preds["multi_quantile_empirical"] = np.broadcast_to(
            consts_arr, (n_test, K)
        ).copy()

    if boundary_log:
        extras["quantile_boundary_clamped"] = boundary_log
        for orig, clamped in boundary_log:
            logger.info(
                "[dummy-baselines] target='%s' alpha=%g: clamped to %g for empirical "
                "baseline (degenerate at boundary)",
                target_name, orig, clamped,
            )
    if n_eff_val:
        extras["quantile_n_eff_val"] = n_eff_val
    if n_eff_test:
        extras["quantile_n_eff_test"] = n_eff_test

    return val_preds, test_preds, extras
