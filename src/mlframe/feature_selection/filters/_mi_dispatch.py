"""Unified MI-estimator dispatcher for MRMR (2026-05-29 Wave 7).

Single entry point ``score_pair_mi(x, y, estimator='plug_in', **kwargs)`` that
routes to any of:

  * **'plug_in'** (default): bins both arrays via the WAVE 1-fixed
    ``per_feature_edges`` strategy and computes the plug-in MI on the binned
    arrays. Identical to MRMR's internal pre-2026-05-29 path.

  * **'mixed_ksg'**: ``mlframe.feature_selection.filters._ksg.mixed_ksg_mi``
    (Gao 2017, NeurIPS). k-NN-based, robust to discrete-continuous mixtures.

  * **'ksg_lnc'**: Local-Nonuniformity-Corrected KSG (Gao 2015, AISTATS).
    Auto-fallback to mixed_ksg when y has low entropy (binary / few-class).

  * **'mine'**: PyTorch Donsker-Varadhan neural MI (Belghazi 2018).

  * **'infonet'**: Pre-trained transformer (Hu 2024). One-time 80s CUDA warm-up,
    then ~70 ms per pair on GTX 1050 Ti.

  * **'mist'**: Pre-trained set-transformer (Gerasimov 2025) with empirical
    calibration to nats via Gaussian copula lookup.

  * **'fastmi'**: Copula FFT-KDE (Purkayastha-Song 2024).

  * **'median'**: median(fd, qs, mixed_ksg) panel aggregator.

  * **'genie'**: GENIE-weighted ensemble (Moon 2021, IEEE TIT). The mega-bench
    v3 honest leader -- MI close to truth across signal types, clean noise floor.

Caveats per the mega-bench v3 leaderboard:
  - 'mist' over-estimates by 90-200% on binary y; use as RANKING signal only.
  - 'mine' needs N >= 1000 per pair; under-converges on small CV val folds.
  - 'fastmi' (silverman variant) over-smooths -- use bandwidth='mise'.
  - 'genie' costs ~23x plug_in's wall-time (it runs 3 sub-estimators).

Public entry: ``score_pair_mi(x, y, estimator=..., **kwargs)``.

The dispatcher is bench-validated (see bench_adaptive_nbins_mega) and ready to
plug into MRMR's per-pair MI loop -- the deep internal-loop integration is
deferred to the next sprint because it requires refactoring the njit
``mi_direct`` kernel chain. Until then callers can use ``MRMR.score_pair_mi``
directly for ad-hoc scoring against any estimator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


_PLUG_IN_DEFAULTS = {
    "nbins_strategy": "freedman_diaconis",
    "miller_madow": False,
}


def score_pair_mi(x: np.ndarray, y: np.ndarray, *,
                   estimator: str = "plug_in",
                   estimator_kwargs: Optional[Dict[str, Any]] = None,
                   nbins_strategy: Optional[str] = None,
                   nbins_strategy_kwargs: Optional[Dict[str, Any]] = None,
                   miller_madow: bool = False) -> float:
    """Single entry point to all MI estimators.

    Args:
        x, y: 1-D arrays of equal length.
        estimator: One of the strings listed in the module docstring.
        estimator_kwargs: Forwarded verbatim to the chosen estimator (e.g.
            ``{'k': 7}`` for KSG, ``{'n_epochs': 800}`` for MINE).
        nbins_strategy: For 'plug_in' only -- the per-column bin-chooser
            strategy ('auto', 'fd', 'qs', 'knuth', 'mdlp', etc.). When None,
            uses fixed 10-quantile bins (legacy default).
        nbins_strategy_kwargs: kwargs for the strategy (e.g. ``{'qs_alpha': 0.25}``).
        miller_madow: For 'plug_in' only -- subtract MM bias correction at the
            scoring step. Restores honest no-signal floor at higher M values.

    Returns:
        I(X; Y) in nats. Clamped at 0 for finite-sample negative noise.
    """
    estimator_kwargs = dict(estimator_kwargs or {})
    x = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y).ravel()

    if estimator == "plug_in":
        return _score_plug_in(x, y_arr, nbins_strategy=nbins_strategy, nbins_strategy_kwargs=nbins_strategy_kwargs, miller_madow=miller_madow)
    if estimator in ("mixed_ksg", "ksg_lnc"):
        from ._ksg import mixed_ksg_mi, ksg_lnc_mi
        fn = mixed_ksg_mi if estimator == "mixed_ksg" else ksg_lnc_mi
        return float(fn(x, y_arr.astype(np.float64), **estimator_kwargs))
    if estimator in ("mine", "infonet", "mist"):
        from ._neural_mi import mine_mi, infonet_mi, mist_mi
        fn_map: dict = {"mine": mine_mi, "infonet": infonet_mi, "mist": mist_mi}
        return float(fn_map[estimator](x, y_arr.astype(np.float64), **estimator_kwargs))
    if estimator == "fastmi":
        from ._fastmi import fastmi
        return float(fastmi(x, y_arr.astype(np.float64), **estimator_kwargs))
    if estimator in ("median", "genie"):
        return _score_aggregator(x, y_arr, kind=estimator, nbins_strategy=nbins_strategy, nbins_strategy_kwargs=nbins_strategy_kwargs)
    raise ValueError(f"score_pair_mi: unknown estimator {estimator!r}")


def _score_plug_in(x: np.ndarray, y_arr: np.ndarray, *, nbins_strategy: Optional[str], nbins_strategy_kwargs: Optional[Dict], miller_madow: bool) -> float:
    from ._adaptive_nbins import per_feature_edges, _plug_in_mi
    strategy_kwargs = dict(nbins_strategy_kwargs or {})
    if nbins_strategy is None:
        nbins_strategy = "freedman_diaconis"
    needs_y = nbins_strategy.lower() in ("mdlp", "fayyad_irani", "optimal_joint", "cv")
    y_for_strategy = y_arr if needs_y else None
    edges_list = per_feature_edges(
        x.reshape(-1, 1), y=y_for_strategy,
        method=nbins_strategy, **strategy_kwargs,
    )
    edges = edges_list[0]
    if edges.size == 0:
        return 0.0
    xb = np.searchsorted(edges, x.astype(np.float64), side="right").astype(np.int64)
    # Coerce low-cardinality float y (binary {0., 1.}, few-class) to int label-
    # codes BEFORE plug_in_mi. The internal quantile-bin-of-y path inside
    # _plug_in_mi treats float y as regression-like and collapses
    # ``np.quantile(binary, 11pts) -> unique==2 -> degenerate 1-bin output``
    # (verified by smoke 2026-05-29). Detecting low cardinality up-front
    # avoids the trap.
    if y_arr.dtype.kind not in "iub":
        uniq = np.unique(y_arr)
        if uniq.size <= 32:
            y_int = np.searchsorted(uniq, y_arr).astype(np.int64)
            return _plug_in_mi(xb, y_int, miller_madow=miller_madow)
        return _plug_in_mi(xb, y_arr, miller_madow=miller_madow)
    return _plug_in_mi(xb, y_arr.astype(np.int64), miller_madow=miller_madow)


def _score_aggregator(x: np.ndarray, y_arr: np.ndarray, *, kind: str, nbins_strategy: Optional[str], nbins_strategy_kwargs: Optional[Dict]) -> float:
    from ._mi_aggregator import median_mi_panel, genie_mi_panel
    from ._ksg import mixed_ksg_mi
    estimators = {
        "fd": lambda a, b: _score_plug_in(a, np.asarray(b), nbins_strategy="freedman_diaconis", nbins_strategy_kwargs={}, miller_madow=True),
        "qs": lambda a, b: _score_plug_in(a, np.asarray(b), nbins_strategy="qs", nbins_strategy_kwargs={}, miller_madow=True),
        "mixed_ksg": lambda a, b: float(mixed_ksg_mi(a, np.asarray(b).astype(np.float64), k=5)),
    }
    fn = median_mi_panel if kind == "median" else genie_mi_panel
    return float(fn(x, y_arr, estimators))


__all__ = ["score_pair_mi"]
