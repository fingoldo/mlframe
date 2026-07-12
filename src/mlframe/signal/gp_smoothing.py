"""``compute_gp_smoothed_features``: Gaussian-Process front end for irregularly-sampled time series.

Source: PLAsTiCC Astronomical Classification 1st place -- "used Gaussian processes to predict the
lightcurves... trained a GP on each object using a Matern Kernel with fixed length scale in one dimension
and variable in another... measured 200 features on the raw data and GP predictions."

Reuses ``sklearn.gaussian_process.GaussianProcessRegressor`` (Matern kernel) rather than implementing GP
regression from scratch. The GP's own POSTERIOR STANDARD DEVIATION at each query point is the "confidence
from local data density" companion feature the idea's own critique calls out as the more transferable half
of the pattern -- no separate distance/density feature needs deriving: a GP's posterior variance is
principled, low near observed points and growing in sparse/extrapolated regions BY CONSTRUCTION.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl


def _fit_predict_gp(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    t_query_arr: np.ndarray,
    length_scale: float,
    nu: float,
    alpha: float,
    normalize_y: bool,
    optimize_hyperparameters: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one Matern-kernel GP and predict ``(mean, std)`` -- shared core reused by CV/ensemble selection."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    kernel = Matern(length_scale=length_scale, nu=nu)
    optimizer = "fmin_l_bfgs_b" if optimize_hyperparameters else None
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, optimizer=optimizer)
    gp.fit(t_arr, y_arr)
    mean, std = gp.predict(t_query_arr, return_std=True)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


def _loo_cv_mse(t_arr: np.ndarray, y_arr: np.ndarray, length_scale: float, nu: float, alpha: float, normalize_y: bool) -> float:
    """Leave-one-out mean squared prediction error for a single candidate ``length_scale`` on this curve.

    Used to pick (or weight) among a small pool of fixed length scales per-entity -- cheap because per-object
    observation counts are small (single digits to low tens) in the irregular-lightcurve use case this module
    targets, so an O(n^2) LOO loop over GP fits is not a bottleneck relative to the final full-data fit.
    """
    n = len(y_arr)
    if n < 3:
        return np.inf
    errors = np.empty(n, dtype=np.float64)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        mask[i] = False
        pred, _ = _fit_predict_gp(t_arr[mask], y_arr[mask], t_arr[i : i + 1], length_scale, nu, alpha, normalize_y, False)
        errors[i] = (pred[0] - y_arr[i]) ** 2
        mask[i] = True
    return float(np.mean(errors))


def gp_smooth_irregular_series(
    t: np.ndarray,
    y: np.ndarray,
    t_query: np.ndarray,
    length_scale: float = 1.0,
    nu: float = 1.5,
    alpha: float = 1e-3,
    normalize_y: bool = True,
    optimize_hyperparameters: bool = False,
    length_scales: Optional[Sequence[float]] = None,
    ensemble_mode: str = "cv_best",
) -> tuple:
    """Fit a Matern-kernel GP on irregular ``(t, y)`` observations, predict ``(mean, std)`` at ``t_query``.

    Parameters
    ----------
    t, y
        Observed (irregularly-sampled) time points and values, ``(n_obs,)``.
    t_query
        Query time points to evaluate the fitted GP at, ``(n_query,)`` -- can include points between/beyond
        observed ``t`` (interpolation/extrapolation).
    length_scale
        Matern kernel length scale (smoothness of the fitted curve). Ignored when ``length_scales`` is given.
    nu
        Matern kernel smoothness parameter (1.5 is once-differentiable, matching typical noisy real-world
        curves; the source used a fixed length scale in one dimension, variable in another -- this single-
        channel primitive is the per-dimension building block, called once per channel for multi-channel
        series).
    alpha
        Observation noise variance added to the kernel diagonal.
    normalize_y
        Standardize ``y`` before fitting (recommended for GPR numerical stability).
    optimize_hyperparameters
        If False (default, matching the source's "FIXED length scale" choice), ``length_scale`` is used
        as-is with no re-optimization -- sklearn's default marginal-likelihood optimizer tends to shrink the
        length scale toward near-zero on sparse/noisy per-object curves (overfitting through the noise
        instead of smoothing it, measured to actively HURT downstream signal quality; see the module's
        biz_value test). Set True to let sklearn's L-BFGS-B optimizer refine ``length_scale`` per curve
        instead. Ignored when ``length_scales`` is given (ensemble candidates are always fit with a fixed
        scale -- the whole point is to sidestep the optimizer's overfitting failure mode by trying a few
        FIXED scales instead of one).
    length_scales
        Opt-in multi-length-scale ensemble: a pool of 2-3 fixed candidate length scales (e.g.
        ``[0.5, 2.0, 8.0]``) spanning fast- to slow-varying curves. ``None`` (default) preserves the
        original single-``length_scale`` behavior bit-for-bit. When given, each candidate is scored per-curve
        by leave-one-out CV MSE on the observed points, then combined per ``ensemble_mode`` -- this replaces
        the single documented compromise (one fixed scale for every object in the pool, underfitting
        fast-varying objects and oversmoothing slow-varying ones) with a per-object data-driven choice, at
        the cost of ``O(len(length_scales) * n_obs)`` extra GP fits for the LOO scoring.
    ensemble_mode
        ``"cv_best"`` (default when ``length_scales`` is given): fit the final GP on ALL observations using
        the single candidate with lowest LOO MSE. ``"cv_blend"``: fit the final GP at every candidate and
        blend their ``(mean, std)`` predictions, weighted by each candidate's inverse LOO MSE (softmin over
        the LOO errors) -- smoother than a hard pick when two candidates are near-tied.

    Returns
    -------
    tuple
        ``(mean, std)``, each ``(n_query,)`` -- ``std`` is the GP posterior standard deviation, the
        "confidence"/observation-density companion feature: LOW near observed points, growing in
        sparse/extrapolated regions.
    """
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64)
    t_query_arr = np.asarray(t_query, dtype=np.float64).reshape(-1, 1)

    if length_scales is None:
        return _fit_predict_gp(t_arr, y_arr, t_query_arr, length_scale, nu, alpha, normalize_y, optimize_hyperparameters)

    candidates = list(length_scales)
    if len(candidates) < 2:
        raise ValueError(f"length_scales must contain at least 2 candidates for ensemble mode, got {candidates!r}")

    cv_errors = np.array([_loo_cv_mse(t_arr, y_arr, ls, nu, alpha, normalize_y) for ls in candidates])
    if not np.isfinite(cv_errors).any():
        # Too few observations for LOO CV (n_obs < 3): fall back to the middle candidate (a neutral default
        # that neither over- nor under-smooths) rather than raising on legitimately sparse real-world curves.
        fallback_scale = candidates[len(candidates) // 2]
        return _fit_predict_gp(t_arr, y_arr, t_query_arr, fallback_scale, nu, alpha, normalize_y, False)

    if ensemble_mode == "cv_best":
        best_scale = candidates[int(np.nanargmin(cv_errors))]
        return _fit_predict_gp(t_arr, y_arr, t_query_arr, best_scale, nu, alpha, normalize_y, False)
    elif ensemble_mode == "cv_blend":
        finite_errors = np.where(np.isfinite(cv_errors), cv_errors, np.nanmax(cv_errors[np.isfinite(cv_errors)]) * 10.0)
        weights = np.exp(-finite_errors / (finite_errors.mean() + 1e-12))
        weights = weights / weights.sum()
        means = np.zeros_like(t_query_arr, dtype=np.float64).reshape(-1)
        stds = np.zeros_like(means)
        for w, ls in zip(weights, candidates):
            m, s = _fit_predict_gp(t_arr, y_arr, t_query_arr, ls, nu, alpha, normalize_y, False)
            means += w * m
            stds += w * s
        return means, stds
    else:
        raise ValueError(f"ensemble_mode must be 'cv_best' or 'cv_blend', got {ensemble_mode!r}")


def compute_gp_smoothed_features(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    value_col: str,
    query_times: Union[Sequence[float], np.ndarray],
    length_scale: float = 1.0,
    nu: float = 1.5,
    alpha: float = 1e-3,
    optimize_hyperparameters: bool = False,
    column_prefix: str = "gp",
    length_scales: Optional[Sequence[float]] = None,
    ensemble_mode: str = "cv_best",
) -> pl.DataFrame:
    """Fit one GP per entity on its own irregularly-sampled observations, evaluate at shared ``query_times``.

    Parameters
    ----------
    df
        Long-format observations: one row per ``(entity_col, time_col, value_col)`` reading.
    entity_col
        Column identifying which object/entity/channel each observation belongs to (one GP fit per unique
        value).
    time_col, value_col
        The irregular observation times and values.
    query_times
        Fixed, shared query grid every entity's GP is evaluated at (produces a REGULAR, entity-comparable
        feature matrix from irregular raw sampling).
    length_scales, ensemble_mode
        Opt-in multi-length-scale ensemble, passed straight through to ``gp_smooth_irregular_series`` and
        applied independently per entity (each entity picks/blends its own best-fitting scale from the same
        candidate pool). ``length_scales=None`` (default) preserves the original single-``length_scale``
        behavior bit-for-bit.

    Returns
    -------
    pl.DataFrame
        One row per unique entity (in first-seen order), columns ``{entity_col}``, then
        ``{prefix}_mean_t{i}`` / ``{prefix}_std_t{i}`` for each query time.
    """
    query_arr = np.asarray(query_times, dtype=np.float64)
    entities = pd.unique(df[entity_col])

    # A per-entity boolean-mask filter (df[df[entity_col] == entity]) rescans the FULL frame for every one
    # of n_entities entities -- O(n_entities * n_rows). A single groupby call is O(n_rows) (hash-based),
    # matching the fix already applied to SegmentedModelFactory's identical pattern earlier this session.
    grouped = {entity: sub for entity, sub in df.groupby(entity_col, sort=False)}

    mean_rows = np.zeros((len(entities), len(query_arr)), dtype=np.float64)
    std_rows = np.zeros((len(entities), len(query_arr)), dtype=np.float64)
    for i, entity in enumerate(entities):
        sub = grouped[entity]
        mean, std = gp_smooth_irregular_series(
            sub[time_col].to_numpy(),
            sub[value_col].to_numpy(),
            query_arr,
            length_scale=length_scale,
            nu=nu,
            alpha=alpha,
            optimize_hyperparameters=optimize_hyperparameters,
            length_scales=length_scales,
            ensemble_mode=ensemble_mode,
        )
        mean_rows[i] = mean
        std_rows[i] = std

    cols: dict = {entity_col: entities}
    for j in range(len(query_arr)):
        cols[f"{column_prefix}_mean_t{j}"] = mean_rows[:, j]
        cols[f"{column_prefix}_std_t{j}"] = std_rows[:, j]
    return pl.DataFrame(cols)


__all__ = ["gp_smooth_irregular_series", "compute_gp_smoothed_features"]
