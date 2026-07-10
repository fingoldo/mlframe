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


def gp_smooth_irregular_series(
    t: np.ndarray,
    y: np.ndarray,
    t_query: np.ndarray,
    length_scale: float = 1.0,
    nu: float = 1.5,
    alpha: float = 1e-3,
    normalize_y: bool = True,
    optimize_hyperparameters: bool = False,
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
        Matern kernel length scale (smoothness of the fitted curve).
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
        instead.

    Returns
    -------
    tuple
        ``(mean, std)``, each ``(n_query,)`` -- ``std`` is the GP posterior standard deviation, the
        "confidence"/observation-density companion feature: LOW near observed points, growing in
        sparse/extrapolated regions.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    t_arr = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64)
    t_query_arr = np.asarray(t_query, dtype=np.float64).reshape(-1, 1)

    kernel = Matern(length_scale=length_scale, nu=nu)
    optimizer = "fmin_l_bfgs_b" if optimize_hyperparameters else None
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, optimizer=optimizer)
    gp.fit(t_arr, y_arr)
    mean, std = gp.predict(t_query_arr, return_std=True)
    return np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)


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
            sub[time_col].to_numpy(), sub[value_col].to_numpy(), query_arr, length_scale=length_scale, nu=nu, alpha=alpha, optimize_hyperparameters=optimize_hyperparameters
        )
        mean_rows[i] = mean
        std_rows[i] = std

    cols: dict = {entity_col: entities}
    for j in range(len(query_arr)):
        cols[f"{column_prefix}_mean_t{j}"] = mean_rows[:, j]
        cols[f"{column_prefix}_std_t{j}"] = std_rows[:, j]
    return pl.DataFrame(cols)


__all__ = ["gp_smooth_irregular_series", "compute_gp_smoothed_features"]
