"""Evaluate test-time-augmentation predictive-uncertainty quality on a held-out frame (Workstream B).

Answers "how good is the TTA uncertainty on val/test?" -- runnable inside the suite where the frames are
live (ctx.{val,test}_df), stamping only small metrics (never carrying frames on saved models). For a fitted
regressor it reports: TTA-mean vs single-pass RMSE (does averaging help?) and the correlation between the
TTA spread and the actual error (is the spread an informative uncertainty signal?).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def evaluate_tta_quality(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n: int = 16,
    sigma_scale: float = 0.02,
    seed: int = 0,
) -> dict[str, float]:
    """Metrics for TTA uncertainty on (X, y): point RMSE, TTA-mean RMSE, gain, and spread<->error correlation."""
    from ._tta import tta_point_mean_spread

    Xf = np.asarray(X, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    point_a, mean_a, spread_a = tta_point_mean_spread(predict_fn, Xf, n=n, sigma_scale=sigma_scale, seed=seed)
    point = np.asarray(point_a, dtype=np.float64).reshape(-1)
    tta_mean = np.asarray(mean_a, dtype=np.float64).reshape(-1)
    spread = np.asarray(spread_a, dtype=np.float64).reshape(-1)

    def _rmse(a):
        d = a - yv
        m = np.isfinite(d)
        return float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else float("nan")

    rmse_point = _rmse(point)
    rmse_tta = _rmse(tta_mean)
    err = np.abs(yv - point)
    fin = np.isfinite(err) & np.isfinite(spread)
    corr = float(np.corrcoef(err[fin], spread[fin])[0, 1]) if fin.sum() > 2 and np.std(spread[fin]) > 0 else float("nan")
    return {
        "rmse_point": rmse_point,
        "rmse_tta": rmse_tta,
        "tta_rmse_gain": rmse_point - rmse_tta,
        "spread_error_corr": corr,
        "mean_spread": float(np.mean(spread[np.isfinite(spread)])) if np.isfinite(spread).any() else float("nan"),
    }


def _narrow_numeric_frame(df, columns: Sequence[str]):
    """Return a (rows, len(columns)) float ndarray from ``df[columns]`` (polars/pandas), or None if not all numeric."""
    cols = list(columns)
    try:
        if hasattr(df, "to_pandas"):  # polars
            sub = df.select([c for c in cols if c in df.columns])
            if sub.width != len(cols):
                return None
            pdf = sub.to_pandas()
        else:  # pandas
            if not all(c in df.columns for c in cols):
                return None
            pdf = df[cols]
        import pandas as pd

        arr = pd.DataFrame(pdf).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            return None
        return arr
    except Exception:
        return None
