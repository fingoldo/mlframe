"""``collapse_predictions_by_group``: broadcast a group-level prediction statistic back to every row.

Source: 6th_ieee-cis-fraud-detection.md -- "Take all predictions from a customer based on our UID6, and
combine the predictions to a single value so that all transactions of a customer have the same value... take
90% quantile of the predictions." Individual per-row predictions for the same entity can disagree even when
within-entity consistency is known to matter more than per-row independence (e.g. fraud risk genuinely is a
property of the CUSTOMER, not the individual transaction) -- collapsing to a group statistic (mean, or a
high quantile to bias toward the entity's worst-observed risk) and broadcasting it back removes that
inconsistency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_predictions_by_group(predictions: np.ndarray, group: np.ndarray, stat: str = "mean", quantile: float = 0.9) -> np.ndarray:
    """Collapse ``predictions`` within each ``group`` to a single statistic, broadcast back to every row.

    Parameters
    ----------
    predictions
        ``(n,)`` per-row predictions.
    group
        ``(n,)`` grouping key (e.g. a customer/entity id).
    stat
        ``"mean"`` or ``"quantile"``.
    quantile
        Used only when ``stat="quantile"`` (e.g. ``0.9`` matches the source's own choice -- biases toward
        the entity's higher-risk transactions rather than averaging them away).

    Returns
    -------
    np.ndarray
        ``(n,)`` -- every row of the same group replaced by that group's collapsed statistic.
    """
    if stat not in ("mean", "quantile"):
        raise ValueError(f"collapse_predictions_by_group: unsupported stat {stat!r}, expected 'mean' or 'quantile'")

    df = pd.DataFrame({"prediction": np.asarray(predictions, dtype=np.float64), "group": group})
    grouped = df.groupby("group", sort=False)["prediction"]
    if stat == "mean":
        group_stat = grouped.transform("mean")
    else:
        # transform(lambda s: s.quantile(q)) invokes one Python-level callback PER GROUP (each paying its
        # own DataFrame/Index construction overhead) -- measured at 27s/50000 groups vs 81ms for the "mean"
        # path. pandas' GroupBy.quantile() is a single vectorized, C-level pass computing every group's
        # quantile at once; map the (small) per-group result back onto every row instead.
        per_group_quantile = grouped.quantile(quantile)
        group_stat = df["group"].map(per_group_quantile)

    return np.asarray(group_stat.to_numpy())


__all__ = ["collapse_predictions_by_group"]
