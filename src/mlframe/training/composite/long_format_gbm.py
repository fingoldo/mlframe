"""``melt_to_long_gbm_features``: reshape wide features to long format to suppress spurious tree interactions.

Source: Santander Customer Transaction Prediction 4th place -- "I put all vars together in one column
(200000*200, 1), add counts as second column, name of features as 3rd categorical column and used LightGBM
... With long model it became much harder to find inter-vars interactions, boosting mainly used only
feature + count pairs." Useful when features are known/suspected to be mutually independent (verifiable via
mlframe's DCD redundancy-detection stats -- low pairwise SU across the feature set) and a tree model trained
on the WIDE table is hallucinating fake cross-feature interactions from noise. Melting to long format
(one row per original (row, feature) pair, with the feature's own value, its within-column frequency count,
and a feature-identity code) removes cross-feature adjacency from the tree's split search entirely -- it can
only ever combine a value with its own count or its own feature identity, never with another feature's value.

Leakage discipline: mirrors :func:`row_level_average.compute_row_level_then_average_predictions` -- the
per-original-row target is broadcast to every long-format row derived from it, and
:func:`composite_oof_predictions`'s ``groups`` support (``GroupKFold`` on the original row id) guarantees one
original row's long-format entries never split across CV folds (they'd all carry the identical label anyway).

Honest empirical note: this mechanism is implemented and leak-safe, but a measurable predictive WIN over a
plain wide-table model could NOT be reproduced in synthetic testing across three attempted configurations
(a purely additive continuous target -- catastrophically worse, MSE +4,200%; a binary target with a handful
of informative features among many noise columns at n=1,200/d=100 -- worse, AUC 0.644 vs. 0.721; the same
shape pushed to n=3,000/d=300 with deeper wide-model trees -- also worse). Diagnosis: a single long-format
row only ever sees ONE feature's value, so the model regresses the FULL row-level target off a single
feature's worth of signal every time -- for an additive-decomposition target this is a severe SNR loss
(each feature explains ~1/d of the variance, so summing d independently-noisy full-target estimates
massively overshoots); for classification the same-column-value evidence is real but the (value, count,
feature-code) representation needs very large n (the source competition used 200,000 rows) for the model to
learn each feature-code's own value->target mapping well, which no synthetic at unit-test-friendly sizes
reproduces. Kept as a correctly-implemented, leak-safe utility (unit-tested for correctness, not biz_value)
rather than deleted -- large-n production data may still see the source technique's benefit; measure before
relying on it.

Opt-in ``context_columns`` fix for the additive-target SNR loss: the diagnosis above is that a pure
long-format row is blind to every OTHER feature of its original row, so an additively-decomposed target
can never be recovered from a single feature's value. Passing ``context_columns`` broadcasts a caller-
chosen set of companion feature values onto every long-format row (in addition to that row's own melted
value/count/feature-code), giving the per-row model enough context to reconstruct the additive sum instead
of regressing the full target off 1/d of the signal. This is strictly opt-in -- omitting the parameter
(default ``None``) reproduces the exact pre-existing long table and is bit-identical to the old behavior.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import polars as pl

from .ensemble.feature_stacking import composite_oof_predictions

logger = logging.getLogger(__name__)

_DEFAULT_AGG_STATS: tuple[str, ...] = ("mean", "max", "min", "std")


def melt_to_long_gbm_features(
    X: pd.DataFrame,
    y: np.ndarray,
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    random_state: int = 42,
    agg_stats: Sequence[str] = _DEFAULT_AGG_STATS,
    column_prefix: str = "long_gbm",
    context_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Reshape ``X`` to long format (value, count, feature-identity), fit one small OOF-safe model on the
    long table with the row's target broadcast to every one of its features, then aggregate the per-
    (row, feature) predictions back to per-row meta-features.

    Parameters
    ----------
    X
        ``(n, d)`` pandas feature frame (numeric columns).
    y
        ``(n,)`` target.
    model_factory
        Zero-arg callable returning a fresh unfitted row-level (long-format-row-level) estimator. Should be
        a REGRESSOR (continuous ``.predict()`` output is what gets aggregated -- see
        ``compute_row_level_then_average_predictions``'s docstring for why a hard-label classifier loses
        most of the averaging signal).
    n_splits, random_state
        Passed to the underlying group-aware OOF CV (grouped by original row id).
    agg_stats
        Which per-row aggregate stats to compute over that row's ``d`` long-format predictions: any of
        pandas' groupby-agg names (``"mean"``, ``"max"``, ``"min"``, ``"std"``, ``"median"``, ...).
    column_prefix
        Column-name prefix for the output aggregate features.
    context_columns
        Opt-in, default ``None`` (pure long format, bit-identical to omitting this parameter). A small set
        of ``X`` column names whose per-row values get broadcast onto every long-format row derived from
        that row, alongside the row's own melted value/count/feature-code -- gives the per-(row, feature)
        model limited cross-feature context, fixing the additive-target SNR loss documented above. Keep
        this small (a handful of companion features, not the whole frame) -- it re-introduces exactly the
        cross-feature adjacency the pure long format was designed to remove, just in a caller-controlled,
        bounded way.

    Returns
    -------
    pl.DataFrame
        ``len(agg_stats)`` columns, one row per original row of ``X``, named ``{column_prefix}_{stat}``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("melt_to_long_gbm_features: X must be a pandas DataFrame.")
    n = X.shape[0]
    y_arr = np.asarray(y, dtype=np.float64)

    X_indexed = X.reset_index(drop=True).copy()
    X_indexed["_row_id"] = np.arange(n)
    melted = X_indexed.melt(id_vars="_row_id", var_name="_feature_name", value_name="_value")
    # Within-column value frequency count (the source technique's "count" column) -- vectorized groupby-
    # transform over (feature, value) pairs, one pass, no per-column Python loop.
    melted["_count"] = melted.groupby(["_feature_name", "_value"])["_value"].transform("count").astype(np.float64)
    melted["_feat_code"] = pd.factorize(melted["_feature_name"])[0].astype(np.float64)

    row_ids = melted["_row_id"].to_numpy()
    base_cols = ["_value", "_count", "_feat_code"]
    if context_columns:
        missing = set(context_columns) - set(X.columns)
        if missing:
            raise ValueError(f"melt_to_long_gbm_features: context_columns not found in X: {sorted(missing)}")
        # Broadcast each context column's per-row value onto every long-format row derived from that row --
        # a plain positional take by _row_id (row ids are 0..n-1, matching X_indexed's reset index).
        context_block = X_indexed[list(context_columns)].to_numpy(dtype=np.float64)[row_ids]
        context_df = pd.DataFrame(context_block, columns=[f"_ctx_{c}" for c in context_columns], index=melted.index)
        X_long = pd.concat([melted[base_cols].astype(np.float64), context_df], axis=1)
    else:
        X_long = melted[base_cols].astype(np.float64)
    y_broadcast = y_arr[row_ids]

    oof_pred = composite_oof_predictions(model_factory, X_long, y_broadcast, n_splits=n_splits, random_state=random_state, groups=row_ids)

    pred_df = pd.DataFrame({"_row_id": row_ids, "_pred": oof_pred})
    agg = pred_df.groupby("_row_id")["_pred"].agg(list(agg_stats)).reindex(range(n))

    return pl.DataFrame({f"{column_prefix}_{stat}": agg[stat].to_numpy(dtype=np.float64) for stat in agg_stats})


__all__ = ["melt_to_long_gbm_features"]
