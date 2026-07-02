"""Causal-lag column detection + honest AR floor for group-sequential targets.

A sequential target y_t (a well log sampled by measured-depth MD, a time series with no explicit timestamp, a per-entity
panel ordered by an index) often carries a strictly-causal lag feature ``{target}_prev`` in the frame: at predict time
y_{t-1} is already observed, so the feature is a legitimate predictor, NOT leakage. The dummy-baseline layer already
deploys it as the ``lag_predict`` failsafe (``y_hat[i] = X[lag_col][i]``), and on strong-AR targets that failsafe is the
production floor every trained model must beat.

This module is the single source of truth for (a) which column is that causal lag and (b) the RMSE the AR failsafe would
achieve on an arbitrary row set. The composite selector uses it to floor every spec against ``min(raw, lag)`` -- a spec
whose honest reconstruction cannot beat the lag it would be deployed alongside is useless (prod incident: an ensemble
that missed lag_predict landed at RMSE 13.30 vs the lag_predict 11.58 floor).

The provenance is deliberately NAME-based, mirroring the ``_dummy_baseline_regression`` probe: only a feature literally
named ``{target}<suffix>`` is trusted as the strictly-causal lag. A contemporaneous near-copy of y (a column that
happens to correlate ~1 with y but is NOT a lag) will not carry this name, so the near-copy gate still protects against
leakage -- the exemption is granted by provenance, never by a marginal correlation match.
"""
from __future__ import annotations

import numpy as np

# Same suffix set as the dummy-baseline lag_predict probe (``_dummy_baseline_regression.py``); keep in sync. A feature
# named ``{target}{suffix}`` is treated as the strictly-causal lag(y) of the sequential target.
CAUSAL_LAG_SUFFIXES = ("_prev", "_lag_1", "_lag1", "_lag")

# Marker minted by the grouped-causal base engineer (``_grouped_causal_bases.py``) into every strictly-causal per-group
# base name (``{target}__gcausal_lag{k}`` / ``__gcausal_tmean{w}`` / ``__gcausal_expmean``). Its presence is unambiguous
# provenance that the column is a within-group causal construct we built, never a contemporaneous near-copy of y.
CAUSAL_BASE_MARKER = "__gcausal_"


def is_causal_base_name(name: str, target_col: str | None = None) -> bool:
    """True when ``name`` is a strictly-causal base by PROVENANCE: a grouped-causal engineered base (carries
    ``CAUSAL_BASE_MARKER``) or a named causal lag ``{target_col}{suffix}``. Provenance only -- a contemporaneous
    near-copy of y that is not a causal construct is never matched, so the near-copy leakage guard still protects it.

    The lag-suffix match requires ``target_col`` (an exact ``{target}{suffix}`` name); with ``target_col=None`` only the
    unambiguous engineered marker matches, so a stray feature merely ending in ``_prev`` is not silently exempted.
    """
    if not name:
        return False
    if CAUSAL_BASE_MARKER in name:
        return True
    if target_col:
        for suffix in CAUSAL_LAG_SUFFIXES:
            if name == f"{target_col}{suffix}":
                return True
    return False


def detect_causal_lag_column(df, target_col: str) -> str | None:
    """Return the causal-lag feature name for ``target_col`` present in ``df`` (first matching suffix), else ``None``.

    Works for both pandas (``.columns``) and polars (``.columns`` / ``.schema``) carriers; a narrow name lookup only, no
    data is materialised.
    """
    if not target_col:
        return None
    cols: set | None = None
    columns = getattr(df, "columns", None)
    if columns is not None:
        try:
            cols = set(columns)
        except TypeError:  # pragma: no cover -- exotic carrier
            cols = None
    if cols is None:
        schema = getattr(df, "schema", None)
        if schema is not None:
            try:
                cols = set(schema)
            except TypeError:  # pragma: no cover
                cols = None
    if cols is None:
        return None
    for suffix in CAUSAL_LAG_SUFFIXES:
        name = f"{target_col}{suffix}"
        if name in cols:
            return name
    return None


def causal_lag_predict_rmse(lag_values: np.ndarray, y_true: np.ndarray) -> float:
    """RMSE of the AR failsafe ``y_hat = lag_values`` vs ``y_true`` over the finite-on-both rows.

    Returns ``nan`` when fewer than 50 finite rows remain (a degenerate MEASUREMENT that must not be used as a floor).
    """
    lag = np.asarray(lag_values, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if lag.shape != y.shape:
        return float("nan")
    finite = np.isfinite(lag) & np.isfinite(y)
    if int(finite.sum()) < 50:
        return float("nan")
    d = y[finite] - lag[finite]
    rmse = float(np.sqrt(np.mean(d * d)))
    return rmse if np.isfinite(rmse) else float("nan")
