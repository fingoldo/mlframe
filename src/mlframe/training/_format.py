"""Small utilities for consistent metric / model-name formatting across
the suite. Centralised so log lines, chart titles, and ensemble headers
all use the same shape.

Two responsibilities:

1. ``format_metric(value, ndigits=2)`` -- adaptive decimal formatter.
   Default 2 d.p. for values >= 1; widens automatically for values < 1
   to preserve ~2 significant digits. Avoids ``f"{11497.4655:.4f}"``
   showing useless precision while still rendering ``f"{0.0034:.4f}"``
   readably (it would otherwise become ``"0.0034"`` -- correct -- vs
   the 2-d.p. default showing ``"0.00"`` which loses the value).

2. ``strip_shim_suffix(name)`` -- drops the ``WithDMatrixReuse`` /
   ``WithDatasetReuse`` suffixes from XGB / LGBM shim class names so
   log lines, ensemble labels and chart titles read
   ``XGBRegressor`` / ``LGBMRegressor`` instead of the implementation
   detail.

3. ``short_model_name(name_or_cls)`` -- compose of the two above plus
   the existing ``CatBoost*``/``XGB*``/``LGBM*`` collapsing used by
   :func:`mlframe.training.core._short_model_tag`. The single entry
   point any new code should call when it needs a short, user-facing
   model name.
"""
from __future__ import annotations

import math
from typing import Any

_SHIM_SUFFIXES = ("WithDMatrixReuse", "WithDatasetReuse")


def strip_shim_suffix(name: str) -> str:
    """Drop the internal-shim suffix from a model class name.

    >>> strip_shim_suffix("XGBRegressorWithDMatrixReuse")
    'XGBRegressor'
    >>> strip_shim_suffix("LGBMClassifierWithDatasetReuse")
    'LGBMClassifier'
    >>> strip_shim_suffix("CatBoostRegressor")
    'CatBoostRegressor'
    """
    if not isinstance(name, str):
        return name
    for suffix in _SHIM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def short_model_tag(name_or_obj: Any) -> str:
    """Map a model name (or instance / class) to the short tag used in
    ensemble headers: ``cb`` / ``xgb`` / ``lgb`` / ``hgb``. Falls back
    to ``strip_shim_suffix(class_name)`` for everything else (so
    ``LinearRegression`` stays as ``LinearRegression``, not collapsed
    further).
    """
    if isinstance(name_or_obj, str):
        cls_name = strip_shim_suffix(name_or_obj)
    else:
        # Object: try .__class__.__name__; fall back to repr.
        cls_name = strip_shim_suffix(type(name_or_obj).__name__)
    if cls_name.startswith("CatBoost"):
        return "cb"
    if cls_name.startswith("XGB"):
        return "xgb"
    if cls_name.startswith("LGBM"):
        return "lgb"
    if cls_name.startswith("HistGradient"):
        return "hgb"
    return cls_name


def format_metric(value: Any, ndigits: int = 2) -> str:
    """Adaptive decimal formatter for metric scalars.

    Contract:
    - non-finite (NaN / inf) values are stringified directly.
    - ``abs(value) >= 1`` or ``value == 0`` -> ``f"{value:.{ndigits}f}"``.
    - ``0 < abs(value) < 1`` -> widen by the count of leading zeros
      after the decimal point so the formatted string carries at least
      ``ndigits`` significant figures.

    Examples (ndigits=2, the default):
    - 11497.4655      -> "11497.47"
    - 13.93           -> "13.93"
    - 0.4655          -> "0.47"           (no widening; 2 sig figs)
    - 0.0034          -> "0.0034"         (widened by 2 leading zeros)
    - 1.23e-05        -> "0.0000123"      (widened by 4 leading zeros)
    - 0.0             -> "0.00"
    - float('nan')    -> "nan"

    For ``ndigits=3`` you get one extra decimal in every case
    (11497.466 / 13.930 / 0.466 / 0.00340 / ...).
    """
    if value is None:
        return "None"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(v)
    abs_v = abs(v)
    if abs_v >= 1 or abs_v == 0:
        return f"{v:.{ndigits}f}"
    # 0 < abs_v < 1 -- count leading zeros after the decimal point.
    # log10(0.5) = -0.30 -> floor = -1 -> 0 leading zeros (just ".5")
    # log10(0.05) = -1.30 -> floor = -2 -> 1 leading zero  ("0.05")
    # log10(0.005) = -2.30 -> floor = -3 -> 2 leading zeros ("0.005")
    try:
        leading_zeros = max(0, -int(math.floor(math.log10(abs_v))) - 1)
    except (ValueError, OverflowError):  # pragma: no cover
        leading_zeros = 0
    # F4 fix (2026-05-11): cap the decimal-widening at 4 leading zeros (= 6 d.p. for ndigits=2). Beyond that switch to scientific notation -- ``MTRESID=0.000000029`` (8 d.p., 11 chars of useless precision) becomes ``MTRESID=2.9e-08`` (7 chars, immediately readable as ~0).
    if leading_zeros > 4:
        return f"{v:.{ndigits}e}"
    return f"{v:.{ndigits + leading_zeros}f}"


__all__ = ["format_metric", "strip_shim_suffix", "short_model_tag"]
