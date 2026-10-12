"""Materialise the feature-base target mix as a plain column, and a fixed-beta offset regressor that reads that column.

The composite machinery reads a base by column name (``training/composite``: ``_extract_base``), and its synthetic-base
recomputation only covers ``mul/add/sub`` of two columns, so an engineered MRMR base is not resolvable by name. The
suite therefore appends the base as a reserved column computed from the raw frame, once at fit and once at predict, and
the regressor reads it from there. ``beta`` is fixed (taken from the held-out-accepted MRMR slope), never refitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MIX_BASE_COLUMN = "__mix_base__"

__all__ = ["MIX_BASE_COLUMN", "with_mix_base_column", "FixedBetaOffsetRegressor"]


def with_mix_base_column(X: Any, provenance: list, *, name: str = MIX_BASE_COLUMN) -> Any:
    """Copy of the pandas frame ``X`` with the first recorded base appended as ``name``; ``X`` itself is not modified.

    ``provenance`` is ``MRMR.fe_target_mix_provenance_``. Returns ``X`` unchanged when no base with a recipe was recorded
    (the caller then keeps the plain model). Polars frames are not supported here: the caller converts first."""
    from ._target_offset_regressor import _replay_base

    entries = [e for e in (provenance or []) if e.get("recipe") is not None]
    if not entries:
        return X
    if not hasattr(X, "assign"):
        raise TypeError("with_mix_base_column expects a pandas DataFrame; convert the polars frame first.")
    base = _replay_base(entries[0]["recipe"], X)
    if not np.all(np.isfinite(base)):
        raise ValueError("the mix base replays to non-finite values on this frame; refusing to fill them silently.")
    return X.assign(**{name: base})


class FixedBetaOffsetRegressor:
    """Regressor fitted on ``y - beta * X[base_column]`` over the other columns, predicting ``f(X) + beta * X[base_column]``.

    The base column is read from the frame at fit and at predict, so it must be present on both; it is removed from the
    inner regressor's input. ``_routes_inner_input`` marks the object for the composite routing, which then hands it the raw
    suite-stage frame. ``beta`` is never refitted.

    Args:
        regressor: The inner regressor, fitted on the frame without the base column.
        base_column: Name of the reserved base column.
        beta: The fixed slope of the offset ``beta * base``; never refitted.
    """

    _routes_inner_input = True

    def __init__(self, regressor: Any, base_column: str, beta: float):
        self.regressor = regressor
        self.base_column = base_column
        self.beta = float(beta)

    def _resolve_base_columns(self) -> list:
        """The base column the routing must find on the frame."""
        return [self.base_column]

    def _split(self, X: Any):
        """The frame without the base column, and the base values; raises when the column is missing or non-finite."""
        if self.base_column not in getattr(X, "columns", ()):
            raise KeyError(f"FixedBetaOffsetRegressor: base column {self.base_column!r} missing from X.")
        base = np.asarray(X[self.base_column].to_numpy())
        if not np.all(np.isfinite(base)):
            raise ValueError(f"FixedBetaOffsetRegressor: base column {self.base_column!r} has non-finite values; refusing to fill them silently.")
        return X.drop(columns=[self.base_column]), base

    def fit(self, X: Any, y: Any) -> "FixedBetaOffsetRegressor":
        """Fit the inner regressor on ``y - beta * base``."""
        Xf, base = self._split(X)
        yy = np.asarray(y).ravel()
        self.regressor_ = self.regressor.fit(Xf, yy - self.beta * base)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """``regressor(X without the base) + beta * base``."""
        Xf, base = self._split(X)
        out: np.ndarray = np.asarray(self.regressor_.predict(Xf)) + self.beta * base
        return out
