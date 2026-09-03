"""Shared helpers for the ``training/composite/`` package: small utilities independently
duplicated across multiple composite modules, consolidated here so a fix can't silently drift
out of sync across copies.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_polars_df(x: Any) -> bool:
    """True iff ``x`` is a polars DataFrame; False (never raises) if polars is absent or ``x`` is any other type.

    Explicit isinstance check, not duck-typing (e.g. ``hasattr(x, "to_pandas")``) -- duck-typing
    mis-detects any object exposing a same-named method (mocks, custom wrappers, sklearn pipeline
    stubs).
    """
    try:
        import polars as pl

        return isinstance(x, pl.DataFrame)
    except ImportError:
        return False


def is_polars_df_logged(x: Any) -> bool:
    """Same contract as :func:`is_polars_df`, but debug-logs the exception when polars is absent or the isinstance check itself fails."""
    try:
        import polars as pl

        return isinstance(x, pl.DataFrame)
    except Exception as exc:
        logger.debug("is_polars_df_logged: polars unavailable or isinstance check failed: %s", exc)
        return False


def sklearn_set_params(self: Any, **params: Any) -> Any:
    """sklearn-compatible bulk attribute setter (used by ``clone`` / grid-search); bind as ``ClassName.set_params = sklearn_set_params``.

    Matches sklearn's own ``BaseEstimator.set_params`` contract: raises ``ValueError`` naming every
    unrecognized parameter instead of silently swallowing a typo'd kwarg via a bare ``setattr`` loop.
    """
    if not params:
        return self
    valid_params = self.get_params(deep=False)
    invalid = sorted(set(params) - set(valid_params))
    if invalid:
        raise ValueError(f"Invalid parameter(s) {invalid} for estimator {self.__class__.__name__}. " f"Valid parameters are: {sorted(valid_params)}.")
    for k, v in params.items():
        setattr(self, k, v)
    return self
