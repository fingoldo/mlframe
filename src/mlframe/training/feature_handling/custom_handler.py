"""
Custom user-supplied transform hatch.

Round-3 future-proofing F14 + plan §1.4 ``CustomParams``: lets users
plug a sklearn-shaped transformer into the handler chain without
forking mlframe. Mirrors the existing
``FeatureSelectionConfig.custom_pre_pipelines`` precedent in
:mod:`mlframe.training.configs`.

Phase P: validates the supplied transformer + wraps it so the
assembler treats it like a built-in handler. The phase ships:

  * :class:`CustomHandler` -- the wrapper consumed by the assembler.
  * :func:`validate_custom_transformer` -- structural check that the
    object has callable ``fit`` and ``transform``.

Users write ``CustomParams(transformer=my_pipeline)`` in their
``CustomHandlerSpec`` (text or cat axis), the resolver runs the
custom pipeline, and the output (dense / sparse / embedding)
flows through the standard assembler routing per
``CustomParams.output_kind``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

import numpy as np

from mlframe.training.feature_handling.handlers import CustomParams

logger = logging.getLogger(__name__)


def validate_custom_transformer(transformer: Any) -> None:
    """Check that ``transformer`` has the sklearn fit/transform contract.

    Round-3 U-R2-16: ``CustomParams.transformer: Any`` accepts anything
    at construct time; without an explicit check, a lambda or
    misconfigured object falls over with a confusing AttributeError
    inside ``fit_transform``. Surface the contract violation early.
    """
    if not hasattr(transformer, "fit") or not callable(getattr(transformer, "fit", None)):
        raise TypeError(
            f"CustomParams.transformer must implement .fit(); got "
            f"{type(transformer).__name__} with no callable fit. "
            f"Pass a sklearn TransformerMixin or compatible Pipeline."
        )
    if not hasattr(transformer, "transform") or not callable(getattr(transformer, "transform", None)):
        raise TypeError(f"CustomParams.transformer must implement .transform(); got " f"{type(transformer).__name__} with no callable transform.")


class CustomHandler:
    """Wrapper around a user-supplied sklearn-shaped transformer.

    Construction is the entry point for ``method="custom"`` handler
    specs. The assembler treats the resulting output exactly like a
    built-in handler -- the ``output_kind`` field on
    :class:`CustomParams` determines whether it lands in the sparse
    block, the dense block, or the model-native embedding-features
    slot.

    The wrapper preserves the user's transformer's state across
    pickling / sklearn.clone so the cache layer can reuse fitted
    instances.
    """

    def __init__(
        self,
        column: str,
        params: CustomParams,
        group_columns: Optional[List[str]] = None,
    ):
        validate_custom_transformer(params.transformer)
        self.column = column
        self.params = params
        self.group_columns = list(group_columns) if group_columns else None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def output_kind(self) -> Literal["dense", "sparse", "embedding"]:
        return self.params.output_kind

    def signature(self) -> str:
        """Stable string for cache keys. Conservative: includes the
        type name + repr() of the transformer. Users with stateful
        transformers that have unstable repr() should override per
        their workflow."""
        return f"custom:{self.column}:{type(self.params.transformer).__name__}:{self.params.output_kind}"

    def fit(self, df: Any, y: Optional[Any] = None) -> CustomHandler:
        column_data = self._extract_column(df)
        if y is not None:
            try:
                self.params.transformer.fit(column_data, y)
            except TypeError as exc:
                # Some sklearn transformers expose ``fit(self, X)`` only; passing y trips a TypeError on the
                # missing positional. Log so the operator can spot misconfigured "supervised" transformers
                # that get silently demoted to unsupervised here.
                logger.warning(
                    "CustomHandler(%r): transformer.fit(X, y) raised TypeError (%s); retrying with fit(X) only.",
                    self.column, exc,
                )
                self.params.transformer.fit(column_data)
        else:
            self.params.transformer.fit(column_data)
        self._fitted = True
        return self

    def transform(self, df: Any) -> Any:
        if not self._fitted:
            # Wave 37 P1 fix (2026-05-20): NotFittedError per sklearn.
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(f"CustomHandler({self.column!r}) not fitted -- call .fit(train_df) first")
        column_data = self._extract_column(df)
        return self.params.transformer.transform(column_data)

    def fit_transform(self, df: Any, y: Optional[Any] = None) -> Any:
        column_data = self._extract_column(df)
        transformer = self.params.transformer
        if not hasattr(transformer, "fit_transform") or not callable(getattr(transformer, "fit_transform", None)):
            # No ``fit_transform`` on the transformer -- structural fallback. Catching AttributeError
            # at call site would also swallow AttributeErrors raised INSIDE a buggy ``fit_transform``,
            # masking the real failure.
            self.fit(df, y)
            out = self.transform(df)
            self._fitted = True
            return out
        try:
            out = transformer.fit_transform(column_data, y) if y is not None else transformer.fit_transform(column_data)
        except TypeError as exc:
            # Narrow the retry to the y-signature mismatch class. Pre-fix any TypeError raised
            # deep inside ``fit_transform`` (shape mismatch, dtype incompatibility, etc.) silently
            # demoted to fit-only and re-ran the transformer twice, hiding the original error.
            msg = str(exc).lower()
            if y is not None and ("argument" in msg or "positional" in msg or "keyword" in msg) and ("y" in msg or "2" in msg):
                logger.warning(
                    "CustomHandler(%r): fit_transform(X, y) raised TypeError (%s); retrying with fit + transform.",
                    self.column, exc,
                )
                self.fit(df, y)
                out = self.transform(df)
            else:
                raise
        self._fitted = True
        return out

    def _extract_column(self, df: Any) -> Any:
        """Pull the column from the dataframe in a backend-agnostic
        way and reshape to ``(n, 1)`` so sklearn transformers that
        require 2-D input (StandardScaler, OneHotEncoder, etc.) work
        without per-handler reshape boilerplate. The downside: a
        transformer that wants 1-D will see (n, 1) -- they typically
        reshape via ``X.ravel()`` themselves."""
        col_array: np.ndarray
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                col_array = df[self.column].to_numpy()
            else:
                raise TypeError("not polars")
        except (ImportError, TypeError):
            try:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    col_array = df[self.column].to_numpy()
                else:
                    raise TypeError("not pandas")
            except (ImportError, TypeError):
                col_array = np.asarray(df)
        # Normalise to 2-D for sklearn compatibility.
        if col_array.ndim == 1:
            col_array = col_array.reshape(-1, 1)
        return col_array

    def __repr__(self) -> str:
        return (
            f"CustomHandler(column={self.column!r}, "
            f"transformer={type(self.params.transformer).__name__}, "
            f"output_kind={self.params.output_kind!r}, "
            f"fitted={self._fitted})"
        )


__all__ = ["CustomHandler", "validate_custom_transformer"]
