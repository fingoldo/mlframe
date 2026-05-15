"""
Content fingerprint + cache-key dataclasses.

Round-3 user-confirmation: in-memory cache uses CHEAP session-keyed
identity (id() within suite -- safe because we hold a strong ref);
content hashing happens ONLY at disk-cache writes, computed once at
suite start in a background thread.

Two key types:

* :class:`InMemoryKey` -- ``(session_id, df_token, train_idx_token,
  column, params_canonical_hash, provider_signature)``. All within
  one ``train_mlframe_models_suite`` call where df / train_idx are
  immutable; ``id()`` is safe because we hold strong refs throughout.

* :class:`DiskKey` -- ``(content_hash, column, params_canonical_hash,
  provider_signature)``. Used only when ``cache.persistence != "off"``.
  ``content_hash`` is :class:`ContentFingerprint`-derived, computed
  ONCE at suite start.

The :func:`fingerprint_df` builder uses a deterministic linspace
stride sample (round-3 R2-3 fix: ``np.unique`` on ``np.linspace``-rounded
indices avoids duplicates for ``n < 4096``; round-3 R3-07 fix:
explicit early-return for tiny frames hashes the entire frame).
Universal across polars and pandas.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =====================================================================
# Session token
# =====================================================================


@dataclass
class SessionToken:
    """Per-suite-call identity. Constructed at the start of
    ``train_mlframe_models_suite``; lives for the duration of that
    call.

    The point: lookup keys based on ``id(train_df)`` / ``id(train_idx)``
    are safe within one session because we hold strong refs. Across
    sessions (next process / next suite call), ``id()`` may collide
    with a freshly-allocated object -- that's OK because the
    ``session_id`` differs.
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)


_CURRENT_SESSION: Optional[SessionToken] = None


def current_session() -> SessionToken:
    """Get or create the suite-level session token."""
    global _CURRENT_SESSION
    if _CURRENT_SESSION is None:
        _CURRENT_SESSION = SessionToken()
    return _CURRENT_SESSION


def reset_session() -> SessionToken:
    """Force a new session. Called at the start of every
    ``train_mlframe_models_suite`` call so cross-suite cache entries
    don't collide."""
    global _CURRENT_SESSION
    _CURRENT_SESSION = SessionToken()
    return _CURRENT_SESSION


# =====================================================================
# In-memory key
# =====================================================================


@dataclass(frozen=True)
class InMemoryKey:
    """Cheap in-memory cache key.

    Safe within ONE ``train_mlframe_models_suite`` call because:
      * ``session_id`` rotates per suite call;
      * ``df_token`` = ``id(train_df)``; we hold a strong ref so
        Python won't recycle the id;
      * ``train_idx_token`` = ``id(train_idx)`` likewise;
      * ``params_canonical_hash`` is deterministic.

    Cross-session reuse is intentionally NOT supported by this key
    type -- use :class:`DiskKey` for that.
    """
    session_id: str
    df_token: int
    train_idx_token: int
    column: str
    params_canonical_hash: str
    provider_signature: str


# =====================================================================
# Disk key (with content fingerprint)
# =====================================================================


@dataclass(frozen=True)
class ContentFingerprint:
    """Content fingerprint for cross-session disk cache identity.

    Computed once at suite start (background thread when persistence
    is enabled). Components:

    * ``n_rows`` -- total row count of the source df
    * ``n_cols`` -- total column count
    * ``column_dtypes_hash`` -- blake2b of sorted ``(name, dtype)``
      tuples; catches schema changes across runs
    * ``sampled_rows_hash`` -- blake2b of stride-sampled rows
      (default 4096 rows; full hash for tiny frames)

    Less strict than full content hash (single-row edits in unsampled
    positions go undetected) but ~1000x cheaper. Round-3 user feedback:
    "subset рядов; меня устроит менее строгий, но быстрый".
    """
    n_rows: int
    n_cols: int
    column_dtypes_hash: str
    sampled_rows_hash: str

    def short(self) -> str:
        """8-char identifier for cache file paths."""
        return self.sampled_rows_hash[:8]


@dataclass(frozen=True)
class DiskKey:
    """Cross-session disk cache key. Computed once per suite via
    :func:`fingerprint_df`."""
    content: ContentFingerprint
    column: str
    params_canonical_hash: str
    provider_signature: str

    def filename(self) -> str:
        """Returns ``{fingerprint_short}__{column_hash}__{params_hash}.bin``.
        Round-3 S2 fix: column name hashed (not embedded literally)
        to neutralise path-traversal via maliciously-crafted column
        names."""
        col_hash = hashlib.blake2b(self.column.encode("utf-8"), digest_size=8).hexdigest()
        return f"{self.content.short()}__{col_hash}__{self.params_canonical_hash[:8]}.bin"


# =====================================================================
# Hashers
# =====================================================================


def canonical_params_hash(params: Any) -> str:
    """Deterministic hash of a params object / dict.

    Accepts: pydantic ``BaseModel``, dict, or scalar. Converts to a
    JSON-canonical-form string (sort_keys=True) then blake2b to
    16 bytes. Round-3 R3-11: ``json.dumps(d, sort_keys=True)`` is the
    pinned canonical form; nested dicts sort recursively via
    ``sort_keys=True``.
    """
    try:
        from pydantic import BaseModel
        if isinstance(params, BaseModel):
            payload = params.model_dump_json()
            h = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
            return h.hexdigest()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(params, dict):
        payload = json.dumps(params, sort_keys=True, default=str)
    else:
        payload = json.dumps(params, default=str)
    h = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return h.hexdigest()


def fingerprint_df(
    df: Any,
    columns: Optional[Tuple[str, ...]] = None,
    n_sample: int = 4096,
) -> ContentFingerprint:
    """Compute a :class:`ContentFingerprint` for a polars or pandas
    DataFrame, sampling deterministically.

    Round-3 R2-3 fix: ``np.linspace(...).round().astype(int64)`` then
    ``np.unique`` to dedupe so tiny frames don't degenerate.
    """
    n_rows = len(df)

    # Determine columns (sorted for deterministic dtype hash).
    if columns is None:
        try:
            cols = list(df.columns)
        except Exception:
            cols = []
    else:
        cols = list(columns)
    cols_sorted = sorted(cols)
    n_cols = len(cols_sorted)

    # Stride-sample row indices.
    if n_rows == 0:
        sample_idx = np.array([], dtype=np.int64)
    elif n_rows <= max(1, n_sample):
        sample_idx = np.arange(n_rows, dtype=np.int64)
    else:
        raw = np.linspace(0, n_rows - 1, n_sample).round().astype(np.int64)
        sample_idx = np.unique(raw)

    # ---- column_dtypes_hash --------------------------------------
    dtype_lines = []
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            for c in cols_sorted:
                dtype_lines.append(f"{c}:{df.schema[c]}")
    except ImportError:  # pragma: no cover
        pass
    if not dtype_lines:
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                for c in cols_sorted:
                    dtype_lines.append(f"{c}:{df[c].dtype}")
        except ImportError:  # pragma: no cover
            pass
    column_dtypes_hash = hashlib.blake2b(
        ";".join(dtype_lines).encode("utf-8"), digest_size=8
    ).hexdigest()

    # ---- sampled_rows_hash ---------------------------------------
    payload_parts: list = [str(n_rows).encode(), str(n_cols).encode()]
    if len(sample_idx) > 0:
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                # polars accepts numpy int64 indices via gather()
                sub = df.select(cols_sorted)[sample_idx.tolist()]
                # Use Arrow IPC stream of the sample -- stable across
                # polars 1.x. NOT zero-copy but the sample is tiny.
                arrow_table = sub.to_arrow()
                payload_parts.append(arrow_table.to_pandas().to_csv(index=False).encode("utf-8"))
            else:
                raise TypeError("not polars")
        except (ImportError, TypeError):
            try:
                import pandas as pd
                if isinstance(df, pd.DataFrame):
                    sub = df.iloc[sample_idx][cols_sorted]
                    payload_parts.append(sub.to_csv(index=False).encode("utf-8"))
            except ImportError:  # pragma: no cover
                pass

    h = hashlib.blake2b(digest_size=16)
    for p in payload_parts:
        h.update(p)
    sampled_rows_hash = h.hexdigest()

    return ContentFingerprint(
        n_rows=n_rows,
        n_cols=n_cols,
        column_dtypes_hash=column_dtypes_hash,
        sampled_rows_hash=sampled_rows_hash,
    )


__all__ = [
    "SessionToken",
    "current_session",
    "reset_session",
    "InMemoryKey",
    "ContentFingerprint",
    "DiskKey",
    "canonical_params_hash",
    "fingerprint_df",
]
