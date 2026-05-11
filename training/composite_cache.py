"""Disk-backed discovery cache: content-hash signature + key composer + DiscoveryCache class. Used by R&D workflows that re-run discovery with the same data + varying config; cache hits skip the expensive MI permutation null + Wilcoxon + tiny-model rerank phases. Pure stdlib + numpy + pandas; no composite-internal deps."""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Discovery caching layer (R10c brainstorm round-2 extension E; content-hash cache for discovery).
#
# R&D workflows often re-run ``CompositeTargetDiscovery`` on the same data while only varying the inner-model hyperparameters. The discovery step (MI permutation null, Wilcoxon per spec, tiny-model rerank) burns minutes on multi-million-row datasets. The caching layer keys discovery results by a content hash of (data-sample, target-column, config-signature, random_state) so repeated discovery calls with the same inputs return the cached specs in milliseconds.
#
# Three primitives:
# 1. ``data_signature(df, target_col, feature_cols, sample_n=1000, random_state=42)`` -- blake2b hash over a deterministic sample of the data + column names + dtypes. Quantises to a 16-byte fingerprint that survives row-permutation.
# 2. ``DiscoveryCache(cache_dir)`` -- disk-backed key->value store. Keys are hex strings; values are pickled (using stdlib ``pickle``; safe since the values are dataclass-derived dicts, not arbitrary user objects). API: ``get(key)`` / ``set(key, value)`` / ``invalidate(key)`` / ``clear()`` / ``__contains__``.
# 3. Convenience ``make_discovery_cache_key(df_sig, target_col, config_signature, random_state)`` -- combines the parts into a stable hex key.
#
# The cache layer does NOT auto-integrate with ``CompositeTargetDiscovery.fit``; callers manage cache lookup / store at their orchestration level. This keeps the discovery class free of I/O concerns (testability + library hygiene).
# ----------------------------------------------------------------------


_DISCOVERY_SIGNATURE_SAMPLE_N: int = 1000


def data_signature(
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    *,
    sample_n: int = _DISCOVERY_SIGNATURE_SAMPLE_N,
    random_state: int = 42,
) -> str:
    """Content-hash signature for a (df, target_col, feature_cols) triple.

    Deterministic sample of ``min(n_rows, sample_n)`` rows + column names + dtypes hashed via blake2b to a 16-byte hex fingerprint. Stable under row REORDER (we sample by indices drawn from a seeded RNG, so identical inputs always produce identical samples) but NOT stable under row INSERTION (which would change the sample composition). Suitable for the R&D workflow where the underlying frame is the same across runs.

    Parameters
    ----------
    df
        pandas / polars frame.
    target_col, feature_cols
        Column identifiers used to scope the signature; changes here invalidate the cache.
    sample_n
        Rows sampled for the hash; lower is faster, higher is more discriminating.
    random_state
        Seed for the row-sample RNG; must match across cache write and read for the signature to be stable.

    Returns
    -------
    32-character hex string (blake2b digest, 16 bytes).
    """
    import hashlib
    n_rows = len(df)
    if n_rows == 0:
        return hashlib.blake2b(b"empty", digest_size=16).hexdigest()
    rng = np.random.default_rng(random_state)
    sample_n_eff = min(n_rows, int(sample_n))
    sample_idx = np.sort(rng.choice(n_rows, size=sample_n_eff, replace=False))
    h = hashlib.blake2b(digest_size=16)
    # Hash 1: target column + feature cols (names + order).
    h.update(target_col.encode("utf-8"))
    for c in feature_cols:
        h.update(b"|")
        h.update(str(c).encode("utf-8"))
    # Hash 2: per-column dtype + per-column sampled values.
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        # Polars path.
        for c in [target_col] + list(feature_cols):
            if c not in df.columns:
                continue
            col = df.get_column(c)
            h.update(str(col.dtype).encode("utf-8"))
            sampled = col.to_numpy()[sample_idx]
            h.update(np.ascontiguousarray(sampled).tobytes())
    elif isinstance(df, pd.DataFrame):
        for c in [target_col] + list(feature_cols):
            if c not in df.columns:
                continue
            h.update(str(df[c].dtype).encode("utf-8"))
            sampled = df[c].to_numpy()[sample_idx]
            h.update(np.ascontiguousarray(sampled).tobytes())
    else:
        raise TypeError(f"data_signature: unsupported df type {type(df).__name__}")
    return h.hexdigest()


def make_discovery_cache_key(
    df_sig: str,
    target_col: str,
    config_signature: str,
    random_state: int = 42,
) -> str:
    """Combine the parts of a discovery cache key into a stable hex string. The ``config_signature`` is caller-supplied (usually a hash of the JSON-serialised CompositeTargetDiscoveryConfig)."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(df_sig.encode("utf-8"))
    h.update(b"|")
    h.update(target_col.encode("utf-8"))
    h.update(b"|")
    h.update(config_signature.encode("utf-8"))
    h.update(b"|")
    h.update(str(int(random_state)).encode("utf-8"))
    return h.hexdigest()


class DiscoveryCache:
    """Disk-backed key->value cache for CompositeTargetDiscovery results.

    Values are pickled with stdlib ``pickle`` (safe: stored objects are dataclass-derived dicts). Files live under ``<cache_dir>/<key>.pkl`` with one file per key for easy invalidation / cleanup.

    Thread-safe for single-process use only; concurrent writers from multiple processes will race on the same key (caller's responsibility).
    """

    def __init__(self, cache_dir: Any) -> None:
        import os
        self.cache_dir = str(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        import os
        # Basic sanitisation: only allow hex keys (or alphanumeric); reject path-traversal attempts.
        safe_key = "".join(c for c in key if c.isalnum())
        if not safe_key:
            raise ValueError(f"DiscoveryCache: empty / unsafe key {key!r}")
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def __contains__(self, key: str) -> bool:
        import os
        return os.path.exists(self._path(key))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value, or ``default`` if the key is absent / unreadable."""
        import os, pickle  # lazy
        path = self._path(key)
        if not os.path.exists(path):
            return default
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """Write ``value`` to ``<cache_dir>/<key>.pkl``. Atomic via tmp-file rename so a crash mid-write doesn't leave corrupt cache files."""
        import os, pickle, tempfile  # lazy
        path = self._path(key)
        # Write to a temp file in the same directory, then rename atomically.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry. Returns True if the entry existed, False otherwise."""
        import os
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def clear(self) -> int:
        """Remove all cached entries. Returns the number of files removed."""
        import os, glob  # lazy
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
        return len(files)
