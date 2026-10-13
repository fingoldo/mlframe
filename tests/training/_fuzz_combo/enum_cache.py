"""On-disk cache for :func:`enumerate_combos`.

Enumeration is deterministic for a given (target, master_seed, exhaustive_pairs) and the code that produces it, but it costs tens of
seconds, and every pytest child of the bug-hunt driver re-runs it at import time. The cache key hashes the source of this package
plus the call parameters, so any change to the enumerator or the axes invalidates the entry and nothing stale is ever replayed.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from .enumerator import enumerate_combos

_CACHE_DIR = Path(__file__).resolve().parent / ".enum_cache"


def _source_digest() -> str:
    """Hash of every ``.py`` file in this package, so a code change invalidates the cache."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def enumerate_combos_cached(target: int, master_seed: int, exhaustive_pairs: bool = False):
    """Return ``enumerate_combos(target, master_seed, exhaustive_pairs=...)``, read from disk when the cache entry matches the current code."""
    key = hashlib.sha256(f"{_source_digest()}|{target}|{master_seed}|{bool(exhaustive_pairs)}".encode()).hexdigest()[:24]
    entry = _CACHE_DIR / f"{key}.pkl"
    if entry.exists():
        try:
            with entry.open("rb") as fh:
                # The cache lives under the repo and is written only by enumerate_combos_cached, so the bytes are trusted local state.
                return pickle.load(fh)  # nosec B301 -- local cache written only by this module, never an external input
        except (OSError, pickle.UnpicklingError, EOFError):
            entry.unlink(missing_ok=True)
    combos = enumerate_combos(target=target, master_seed=master_seed, exhaustive_pairs=exhaustive_pairs)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with entry.open("wb") as fh:
            pickle.dump(combos, fh)
    except OSError:
        pass
    return combos
