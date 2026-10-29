"""Detector of device-to-host transfers that move the same content more than once.

``residency_audit`` counts bytes, which says how much crossed the bus but not whether the same column crossed twice. Two readers of one device array that each take their own
copy (one dense, one narrowed, or two stages replaying the same recipe) show up there only as a larger total. This audit hashes the host array of every transfer at or above
``MIN_BYTES`` and groups transfers of equal content, so each redundant copy appears with the call sites that made it.

Content is compared after a normalising cast (floats as float32, integers and booleans as int64), so a column copied once in double and once in single precision, or as int8 and as
int64 codes, counts as one array copied twice. Two double precision arrays that differ only below single precision are therefore reported as equal; the audit is a finder of
candidates, not a proof.

Like ``residency_audit`` it monkeypatches ``cp.asnumpy`` and ``cupy.ndarray.get`` for the duration of the block, serialised by the same lock, and is meant for tests and profiling.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from ._audit import _AUDIT_LOCK

logger = logging.getLogger(__name__)

# Transfers below this many bytes are scalars and small tables, not columns.
MIN_BYTES = 2048
# Call-site frames kept per transfer, innermost first.
_SITE_DEPTH = 3
_SKIP_FILES = ("_duplicate_transfers.py", "_audit.py")
_NESTED = threading.local()  # set while ``cp.asnumpy`` runs, because it calls ``ndarray.get`` itself and the pair must be tallied once


@dataclass(frozen=True)
class Transfer:
    """One device-to-host copy: its content key, size, host dtype and shape, and the mlframe call sites that made it."""

    key: str
    nbytes: int
    dtype: str
    shape: tuple
    site: str


@dataclass
class DuplicateTransferReport:
    """The transfers seen inside a ``duplicate_transfer_audit`` block, with the groups that repeat the same content."""

    transfers: list[Transfer] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        """Bytes of every recorded transfer."""
        return sum(t.nbytes for t in self.transfers)

    def groups(self) -> list[list[Transfer]]:
        """Groups of transfers with equal content (two or more), largest redundant volume first."""
        by_key: dict[str, list[Transfer]] = {}
        for t in self.transfers:
            by_key.setdefault(t.key, []).append(t)
        found = [g for g in by_key.values() if len(g) > 1]
        found.sort(key=_redundant_bytes, reverse=True)
        return found

    @property
    def redundant_bytes(self) -> int:
        """Bytes beyond the cheapest copy of each repeated content: what one shared copy per array would have saved."""
        return sum(_redundant_bytes(g) for g in self.groups())

    def summary(self, limit: int = 10) -> str:
        """Totals and the ``limit`` largest repeated groups, one line per copy with its call sites."""
        lines = [f"D2H >= {MIN_BYTES} B: {self.total_bytes} B in {len(self.transfers)} transfers; redundant {self.redundant_bytes} B"]
        for g in self.groups()[:limit]:
            lines.append(f"  {len(g)} copies of one content, {_redundant_bytes(g)} B redundant")
            lines.extend(f"    {t.nbytes} B {t.dtype} {t.shape} {t.site}" for t in g)
        return "\n".join(lines)


def _redundant_bytes(group: list[Transfer]) -> int:
    """Bytes of ``group`` beyond its cheapest member."""
    return sum(t.nbytes for t in group) - min(t.nbytes for t in group)


def _content_key(arr: np.ndarray) -> str:
    """Hash of ``arr`` after the normalising cast, so a dense and a narrowed copy of one array share a key."""
    kind = arr.dtype.kind
    if kind == "f":
        norm = arr.astype(np.float32, copy=False)
    elif kind in "iub":
        norm = arr.astype(np.int64, copy=False)
    else:
        norm = arr
    return hashlib.blake2b(np.ascontiguousarray(norm).tobytes(), digest_size=16).hexdigest()


def _call_site() -> str:
    """The innermost ``_SITE_DEPTH`` mlframe frames of the caller as ``file:line(function)``, joined with ``<``."""
    frame: "types.FrameType | None" = sys._getframe(2)
    parts: list[str] = []
    while frame is not None and len(parts) < _SITE_DEPTH:
        path = frame.f_code.co_filename
        if "mlframe" in path and not path.endswith(_SKIP_FILES):
            parts.append(f"{os.path.basename(path)}:{frame.f_lineno}({frame.f_code.co_name})")
        frame = frame.f_back
    return " < ".join(parts)


@contextlib.contextmanager
def duplicate_transfer_audit(min_bytes: int = MIN_BYTES) -> Iterator[DuplicateTransferReport]:
    """Yield a :class:`DuplicateTransferReport` filled with the device-to-host copies of at least ``min_bytes`` made inside the block.

    No-op (an empty report) when cupy is unavailable. Not for the production path: every transfer is hashed on the host.
    """
    rep = DuplicateTransferReport()
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("duplicate_transfer_audit: cupy unavailable, yielding an empty report: %s", e)
        yield rep
        return

    orig_asnumpy = cp.asnumpy
    orig_get = cp.ndarray.get

    def _note(host: Any) -> None:
        """Record the host array of one finished transfer when it is large enough."""
        try:
            arr = np.asarray(host)
            if arr.nbytes >= min_bytes:
                rep.transfers.append(Transfer(_content_key(arr), int(arr.nbytes), str(arr.dtype), tuple(arr.shape), _call_site()))
        except Exception as e:  # nosec B110 - best-effort recording
            logger.debug("duplicate_transfer_audit: recording a transfer failed: %s", e)

    def _asnumpy(obj, *a, **k):
        """Monkeypatched ``cp.asnumpy``: delegates unchanged, then records the host array."""
        outer = getattr(_NESTED, "active", False)
        _NESTED.active = True
        try:
            out = orig_asnumpy(obj, *a, **k)
        finally:
            _NESTED.active = outer
        if not outer:
            _note(out)
        return out

    def _get(self, *a, **k):
        """Monkeypatched ``cupy.ndarray.get``: delegates unchanged, then records the host array (the ``out`` buffer when one is passed)."""
        outer = getattr(_NESTED, "active", False)
        _NESTED.active = True
        try:
            out = orig_get(self, *a, **k)
        finally:
            _NESTED.active = outer
        if not outer:
            target = out if out is not None else k.get("out", a[1] if len(a) > 1 else None)
            if target is not None:
                _note(target)
        return out

    with _AUDIT_LOCK:
        cp.asnumpy = _asnumpy
        try:
            cp.ndarray.get = _get  # may be read-only on some cupy builds; guarded
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("duplicate_transfer_audit: patching cupy.ndarray.get failed, .get() transfers won't be tracked: %s", e)
        try:
            yield rep
        finally:
            cp.asnumpy = orig_asnumpy
            try:
                cp.ndarray.get = orig_get
            except Exception as e:  # nosec B110 - best-effort path
                logger.debug("duplicate_transfer_audit: restoring the original cupy.ndarray.get failed: %s", e)


__all__ = ["MIN_BYTES", "DuplicateTransferReport", "Transfer", "duplicate_transfer_audit"]
