"""Residency audit harness for the GPU-strict resident FE path.

cupy exposes a malloc/free MemoryHook but NOT a memcpy hook, so this counts host<->device transfers by
monkeypatching the Python-level transfer entry points (``cp.asarray`` for H2D, ``cupy.ndarray.get`` /
``cp.asnumpy`` for D2H) and classifying each by BYTE SIZE. The contract is audited by size, not by call count:
the branchy selection legitimately pulls O(rounds + stages) tiny SCALAR values, so a `.get()`-count assertion
would false-fail; what must be zero is BULK transfer (arrays whose size scales with n_sub or n_candidates)."""
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Iterator

logger = logging.getLogger(__name__)

# bytes at/above which a transfer is "bulk" (a scalar / tiny index is far below; one operand column at
# n_sub=30k f32 is 120KB, so 8KB cleanly separates scalar decisions from bulk data).
BULK_BYTES = 8192

# residency_audit monkeypatches
# process-wide cp.asarray/cp.asnumpy/cp.ndarray.get with no reentrancy guard. Two overlapping
# residency_audit() regions on different threads would have the second region's "_orig_*" capture be
# the first region's wrapper (not the true original), and whichever region exits first restores to a
# stale value - silently corrupting the surviving region's byte tally with no error. Serialize entry
# so only one region's monkeypatch is ever installed at a time.
_AUDIT_LOCK = threading.RLock()  # RLock: a same-thread nested residency_audit() must not deadlock
_TAG = threading.local()
_NESTED = threading.local()  # set while cp.asnumpy runs, because it calls ndarray.get itself and must not be tallied twice
_EXEMPT_TAG = "__exempt__"


@contextlib.contextmanager
def audit_tag(name: str) -> Iterator[None]:
    """Transfers made inside this block (on this thread) are tallied under ``name`` in ``ResidencyReport.tagged``, not as bulk traffic.

    For transfers the product makes on purpose and documents, so the resident-FE contract can still be asserted on everything else: the float
    candidate block copied to host for the downstream survivor reads is one, the benchmark sweeps of :func:`audit_exempt` another.
    """
    prev = getattr(_TAG, "name", None)
    _TAG.name = name
    try:
        yield
    finally:
        _TAG.name = prev


def audit_exempt() -> contextlib.AbstractContextManager:
    """Transfers made inside this block are not tallied at all.

    For code whose job is to MEASURE transfers, such as the kernel-tuning sweeps that time a numpy variant against a cupy one and pay the
    round trip on purpose: that traffic is benchmark input, not the production data path the residency contract is about, and which
    fit it lands in depends on whether the tuning cache already holds a region for the shape.
    """
    return audit_tag(_EXEMPT_TAG)


def _record(rep: "ResidencyReport", direction: list, nbytes: int) -> None:
    """File ``nbytes`` under the active tag when there is one (dropped for the exempt tag), else in ``direction``."""
    tag = getattr(_TAG, "name", None)
    if tag is None:
        direction.append(nbytes)
    elif tag != _EXEMPT_TAG:
        rep.tagged.setdefault(tag, []).append(nbytes)


class ResidencyReport:
    """Tally of host<->device transfer byte sizes recorded by :func:`residency_audit`, split into bulk
    (``>= BULK_BYTES``) vs scalar transfers so the resident-FE contract can be asserted on bulk volume."""

    def __init__(self):
        self.h2d = []  # list of byte sizes
        self.d2h = []
        # Implicit scalar syncs: ``float(dev)`` / ``int(dev)`` / ``bool(dev)`` / ``dev.item()`` read one value back and BLOCK the stream, but never show up as
        # an explicit ``.get()``. One entry per call, the value is the element size in bytes.
        self.scalar_syncs: list = []
        self.tagged: dict = {}  # tag -> byte sizes of transfers filed under :func:`audit_tag`

    @property
    def bulk_h2d(self):
        """H2D transfer byte sizes that are >= :data:`BULK_BYTES` - the ones the resident-FE contract forbids."""
        return [b for b in self.h2d if b >= BULK_BYTES]

    @property
    def bulk_d2h(self):
        """D2H transfer byte sizes that are >= :data:`BULK_BYTES` - the ones the resident-FE contract forbids."""
        return [b for b in self.d2h if b >= BULK_BYTES]

    @property
    def scalar_d2h_bytes(self):
        """Total bytes of D2H transfers below :data:`BULK_BYTES` - the tolerated scalar/branch-decision traffic."""
        return sum(b for b in self.d2h if b < BULK_BYTES)

    def summary(self) -> str:
        """One-line ``"H2D: N ops (B bulk, T B); D2H: ..."`` string for logging / assertion messages."""
        return (f"H2D: {len(self.h2d)} ops ({len(self.bulk_h2d)} bulk, {sum(self.h2d)} B); "
                f"D2H: {len(self.d2h)} ops ({len(self.bulk_d2h)} bulk, {self.scalar_d2h_bytes} B scalar); "
                f"implicit scalar syncs: {len(self.scalar_syncs)}")


@contextlib.contextmanager
def residency_audit() -> Iterator[ResidencyReport]:
    """Context manager yielding a :class:`ResidencyReport`. Records H2D bytes (``cp.asarray`` of a host array)
    and D2H bytes (``ndarray.get`` / ``cp.asnumpy``) for the enclosed region. No-op (empty report) when cupy is
    unavailable. Intended for tests / profiling, not the production path."""
    rep = ResidencyReport()
    try:
        import cupy as cp
        import numpy as np
    except ImportError as e:
        logger.debug("residency_audit: cupy unavailable, yielding an empty report: %s", e)
        yield rep
        return

    _orig_asarray = cp.asarray
    _orig_asnumpy = cp.asnumpy
    _orig_get = cp.ndarray.get

    def _asarray(obj, *a, **k):
        """Monkeypatched ``cp.asarray``: records host-array byte size as an H2D transfer, then delegates unchanged."""
        try:
            if isinstance(obj, np.ndarray):
                _record(rep, rep.h2d, int(obj.nbytes))
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("residency_audit: recording an H2D transfer failed: %s", e)
        return _orig_asarray(obj, *a, **k)

    def _asnumpy(obj, *a, **k):
        """Monkeypatched ``cp.asnumpy``: records the source array's byte size as a D2H transfer, then delegates unchanged."""
        try:
            nb = int(getattr(obj, "nbytes", 0))
            if nb:
                _record(rep, rep.d2h, nb)
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("residency_audit: recording a D2H transfer (asnumpy) failed: %s", e)
        outer = getattr(_NESTED, "in_asnumpy", False)
        _NESTED.in_asnumpy = True
        try:
            return _orig_asnumpy(obj, *a, **k)
        finally:
            _NESTED.in_asnumpy = outer

    def _get(self, *a, **k):
        """Monkeypatched ``cupy.ndarray.get``: records ``self``'s byte size as a D2H transfer, then delegates unchanged."""
        try:
            if not getattr(_NESTED, "in_asnumpy", False):
                _record(rep, rep.d2h, int(self.nbytes))
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("residency_audit: recording a D2H transfer (ndarray.get) failed: %s", e)
        return _orig_get(self, *a, **k)

    _sync_names = ("item", "__float__", "__int__", "__bool__")
    _orig_sync = {}

    def _wrap_sync(name, orig):
        """Counting wrapper for one implicit scalar-read method."""
        def _w(self, *a, **k):
            """Record one scalar sync (unless inside an explicit transfer already counted), then delegate."""
            if not getattr(_NESTED, "in_asnumpy", False):
                try:
                    rep.scalar_syncs.append(int(self.dtype.itemsize))
                except Exception as e:  # nosec B110 - best-effort path
                    logger.debug("residency_audit: recording a scalar sync failed: %s", e)
            return orig(self, *a, **k)

        return _w

    with _AUDIT_LOCK:
        cp.asarray = _asarray
        cp.asnumpy = _asnumpy
        for _n in _sync_names:
            try:
                _orig_sync[_n] = getattr(cp.ndarray, _n)
                setattr(cp.ndarray, _n, _wrap_sync(_n, _orig_sync[_n]))
            except Exception as e:  # nosec B110 - best-effort path  # noqa: PERF203 - per-method fault isolation
                _orig_sync.pop(_n, None)
                logger.debug("residency_audit: patching cupy.ndarray.%s failed, those syncs won't be tracked: %s", _n, e)
        try:
            cp.ndarray.get = _get  # may be read-only on some cupy builds; guarded
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("residency_audit: patching cupy.ndarray.get failed, .get() transfers won't be tracked: %s", e)
        try:
            yield rep
        finally:
            cp.asarray = _orig_asarray
            cp.asnumpy = _orig_asnumpy
            for _n, _o in _orig_sync.items():
                try:
                    setattr(cp.ndarray, _n, _o)
                except Exception as e:  # nosec B110 - best-effort path  # noqa: PERF203 - per-method fault isolation
                    logger.debug("residency_audit: restoring cupy.ndarray.%s failed: %s", _n, e)
            try:
                cp.ndarray.get = _orig_get
            except Exception as e:  # nosec B110 - best-effort path
                logger.debug("residency_audit: restoring the original cupy.ndarray.get failed: %s", e)
