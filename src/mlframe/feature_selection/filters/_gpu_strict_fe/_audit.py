"""Residency audit harness for the GPU-strict resident FE path.

cupy exposes a malloc/free MemoryHook but NOT a memcpy hook, so this counts host<->device transfers by
monkeypatching the Python-level transfer entry points (``cp.asarray`` for H2D, ``cupy.ndarray.get`` /
``cp.asnumpy`` for D2H) and classifying each by BYTE SIZE. The contract is audited by size, not by call count:
the branchy selection legitimately pulls O(rounds + stages) tiny SCALAR values, so a `.get()`-count assertion
would false-fail; what must be zero is BULK transfer (arrays whose size scales with n_sub or n_candidates)."""
from __future__ import annotations

import contextlib
import threading
from typing import Iterator

# bytes at/above which a transfer is "bulk" (a scalar / tiny index is far below; one operand column at
# n_sub=30k f32 is 120KB, so 8KB cleanly separates scalar decisions from bulk data).
BULK_BYTES = 8192

# X_EDGE_CASES_BEST_PRACTICES-4 fix: residency_audit monkeypatches
# process-wide cp.asarray/cp.asnumpy/cp.ndarray.get with no reentrancy guard. Two overlapping
# residency_audit() regions on different threads would have the second region's "_orig_*" capture be
# the first region's wrapper (not the true original), and whichever region exits first restores to a
# stale value -- silently corrupting the surviving region's byte tally with no error. Serialize entry
# so only one region's monkeypatch is ever installed at a time.
_AUDIT_LOCK = threading.RLock()  # RLock: a same-thread nested residency_audit() must not deadlock


class ResidencyReport:
    """Tally of host<->device transfer byte sizes recorded by :func:`residency_audit`, split into bulk
    (``>= BULK_BYTES``) vs scalar transfers so the resident-FE contract can be asserted on bulk volume."""

    def __init__(self):
        self.h2d = []  # list of byte sizes
        self.d2h = []

    @property
    def bulk_h2d(self):
        """H2D transfer byte sizes that are >= :data:`BULK_BYTES` -- the ones the resident-FE contract forbids."""
        return [b for b in self.h2d if b >= BULK_BYTES]

    @property
    def bulk_d2h(self):
        """D2H transfer byte sizes that are >= :data:`BULK_BYTES` -- the ones the resident-FE contract forbids."""
        return [b for b in self.d2h if b >= BULK_BYTES]

    @property
    def scalar_d2h_bytes(self):
        """Total bytes of D2H transfers below :data:`BULK_BYTES` -- the tolerated scalar/branch-decision traffic."""
        return sum(b for b in self.d2h if b < BULK_BYTES)

    def summary(self) -> str:
        """One-line ``"H2D: N ops (B bulk, T B); D2H: ..."`` string for logging / assertion messages."""
        return (f"H2D: {len(self.h2d)} ops ({len(self.bulk_h2d)} bulk, {sum(self.h2d)} B); "
                f"D2H: {len(self.d2h)} ops ({len(self.bulk_d2h)} bulk, {self.scalar_d2h_bytes} B scalar)")


@contextlib.contextmanager
def residency_audit() -> Iterator[ResidencyReport]:
    """Context manager yielding a :class:`ResidencyReport`. Records H2D bytes (``cp.asarray`` of a host array)
    and D2H bytes (``ndarray.get`` / ``cp.asnumpy``) for the enclosed region. No-op (empty report) when cupy is
    unavailable. Intended for tests / profiling, not the production path."""
    rep = ResidencyReport()
    try:
        import cupy as cp
        import numpy as np
    except Exception:
        yield rep
        return

    _orig_asarray = cp.asarray
    _orig_asnumpy = cp.asnumpy
    _orig_get = cp.ndarray.get

    def _asarray(obj, *a, **k):
        """Monkeypatched ``cp.asarray``: records host-array byte size as an H2D transfer, then delegates unchanged."""
        try:
            if isinstance(obj, np.ndarray):
                rep.h2d.append(int(obj.nbytes))
        except Exception:  # nosec B110 - best-effort path
            pass
        return _orig_asarray(obj, *a, **k)

    def _asnumpy(obj, *a, **k):
        """Monkeypatched ``cp.asnumpy``: records the source array's byte size as a D2H transfer, then delegates unchanged."""
        try:
            nb = int(getattr(obj, "nbytes", 0))
            if nb:
                rep.d2h.append(nb)
        except Exception:  # nosec B110 - best-effort path
            pass
        return _orig_asnumpy(obj, *a, **k)

    def _get(self, *a, **k):
        """Monkeypatched ``cupy.ndarray.get``: records ``self``'s byte size as a D2H transfer, then delegates unchanged."""
        try:
            rep.d2h.append(int(self.nbytes))
        except Exception:  # nosec B110 - best-effort path
            pass
        return _orig_get(self, *a, **k)

    with _AUDIT_LOCK:
        cp.asarray = _asarray
        cp.asnumpy = _asnumpy
        try:
            cp.ndarray.get = _get  # may be read-only on some cupy builds; guarded
        except Exception:  # nosec B110 - best-effort path
            pass
        try:
            yield rep
        finally:
            cp.asarray = _orig_asarray
            cp.asnumpy = _orig_asnumpy
            try:
                cp.ndarray.get = _orig_get
            except Exception:  # nosec B110 - best-effort path
                pass
