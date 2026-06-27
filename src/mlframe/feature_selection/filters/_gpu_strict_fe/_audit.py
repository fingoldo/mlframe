"""Residency audit harness for the GPU-strict resident FE path.

cupy exposes a malloc/free MemoryHook but NOT a memcpy hook, so this counts host<->device transfers by
monkeypatching the Python-level transfer entry points (``cp.asarray`` for H2D, ``cupy.ndarray.get`` /
``cp.asnumpy`` for D2H) and classifying each by BYTE SIZE. The contract is audited by size, not by call count:
the branchy selection legitimately pulls O(rounds + stages) tiny SCALAR values, so a `.get()`-count assertion
would false-fail; what must be zero is BULK transfer (arrays whose size scales with n_sub or n_candidates)."""
from __future__ import annotations

import contextlib

# bytes at/above which a transfer is "bulk" (a scalar / tiny index is far below; one operand column at
# n_sub=30k f32 is 120KB, so 8KB cleanly separates scalar decisions from bulk data).
BULK_BYTES = 8192


class ResidencyReport:
    def __init__(self):
        self.h2d = []   # list of byte sizes
        self.d2h = []

    @property
    def bulk_h2d(self):
        return [b for b in self.h2d if b >= BULK_BYTES]

    @property
    def bulk_d2h(self):
        return [b for b in self.d2h if b >= BULK_BYTES]

    @property
    def scalar_d2h_bytes(self):
        return sum(b for b in self.d2h if b < BULK_BYTES)

    def summary(self) -> str:
        return (f"H2D: {len(self.h2d)} ops ({len(self.bulk_h2d)} bulk, {sum(self.h2d)} B); "
                f"D2H: {len(self.d2h)} ops ({len(self.bulk_d2h)} bulk, {self.scalar_d2h_bytes} B scalar)")


@contextlib.contextmanager
def residency_audit():
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
        try:
            if isinstance(obj, np.ndarray):
                rep.h2d.append(int(obj.nbytes))
        except Exception:
            pass
        return _orig_asarray(obj, *a, **k)

    def _asnumpy(obj, *a, **k):
        try:
            nb = int(getattr(obj, "nbytes", 0))
            if nb:
                rep.d2h.append(nb)
        except Exception:
            pass
        return _orig_asnumpy(obj, *a, **k)

    def _get(self, *a, **k):
        try:
            rep.d2h.append(int(self.nbytes))
        except Exception:
            pass
        return _orig_get(self, *a, **k)

    cp.asarray = _asarray
    cp.asnumpy = _asnumpy
    try:
        cp.ndarray.get = _get   # may be read-only on some cupy builds; guarded
    except Exception:
        pass
    try:
        yield rep
    finally:
        cp.asarray = _orig_asarray
        cp.asnumpy = _orig_asnumpy
        try:
            cp.ndarray.get = _orig_get
        except Exception:
            pass
