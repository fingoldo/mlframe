"""Fit-constant FE operand resident-device cache.

Under ``MLFRAME_FE_GPU_STRICT`` the GPU twins are applied, but several of them re-upload the SAME host
fit-constant operands (the label ``y``, the conditioning support ``z``, and the base feature columns the
candidate matrices are built from) on EVERY call. An H2D instrumentation of a 250k F2 strict fit (monkeypatch
of ``cupy.asarray``/``cupy.array`` logging shape/dtype/caller) showed the leak concretely: the fit-constant
operands are uploaded dozens of times per fit (e.g. ``_resident_candidate_mi.py:153`` base operand columns
80x / 140 MB, ``_orth_mi_backends.py:315`` y 54x / 86 MB, the ``_mi_greedy_cmi_fe`` y/z sites ~120x). The
candidate matrices themselves are generated ON device (no host source) and are transient -- they MUST NOT be
cached. Only the fit-constant base operands / y / z are uploaded once and reused.

This mirrors the proven y/z resident cache in ``info_theory/_cmi_cuda.py``
(``_resident_upload`` / ``_CMI_RESIDENT_CACHE``): a module-level dict keyed on a STABLE identity (parent array
id + a discriminator + dtype + shape) WITH a cheap O(n) content fingerprint to guard against id recycling
(``id()`` is reused by the allocator after the parent is GC'd, so a recycled id with different VALUES would
otherwise alias a stale device buffer -- a silent correctness bug). On a hit the cached device array is
returned; on a miss the operand is uploaded in the FINAL kernel dtype (so repeated ``astype`` collapse) and
cached. Bounded; NEVER ``free_all_blocks`` (that is the mempool teardown's job). Cleared at FE-step teardown
alongside the mempool free + the cmi resident cache.

Pickle-safe: this cache is module-level and NEVER stored on an MRMR instance (mirrors ``_cmi_cuda``'s note).
"""

import os as _os

# id(parent) + discriminator + dtype-str + shape  ->  (device_array, content_signature)
_FE_RESIDENT_OPERANDS: dict = {}

# Bound: the F2 strict fit holds at most a handful of distinct fit-constants live at once (y, z, and the few
# base operand columns of the current candidate group). 64 entries leaves ample headroom; on overflow we clear
# the whole table (a fit-scoped cache, re-warmed on the next call) rather than evicting one-by-one.
_MAX_ENTRIES = 64


def _disabled() -> bool:
    # Diagnostic A/B switch only (default ON): MLFRAME_FE_RESIDENT_OPERANDS=0 forces a fresh upload every call
    # to reproduce the pre-cache H2D churn for the wall/util A/B. NOT a perf gate -- always on in prod.
    return _os.environ.get("MLFRAME_FE_RESIDENT_OPERANDS", "1").strip().lower() in ("0", "false", "off", "no")


def resident_operand(arr, key, *, dtype=None, contiguous: bool = True):
    """Return a cached device copy of host ``arr`` (uploaded ONCE per fit), keyed on the caller's ``key``.

    ``key`` is a ROLE discriminator (e.g. ``"cmi_y"``, ``("op", col_name)``). The operand's own shape + dtype
    are folded into the cache key automatically, so a single role naturally holds one entry per (shape, dtype)
    -- e.g. the FE subsample y (200k) and the full y (250k) coexist as two entries. Do NOT key on ``id()`` of
    the operand: the host operand is frequently re-raveled / re-derived per call (a fresh object, fresh id), so
    an id-based key would miss every time; the role + shape + content fingerprint give a STABLE hit while the
    fingerprint still guards correctness. ``dtype`` (optional) is the FINAL dtype the kernel needs (e.g.
    ``cp.float64`` / ``np.int64``); caching in that dtype collapses repeated ``astype(copy=False)`` calls.

    A cheap O(n) host content fingerprint (shape + dtype + ``hash(arr.tobytes())``) is folded into the cached
    entry so a RECYCLED id with different VALUES misses and re-uploads rather than returning a stale buffer
    (recycled-id-no-alias guard). The hash is pure CPU and far cheaper than the H2D it guards.
    """
    import cupy as cp
    import numpy as np

    host = np.asarray(arr)
    if dtype is not None:
        host = host.astype(dtype, copy=False)
    if contiguous:
        host = np.ascontiguousarray(host)

    if _disabled():
        return cp.asarray(host)

    # Fold shape + dtype into the lookup key so one role holds one entry per (shape, dtype); the content hash
    # then guards against a same-shape operand with different VALUES aliasing a stale device buffer.
    full_key = (key, host.shape, host.dtype.str)
    sig = (host.shape, host.dtype.str, hash(host.tobytes()))
    cached = _FE_RESIDENT_OPERANDS.get(full_key)
    if cached is not None:
        g, csig = cached
        if csig == sig:
            return g
    if len(_FE_RESIDENT_OPERANDS) > _MAX_ENTRIES:
        _FE_RESIDENT_OPERANDS.clear()
    g = cp.asarray(host)
    _FE_RESIDENT_OPERANDS[full_key] = (g, sig)
    return g


def clear_fe_resident_operands() -> None:
    """Drop the fit-constant FE operand device cache (call at FE-step teardown; mirrors the mempool free)."""
    _FE_RESIDENT_OPERANDS.clear()


__all__ = ["resident_operand", "clear_fe_resident_operands"]
