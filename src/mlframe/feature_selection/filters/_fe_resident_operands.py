"""Fit-constant FE operand resident-device cache.

Under ``MLFRAME_FE_GPU_STRICT`` the GPU twins are applied, but several of them re-upload the SAME host
fit-constant operands (the label ``y``, the conditioning support ``z``, and the base feature columns the
candidate matrices are built from) on EVERY call. An H2D instrumentation of a 250k F2 strict fit (monkeypatch
of ``cupy.asarray``/``cupy.array`` logging shape/dtype/caller) showed the leak concretely: the fit-constant
operands are uploaded dozens of times per fit (e.g. ``_resident_candidate_mi.py:153`` base operand columns
80x / 140 MB, ``_orth_mi_backends.py:315`` y 54x / 86 MB, the ``_mi_greedy_cmi_fe`` y/z sites ~120x). The
candidate matrices themselves are generated ON device (no host source) and are transient -- they MUST NOT be
cached. Only the fit-constant base operands / y / z are uploaded once and reused.

The cache is keyed PURELY on a content fingerprint (shape + dtype + a cheap O(n) ``hash(tobytes())``), NOT on
the caller's role discriminator. An H2D audit of a 1M strict-resident fit (monkeypatch of ``resident_operand``
counting misses by content signature) showed 615 MB / 90 ops -- 65% of the operand H2D -- were CROSS-ROLE
re-uploads of IDENTICAL content: the label ``y`` is uploaded by ~6 roles (``cmi_y`` / ``card_y`` /
``fixedyz_y`` / ``y_mi_classif`` / ``orth_uni_y`` / ``cmi_greedy_y_fixed``) and the conditioning support ``z``
by 3 (``cmi_z`` / ``card_z`` / ``fixedyz_z``), each a separate upload under the old ``(role, shape, dtype)``
key. Keying on the content signature collapses every same-content upload to ONE resident device buffer shared
across all roles. The content hash is exactly what already guarded against id recycling (a recycled ``id()``
with different VALUES aliasing a stale buffer), so keying on it directly is strictly safer than the old key.
Operands are fit-constant INPUTS consumed READ-ONLY by the MI / CMI / entropy / ``.max()`` kernels (FE
generation writes to fresh output buffers, never the operands -- verified: no in-place mutation of any
``resident_operand`` result), so one shared buffer per content is correct and selection-IDENTICAL.

This mirrors the proven y/z resident cache in ``info_theory/_cmi_cuda.py`` (``_resident_upload`` /
``_CMI_RESIDENT_CACHE``). On a hit the cached device array is returned; on a miss the operand is uploaded in the
FINAL kernel dtype (so repeated ``astype`` collapse) and cached. Bounded by an LRU (the single COLDEST entry is
evicted on overflow, NOT the whole table -- a clear-all forces a re-upload storm of the still-hot operands);
NEVER ``free_all_blocks`` (that is the mempool teardown's job). Cleared at FE-step teardown alongside the
mempool free + the cmi resident cache.

Pickle-safe: this cache is module-level and NEVER stored on an MRMR instance (mirrors ``_cmi_cuda``'s note).
"""

import os as _os
from collections import OrderedDict

# content signature (shape + dtype-str + content hash)  ->  device_array. OrderedDict gives O(1) LRU: hits
# move-to-end (hot), overflow pops the front (coldest).
_FE_RESIDENT_OPERANDS: "OrderedDict" = OrderedDict()

# Bound: content-keyed, so entries == DISTINCT fit-constant operands live this fit. A 1M strict-resident fit
# touched 118 distinct contents (~330 MB) over its lifetime; 192 holds that whole working set with headroom so
# nothing hot is ever evicted (and the operands are <=200k-row subsample columns, a few MB each -> ~330 MB of
# VRAM worst case, comfortable alongside the chunked candidate buffers on a 4 GB card). On overflow the LRU
# evicts only the single coldest entry (a fit-scoped re-upload of that one if reused), never the whole table.
_MAX_ENTRIES = 192


def _disabled() -> bool:
    # Diagnostic A/B switch only (default ON): MLFRAME_FE_RESIDENT_OPERANDS=0 forces a fresh upload every call
    # to reproduce the pre-cache H2D churn for the wall/util A/B. NOT a perf gate -- always on in prod.
    return _os.environ.get("MLFRAME_FE_RESIDENT_OPERANDS", "1").strip().lower() in ("0", "false", "off", "no")


def resident_operand(arr, key, *, dtype=None, contiguous: bool = True):
    """Return a cached device copy of host ``arr`` (uploaded ONCE per fit), keyed on its CONTENT.

    ``key`` is a ROLE label (e.g. ``"cmi_y"``, ``("op", col_name)``) retained only for call-site readability
    and A/B tracing: it is NOT part of the cache key. The key is the content signature
    (shape + dtype + ``hash(arr.tobytes())``), so the SAME fit-constant content uploaded under different roles
    (the label ``y`` by cmi_y / card_y / fixedyz_y / y_mi_classif / ...; the support ``z`` by cmi_z / card_z /
    fixedyz_z) shares ONE resident device buffer instead of one upload per role -- the 65%-of-operand-H2D
    cross-role re-upload leak (see module docstring). ``dtype`` (optional) is the FINAL dtype the kernel needs
    (e.g. ``cp.float64`` / ``np.int64``); caching in that dtype collapses repeated ``astype(copy=False)`` calls
    AND keeps two roles that want the same values in different dtypes as distinct (correct) entries.

    The content hash both deduplicates and guards correctness: a same-shape operand with different VALUES (or a
    recycled host id) gets a different signature -> a fresh upload, never a stale-buffer alias. The hash is pure
    CPU and far cheaper than the H2D it guards. Operands are read-only inputs, so the shared buffer is safe;
    returned bytes are identical -> selection-IDENTICAL.
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

    # Content signature is the WHOLE key: shape + dtype distinguish dtype/length variants; the content hash
    # deduplicates identical operands across roles and guards against a different-values alias.
    sig = (host.shape, host.dtype.str, hash(host.tobytes()))
    g = _FE_RESIDENT_OPERANDS.get(sig)
    if g is not None:
        _FE_RESIDENT_OPERANDS.move_to_end(sig)        # LRU: this content is hot
        return g
    g = cp.asarray(host)
    _FE_RESIDENT_OPERANDS[sig] = g
    if len(_FE_RESIDENT_OPERANDS) > _MAX_ENTRIES:
        _FE_RESIDENT_OPERANDS.popitem(last=False)     # evict ONLY the coldest entry, never the whole table
    return g


def clear_fe_resident_operands() -> None:
    """Drop the fit-constant FE operand device cache (call at FE-step teardown; mirrors the mempool free)."""
    _FE_RESIDENT_OPERANDS.clear()


__all__ = ["resident_operand", "clear_fe_resident_operands"]
