"""Fit-constant FE operand resident-device cache.

Under ``MLFRAME_FE_GPU_STRICT`` the GPU twins are applied, but several of them re-upload the SAME host
fit-constant operands (the label ``y``, the conditioning support ``z``, and the base feature columns the
candidate matrices are built from) on EVERY call. An H2D instrumentation of a 250k F2 strict fit (monkeypatch
of ``cupy.asarray``/``cupy.array`` logging shape/dtype/caller) showed the leak concretely: the fit-constant
operands are uploaded dozens of times per fit (e.g. ``_resident_candidate_mi.py:153`` base operand columns
80x / 140 MB, ``_orth_mi_backends.py:315`` y 54x / 86 MB, the ``_mi_greedy_cmi_fe`` y/z sites ~120x). The
candidate matrices themselves are generated ON device (no host source) and are transient -- they MUST NOT be
cached. Only the fit-constant base operands / y / z are uploaded once and reused.

The cache is keyed PURELY on a content fingerprint (shape + dtype + a cheap copy-free O(n) content hash), NOT on
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

import logging
import os as _os
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as _np

logger = logging.getLogger(__name__)

try:
    from numba import njit as _njit

    @_njit(cache=True, nogil=True)
    def _njit_content_hash(words: "_np.ndarray", tail: "_np.ndarray") -> int:
        """MurmurHash3-finalizer-mixed 64-bit hash over a buffer viewed as uint64 words + <=7 tail bytes.
        Copy-free (numba reads the array buffer directly) and ~3x the ``hash(tobytes())`` fallback (677 vs
        2054 us on a 300k-f64 operand) with strong avalanche so the stale-alias collision guard is preserved."""
        h = _np.uint64(1469598103934665603)
        prime = _np.uint64(1099511628211)
        for i in range(words.shape[0]):
            k = words[i]
            k ^= k >> _np.uint64(33)
            k *= _np.uint64(0xFF51AFD7ED558CCD)
            h ^= k
            h *= prime
        for j in range(tail.shape[0]):
            h ^= _np.uint64(tail[j])
            h *= prime
        h ^= h >> _np.uint64(33)
        h *= _np.uint64(0xC4CEB9FE1A85EC53)
        h ^= h >> _np.uint64(33)
        return int(h)
except Exception as e:  # numba optional: fall through to the tobytes hash below
    logger.debug("numba unavailable, falling back to tobytes-based content hash: %s", e)
    _njit_content_hash = None

# Copy-free content hash. The signature must O(n)-hash the WHOLE operand buffer to guard against a
# same-shape/dtype operand with different VALUES aliasing a stale device buffer (see resident_operand). The old
# ``hash(host.tobytes())`` walked the buffer AND allocated a full host copy first (an ~8 MB tobytes churn for a
# 1M-row f64 operand). ``xxh3_64`` walks the array buffer directly via the buffer protocol -- no intermediate
# bytes copy -- at ~8x the throughput, and stays in the SAME 64-bit collision domain as the old key (shape +
# dtype-str still split the space; a 64-bit content hash collides no more than Python's siphash-based
# ``hash(bytes)`` did), so keying on it is exactly as safe against a stale-buffer alias. If xxhash is absent, or
# the array is not C-contiguous (buffer protocol would reject it), fall back to the original tobytes hash.
_xxh3_64: Optional[Callable] = None
try:
    import xxhash as _xxhash

    _xxh3_64 = _xxhash.xxh3_64_intdigest
except Exception as e:  # nosec B110 - xxhash optional: correctness identical via the tobytes fallback, only the copy churn returns
    logger.debug("xxhash unavailable, falling back to njit/tobytes content hash: %s", e)


def _content_hash(host: Any) -> int:
    """O(n) content hash of a host operand, copy-free when possible (see module note on the collision domain)."""
    if _xxh3_64 is not None and host.flags["C_CONTIGUOUS"]:
        return int(_xxh3_64(host))
    # xxhash-absent fallback: copy-free njit word hash (~3x the tobytes path) when the array is C-contiguous.
    if _njit_content_hash is not None and host.flags["C_CONTIGUOUS"]:
        b = host.view(_np.uint8).reshape(-1)
        nwords = b.shape[0] // 8
        words = b[: nwords * 8].view(_np.uint64)
        tail = b[nwords * 8 :]
        return int(_njit_content_hash(words, tail))
    return hash(host.tobytes())


# HASH MEMO (mrmr_audit_2026-07-20 gpu_residency.md #6, 2026-07-21): the docstring below used to
# document this as unaddressed -- "the fit-constant y/z are re-hashed on every role's call". A full
# caller-side handle-threading rewrite (upload y/z ONCE at the FE-step entry and hand every one of
# the ~9 documented roles the same resident cupy array by reference) would touch every call site in
# this file AND _cmi_cuda.py; instead, memoize the O(n) HASH itself keyed on the host array's id(),
# mirroring the identical id()-plus-weakref-plus-shape/dtype guard convention
# info_theory._cmi_cuda._cmi_forder_view already uses for its own per-fit cache. When the SAME y/z
# object is handed to ``resident_operand``/``resident_qbin_codes`` across every role each round (the
# common case: same dtype request, already contiguous -> ``np.asarray``/``astype(copy=False)``/
# ``ascontiguousarray`` all return the identical object, not a fresh copy), the hash is computed
# ONCE per fit instead of once per role-call. A recycled id (the original array GC'd, a new one
# allocated at the same address) fails the ``ref() is host`` check and falls back to a full
# recompute -- never a stale hash for different content.
import weakref as _weakref

_HASH_MEMO: "OrderedDict" = OrderedDict()  # id(host) -> (weakref, shape, dtype_str, hash)
_HASH_MEMO_MAX_ENTRIES = 64


def _content_hash_memoized(host: Any) -> int:
    """``_content_hash(host)``, memoized on ``id(host)`` with a weakref+shape/dtype recycled-id guard.
    Falls back to a full recompute whenever the object identity, shape, or dtype does not match the
    cached entry (a genuinely different array, or the SAME id recycled with different content)."""
    key = id(host)
    ent = _HASH_MEMO.get(key)
    if ent is not None:
        ref, shape, dtype_str, h = ent
        if ref() is host and shape == host.shape and dtype_str == host.dtype.str:
            _HASH_MEMO.move_to_end(key)  # LRU: this identity is hot
            return int(h)
    h = _content_hash(host)
    _HASH_MEMO[key] = (_weakref.ref(host), host.shape, host.dtype.str, h)
    if len(_HASH_MEMO) > _HASH_MEMO_MAX_ENTRIES:
        _HASH_MEMO.popitem(last=False)  # evict ONLY the coldest entry, never the whole table
    return h


def clear_hash_memo() -> None:
    """Drop the id()-keyed hash memo (call at FE-step teardown; mirrors the other resident caches)."""
    _HASH_MEMO.clear()

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
    """Whether the resident-operand cache is disabled via ``MLFRAME_FE_RESIDENT_OPERANDS`` (diagnostic A/B switch, default on)."""
    # Diagnostic A/B switch only (default ON): MLFRAME_FE_RESIDENT_OPERANDS=0 forces a fresh upload every call
    # to reproduce the pre-cache H2D churn for the wall/util A/B. NOT a perf gate -- always on in prod.
    return _os.environ.get("MLFRAME_FE_RESIDENT_OPERANDS", "1").strip().lower() in ("0", "false", "off", "no")


def resident_operand(arr: Any, key: Any, *, dtype: Any = None, contiguous: bool = True) -> Any:
    """Return a cached device copy of host ``arr`` (uploaded ONCE per fit), keyed on its CONTENT.

    ``key`` is a ROLE label (e.g. ``"cmi_y"``, ``("op", col_name)``) retained only for call-site readability
    and A/B tracing: it is NOT part of the cache key. The key is the content signature
    (shape + dtype + a copy-free content hash), so the SAME fit-constant content uploaded under different roles
    (the label ``y`` by cmi_y / card_y / fixedyz_y / y_mi_classif / ...; the support ``z`` by cmi_z / card_z /
    fixedyz_z) shares ONE resident device buffer instead of one upload per role -- the 65%-of-operand-H2D
    cross-role re-upload leak (see module docstring). The content hash is copy-free (``xxh3_64`` over the array
    buffer, tobytes fallback for non-contiguous / no-xxhash), in the same 64-bit collision domain as before.
    NOTE (2026-07-21, mrmr_audit_2026-07-20 gpu_residency.md #6): the O(n) HASH of the fit-constant
    ``y`` / ``z`` is now memoized on the host array's ``id()`` (``_content_hash_memoized``, weakref
    + shape/dtype recycled-id guard) -- when the SAME object is handed to this function across every
    role each round (the common case), the hash is computed ONCE per fit instead of once per
    role-call. A full caller-side upload-dedup (upload y/z ONCE at the FE-step entry and pass the
    resident handle directly, skipping ``resident_operand`` entirely for them) would still remove
    the dict-lookup itself; that is a bigger caller-side plumbing change across ~9 call sites in
    this file and ``_cmi_cuda.py``, left as further future work. ``dtype`` (optional) is the FINAL dtype the kernel needs
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
    # _content_hash_memoized skips the O(n) recompute when the SAME array object (e.g. a fit-constant
    # y/z passed unchanged across every role) was already hashed this fit -- see its own docstring.
    sig = (host.shape, host.dtype.str, _content_hash_memoized(host))
    g = _FE_RESIDENT_OPERANDS.get(sig)
    if g is not None:
        _FE_RESIDENT_OPERANDS.move_to_end(sig)  # LRU: this content is hot
        return g
    g = cp.asarray(host)
    _FE_RESIDENT_OPERANDS[sig] = g
    if len(_FE_RESIDENT_OPERANDS) > _MAX_ENTRIES:
        _FE_RESIDENT_OPERANDS.popitem(last=False)  # evict ONLY the coldest entry, never the whole table
    return g


def assemble_resident_matrix(host, names, fallback_key, *, dtype=None):
    """Return a resident ``(n, k)`` device matrix DEVICE-ASSEMBLED from its per-column resident operands.

    Several FE scorers upload a fit-constant raw baseline matrix WHOLE via a single ``resident_operand`` keyed
    on the ``(n, k)`` blob's content -- but that blob is a distinct fingerprint that never dedups, even though
    every one of its columns is the SAME raw source column already uploaded ONCE elsewhere (the basis /
    cross-basis device builders upload each source column column-by-column). Stacking the resident PER-COLUMN
    operands on device instead lets each column content-hit the operand cache, so the whole matrix never
    crosses H2D (only the few distinct columns upload once, shared with every other consumer of that column).

    ``host`` is the ``(n, k)`` (or ``(n,)``) host matrix; column ``j`` MUST be exactly the raw column named
    ``names[j]`` (same float bytes) so the per-column upload is the SAME content -> content-keyed dedup ->
    selection-IDENTICAL. When ``names`` is missing / length-mismatched / STRICT-residency is off / any per-column
    upload faults, falls back to uploading the whole matrix under ``fallback_key`` (the prior behaviour, so a
    layout mismatch is a perf no-op, never a correctness change). ``dtype`` is the final kernel dtype (folded
    into every per-column upload so all consumers of a column share one buffer)."""
    import cupy as cp
    import numpy as np

    host = np.asarray(host)
    if dtype is not None:
        host = host.astype(dtype, copy=False)
    host = np.ascontiguousarray(host)
    if host.ndim == 1:
        host = host[:, None]
    _n, _k = host.shape
    if (not _disabled()) and names is not None and len(names) == _k and _k >= 1:
        try:
            cols = [resident_operand(np.ascontiguousarray(host[:, j]), ("xbasis_op", names[j]), dtype=dtype) for j in range(_k)]
            return cp.stack(cols, axis=1) if _k > 1 else cols[0][:, None]
        except Exception as e:  # nosec B110 - best-effort path
            logger.debug("per-column resident-operand assembly failed, falling back to whole-matrix upload: %s", e)
    g = resident_operand(host, fallback_key, dtype=dtype)
    return g[:, None] if g.ndim == 1 else g


def resident_code_operand(codes, role):
    """Upload candidate BIN CODES once as int16 (with an int64 escape for genuinely high-cardinality inputs) and
    return them cast to int64 on the device for the joint-histogram kernels.

    Candidate codes are equi-frequency bin indices in ``0..nbins-1`` (nbins a few dozen), so int16 (2 B) holds
    them with a wide safety margin (up to 32767 bins) -- uploading them as int64 (8 B) was 4x the bytes at every
    ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x`` (and the fit-constant y-code) re-upload site. int16 is
    chosen over int8 deliberately: int8 would save another 2x but only holds 0..127, so a higher-resolution
    binning (or a code array that is not a plain per-column bin index) would need a per-array range check to stay
    safe; int16 needs no such reasoning and is robust to any realistic nbins. The rare array whose max does not
    fit int16 (e.g. a densified high-cardinality JOINT accidentally routed here) falls back to int64. Upload the
    narrow codes (content-keyed, so identical content still dedups) and return the int64 cast the CMI / entropy
    kernels index histograms with. BIT-IDENTICAL: int16 holds ``0..nbins-1`` exactly and the widening cast
    restores the same int64 values the kernels saw before, so the partition -- and every downstream MI /
    cardinality -- is unchanged, not merely selection-equivalent. The narrow array is cached + reused across
    calls; only the transient int64 view is rebuilt per call (a cheap device widen), so the H2D shrinks 4x with
    no re-upload."""
    import cupy as cp
    import numpy as np

    host = np.ascontiguousarray(np.asarray(codes).ravel())
    _m = int(host.max()) if host.size else 0
    _mn = int(host.min()) if host.size else 0
    _dt = np.int16 if (0 <= _mn and _m < 32768) else np.int64
    dev = resident_operand(host, role, dtype=_dt)
    return dev.astype(cp.int64) if dev.dtype != cp.int64 else dev


# content signature (shape + dtype-str + content hash + nbins + bin-dtype)  ->  RESIDENT int16 bin codes. The
# same candidate column is re-binned across greedy steps with IDENTICAL content (H2D audit of a 1M strict fit:
# 42 qbin calls / 20 distinct contents), yet each call re-ran the full O(n log n) ``cp.sort`` in
# ``_sync_free_qbin_codes`` (the cub DeviceMergeSort at 5.6% of GPU time). The float operand was already
# content-deduped (``qbin_x`` via ``resident_operand``), but nothing cached the CODES, so a repeat bin of
# identical content re-sorted from scratch. Caching the codes on the SAME content signature skips the sort
# ENTIRELY on the ~half of calls that re-bin an already-seen column: same float bytes + same nbins + same bin
# dtype -> same sorted order -> same edges -> BIT-IDENTICAL codes (not merely selection-equivalent), with NO
# device->host sync (the key is a pure host content hash, the same one the operand cache computes). Stored as
# int16 (codes are ``0..nbins-1``, a few dozen bins; the widening cast on retrieval restores the exact int64 the
# kernels index -- the same narrow-store/widen discipline as ``resident_code_operand``), so the whole 20-content
# working set is a few MB of VRAM, not the 160 MB an int64 store would cost. Same LRU + teardown as the operand
# cache below.
_FE_RESIDENT_QBIN_CODES: "OrderedDict" = OrderedDict()
_MAX_QBIN_CODE_ENTRIES = 64


def resident_qbin_codes(a: Any, nbins: int, dtype: Any, compute_fn: Any) -> Any:
    """Return RESIDENT int64 equi-frequency bin codes for host column ``a``, content-cached to skip a repeat sort.

    ``compute_fn(cp, xd, nbins)`` is the sync-free device binner (``_sync_free_qbin_codes``): called only on a
    cache MISS, on the content-deduped resident float column ``xd`` (uploaded via ``resident_operand`` in the
    ``dtype`` the binner needs). On a HIT the previously computed codes are returned WITHOUT re-uploading the
    float or re-running the sort. The cache key is the host content signature (shape + dtype + copy-free content
    hash) plus ``nbins`` and the bin dtype, so only genuinely identical bin requests share a result -- a
    different column, a different nbins, or a different bin dtype gets a fresh compute. Codes are stored int16 and
    widened to int64 on return (values are ``0..nbins-1`` -> bit-identical), mirroring ``resident_code_operand``.
    Fully sync-free: the key is a host hash, ``compute_fn`` is the sync-free binner, and the widen is a device
    op -- nothing scalar crosses the bus."""
    import cupy as cp
    import numpy as np

    host = np.ascontiguousarray(np.asarray(a))
    if _disabled():
        xd = resident_operand(host, "qbin_x", dtype=dtype)
        return compute_fn(cp, xd, int(nbins))

    sig = (host.shape, host.dtype.str, _content_hash_memoized(host), int(nbins), str(dtype))
    dev = _FE_RESIDENT_QBIN_CODES.get(sig)
    if dev is not None:
        _FE_RESIDENT_QBIN_CODES.move_to_end(sig)  # LRU: this content is hot
        return dev.astype(cp.int64)  # widen the narrow store to the int64 the kernels index
    xd = resident_operand(host, "qbin_x", dtype=dtype)
    codes = compute_fn(cp, xd, int(nbins))  # sync-free binner, resident int64 codes
    # Store narrow: codes are 0..nbins-1 (a few dozen bins), int16 holds them exactly (int64 escape only if some
    # binner ever emits a value outside int16, keeping the store bit-identical either way).
    narrow = codes.astype(cp.int16) if (int(nbins) <= 32767) else codes
    _FE_RESIDENT_QBIN_CODES[sig] = narrow
    if len(_FE_RESIDENT_QBIN_CODES) > _MAX_QBIN_CODE_ENTRIES:
        _FE_RESIDENT_QBIN_CODES.popitem(last=False)  # evict ONLY the coldest entry, never the whole table
    return codes


def clear_fe_resident_operands() -> None:
    """Drop the fit-constant FE operand + qbin-code device caches (call at FE-step teardown; mirrors mempool free)."""
    _FE_RESIDENT_OPERANDS.clear()
    _FE_RESIDENT_QBIN_CODES.clear()
    clear_hash_memo()


__all__ = [
    "resident_operand",
    "resident_code_operand",
    "resident_qbin_codes",
    "assemble_resident_matrix",
    "clear_fe_resident_operands",
    "clear_hash_memo",
]
