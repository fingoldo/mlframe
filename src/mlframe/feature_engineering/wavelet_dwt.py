"""Numba-jitted Discrete Wavelet Transform (DWT) for feature
engineering.

Faster numerically-identical alternative to ``pywt.wavedec`` /
``pywt.waverec`` for the common case of repeated DWT/iDWT calls on
many small-to-medium signals (rolling windows, per-row scans,
per-group denoising). Reference numba kernels ported from
https://github.com/fingoldo/astronomy/blob/main/test_numba_wavelet.py
and extended with:

* a parallel batched ``wavedec_batched`` that processes N signals at
  once via ``numba.prange`` (use when N > a few hundred);
* an inverse-DWT (``waverec_numba`` + ``_idwt_single_level_numba``)
  matched bit-for-bit to pywt's symmetric reconstruction so the full
  denoise workflow (wavedec -> threshold -> waverec) can run without
  any pywt round trip;
* a dispatcher (``wavedec_dispatch``) that picks single vs batched
  based on input shape;
* filter helpers that cache pywt-derived (dec_lo, dec_hi, rec_lo,
  rec_hi) tuples per wavelet name so repeated calls don't re-import
  pywt for the constants.

Public API:
    * ``wavedec(signal, wavelet, max_level)``      -- single signal
    * ``wavedec_batched(signals, wavelet, max_level)`` -- (N, T)
    * ``waverec(coeffs, wavelet)``                  -- inverse single
    * ``wavedec_dispatch(signals, wavelet, max_level)`` -- pick best
    * ``wavelet_denoise(signal, wavelet, level, threshold, mode='soft')``
    * ``get_wavelet_filters(name)`` -- (lo, hi, rec_lo, rec_hi)
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    numba = None
    _HAS_NUMBA = False

try:
    from ..metrics._numba_params import NUMBA_NJIT_PARAMS
except ImportError:
    NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


# ---------------------------------------------------------------------
# Filter lookup (uses pywt only as a constants source; the constants
# are cached so subsequent calls don't re-touch pywt).
# ---------------------------------------------------------------------

_FILTER_CACHE: dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def get_wavelet_filters(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(dec_lo, dec_hi, rec_lo, rec_hi)`` filters for the
    named wavelet. Cached so repeated calls don't re-invoke pywt.
    Supported names: any pywt wavelet name (``"haar"``, ``"db4"``,
    ``"db6"``, ``"coif3"``, ``"sym4"`` etc).
    """
    if name in _FILTER_CACHE:
        return _FILTER_CACHE[name]
    import pywt
    w = pywt.Wavelet(name)
    dec_lo = np.asarray(w.dec_lo, dtype=np.float64)
    dec_hi = np.asarray(w.dec_hi, dtype=np.float64)
    rec_lo = np.asarray(w.rec_lo, dtype=np.float64)
    rec_hi = np.asarray(w.rec_hi, dtype=np.float64)
    _FILTER_CACHE[name] = (dec_lo, dec_hi, rec_lo, rec_hi)
    return _FILTER_CACHE[name]


# ---------------------------------------------------------------------
# Numba kernels: single-signal forward + inverse DWT.
# ---------------------------------------------------------------------


if _HAS_NUMBA:

    @numba.njit(**NUMBA_NJIT_PARAMS)
    def _symmetric_reflect(idx: int, n: int) -> int:
        """pywt 'symmetric' boundary: ``[..., x1, x0, x0, x1, ...]``."""
        if idx < 0:
            idx = -idx - 1
        if idx >= n:
            idx = 2 * n - 1 - idx
        while idx < 0 or idx >= n:
            if idx < 0:
                idx = -idx - 1
            if idx >= n:
                idx = 2 * n - 1 - idx
        return idx

    @numba.njit(**NUMBA_NJIT_PARAMS)
    def _dwt_single_level_numba(
        signal: np.ndarray, lo_filter: np.ndarray, hi_filter: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-level DWT (convolve + decimate by 2) with pywt
        symmetric boundary.

        Output length matches ``pywt.dwt`` mode='symmetric':
        ``(len(signal) + filter_len - 1) // 2``.
        """
        n = signal.shape[0]
        flen = lo_filter.shape[0]
        out_len = (n + flen - 1) // 2
        approx = np.empty(out_len, dtype=np.float64)
        detail = np.empty(out_len, dtype=np.float64)
        for i in range(out_len):
            lo_sum = 0.0
            hi_sum = 0.0
            for j in range(flen):
                sig_idx = 2 * i + 1 - (flen - 1) + j
                sig_idx = _symmetric_reflect(sig_idx, n)
                v = signal[sig_idx]
                lo_sum += v * lo_filter[flen - 1 - j]
                hi_sum += v * hi_filter[flen - 1 - j]
            approx[i] = lo_sum
            detail[i] = hi_sum
        return approx, detail

    # NOTE: the legacy ``_wavedec_numba`` returned a
    # ``numba.typed.List`` of detail arrays. Building/consuming a
    # typed list at the Python/JIT boundary cost ~5-15 ms per call
    # (observed in production profiles: mean wavedec wrapper
    # latency ~300 ms when called on fresh-signature signals, vs
    # ~1 ms after warmup). The new wavedec() Python loop calls
    # ``_dwt_single_level_numba`` per level directly so no typed
    # list ever crosses the boundary. Kept the legacy function
    # below as ``_wavedec_numba_typedlist`` for any external caller
    # that needs the typed-list shape.

    @numba.njit(**NUMBA_NJIT_PARAMS)
    def _wavedec_numba_typedlist(
        signal: np.ndarray,
        lo_filter: np.ndarray,
        hi_filter: np.ndarray,
        max_level: int,
    ):
        """LEGACY: multi-level DWT returning a typed.List of details.
        Typed-list construction at the Python boundary is slow; new
        callers should use ``wavedec()`` which loops single-level
        kernels in Python. Retained for back-compat.
        """
        current = signal.copy()
        details = numba.typed.List()
        for _ in range(max_level):
            n = current.shape[0]
            flen = lo_filter.shape[0]
            out_len = (n + flen - 1) // 2
            if out_len < 1:
                break
            approx, detail = _dwt_single_level_numba(
                current, lo_filter, hi_filter,
            )
            details.append(detail)
            current = approx
        return current, details

    @numba.njit(**NUMBA_NJIT_PARAMS)
    def _idwt_single_level_numba(
        approx: np.ndarray,
        detail: np.ndarray,
        rec_lo: np.ndarray,
        rec_hi: np.ndarray,
        out_len: int,
    ) -> np.ndarray:
        """Single-level inverse DWT matching pywt.idwt mode='symmetric'.

        ``out_len`` is the desired reconstruction length (must equal the
        signal length at this level since DWT uses symmetric padding).
        Algorithm: upsample by 2 + convolve with reconstruction
        filters + sum approx and detail contributions, then trim
        boundary so the output length matches ``out_len``.
        """
        flen = rec_lo.shape[0]
        n = approx.shape[0]
        # Upsampled length before trim.
        up_len = 2 * n + flen - 2
        recon = np.zeros(up_len, dtype=np.float64)
        # Convolve each branch with its reconstruction filter at the
        # upsampled positions (2*k for k=0..n-1). pywt convention.
        for k in range(n):
            a = approx[k]
            d = detail[k]
            base = 2 * k
            for j in range(flen):
                recon[base + j] += a * rec_lo[j] + d * rec_hi[j]
        # Trim the boundary: pywt drops the first (flen-2) samples and
        # the trailing samples beyond out_len.
        start = flen - 2
        end = start + out_len
        if start < 0:
            start = 0
        if end > up_len:
            end = up_len
        return recon[start:end]

    # NOTE: legacy multi-level waverec that took a typed.List of
    # details. Same typed-list overhead as ``_wavedec_numba_typedlist``;
    # new callers should use ``waverec()`` Python wrapper which loops
    # single-level kernels in Python.

    @numba.njit(**NUMBA_NJIT_PARAMS)
    def _waverec_numba_typedlist(
        approx: np.ndarray,
        details_typed_list,
        rec_lo: np.ndarray,
        rec_hi: np.ndarray,
        target_lengths_per_level,
    ) -> np.ndarray:
        """LEGACY (typed-list inputs)."""
        current = approx
        n_levels = len(details_typed_list)
        for i in range(n_levels - 1, -1, -1):
            detail = details_typed_list[i]
            tlen = target_lengths_per_level[i]
            if current.shape[0] > detail.shape[0]:
                current = current[: detail.shape[0]]
            current = _idwt_single_level_numba(
                current, detail, rec_lo, rec_hi, tlen,
            )
        return current

    @numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
    def _wavedec_batched_njit(
        signals: np.ndarray,
        lo_filter: np.ndarray,
        hi_filter: np.ndarray,
        max_level: int,
        out_lengths: np.ndarray,
        out_approx: np.ndarray,
        out_details: np.ndarray,
    ) -> None:
        """@njit-parallel batched DWT for ``signals`` shape ``(N, T)``.

        Layout: ``out_lengths`` shape ``(max_level+1,)`` precomputed
        per-level output lengths (level 0 = signal length, level k =
        approximation length after k DWT steps); the caller computes
        these once based on T and the filter length. Output arrays:
          * ``out_approx`` shape ``(N, out_lengths[max_level])``
          * ``out_details`` shape ``(N, sum(out_lengths[1:]))`` -- a
            flat per-signal buffer in which level-1 details fill the
            first ``out_lengths[1]`` entries, level-2 the next
            ``out_lengths[2]``, etc.
        """
        N = signals.shape[0]
        T = signals.shape[1]
        flen = lo_filter.shape[0]
        for i in numba.prange(N):
            # Two ping-pong buffers sized for the worst-case (level 0)
            # signal length. Each thread allocates its own pair so the
            # prange has no shared mutable state. We can't write the
            # approx in place into ``current`` because the inner
            # convolution at row k may still need to read positions <k
            # (filter footprint extends behind the centre after
            # symmetric reflection).
            buf_a = signals[i].copy()
            buf_b = np.empty(T, dtype=np.float64)
            cur_len = T
            src_is_a = True
            offset = 0
            for level in range(max_level):
                out_len = (cur_len + flen - 1) // 2
                if out_len < 1:
                    break
                src = buf_a if src_is_a else buf_b
                dst = buf_b if src_is_a else buf_a
                for k in range(out_len):
                    lo_sum = 0.0
                    hi_sum = 0.0
                    for j in range(flen):
                        sig_idx = 2 * k + 1 - (flen - 1) + j
                        if sig_idx < 0:
                            sig_idx = -sig_idx - 1
                        if sig_idx >= cur_len:
                            sig_idx = 2 * cur_len - 1 - sig_idx
                        while sig_idx < 0 or sig_idx >= cur_len:
                            if sig_idx < 0:
                                sig_idx = -sig_idx - 1
                            if sig_idx >= cur_len:
                                sig_idx = 2 * cur_len - 1 - sig_idx
                        v = src[sig_idx]
                        lo_sum += v * lo_filter[flen - 1 - j]
                        hi_sum += v * hi_filter[flen - 1 - j]
                    out_details[i, offset + k] = hi_sum
                    dst[k] = lo_sum
                offset += out_lengths[level + 1]
                cur_len = out_len
                src_is_a = not src_is_a
            final = buf_a if src_is_a else buf_b
            for k in range(cur_len):
                out_approx[i, k] = final[k]


# ---------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------


def _precompute_out_lengths(
    signal_len: int, filter_len: int, max_level: int,
) -> List[int]:
    """Per-level output length given pywt 'symmetric' mode."""
    lengths = [signal_len]
    cur = signal_len
    for _ in range(max_level):
        cur = (cur + filter_len - 1) // 2
        lengths.append(cur)
        if cur < 1:
            break
    return lengths


def wavedec(
    signal: np.ndarray, wavelet: str, max_level: int,
) -> List[np.ndarray]:
    """Multi-level DWT decomposition of a single signal.

    Returns coefficients in ``pywt.wavedec`` format:
    ``[approx, detail_N, detail_{N-1}, ..., detail_1]``.

    Implementation: Python loop over ``_dwt_single_level_numba`` (one
    JIT call per level). Avoids ``numba.typed.List`` at the boundary
    -- typed-list construction at the Python/JIT seam paid
    ~5-15 ms per call on cold signatures (observed in production:
    50 distinct cold-signature signals -> 50 typed-list constructions
    -> wavelet wrapper mean 300 ms vs 1 ms after warmup). The per-level loop
    pays only the cached @njit dispatch overhead (~30 us each).
    """
    if not _HAS_NUMBA:
        import pywt
        return list(pywt.wavedec(signal, wavelet, level=max_level, mode="symmetric"))
    lo, hi, _, _ = get_wavelet_filters(wavelet)
    sig = np.ascontiguousarray(signal, dtype=np.float64)
    details_lvl1_first: list = []
    current = sig
    for _ in range(int(max_level)):
        n = current.shape[0]
        flen = lo.shape[0]
        out_len = (n + flen - 1) // 2
        if out_len < 1:
            break
        approx, detail = _dwt_single_level_numba(current, lo, hi)
        details_lvl1_first.append(detail)
        current = approx
    out = [current]
    for d in reversed(details_lvl1_first):
        out.append(d)
    return out


def wavedec_batched(
    signals: np.ndarray, wavelet: str, max_level: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel batched multi-level DWT.

    Parameters
    ----------
    signals : ndarray, shape ``(N, T)``
        N independent signals of length T.
    wavelet : str
        Wavelet name (e.g. ``"db4"``).
    max_level : int

    Returns
    -------
    approx : ndarray, shape ``(N, approx_len)``
    details_flat : ndarray, shape ``(N, sum_detail_len)``
        Flat layout: details[i, :out_lengths[1]] is level-1 detail of
        signal i; the next ``out_lengths[2]`` entries are level-2, etc.
    out_lengths : ndarray, shape ``(max_level + 1,)``
        ``out_lengths[0]`` = T; ``out_lengths[k]`` = approx length
        after k DWT steps. Use to slice ``details_flat`` into per-
        level arrays.
    """
    if not _HAS_NUMBA:
        raise ImportError("wavedec_batched requires numba; install it or fall back " "to a loop over wavedec().")
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2-D (N, T), got shape {signals.shape}")
    lo, hi, _, _ = get_wavelet_filters(wavelet)
    sig = np.ascontiguousarray(signals, dtype=np.float64)
    N, T = sig.shape
    flen = lo.shape[0]
    lengths = _precompute_out_lengths(T, flen, int(max_level))
    out_lengths = np.asarray(lengths, dtype=np.int64)
    approx_len = int(out_lengths[-1])
    detail_total = int(out_lengths[1:].sum())
    out_approx = np.zeros((N, approx_len), dtype=np.float64)
    out_details = np.zeros((N, detail_total), dtype=np.float64)
    _wavedec_batched_njit(
        sig, lo, hi, int(max_level), out_lengths, out_approx, out_details,
    )
    return out_approx, out_details, out_lengths


_DISPATCH_BATCHED_MIN_N = 64


def set_wavedec_dispatch_threshold(n_signals: int) -> None:
    """Update the row-count threshold at which ``wavedec_dispatch``
    switches from the Python-loop fallback to ``wavedec_batched``.

    Default ``_DISPATCH_BATCHED_MIN_N=64`` was picked from a micro-bench
    on a typical numba-enabled host; callers on slower CPUs can raise
    this to defer the batched path further.
    """
    global _DISPATCH_BATCHED_MIN_N
    _DISPATCH_BATCHED_MIN_N = int(n_signals)


def wavedec_dispatch(
    signals: np.ndarray, wavelet: str, max_level: int,
) -> Union[List[np.ndarray], List[List[np.ndarray]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Auto-pick between single-signal loop and batched parallel DWT.

    1-D input -> single ``wavedec``. 2-D input with N below threshold
    -> Python loop over ``wavedec``. N >= threshold -> ``wavedec_batched``.

    CAVEAT (no current callers, so left as-is rather than a silent behavior change):
    the batched branch returns ``wavedec_batched``'s flat ``(approx, details_flat,
    out_lengths)`` tuple shape, which is NOT the same shape as the other two branches'
    ``list``-of-arrays coefficient format -- a caller relying on this dispatcher across
    the ``_DISPATCH_BATCHED_MIN_N`` threshold would see the output shape change under
    it. The return annotation above is now honest about this; unifying the shapes is a
    real behavioral decision for whoever adds the first caller, not a type-only fix.
    """
    if signals.ndim == 1:
        return wavedec(signals, wavelet, max_level)
    if signals.shape[0] < _DISPATCH_BATCHED_MIN_N or not _HAS_NUMBA:
        return [wavedec(signals[i], wavelet, max_level) for i in range(signals.shape[0])]
    return wavedec_batched(signals, wavelet, max_level)


def waverec(coeffs: List[np.ndarray], wavelet: str) -> np.ndarray:
    """Multi-level inverse DWT matching ``pywt.waverec`` mode='symmetric'.

    ``coeffs`` is the same shape as ``wavedec`` output:
    ``[approx, detail_N, ..., detail_1]``. The reconstruction target
    length at each level is derived from the detail coefficient
    length at that level (pywt convention: reconstructed length = 2 *
    detail_len - filter_len + 2, then trimmed by symmetric boundary).

    Implementation: Python loop over ``_idwt_single_level_numba``
    (one JIT call per level reconstruction). Avoids
    ``numba.typed.List`` at the boundary, matching the wavedec()
    rewrite -- see its docstring for the typed-list overhead
    rationale.
    """
    if not _HAS_NUMBA:
        import pywt
        return np.asarray(pywt.waverec(coeffs, wavelet, mode="symmetric"))
    _, _, rec_lo, rec_hi = get_wavelet_filters(wavelet)
    approx = np.ascontiguousarray(coeffs[0], dtype=np.float64)
    # Order details level-1-first: pywt returns
    # ``[approx, detail_N, ..., detail_1]``; we iterate level-N-first
    # to N=1 by walking coeffs[1:] in order.
    flen = rec_lo.shape[0]
    current = approx
    for d_any in coeffs[1:]:
        detail = np.ascontiguousarray(d_any, dtype=np.float64)
        # Detail may be 1 shorter than current approx for odd boundary
        # cases; truncate approx to match detail length before idwt.
        if current.shape[0] > detail.shape[0]:
            current = current[: detail.shape[0]]
        target_len = 2 * detail.shape[0] - flen + 2
        current = _idwt_single_level_numba(
            current, detail, rec_lo, rec_hi, target_len,
        )
    return current


def wavelet_denoise(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold: float | None = None,
    mode: str = "soft",
) -> np.ndarray:
    """Wavelet denoise: ``wavedec`` -> threshold details -> ``waverec``.

    A common geophysical / signal denoising recipe. If ``threshold``
    is None, uses VisuShrink universal threshold
    ``sigma * sqrt(2 * ln(n))`` where sigma is estimated from the
    finest-level details via the MAD estimator (sigma = MAD / 0.6745).

    Parameters
    ----------
    signal : 1-D ndarray
    wavelet : str
    level : int
    threshold : float or None
    mode : {"soft", "hard"}
    """
    coeffs = wavedec(signal, wavelet, level)
    # finest detail = last element in coeffs (level 1)
    finest = coeffs[-1]
    if threshold is None:
        sigma = np.median(np.abs(finest)) / 0.6745
        if sigma <= 0:
            return signal.astype(np.float64, copy=True)
        n = len(signal)
        threshold = sigma * np.sqrt(2.0 * np.log(n))
    new_coeffs = [coeffs[0]]
    for d in coeffs[1:]:
        if mode == "soft":
            d2 = np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0)
        elif mode == "hard":
            d2 = np.where(np.abs(d) >= threshold, d, 0.0)
        else:
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
        new_coeffs.append(d2)
    return waverec(new_coeffs, wavelet)
