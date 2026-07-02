"""GPU-RESIDENT twin of the multi-frequency Fourier detector
(:func:`_orth_extra_basis_fe._detect_fourier_freqs_for_col`).

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF.
On this GTX 1050 Ti the detector's operand is small (<=66k subsampled train
slice) and the body is a sequential deflation loop, so the GPU twin is EXPECTED
to be slower than the fused njit CPU path -- and that is a PASS by the residency
contract. The bench-note in the CPU detector documents why it stays CPU for the
WALL; this twin exists for RESIDENCY COMPLETENESS so that, under the resident
flag, the column operand + target stay resident on the device and the per-stage
vector math (coarse periodogram plane, refine scans, held-out confirm,
least-squares deflation, polynomial detrend) runs on cupy with NO per-iteration
bulk (n-scaled) H2D/D2H.

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT: z_tr/z_va/y_tr/y_va columns, the (nf, n) coarse sin/cos plane, all
    centered dot products, the refine-scan power vectors, the 3-column lstsq
    deflation, the cubic detrend -- one bulk H2D of the 4 columns at entry, then
    no n-scaled transfer until return.
  * HOST scalar D2H (bounded, O(max_freqs + grid + refine_steps)): the coarse
    argmax index, the refined-freq value, the held-out gate compare, the
    std/stop scalars. These are tiny (< BULK_BYTES) and the residency audit
    classifies them as scalar, not bulk.

Selection-equivalence: the returned frequency list matches the CPU detector
within the coarse-grid tolerance (the downstream skip/lock test uses a 0.25
window and ``_refine_peak_freq`` re-localises the coarse argmax, so single-ULP
reduction-order shifts between cupy and the njit kernels do not move the result).

Any cupy / device error falls back to the CPU njit detector, so the default
(flag-off) path stays byte-identical and a GPU fault never breaks a fit.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_TWO_PI = 2.0 * np.pi


# HOST-preamble memoisation (2026-07-02). The seeded held-out split and the seeded row-subsample indices are a
# pure deterministic function of ``n`` (seed 0) / ``(size, cap)`` (seed 0xF0F0_1234) -- IDENTICAL for every column
# the detector scans in a fit (same 1M n, same train/val sizes). Recomputing ``default_rng(0).permutation(n)`` +
# the ``rng.choice`` subsample on every column was ~30 full 1M-row shuffles per fit (F2 host hotspot: this preamble
# is 0.285s tottime / 34 calls). Cache them so the shuffle/choice runs ONCE per distinct shape. The cached arrays
# are used read-only (boolean-mask indexing / fancy-index gather -- never mutated), so returning the shared object
# is byte-identical to a fresh recompute. Bounded: keyed by scalar shape, a handful of entries per process.
_SPLIT_MASK_CACHE: dict = {}
_SUBSAMPLE_IDX_CACHE: dict = {}


def _seeded_split_masks(n: int):
    """(train_mask, val_mask) for the seed-0 permutation held-out split -- memoised by ``n``. Byte-identical to
    recomputing ``default_rng(0).permutation(n)`` each call (deterministic), read-only for the caller."""
    cached = _SPLIT_MASK_CACHE.get(n)
    if cached is None:
        val_mask = np.zeros(n, dtype=bool)
        val_mask[np.random.default_rng(0).permutation(n)[: n // 3]] = True
        cached = (~val_mask, val_mask)
        _SPLIT_MASK_CACHE[n] = cached
    return cached


def _seeded_subsample_idx(tr_size: int, cap: int, va_size: int, va_cap: int):
    """(sub_tr, sub_va) row-subsample gather indices for the seed-0xF0F0_1234 cap -- memoised by the shape tuple.

    Reproduces the ORIGINAL single-generator sequence EXACTLY: one ``default_rng(0xF0F0_1234)`` draws ``sub_tr``
    first and then (only when ``va_size > va_cap``) ``sub_va`` from the SAME advanced state -- NOT two fresh seeds
    (which would give a different ``sub_va``). ``sub_va`` is None when the val slice is not subsampled. Byte-identical
    to recomputing per call, read-only gather indices for the caller."""
    key = (tr_size, cap, va_size, va_cap)
    cached = _SUBSAMPLE_IDX_CACHE.get(key)
    if cached is None:
        _rng = np.random.default_rng(0xF0F0_1234)
        sub_tr = _rng.choice(tr_size, size=cap, replace=False)
        sub_va = _rng.choice(va_size, size=va_cap, replace=False) if va_size > va_cap else None
        cached = (sub_tr, sub_va)
        _SUBSAMPLE_IDX_CACHE[key] = cached
    return cached


def _corr_sq_centered_gpu(cp, v, yc, y_ss: float) -> float:
    """Squared Pearson correlation of resident ``v`` with a pre-centered resident
    ``yc`` (sum-of-squares ``y_ss``), mirroring the CPU ``_corr_sq_centered``
    raw-moment form ``v_ss = v@v - sum(v)^2/n``; ``num = v@yc`` (identity-equal to
    centered because yc sums to zero). Returns a host float (bounded scalar D2H).
    Same relative degeneracy guard as the CPU path."""
    n = v.shape[0]
    # Stack the three independent reductions and read them back in ONE D2H (was three separate float() syncs).
    sv, vv, vy = (float(_x) for _x in cp.asnumpy(cp.stack([cp.sum(v), cp.dot(v, v), cp.dot(v, yc)])))
    v_ss = vv - sv * sv / n
    if v_ss <= 1e-12 * vv or v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    return (vy * vy) / (v_ss * y_ss)


def _power_centered_gpu(cp, z, yc, y_ss: float, freq: float) -> float:
    """Periodogram power at ``freq`` against pre-centered resident ``yc`` -- GPU
    twin of ``_power_centered`` / ``_power_centered_fused_par_njit``. sin/cos
    planes are resident; only the final scalar power comes back."""
    ang = (_TWO_PI * float(freq)) * z
    s = cp.sin(ang)
    c = cp.cos(ang)
    return _corr_sq_centered_gpu(cp, s, yc, y_ss) + _corr_sq_centered_gpu(cp, c, yc, y_ss)


def _power_grid_centered_gpu(cp, z, yc, y_ss: float, freqs_dev):
    """Periodogram power against pre-centered resident ``yc`` for EVERY frequency in
    resident ``freqs_dev`` (F,) in ONE batched pass -- the (F, n) sin/cos planes and
    all centered dots run as matmuls, returning a resident (F,) power vector (no
    per-frequency scalar D2H). Selection-equivalent to calling ``_power_centered_gpu``
    per frequency: same raw-moment v_ss = v@v - sum(v)^2/n, same num = v@yc, same
    relative degeneracy guard, summed over the sin + cos planes."""
    n = z.shape[0]
    ang = (_TWO_PI * freqs_dev)[:, None] * z[None, :]      # (F, n)
    s = cp.sin(ang)
    c = cp.cos(ang)

    def _corr_sq_rows(plane):
        sv = plane.sum(axis=1)                              # (F,)
        vv = (plane * plane).sum(axis=1)                    # (F,)
        vy = plane @ yc                                     # (F,)
        v_ss = vv - sv * sv / n
        ok = (v_ss > 1e-12 * vv) & (v_ss >= 1e-24)
        denom = v_ss * y_ss
        val = cp.where(denom > 0.0, (vy * vy) / cp.where(denom > 0.0, denom, 1.0), 0.0)
        return cp.where(ok & (y_ss >= 1e-24), val, 0.0)

    return _corr_sq_rows(s) + _corr_sq_rows(c)


def _refine_peak_freq_gpu(cp, z_tr, yc, y_ss: float, coarse_f: float) -> float:
    """GPU twin of ``_refine_peak_freq``: two-stage local scan (+-0.25 @ 0.05,
    then +-0.05 @ 0.0125) maximising resident periodogram power. Each scan's
    candidate frequencies (the center + the swept grid) are powered in ONE batched
    matmul pass and the argmax is a single scalar D2H -- selection-equivalent to the
    per-frequency scan (same powers, ``cp.argmax`` first-max ties matching the
    original strictly-greater earliest-wins order)."""
    def _scan(center: float, half_width: float, step: float):
        lo_r = max(0.05, center - half_width)
        hi_r = center + half_width
        n_steps = int(round((hi_r - lo_r) / step)) + 1
        # freq order = [center, lo_r+0*step, ..., lo_r+(n_steps-1)*step]: the center is
        # the initial best, then strictly-greater earliest-wins -> argmax first-max.
        freqs_host = np.empty(n_steps + 1, dtype=np.float64)
        freqs_host[0] = center
        freqs_host[1:] = lo_r + step * np.arange(n_steps, dtype=np.float64)
        freqs_dev = cp.asarray(freqs_host)
        powers = _power_grid_centered_gpu(cp, z_tr, yc, y_ss, freqs_dev)
        bi = int(cp.argmax(powers))                        # single scalar D2H per scan
        return float(freqs_host[bi]), 0.0
    f1, _ = _scan(coarse_f, 0.25, 0.05)
    f2, _ = _scan(f1, 0.05, 0.0125)
    return float(f2)


def _deflate_sincos_gpu(cp, z, y, freq: float):
    """Residual of resident ``y`` after least-squares projection onto resident
    ``[1, sin, cos]`` -- GPU twin of ``_deflate_sincos``. Normal-equations solve
    on the 3-col design (lstsq fallback on singular A^T A). Stays resident: the
    returned residual is a device array, no D2H of y."""
    ang = (_TWO_PI * float(freq)) * z
    A = cp.stack([cp.ones_like(z), cp.sin(ang), cp.cos(ang)], axis=1)  # (n, 3)
    try:
        AtA = A.T @ A
        Aty = A.T @ y
        try:
            coef = cp.linalg.solve(AtA, Aty)
        except np.linalg.LinAlgError:
            coef = cp.linalg.lstsq(A, y, rcond=None)[0]
        return y - A @ coef
    except np.linalg.LinAlgError:
        # Degenerate [1,sin,cos] design (singular even for lstsq) -> no deflation this freq (graceful).
        # Narrowed from bare Exception: a device fault now propagates to the dispatch-site CPU fallback and a
        # logic bug surfaces, instead of silently returning an UN-DEFLATED y (which would diverge from CPU).
        return y


def _vander4_gpu(cp, z):
    """[z^3, z^2, z, 1] resident, matching np.vander(z, 4) column order."""
    z2 = z * z
    return cp.stack([z2 * z, z2, z, cp.ones_like(z)], axis=1)


def detect_fourier_freqs_for_col_gpu(
    z01: np.ndarray,
    y: np.ndarray,
    *,
    f_grid: Sequence[float],
    min_val_corr: float = 0.15,
    min_rows: int = 800,
    max_freqs: int = 4,
    fourier_detect_max_n: int = 0,
) -> list:
    """GPU-resident multi-frequency Fourier detector. Returns the SAME list of
    detected z-space frequencies as the CPU :func:`_detect_fourier_freqs_for_col`
    (within the coarse-grid tolerance). Raises on any cupy/device error so the
    caller falls back to the CPU njit path.

    The host-side preamble (finite/std guards, the SEEDED held-out split, the
    optional row-subsample cap) mirrors the CPU detector EXACTLY using numpy +
    the identical RNG seeds, so the resident path operates on the byte-identical
    (z_tr, z_va, y_tr, y_va) the CPU path would. Only AFTER that split are the
    four columns uploaded ONCE to the device; the deflation loop is then fully
    resident."""
    import cupy as cp

    z01 = np.asarray(z01, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = z01.size
    if n != y.size or n < int(min_rows):
        return []
    if not np.all(np.isfinite(z01)) or not np.all(np.isfinite(y)):
        return []
    if float(np.std(z01)) < 1e-12 or float(np.std(y)) < 1e-12:
        return []
    grid = [float(f) for f in f_grid if float(f) > 0.0]
    if not grid:
        return []
    # SEEDED held-out split -- IDENTICAL RNG seed/recipe to the CPU detector so the resident path operates on
    # the byte-identical train/val rows (selection-equivalence depends on this matching exactly).
    train_mask, val_mask = _seeded_split_masks(n)
    z_tr_h, z_va_h = z01[train_mask], z01[val_mask]
    y_tr_h = y[train_mask].copy()
    y_va_h = y[val_mask].copy()
    if z_tr_h.size < 16 or z_va_h.size < 8:
        return []
    # Row-subsample cap -- IDENTICAL seed/recipe to the CPU detector.
    _fdet_cap = int(fourier_detect_max_n)
    if _fdet_cap > 0 and z_tr_h.size > _fdet_cap:
        _va_cap = max(8, _fdet_cap // 2)
        _sub_tr, _sub_va = _seeded_subsample_idx(z_tr_h.size, _fdet_cap, z_va_h.size, _va_cap)
        z_tr_h = np.ascontiguousarray(z_tr_h[_sub_tr]); y_tr_h = np.ascontiguousarray(y_tr_h[_sub_tr])
        if _sub_va is not None:
            z_va_h = np.ascontiguousarray(z_va_h[_sub_va]); y_va_h = np.ascontiguousarray(y_va_h[_sub_va])
    if float(np.std(y_tr_h)) < 1e-12 or float(np.std(y_va_h)) < 1e-12:
        return []

    # ---- ONE bulk H2D of the 4 columns; everything below stays resident -------------------------------
    # z_tr/z_va are the per-column z split -> distinct, a genuine upload each. y_tr/y_va are the FIXED held-out
    # target split (the seed-0 val mask + seed-0xF0F0 row-subsample above are identical for EVERY column the
    # detector scans), so they re-uploaded once per column (H2D audit: 34x). Route them through the content-keyed
    # resident cache so the target split uploads ONCE; read-only here (line below reassigns y_tr/y_va to fresh
    # detrended arrays rather than mutating the cached buffer) -> selection-equivalent.
    from .._fe_resident_operands import resident_operand
    # Under MLFRAME_CRIT_DTYPE_RELAXED (default ON) the whole per-column fourier detect runs in FLOAT32: the two
    # z splits + the held-out target splits upload as f32 (half the H2D) and the polynomial detrend + periodogram
    # peak-pick compute in f32 (faster, no widening). The f32 rounding (~1e-6) is far below the FE decision
    # margin so the detected frequencies + the downstream FE selection are unchanged (validated on F2 across all
    # distributions + the extra-basis / robust-basis biz suites). MLFRAME_CRIT_DTYPE_RELAXED=0 restores strict f64.
    try:
        from .._fe_gpu_batch._devices import crit_float_dtype
        _zdt = crit_float_dtype()
    except Exception:
        _zdt = cp.float64
    z_tr = cp.asarray(np.ascontiguousarray(np.asarray(z_tr_h, dtype=_zdt)))
    z_va = cp.asarray(np.ascontiguousarray(np.asarray(z_va_h, dtype=_zdt)))
    y_tr = resident_operand(y_tr_h, "fourier_y_tr", dtype=_zdt)
    y_va = resident_operand(y_va_h, "fourier_y_va", dtype=_zdt)

    # POLYNOMIAL DETREND (cubic in z, train-fit / val-applied) -- resident.
    _V_tr = _vander4_gpu(cp, z_tr)
    try:
        try:
            _poly_coef = cp.linalg.solve(_V_tr.T @ _V_tr, _V_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            _poly_coef = cp.linalg.lstsq(_V_tr, y_tr, rcond=None)[0]
        y_tr = y_tr - _V_tr @ _poly_coef
        y_va = y_va - _vander4_gpu(cp, z_va) @ _poly_coef
    except np.linalg.LinAlgError:
        # Singular cubic Vandermonde (constant/degenerate z) -> skip detrend, run detection on raw y (graceful).
        # Narrowed from bare Exception so a device fault propagates to the dispatch CPU fallback and a logic bug
        # surfaces instead of silently skipping the detrend (which would diverge from the CPU detector).
        pass
    _std_tr, _std_va = cp.asnumpy(cp.stack([cp.std(y_tr), cp.std(y_va)]))   # one D2H for the pair
    if _std_tr < 1e-9 or _std_va < 1e-9:
        return []

    _eff_min_val_corr = max(float(min_val_corr), 0.30)

    # Coarse-grid sin/cos plane on TRAIN, built ONCE (depends only on z). Resident (nf, n).
    grid_dev = cp.asarray(np.asarray(grid, dtype=np.float64))          # tiny H2D (grid is O(48))
    ang_plane = (_TWO_PI * grid_dev)[:, None] * z_tr[None, :]          # (nf, n)
    sin_plane = cp.sin(ang_plane)
    cos_plane = cp.cos(ang_plane)
    sin_mean = sin_plane.mean(axis=1, keepdims=True)
    cos_mean = cos_plane.mean(axis=1, keepdims=True)
    sc_plane = sin_plane - sin_mean                                    # centered (nf, n)
    cc_plane = cos_plane - cos_mean
    s_ss_vec = (sc_plane * sc_plane).sum(axis=1)                       # (nf,)
    c_ss_vec = (cc_plane * cc_plane).sum(axis=1)

    out: list = []
    for _ in range(max(1, int(max_freqs))):
        # std_tr, std_va and y_ss (the three per-iteration guard scalars) in ONE D2H. yc is cheap and independent
        # of the std guard, so computing it first to fold y_ss into the same pull costs nothing.
        yc = y_tr - y_tr.mean()
        _s_tr, _s_va, y_ss = (float(_v) for _v in cp.asnumpy(cp.stack([cp.std(y_tr), cp.std(y_va), cp.dot(yc, yc)])))
        if _s_tr < 1e-9 or _s_va < 1e-9:
            break
        if y_ss < 1e-24:
            break
        # Coarse peak-pick: batched matvec over the resident plane -> power per grid freq, scalar argmax.
        num_s = sc_plane @ yc                                          # (nf,)
        num_c = cc_plane @ yc
        p_s = cp.where(s_ss_vec >= 1e-24, (num_s * num_s) / (s_ss_vec * y_ss), 0.0)
        p_c = cp.where(c_ss_vec >= 1e-24, (num_c * num_c) / (c_ss_vec * y_ss), 0.0)
        power_vec = p_s + p_c                                          # (nf,)
        best_gi = int(cp.argmax(power_vec))                            # scalar D2H
        best_f = grid[best_gi]
        refined_f = _refine_peak_freq_gpu(cp, z_tr, yc, y_ss, best_f)
        if any(abs(refined_f - g) < 0.25 for g in out):
            y_tr = _deflate_sincos_gpu(cp, z_tr, y_tr, refined_f)
            y_va = _deflate_sincos_gpu(cp, z_va, y_va, refined_f)
            continue
        # Held-out confirm: center y_va resident, scalar power back.
        yvc = y_va - y_va.mean()
        yv_ss = float(cp.dot(yvc, yvc))
        if yv_ss < 1e-24:
            break
        val_power = _power_centered_gpu(cp, z_va, yvc, yv_ss, refined_f)
        # Held-out admission razor (not grid-snapped): the cupy-vs-numpy reduction-order delta (~1e-12) is far
        # below the >=0.30 _eff_min_val_corr margin, so it cannot flip this gate in practice; F2 + the fourier
        # parity suite confirm the resident detector returns the same frequency list as CPU. Accept-and-documented.
        if val_power <= 0.0 or np.sqrt(val_power) < _eff_min_val_corr:
            break
        out.append(float(refined_f))
        y_tr = _deflate_sincos_gpu(cp, z_tr, y_tr, refined_f)
        y_va = _deflate_sincos_gpu(cp, z_va, y_va, refined_f)
    return out
