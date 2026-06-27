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

import numpy as np

_TWO_PI = 2.0 * np.pi


def _corr_sq_centered_gpu(cp, v, yc, y_ss: float) -> float:
    """Squared Pearson correlation of resident ``v`` with a pre-centered resident
    ``yc`` (sum-of-squares ``y_ss``), mirroring the CPU ``_corr_sq_centered``
    raw-moment form ``v_ss = v@v - sum(v)^2/n``; ``num = v@yc`` (identity-equal to
    centered because yc sums to zero). Returns a host float (bounded scalar D2H).
    Same relative degeneracy guard as the CPU path."""
    n = v.shape[0]
    sv = float(cp.sum(v))
    vv = float(cp.dot(v, v))
    vy = float(cp.dot(v, yc))
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


def _refine_peak_freq_gpu(cp, z_tr, yc, y_ss: float, coarse_f: float) -> float:
    """GPU twin of ``_refine_peak_freq``: two-stage local scan (+-0.25 @ 0.05,
    then +-0.05 @ 0.0125) maximising resident periodogram power. Scalar argmax."""
    def _scan(center: float, half_width: float, step: float):
        lo_r = max(0.05, center - half_width)
        hi_r = center + half_width
        n_steps = int(round((hi_r - lo_r) / step)) + 1
        best_f = center
        best_p = _power_centered_gpu(cp, z_tr, yc, y_ss, center)
        for k in range(n_steps):
            f = lo_r + step * k
            p = _power_centered_gpu(cp, z_tr, yc, y_ss, f)
            if p > best_p:
                best_p = p
                best_f = f
        return best_f, best_p
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
        except Exception:
            coef = cp.linalg.lstsq(A, y, rcond=None)[0]
        return y - A @ coef
    except Exception:
        return y


def _vander4_gpu(cp, z):
    """[z^3, z^2, z, 1] resident, matching np.vander(z, 4) column order."""
    z2 = z * z
    return cp.stack([z2 * z, z2, z, cp.ones_like(z)], axis=1)


def detect_fourier_freqs_for_col_gpu(
    z01: np.ndarray,
    y: np.ndarray,
    *,
    f_grid,
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
    val_mask = np.zeros(n, dtype=bool)
    val_mask[np.random.default_rng(0).permutation(n)[: n // 3]] = True
    train_mask = ~val_mask
    z_tr_h, z_va_h = z01[train_mask], z01[val_mask]
    y_tr_h = y[train_mask].copy()
    y_va_h = y[val_mask].copy()
    if z_tr_h.size < 16 or z_va_h.size < 8:
        return []
    # Row-subsample cap -- IDENTICAL seed/recipe to the CPU detector.
    _fdet_cap = int(fourier_detect_max_n)
    if _fdet_cap > 0 and z_tr_h.size > _fdet_cap:
        _fdet_rng = np.random.default_rng(0xF0F0_1234)
        _sub_tr = _fdet_rng.choice(z_tr_h.size, size=_fdet_cap, replace=False)
        z_tr_h = np.ascontiguousarray(z_tr_h[_sub_tr]); y_tr_h = np.ascontiguousarray(y_tr_h[_sub_tr])
        _va_cap = max(8, _fdet_cap // 2)
        if z_va_h.size > _va_cap:
            _sub_va = _fdet_rng.choice(z_va_h.size, size=_va_cap, replace=False)
            z_va_h = np.ascontiguousarray(z_va_h[_sub_va]); y_va_h = np.ascontiguousarray(y_va_h[_sub_va])
    if float(np.std(y_tr_h)) < 1e-12 or float(np.std(y_va_h)) < 1e-12:
        return []

    # ---- ONE bulk H2D of the 4 columns; everything below stays resident -------------------------------
    z_tr = cp.asarray(z_tr_h)
    z_va = cp.asarray(z_va_h)
    y_tr = cp.asarray(y_tr_h)
    y_va = cp.asarray(y_va_h)

    # POLYNOMIAL DETREND (cubic in z, train-fit / val-applied) -- resident.
    _V_tr = _vander4_gpu(cp, z_tr)
    try:
        try:
            _poly_coef = cp.linalg.solve(_V_tr.T @ _V_tr, _V_tr.T @ y_tr)
        except Exception:
            _poly_coef = cp.linalg.lstsq(_V_tr, y_tr, rcond=None)[0]
        y_tr = y_tr - _V_tr @ _poly_coef
        y_va = y_va - _vander4_gpu(cp, z_va) @ _poly_coef
    except Exception:
        pass
    if float(cp.std(y_tr)) < 1e-9 or float(cp.std(y_va)) < 1e-9:
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
        if float(cp.std(y_tr)) < 1e-9 or float(cp.std(y_va)) < 1e-9:
            break
        yc = y_tr - y_tr.mean()
        y_ss = float(cp.dot(yc, yc))
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
        if val_power <= 0.0 or np.sqrt(val_power) < _eff_min_val_corr:
            break
        out.append(float(refined_f))
        y_tr = _deflate_sincos_gpu(cp, z_tr, y_tr, refined_f)
        y_va = _deflate_sincos_gpu(cp, z_va, y_va, refined_f)
    return out
