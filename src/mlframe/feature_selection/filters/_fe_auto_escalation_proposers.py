"""Candidate proposers of the FE auto-escalation: the signal-adaptive orthogonal-polynomial warp and the demodulated adaptive-frequency Fourier / chirp warp.

Carved out of ``_fe_auto_escalation``, which re-exports every name so the historical import path keeps working.
"""

from __future__ import annotations

import logging
import numpy as np

from mlframe.feature_selection.filters._safe_scale import guarded_scale, scale_is_usable

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Signal-adaptive poly basis routing: try all four shipped families, best by held-out
# reconstruction |corr|. Chebyshev first (the production prewarp default).
_ESCALATION_POLY_BASES = ("chebyshev", "hermite", "legendre", "laguerre")


# Coarse z-space frequency grids - VERBATIM the shipped univariate adaptive grids
# (``_orth_extra_basis_fe``): linear axis 0.5..8.0, chirp axis 0.5..24.0.
_ADAPTIVE_F_GRID = tuple(0.5 * k for k in range(1, 17))


_CHIRP_F_GRID = tuple(0.5 * k for k in range(1, 49))


# Identity warp degree for the mate operand (coef [0, 1] -> the basis' affine z map).
_IDENTITY_BASIS = "chebyshev"


def _finite_filled(x: np.ndarray) -> np.ndarray:
    """Copy of ``x`` with non-finite entries replaced by the finite mean (0.0 when no
    finite entries). Used ONLY for detector / ALS FITTING; candidate VALUES are always
    computed from the RAW column through the replay path (warp -> mul -> nan_to_num),
    so fit-time and transform-time values agree bit-for-bit."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.all():
        return np.asarray(x)
    fill = float(np.mean(x[finite])) if finite.any() else 0.0
    out = x.copy()
    out[~finite] = fill
    return np.asarray(out)


def _identity_prewarp_spec(x: np.ndarray) -> dict | None:
    """Closed-form IDENTITY prewarp spec for the mate operand: chebyshev degree-1 with
    coef [0, 1] evaluates to the basis' affine z-map of x (replayed by
    ``apply_operand_prewarp`` exactly like any learned warp). An affine map of the mate
    keeps the product's MI/correlation structure intact while staying on the standard
    ``prewarp`` recipe path (no new pseudo-unary needed)."""
    from mlframe.feature_selection.filters.hermite_fe.shared import POLY_BASES as _POLY_BASES
    bi = _POLY_BASES[_IDENTITY_BASIS]
    xf = _finite_filled(x)
    if float(np.std(xf)) < 1e-12:
        return None
    _, params = bi["fit"](xf)
    coef = np.zeros(2, dtype=np.float64)
    coef[1] = 1.0
    return {"basis": _IDENTITY_BASIS, "degree": 1, "coef": coef, "preprocess": dict(params)}


def _candidate_values(x_a: np.ndarray, spec_a: dict, x_b: np.ndarray, spec_b: dict) -> np.ndarray | None:
    """Replay-exact candidate column: ``nan_to_num(mul(prewarp_a(x_a), prewarp_b(x_b)))``
    - the same chain ``_apply_unary_binary`` executes at transform() time."""
    from .hermite_fe import apply_operand_prewarp
    try:
        wa = apply_operand_prewarp(np.asarray(x_a, dtype=np.float64), spec_a)
        wb = apply_operand_prewarp(np.asarray(x_b, dtype=np.float64), spec_b)
    except Exception as exc:
        # Was unlogged, unlike every other except-block in
        # this file (which all log via logger.debug); low blast radius (degrades to skipping this
        # escalation candidate, per the file's own "never raises" design), but a real regression in
        # apply_operand_prewarp would be invisible in logs otherwise.
        logger.debug("_candidate_values: apply_operand_prewarp failed; skipping candidate: %r", exc)
        return None
    out = np.multiply(wa, wb)
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(out)) or float(np.std(out)) < 1e-12:
        return None
    return np.asarray(out)


def _heldout_abs_corr(vals_va, y_va) -> float:
    """Absolute correlation between candidate values ``vals_va`` and the held-out (mean-centered) target ``y_va``, sanitizing non-finite inputs and returning 0.0 for a constant/degenerate candidate."""
    vals_va = np.nan_to_num(np.asarray(vals_va, dtype=np.float64), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.std(vals_va)) < 1e-12:
        return 0.0
    cc = float(np.corrcoef(vals_va, y_va)[0, 1])
    return abs(cc) if np.isfinite(cc) else 0.0


def _best_pair_basis(_blocks_a, _blocks_b, y_tr, y_va, xa_tr, xb_tr):
    """Basis whose rank-1 ALS pair reconstruction has the best held-out |corr|; returns ``(basis_or_None, corr)``."""
    from .hermite_fe import warm_start_als_seed

    best_corr = -1.0
    best_basis = None
    for basis in _ESCALATION_POLY_BASES:
        try:
            blk_a = _blocks_a.get(basis); blk_b = _blocks_b.get(basis)
            if blk_a is None or blk_b is None:
                continue
            Ba_tr, Ba_va, za_tr = blk_a; Bb_tr, Bb_va, zb_tr = blk_b
            # Rank-1 ALS pair warp = fit_pair_prewarp_als' solve on the SAME basis
            # matrices (warm_start_als_seed; OLS-ALS, no robustify - matches legacy).
            # z_a/z_b/basis are passed too so the resident GPU branch takes the
            # DEVICE-BORN path (warm_start_als_seed_gpu_from_z): Ba/Bb are rebuilt on
            # device from these standardised columns (the SAME basis_fit the prebuilt
            # Ba_tr/Bb_tr used), collapsing the ~358MB design H2D at 300k. Ba_tr/Bb_tr
            # still feed the byte-identical CPU fallback (default, flag-off) path.
            coef_a, coef_b = warm_start_als_seed(Ba_tr, Bb_tr, y_tr, iters=3, x_a=xa_tr, x_b=xb_tr, z_a=za_tr, z_b=zb_tr, basis=basis)
            if coef_a is None or coef_b is None:
                continue
            c = _heldout_abs_corr((Ba_va @ np.ascontiguousarray(coef_a, dtype=np.float64)) * (Bb_va @ np.ascontiguousarray(coef_b, dtype=np.float64)), y_va)
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            continue
        if c > best_corr:
            best_corr = c
            best_basis = basis
    return best_basis, best_corr


def _propose_poly(x_a, x_b, y_f, *, degree: int, min_val_corr: float, pairness_margin: float = 1.15):
    """Signal-adaptive orth-poly proposer: rank-1 ALS pair warp per shipped basis,
    held-out stride validation of the rank-1 reconstruction, best basis wins.

    PAIR-NESS GUARD: the rank-1 PAIR reconstruction's held-out |corr| must beat the
    best SINGLE-OPERAND warp's held-out |corr| by ``pairness_margin`` (default 1.15,
    mirroring ``fe_synergy_min_prevalence``). Without it, a (genuine-marginal x noise)
    cross-mix pair passes trivially - the ALS collapses the noise side to ~constant
    and the "pair" reconstruction is just a wrapped univariate trend (measured on the
    weak F2: 6 cross-mix wrappers admitted without the guard, 0 with it; the genuine
    product terms keep ratios >= 1.5 because no single operand carries the product).

    Returns ``(spec_a, spec_b, basis, val_corr)`` or ``None`` (no basis generalises /
    the pair adds nothing over its best single operand).

    PERF: the basis matrices for ``xa[tr]`` / ``xb[tr]`` are built ONCE
    per (operand, basis) and SHARED by the single-operand OLS baseline AND the pair
    ALS sweep, instead of legacy's ``fit_operand_prewarp`` + ``fit_pair_prewarp_als``
    each rebuilding them per basis (z-map cached per FAMILY - cheb+leg share min-max).
    Inlines the SAME library solves (``fit_basis_coef_robust`` single, ``warm_start_als_seed``
    pair) so held-out corr / coefficients match the legacy calls; selection-equivalent
    (interleaved isolated A/B on the canonical n=100k first-escalation call: same 8
    eligible pairs, same 0 proposed; OLD median 4.382s -> NEW 4.083s, 1.073x). The held-
    out apply uses ``B_va @ coef`` (matrix) vs legacy ``apply_operand_prewarp`` (Horner);
    they agree to ~1e-13, which only perturbs the threshold comparisons, never selection.
    # bench-attempt-rejected (2026-06-21): the remaining poly cost is irreducible compute
    # - ``warm_start_als_seed`` (3 lstsq/ALS x 4 bases, ~0.95s/call) + the 8 single-operand
    # ``fit_basis_coef_robust`` solves (~0.52s/call); skipping bases or the single baseline
    # changes the pairness-guard verdict and is NOT selection-safe."""
    from .hermite_fe import build_basis_matrix, fit_pair_prewarp_als
    from mlframe.feature_selection.filters.hermite_fe.shared import POLY_BASES as _POLY_BASES
    from mlframe.feature_selection.filters.hermite_fe.shared import fit_basis_coef_robust
    xa = _finite_filled(x_a)
    xb = _finite_filled(x_b)
    n = xa.size
    if n < 60 or float(np.std(xa)) < 1e-12 or float(np.std(xb)) < 1e-12:
        return None
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    # Materialise the train / val operand slices ONCE (legacy re-sliced xa[tr]/xb[tr]/
    # xa[va]/xb[va] inside every per-basis iteration - 8x the single sweep + 8x the
    # pair sweep - each a fresh boolean-mask gather over the ~n-row column).
    xa_tr = xa[tr]; xb_tr = xb[tr]; xa_va = xa[va]; xb_va = xb[va]
    y_tr = y_f[tr]
    y_va = y_f[va] - float(np.mean(y_f[va]))
    if float(np.std(y_tr)) < 1e-12 or float(np.std(y_va)) < 1e-12:
        return None
    deg = max(1, int(degree))

    # Per-(operand-slice, basis) z-map + train/val basis matrices, built ONCE and
    # shared by BOTH the single-operand baseline (1-D OLS) and the pair ALS sweep.
    # Legacy rebuilt these inside ``fit_operand_prewarp`` (single) AND again inside
    # ``fit_pair_prewarp_als`` (pair) for EVERY basis - the dominant escalation cost
    # (cProfile: _propose_poly 5.46s/8 calls). The z-map is the basis FAMILY's
    # preprocessing (chebyshev & legendre share min-max), so it is keyed by the
    # ``fit`` callable; the basis matrix (chebval vs legval recurrence) is per-basis.
    # This inlines the SAME math the library helpers run (build_basis_matrix +
    # fit_basis_coef_robust for the single OLS, warm_start_als_seed for the pair):
    # coefficients are bit-identical; the held-out apply (B_va @ coef) vs legacy
    # Horner agrees to ~1e-13, perturbing only the threshold compares, never selection.
    _y_tr_c = y_tr - float(np.mean(y_tr))
    _zmap_cache: dict = {}  # id(fit_fn) -> (z_tr, params, z_va) per operand handled below

    def _basis_block(x_tr, x_va, basis):
        """(B_tr, B_va, params) for ``basis`` on the given operand slices, z-map cached
        per basis FAMILY (preprocessing callable) so cheb+leg reuse one z computation."""
        bi = _POLY_BASES[basis]
        fit_fn = bi["fit"]; apply_fn = bi["apply"]
        key = id(fit_fn)
        zc = _zmap_cache.get(key)
        if zc is None:
            z_tr, params = fit_fn(x_tr)
            z_tr = np.ascontiguousarray(z_tr, dtype=np.float64)
            z_va = np.ascontiguousarray(apply_fn(x_va, dict(params)), dtype=np.float64)
            _zmap_cache[key] = (z_tr, params, z_va)
        else:
            z_tr, params, z_va = zc
        B_tr = build_basis_matrix(basis, z_tr, deg)
        B_va = build_basis_matrix(basis, z_va, deg)
        # z_tr returned too so the pair ALS sweep can route through the DEVICE-BORN
        # warm-start (warm_start_als_seed_gpu_from_z): the resident branch rebuilds
        # B_tr ON DEVICE from these standardised columns instead of uploading the
        # prebuilt (n x degree+1) matrices (collapses the ~358MB Ba/Bb H2D at 300k).
        return B_tr, B_va, params, z_tr

    # Best SINGLE-operand warp baseline (1-D fit per side, train-fit / val-scored) - the bar the PAIR
    # reconstruction must clear by the margin. Sweep the SAME bases the PAIR ALS sweeps below (not chebyshev
    # alone): the pair search picks its best basis over all of ``_ESCALATION_POLY_BASES``, so a fair pair-ness
    # bar must give the single-operand baseline the same freedom. A chebyshev-only single baseline UNDER-states
    # the genuine single-source recovery for an even target like ``exp(-a**2)`` (the bounded chebyshev warp of
    # ``a`` under-recovers while the hermite single warp recovers ~0.85), letting a noise-wrap pair whose ALS
    # collapses the noise side to ~const beat the under-stated bar and admit ``esc_poly_*_mul(a,e)``.
    single_best = 0.0
    # Per-operand basis-block caches (reused by the pair sweep below).
    _blocks_a: dict = {}; _blocks_b: dict = {}
    for x_tr, x_va, xraw, blocks in ((xa_tr, xa_va, xa, _blocks_a), (xb_tr, xb_va, xb, _blocks_b)):
        _zmap_cache = {}  # z-map cache is per OPERAND (different column -> different z)
        if float(np.std(xraw)) < 1e-12 or float(np.std(y_tr)) < 1e-12:
            continue
        # ONE heavy-tail memo scope per operand (cProfile-driven): _basis_block fits each escalation
        # basis on the SAME x_tr, and every basis preprocess re-runs the robust heavy-tail np.median/MAD detect
        # on that identical column. Wrapping the per-basis probe in one nesting-safe, identity-verified scope
        # collapses the ~5 detects/operand to 1 (bit-identical: the memo returns a cached verdict only when the
        # stored array IS x_tr). Cleared at operand exit, so no cross-operand ref retention.
        from mlframe.feature_selection.filters.hermite_fe.shared import heavy_tail_memo_scope
        with heavy_tail_memo_scope():
            for _sb_basis in _ESCALATION_POLY_BASES:
                try:
                    B_tr, B_va, _params, z_tr = _basis_block(x_tr, x_va, _sb_basis)
                    blocks[_sb_basis] = (B_tr, B_va, z_tr)
                    # 1-D OLS warp = fit_operand_prewarp's solve (robust gate folded in).
                    coef, _rb, _wn = fit_basis_coef_robust(B_tr, _y_tr_c, x_tr)
                    if coef is None or not np.all(np.isfinite(coef)):
                        continue
                    single_best = max(single_best, _heldout_abs_corr(B_va @ np.ascontiguousarray(coef, dtype=np.float64), y_va))
                except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed: %s", e)
                    continue

    best_basis, best_corr = _best_pair_basis(_blocks_a, _blocks_b, y_tr, y_va, xa_tr, xb_tr)
    if best_basis is None or best_corr < float(min_val_corr):
        return None
    if best_corr < float(pairness_margin) * single_best:
        # Wrapped-marginal cross-mix: the pair form adds nothing over the best single
        # operand's 1-D warp -> not a PAIR signal, leave it to the univariate stages.
        return None
    try:
        sa, sb = fit_pair_prewarp_als(xa, xb, y_f, basis=best_basis, max_degree=degree)
    except Exception as exc:
        # See _candidate_values's matching fix above.
        logger.debug("_propose_poly: fit_pair_prewarp_als failed; skipping candidate: %r", exc)
        return None
    if sa is None or sb is None:
        return None
    return sa, sb, best_basis, best_corr


def _fit_fourier_amplitude_spec(axis01: np.ndarray, t: np.ndarray, freqs, preprocess: dict) -> dict | None:
    """Least-squares sin/cos amplitudes of the demodulated target ``t`` at the detected
    frequencies over the fitted axis. Returns the closed-form ``fourier_adaptive``
    prewarp spec (``coef`` packs ``[a_1, b_1, ..., a_K, b_K]``; ``preprocess`` carries
    the axis params + freqs) consumable by ``apply_operand_prewarp``."""
    K = len(freqs)
    if K == 0:
        return None
    D = np.empty((axis01.size, 2 * K), dtype=np.float64)
    for i, f in enumerate(freqs):
        ang = 2.0 * np.pi * float(f) * axis01
        D[:, 2 * i] = np.sin(ang)
        D[:, 2 * i + 1] = np.cos(ang)
    try:
        coef, *_ = np.linalg.lstsq(D, t - float(np.mean(t)), rcond=None)
    except Exception as e:
        logger.debug("_fit_fourier_amplitude_spec: lstsq failed for freqs=%s, no prewarp spec produced: %s", freqs, e)
        return None
    if coef is None or not np.all(np.isfinite(coef)) or float(np.max(np.abs(coef))) < 1e-12:
        return None
    pp = dict(preprocess)
    pp["freqs"] = [float(f) for f in freqs]
    return {
        "basis": "fourier_adaptive",
        "degree": int(K),
        "coef": np.ascontiguousarray(coef, dtype=np.float64),
        "preprocess": pp,
    }


def _fourier_jobs(x_w, x_m, y_f, *, chirp: bool = True) -> list[dict]:
    """Detection jobs of the demodulated Fourier (+ chirp) proposer for ``y ~ g(x_w) * x_m``: the linear-axis job and, when usable, the quadratic-chirp-axis job.

    Each job holds the axis ``z`` the detector scans, the demodulated target ``t = y_c * zscore(x_m)``, the frequency grid, the warp ``kind`` and the ``axis`` description the amplitude spec
    stores. Empty when the pair carries no usable axis (an integer-code axis, a constant column or target)."""
    from ._orthogonal_univariate_fe._orth_extra_basis_fe import _chirp_axis, _fit_chirp_warp_for_col, _fit_fourier_for_col, _is_int_as_cat_axis

    jobs: list[dict] = []
    xw = _finite_filled(x_w)
    xm = _finite_filled(x_m)
    if _is_int_as_cat_axis(xw):
        # Arbitrary integer label codes carry no real oscillation - mirror the shipped
        # univariate guard (sin/cos of a region code is spurious periodicity).
        return jobs
    std_m = float(np.std(xm))
    if std_m < 1e-12 or float(np.std(xw)) < 1e-12:
        return jobs
    z_m = (xm - float(np.mean(xm))) / std_m
    y_c = y_f - float(np.mean(y_f))
    if float(np.std(y_c)) < 1e-12:
        return jobs
    t = y_c * z_m
    # Linear axis (shipped robust min-max normalisation).
    lo, span = _fit_fourier_for_col(xw)
    span = float(guarded_scale(span, np.abs(xw).max()))  # stored below, so replay divides by exactly this
    z01 = (xw - float(lo)) / span
    jobs.append({"kind": "fourier", "z": z01, "t": t, "grid": _ADAPTIVE_F_GRID, "axis": {"arg": "linear", "lo": float(lo), "span": float(span)}})
    # Quadratic-argument chirp axis (shipped warp): stationary in u for growing-frequency inners.
    if chirp:
        c_mean, c_std, c_lo, c_span = _fit_chirp_warp_for_col(xw)
        if scale_is_usable(c_span, xw) and scale_is_usable(c_std, xw):
            u = _chirp_axis(xw, c_mean, c_std, c_lo, c_span)
            if np.all(np.isfinite(u)) and float(np.std(u)) > 1e-12:
                jobs.append(
                    {
                        "kind": "chirp", "z": u, "t": t, "grid": _CHIRP_F_GRID,
                        "axis": {"arg": "quadratic", "mean": float(c_mean), "std": float(c_std), "lo": float(c_lo), "span": float(c_span)},
                    }
                )
    return jobs


def _fourier_proposal(job: dict, freqs: list):
    """The proposal ``{"kind", "spec_w", "freqs"}`` of one detection job given its detected frequencies, or ``None`` when nothing was detected or no amplitude spec could be fitted."""
    if not freqs:
        return None
    spec = _fit_fourier_amplitude_spec(job["z"], job["t"], freqs, job["axis"])
    if spec is None:
        return None
    return {"kind": job["kind"], "spec_w": spec, "freqs": [float(f) for f in freqs]}


def _detect_jobs(jobs: list[dict], *, min_val_corr: float, max_freqs: int) -> list[list]:
    """Detected frequencies of every job (one resident batch on the device when the GPU-resident mode is on, else the single-column detector per job)."""
    from ._orthogonal_univariate_fe._fourier_detect_batch import detect_fourier_freqs_batch

    return detect_fourier_freqs_batch([(j["z"], j["t"], j["grid"]) for j in jobs], min_val_corr=float(min_val_corr), min_rows=800, max_freqs=int(max_freqs))


def _propose_fourier(x_w, x_m, y_f, *, min_val_corr: float, max_freqs: int, chirp: bool = True):
    """Adaptive-frequency Fourier (+ chirp) proposer for the multiplicative pair form
    ``y ~ g(x_w) * x_m`` via DEMODULATION: the shipped held-out multitone detector is run
    on ``(axis(x_w), t = y_c * zscore(x_m))``. Returns a list of fitted warp specs (0-2:
    linear-axis and/or quadratic-chirp-axis), each a ``fourier_adaptive`` prewarp spec."""
    jobs = _fourier_jobs(x_w, x_m, y_f, chirp=chirp)
    freqs = _detect_jobs(jobs, min_val_corr=min_val_corr, max_freqs=max_freqs) if jobs else []
    return [p for p in (_fourier_proposal(j, f) for j, f in zip(jobs, freqs)) if p is not None]
