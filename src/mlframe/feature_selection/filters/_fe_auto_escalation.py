"""AUTO-ESCALATION of the pair-FE search to the richer SHIPPED bases (2026-06-10, backlog idea B).

When a prospective pair PASSED the pair-MI prescreen (joint-MI ratio gate + order-2 maxT
floor) but the unary/binary operator search admitted NOTHING for it, the legacy behaviour
was a WARNING ("FE produced 0 engineered features despite N pair(s) passing the pair-MI
gate") -- the signal was DETECTED and then silently abandoned. This module escalates
instead: for each such pair it PROPOSES candidates from the two richer shipped basis
families and lets the EXISTING admission gates decide (escalation proposes, gates decide
-- the iron rule):

* SIGNAL-ADAPTIVE ORTHOGONAL-POLY pair warp: the rank-1 ALS per-operand warp
  (``hermite_fe.fit_pair_prewarp_als``) re-run at a HIGHER degree across the four shipped
  polynomial bases (chebyshev / hermite / legendre / laguerre), with the best basis
  selected by held-out rank-1-reconstruction |corr| on a deterministic stride slice --
  catches a poly inner the default degree-4 chebyshev prewarp under-fits or that the
  default prewarp's own held-out gate rejected at its fixed basis.

* ADAPTIVE-FREQUENCY FOURIER / CHIRP pair warp via DEMODULATION: for a multiplicative
  pair signal ``y ~ g(a) * b`` the univariate adaptive-Fourier detector sees nothing
  (``E[y | a] ~ 0`` when b is ~zero-mean); but the DEMODULATED target
  ``t = (y - mean(y)) * zscore(b)`` satisfies ``E[t | a] ~ g(a) * E[zscore(b)^2]``, so
  the SHIPPED held-out-validated multitone detector
  (``_orth_extra_basis_fe._detect_fourier_freqs_for_col``) run on ``(z01(a), t)`` locks
  g's frequency -- e.g. the ``sin(3.7*a)`` INNER frequency no library unary can express.
  The fitted sin/cos mix is stored as a closed-form ``fourier_adaptive`` prewarp spec
  replayed by ``hermite_fe.apply_operand_prewarp`` (a pure function of x -- leak-safe,
  no y at transform time). The CHIRP variant runs the same detector on the shipped
  quadratic-argument warp ``u = sign(z) * z**2`` so a growing-frequency inner is also
  reachable.

GATES (all existing -- escalation only PROPOSES):
  1. held-out validation floors inside the proposers (the shipped detector's >= 0.30
     held-out periodogram floor / the ALS stride-slice reconstruction-|corr| floor);
  2. the Miller-Madow-debiased candidate MI must clear the order-2 maxT permutation
     floor computed over the prospective-pair pool (the same floor that gated the pair);
  3. a marginal-permutation MI floor (``_fe_cmi_redundancy_gate._conditional_perm_null``);
  4. the S5 conditional-MI redundancy gate over {already-admitted engineered survivors}
     UNION {escalation candidates} -- a candidate redundant given the admitted support is
     dropped; verdicts are applied to ESCALATION candidates only (main-path admissions
     are never revoked here).
A pure-noise pair that slipped the prescreen by chance proposes nothing (the detectors
return no validated frequency, the ALS reconstruction fails the held-out floor) or is
rejected by floors 2-4 -- measured 0 admissions on pure-noise controls (see
``tests/feature_selection/test_fe_auto_escalation.py``).

COST: structurally a no-op when every prescreen-surviving pair produced an admitted
engineered column (the common case -- one set-difference). When it fires, the cost is a
handful of ``lstsq`` solves + one detector sweep per escalated pair, bounded by
``fe_escalation_max_pairs``.

Replay/persistence: every admitted candidate carries a standard ``unary_binary``
EngineeredRecipe with ``prewarp`` pseudo-unaries on both sides and the ``mul`` binary, so
``transform()`` replay, pickling and the cross-fold stability vote treat it exactly like
a default-prewarp pair feature. The candidate's fit-time values are computed through the
SAME ``apply_operand_prewarp`` + ``np.multiply`` + ``nan_to_num`` path the recipe replays,
so fit and transform are bit-identical on the same rows.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

__all__ = ["run_fe_auto_escalation", "find_underdelivering_pairs"]

# Signal-adaptive poly basis routing: try all four shipped families, best by held-out
# reconstruction |corr|. Chebyshev first (the production prewarp default).
_ESCALATION_POLY_BASES = ("chebyshev", "hermite", "legendre", "laguerre")

# Coarse z-space frequency grids -- VERBATIM the shipped univariate adaptive grids
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
        return x
    fill = float(np.mean(x[finite])) if finite.any() else 0.0
    out = x.copy()
    out[~finite] = fill
    return out


def _identity_prewarp_spec(x: np.ndarray) -> dict | None:
    """Closed-form IDENTITY prewarp spec for the mate operand: chebyshev degree-1 with
    coef [0, 1] evaluates to the basis' affine z-map of x (replayed by
    ``apply_operand_prewarp`` exactly like any learned warp). An affine map of the mate
    keeps the product's MI/correlation structure intact while staying on the standard
    ``prewarp`` recipe path (no new pseudo-unary needed)."""
    from .hermite_fe import _POLY_BASES
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
    -- the same chain ``_apply_unary_binary`` executes at transform() time."""
    from .hermite_fe import apply_operand_prewarp
    try:
        wa = apply_operand_prewarp(np.asarray(x_a, dtype=np.float64), spec_a)
        wb = apply_operand_prewarp(np.asarray(x_b, dtype=np.float64), spec_b)
    except Exception:
        return None
    out = np.multiply(wa, wb)
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(out)) or float(np.std(out)) < 1e-12:
        return None
    return out


def _propose_poly(x_a, x_b, y_f, *, degree: int, min_val_corr: float, pairness_margin: float = 1.15):
    """Signal-adaptive orth-poly proposer: rank-1 ALS pair warp per shipped basis,
    held-out stride validation of the rank-1 reconstruction, best basis wins.

    PAIR-NESS GUARD: the rank-1 PAIR reconstruction's held-out |corr| must beat the
    best SINGLE-OPERAND warp's held-out |corr| by ``pairness_margin`` (default 1.15,
    mirroring ``fe_synergy_min_prevalence``). Without it, a (genuine-marginal x noise)
    cross-mix pair passes trivially -- the ALS collapses the noise side to ~constant
    and the "pair" reconstruction is just a wrapped univariate trend (measured on the
    weak F2: 6 cross-mix wrappers admitted without the guard, 0 with it; the genuine
    product terms keep ratios >= 1.5 because no single operand carries the product).

    Returns ``(spec_a, spec_b, basis, val_corr)`` or ``None`` (no basis generalises /
    the pair adds nothing over its best single operand).

    PERF (2026-06-21): the basis matrices for ``xa[tr]`` / ``xb[tr]`` are built ONCE
    per (operand, basis) and SHARED by the single-operand OLS baseline AND the pair
    ALS sweep, instead of legacy's ``fit_operand_prewarp`` + ``fit_pair_prewarp_als``
    each rebuilding them per basis (z-map cached per FAMILY -- cheb+leg share min-max).
    Inlines the SAME library solves (``fit_basis_coef_robust`` single, ``warm_start_als_seed``
    pair) so held-out corr / coefficients match the legacy calls; selection-equivalent
    (interleaved isolated A/B on the canonical n=100k first-escalation call: same 8
    eligible pairs, same 0 proposed; OLD median 4.382s -> NEW 4.083s, 1.073x). The held-
    out apply uses ``B_va @ coef`` (matrix) vs legacy ``apply_operand_prewarp`` (Horner);
    they agree to ~1e-13, which only perturbs the threshold comparisons, never selection.
    # bench-attempt-rejected (2026-06-21): the remaining poly cost is irreducible compute
    # -- ``warm_start_als_seed`` (3 lstsq/ALS x 4 bases, ~0.95s/call) + the 8 single-operand
    # ``fit_basis_coef_robust`` solves (~0.52s/call); skipping bases or the single baseline
    # changes the pairness-guard verdict and is NOT selection-safe."""
    from .hermite_fe import (
        _POLY_BASES, build_basis_matrix, fit_pair_prewarp_als, warm_start_als_seed,
    )
    from .hermite_fe._hermite_robust import fit_basis_coef_robust
    xa = _finite_filled(x_a)
    xb = _finite_filled(x_b)
    n = xa.size
    if n < 60 or float(np.std(xa)) < 1e-12 or float(np.std(xb)) < 1e-12:
        return None
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    # Materialise the train / val operand slices ONCE (legacy re-sliced xa[tr]/xb[tr]/
    # xa[va]/xb[va] inside every per-basis iteration -- 8x the single sweep + 8x the
    # pair sweep -- each a fresh boolean-mask gather over the ~n-row column).
    xa_tr = xa[tr]; xb_tr = xb[tr]; xa_va = xa[va]; xb_va = xb[va]
    y_tr = y_f[tr]
    y_va = y_f[va] - float(np.mean(y_f[va]))
    if float(np.std(y_tr)) < 1e-12 or float(np.std(y_va)) < 1e-12:
        return None
    deg = max(1, int(degree))

    def _heldout_corr(vals_va) -> float:
        vals_va = np.nan_to_num(np.asarray(vals_va, dtype=np.float64), copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.std(vals_va)) < 1e-12:
            return 0.0
        cc = float(np.corrcoef(vals_va, y_va)[0, 1])
        return abs(cc) if np.isfinite(cc) else 0.0

    # Per-(operand-slice, basis) z-map + train/val basis matrices, built ONCE and
    # shared by BOTH the single-operand baseline (1-D OLS) and the pair ALS sweep.
    # Legacy rebuilt these inside ``fit_operand_prewarp`` (single) AND again inside
    # ``fit_pair_prewarp_als`` (pair) for EVERY basis -- the dominant escalation cost
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

    # Best SINGLE-operand warp baseline (1-D fit per side, train-fit / val-scored) -- the bar the PAIR
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
        # ONE heavy-tail memo scope per operand (2026-07-02, cProfile-driven): _basis_block fits each escalation
        # basis on the SAME x_tr, and every basis preprocess re-runs the robust heavy-tail np.median/MAD detect
        # on that identical column. Wrapping the per-basis probe in one nesting-safe, identity-verified scope
        # collapses the ~5 detects/operand to 1 (bit-identical: the memo returns a cached verdict only when the
        # stored array IS x_tr). Cleared at operand exit, so no cross-operand ref retention.
        from .hermite_fe._hermite_robust import heavy_tail_memo_scope
        with heavy_tail_memo_scope():
            for _sb_basis in _ESCALATION_POLY_BASES:
                try:
                    B_tr, B_va, _params, z_tr = _basis_block(x_tr, x_va, _sb_basis)
                    blocks[_sb_basis] = (B_tr, B_va, z_tr)
                    # 1-D OLS warp = fit_operand_prewarp's solve (robust gate folded in).
                    coef, _rb, _wn = fit_basis_coef_robust(B_tr, _y_tr_c, x_tr)
                    if coef is None or not np.all(np.isfinite(coef)):
                        continue
                    single_best = max(single_best, _heldout_corr(B_va @ np.ascontiguousarray(coef, dtype=np.float64)))
                except Exception:
                    continue

    best_corr = -1.0
    best_basis = None
    for basis in _ESCALATION_POLY_BASES:
        try:
            blk_a = _blocks_a.get(basis); blk_b = _blocks_b.get(basis)
            if blk_a is None or blk_b is None:
                continue
            Ba_tr, Ba_va, za_tr = blk_a; Bb_tr, Bb_va, zb_tr = blk_b
            # Rank-1 ALS pair warp = fit_pair_prewarp_als' solve on the SAME basis
            # matrices (warm_start_als_seed; OLS-ALS, no robustify -- matches legacy).
            # z_a/z_b/basis are passed too so the resident GPU branch takes the
            # DEVICE-BORN path (warm_start_als_seed_gpu_from_z): Ba/Bb are rebuilt on
            # device from these standardised columns (the SAME basis_fit the prebuilt
            # Ba_tr/Bb_tr used), collapsing the ~358MB design H2D at 300k. Ba_tr/Bb_tr
            # still feed the byte-identical CPU fallback (default, flag-off) path.
            coef_a, coef_b = warm_start_als_seed(Ba_tr, Bb_tr, y_tr, iters=3, x_a=xa_tr, x_b=xb_tr, z_a=za_tr, z_b=zb_tr, basis=basis)
            if coef_a is None or coef_b is None:
                continue
            c = _heldout_corr((Ba_va @ np.ascontiguousarray(coef_a, dtype=np.float64)) * (Bb_va @ np.ascontiguousarray(coef_b, dtype=np.float64)))
        except Exception:
            continue
        if c > best_corr:
            best_corr = c
            best_basis = basis
    if best_basis is None or best_corr < float(min_val_corr):
        return None
    if best_corr < float(pairness_margin) * single_best:
        # Wrapped-marginal cross-mix: the pair form adds nothing over the best single
        # operand's 1-D warp -> not a PAIR signal, leave it to the univariate stages.
        return None
    try:
        sa, sb = fit_pair_prewarp_als(xa, xb, y_f, basis=best_basis, max_degree=degree)
    except Exception:
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
    except Exception:
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


def _propose_fourier(x_w, x_m, y_f, *, min_val_corr: float, max_freqs: int, chirp: bool = True):
    """Adaptive-frequency Fourier (+ chirp) proposer for the multiplicative pair form
    ``y ~ g(x_w) * x_m`` via DEMODULATION: the shipped held-out multitone detector is run
    on ``(axis(x_w), t = y_c * zscore(x_m))``. Returns a list of fitted warp specs (0-2:
    linear-axis and/or quadratic-chirp-axis), each a ``fourier_adaptive`` prewarp spec."""
    from ._orthogonal_univariate_fe._orth_extra_basis_fe import (
        _chirp_axis,
        _detect_fourier_freqs_for_col,
        _fit_chirp_warp_for_col,
        _fit_fourier_for_col,
        _is_int_as_cat_axis,
    )
    out: list[dict] = []
    xw = _finite_filled(x_w)
    xm = _finite_filled(x_m)
    if _is_int_as_cat_axis(xw):
        # Arbitrary integer label codes carry no real oscillation -- mirror the shipped
        # univariate guard (sin/cos of a region code is spurious periodicity).
        return out
    std_m = float(np.std(xm))
    if std_m < 1e-12 or float(np.std(xw)) < 1e-12:
        return out
    z_m = (xm - float(np.mean(xm))) / std_m
    y_c = y_f - float(np.mean(y_f))
    if float(np.std(y_c)) < 1e-12:
        return out
    t = y_c * z_m
    # Linear axis (shipped robust min-max normalisation).
    lo, span = _fit_fourier_for_col(xw)
    z01 = (xw - float(lo)) / max(float(span), 1e-12)
    freqs = _detect_fourier_freqs_for_col(
        z01, t, f_grid=_ADAPTIVE_F_GRID, min_val_corr=float(min_val_corr),
        min_rows=800, max_freqs=int(max_freqs),
    )
    if freqs:
        spec = _fit_fourier_amplitude_spec(
            z01, t, freqs, {"arg": "linear", "lo": float(lo), "span": float(span)},
        )
        if spec is not None:
            out.append({"kind": "fourier", "spec_w": spec, "freqs": [float(f) for f in freqs]})
    # Quadratic-argument chirp axis (shipped warp): stationary in u for growing-frequency inners.
    if chirp:
        c_mean, c_std, c_lo, c_span = _fit_chirp_warp_for_col(xw)
        if c_span > 1e-12 and c_std > 1e-12:
            u = _chirp_axis(xw, c_mean, c_std, c_lo, c_span)
            if np.all(np.isfinite(u)) and float(np.std(u)) > 1e-12:
                cfreqs = _detect_fourier_freqs_for_col(
                    u, t, f_grid=_CHIRP_F_GRID, min_val_corr=float(min_val_corr),
                    min_rows=800, max_freqs=int(max_freqs),
                )
                if cfreqs:
                    spec = _fit_fourier_amplitude_spec(
                        u, t, cfreqs,
                        {"arg": "quadratic", "mean": float(c_mean), "std": float(c_std),
                         "lo": float(c_lo), "span": float(c_span)},
                    )
                    if spec is not None:
                        out.append({"kind": "chirp", "spec_w": spec, "freqs": [float(f) for f in cfreqs]})
    return out


def _resolve_operand(X, name: str, engineered_continuous: dict | None) -> np.ndarray | None:
    """Continuous values for a RAW column ``name`` from the (possibly augmented) frame.
    Prefers the continuous engineered store (not expected for raw operands, kept for
    symmetry); pandas / polars by-name extraction; ``None`` when unresolvable."""
    if engineered_continuous:
        v = engineered_continuous.get(name)
        if v is not None and np.asarray(v).shape[0] == len(X):
            return np.asarray(v, dtype=np.float64)
    try:
        if hasattr(X, "columns") and name in list(X.columns):
            col = X[name]
            vals = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            return np.asarray(vals, dtype=np.float64)
    except Exception:
        return None
    return None


def find_underdelivering_pairs(
    self: Any,
    *,
    prospective_pairs: Any,
    prospective_additions: dict,
    X: Any,
    cols: Sequence[str],
    classes_y: Any,
    done: Any,
    max_rows: int = 20000,
    n_permutations: int = 8,
) -> list[tuple]:
    """UNDERDELIVERY trigger for the auto-escalation (2026-06-10): pairs whose
    unary/binary search DID admit a column, but whose best admitted capture leaves
    SIGNIFICANT conditional pair signal on the table.

    Why a leftover-CMI test and not an MI-ratio bar: the prescreen ``pair_mi`` is a
    2-D joint MI over the (possibly adaptive) operand codes and structurally
    UNDER-estimates the pair information, so ``best_admitted_mi / pair_mi`` does not
    separate a weak envelope capture from a genuine one (measured on the
    ``y=sin(3.7*a)*b`` fixture: the junk ``mul(sin(a),qubed(b))`` envelope capture
    scores ratio 1.20 -- ABOVE the genuine He3 capture's own scale). The leftover
    conditional MI ``CMI(joint(a,b) codes; y | best admitted column's codes)`` is the
    exact quantity of interest: ~bias when the capture is complete (He3 fixture),
    large when the library form only caught an envelope of the detected signal (the
    sin fixture, where most of ``sin(3.7a)*b`` lies beyond ``sin(a)*b**3``).

    TRIGGER (three legs): the leftover CMI must clear (1) the conditional-permutation
    null's quantile floor (same-bias null: the pair codes are permuted WITHIN
    admitted-code strata), (2) a small debiased-excess bar relative to the captured MI
    (``fe_escalation_underdelivery_excess_frac``, default 0.05) so a floor-grazing
    fluctuation cannot fire it, and (3) a DISCRETISATION-RESIDUAL control: even a
    functionally COMPLETE capture leaves leftover CMI in the 2-D joint, because its
    own ``nbins`` quantile code is coarse (within-bin variation of the captured value
    still predicts y) -- so the joint's leftover must exceed
    ``fe_escalation_underdelivery_self_ratio`` (default 3.0) times the capture's OWN
    finer-binning refinement ``CMI(capture @ 2*nbins; y | capture @ nbins)``. A
    complete capture refines itself about as much as the joint refines it (measured:
    He3 perfect capture ratio 0.70, F2 a**2/b capture 0.83, F2 log*sin capture 2.44),
    while an envelope junk capture cannot (sin-fixture ``mul(sin(a),qubed(b))``
    measures 14.6 -- most of ``sin(3.7a)*b`` lies beyond any binning of the envelope).
    A FALSE trigger is safe -- escalation only PROPOSES and every candidate still
    faces the full admission gates (incl. the S5 CMI gate conditioned on the pair's
    own admitted column) -- so the trigger is tuned cheap, not razor-sharp: all
    arrays are stride-subsampled to ``max_rows`` and the null uses
    ``n_permutations=8`` (the real gates re-verify at full rigor / full n).

    Returns ``[(pair_idx_tuple, pair_mi), ...]`` ready to append to the escalation's
    ``failed_pairs`` argument. Never raises (skips a pair on any internal hiccup)."""
    from ._fe_cmi_redundancy_gate import _conditional_perm_null
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

    out = []
    n = int(len(X))
    if n <= 0:
        return out
    step = max(1, int(np.ceil(n / float(max_rows))))
    sl = slice(None, None, step)
    y_arr = np.asarray(classes_y)[sl]
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    if np.unique(y_dense).size < 2:
        return out
    nbq = int(self.quantization_nbins)
    raw_names = set(getattr(self, "feature_names_in_", []) or [])
    eng_cont = getattr(self, "_engineered_continuous_", None)
    seed = int(getattr(self, "random_seed", 0) or 0)
    excess_frac = float(getattr(self, "fe_escalation_underdelivery_excess_frac", 0.05))
    self_ratio = float(getattr(self, "fe_escalation_underdelivery_self_ratio", 3.0))
    for _k in prospective_pairs:
        try:
            pair, pair_mi = _k[0], float(_k[1])
            if pair in done:
                continue
            v = prospective_additions.get(pair)
            # Zero-admission pairs are the PRIMARY trigger's job; here we only look at
            # pairs that admitted something (values matrix + names present).
            if not v or not v[0] or v[1] is None or not v[2]:
                continue
            na, nb = cols[pair[0]], cols[pair[1]]
            if na not in raw_names or nb not in raw_names:
                continue
            x_a = _resolve_operand(X, na, eng_cont)
            x_b = _resolve_operand(X, nb, eng_cont)
            if x_a is None or x_b is None or x_a.size != n or x_b.size != n:
                continue
            ca = _quantile_bin(x_a[sl], nbins=nbq).astype(np.int64)
            cb = _quantile_bin(x_b[sl], nbins=nbq).astype(np.int64)
            joint = ca * (int(cb.max()) + 1) + cb
            _, joint = np.unique(joint, return_inverse=True)
            joint = joint.astype(np.int64)
            # Conditioning support: the SINGLE best admitted column (by marginal MI on
            # the same subsample). Conditioning on one column keeps the null's strata
            # populated; a multi-column capture that is only jointly complete merely
            # causes a false trigger, which the downstream S5 gate (conditioned on ALL
            # admitted columns) absorbs.
            tvals, ncols_names = v[1], v[2]
            best_mi, best_codes, best_vals = -1.0, None, None
            for j in range(min(len(ncols_names), int(tvals.shape[1]))):
                vj = np.asarray(tvals[sl, j], dtype=np.float64)
                cj = _quantile_bin(vj, nbins=nbq).astype(np.int64)
                mij = float(_cmi_from_binned(cj, y_dense, None))
                if mij > best_mi:
                    best_mi, best_codes, best_vals = mij, cj, vj
            if best_codes is None or best_mi <= 0.0:
                continue
            leftover = float(_cmi_from_binned(joint, y_dense, best_codes))
            floor, null_mean = _conditional_perm_null(
                joint, y_dense, best_codes,
                n_permutations=int(n_permutations),
                seed=seed + 1009 * int(pair[0]) + int(pair[1]),
            )
            if leftover <= floor or (leftover - null_mean) < excess_frac * best_mi:
                continue
            # Leg 3 -- discretisation-residual control (see docstring): the capture's
            # OWN finer-binning refinement bounds the leftover a COMPLETE capture shows.
            cap_fine = _quantile_bin(best_vals, nbins=2 * nbq).astype(np.int64)
            leftover_self = max(0.0, float(_cmi_from_binned(cap_fine, y_dense, best_codes)))
            if leftover > self_ratio * leftover_self:
                out.append((pair, pair_mi))
        except Exception:  # pragma: no cover - trigger must never break the FE step
            continue
    return out


def run_fe_auto_escalation(
    self: Any,
    *,
    failed_pairs: Any,
    X: Any,
    cols: Sequence[str],
    classes_y: Any,
    pair_maxt_floor: float,
    admitted_pool: dict,
    verbose: int = 0,
    capture_vals: dict | None = None,
    rescue_pairs: set | None = None,
) -> list[dict]:
    """Escalate the FE search to the richer shipped bases for prescreen-surviving pairs
    the unary/binary step admitted NOTHING for (or, via the UNDERDELIVERY trigger,
    admitted only a partial capture for). PROPOSES candidates (signal-adaptive
    orth-poly ALS warps + demodulated adaptive-frequency Fourier/chirp warps), then runs
    them through the EXISTING admission gates (order-2 maxT floor on MM-debiased MI,
    marginal-permutation floor, S5 conditional-MI redundancy gate vs the admitted
    engineered support).

    ``capture_vals`` (optional): per-pair matrix of the pair's ALREADY-ADMITTED
    engineered column values (full n). When present for a pair, the proposers fit the
    RESIDUAL of the supervised target after removing each capture column's BINNED
    CONDITIONAL MEAN (nonparametric -- removes ANY function of the capture at bin
    resolution, crucially including the rank-vs-raw monotone remap a linear residual
    leaves behind), so they hunt for the MISSING part of the signal: a candidate that
    merely re-expresses the existing capture finds ~no residual correlation and dies
    at the proposers' held-out floors -- measured on the He3(a)*b fixture, where the
    default prewarp capture's leftover triggers underdelivery but the full-target /
    lstsq-residual re-fits both produced a +0.0008-held-out-R^2 remap candidate (the
    S5 gate's train-side conditional MI admitted it); the binned-mean residual kills
    it at the proposal stage while the sin(3.7a)*b inner frequency -- genuinely
    ABSENT from its envelope capture -- survives residualisation and is recovered.

    Returns a list of admitted candidate dicts ``{name, values, recipe, mi, kind,
    pair}`` for the caller to materialise; stamps ``self.fe_escalation_info_``
    provenance. Never raises (degrades to ``[]``)."""
    from ._fe_cmi_redundancy_gate import _conditional_perm_null, apply_cmi_redundancy_gate
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
    from .engineered_recipes import build_unary_binary_recipe

    info: dict = {"eligible_pairs": [], "proposed": 0, "admitted": [], "rejected": {}, "pair_maxt_floor": float(pair_maxt_floor)}
    self.fe_escalation_info_ = info
    # Accumulated per-call history (one entry per FE step that ran escalation), so a
    # late no-op step does not erase the provenance of an earlier admitting step.
    if not isinstance(getattr(self, "fe_escalation_history_", None), list):
        self.fe_escalation_history_ = []
    self.fe_escalation_history_.append(info)
    if not failed_pairs:
        return []

    n_rows = int(len(X))
    min_rows = int(getattr(self, "fe_escalation_min_rows", 500))
    if n_rows < min_rows:
        info["skipped"] = f"n_rows={n_rows} < fe_escalation_min_rows={min_rows}"
        return []

    raw_names = set(getattr(self, "feature_names_in_", []) or [])
    eng_cont = getattr(self, "_engineered_continuous_", None)
    max_pairs = int(getattr(self, "fe_escalation_max_pairs", 8))
    min_val_corr = float(getattr(self, "fe_escalation_min_val_corr", 0.15))
    poly_degree = int(getattr(self, "fe_escalation_poly_degree", 6))
    max_freqs = int(getattr(self, "fe_escalation_fourier_max_freqs", 3))
    per_pair_cap = int(getattr(self, "fe_escalation_max_candidates_per_pair", 3))
    seed = int(getattr(self, "random_seed", 0) or 0)
    nbins = int(self.quantization_nbins)

    # SUBSAMPLED DECISION (2026-06-21). The escalation proposers (orth-poly ALS warp fit +
    # adaptive Fourier/chirp periodogram DETECTION) are the dominant active orth-FE CPU cost
    # and ran on the FULL frame. Decide on the SAME row-subsample the rest of FE uses
    # (fe_check_pairs_subsample_n + random_seed), then rebuild each ADMITTED candidate's
    # ``values`` at full n via its closed-form recipe before returning (output-safe). Operands,
    # target, residualisation captures and the admitted-support pool are all subsampled in
    # lockstep so the gates decide on a consistent slice. Default off -> full-data decision.
    _X_full = X
    _esc_ss_n = int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0)
    _esc_do_sub = isinstance(_esc_ss_n, int) and 0 < _esc_ss_n < n_rows
    _y_rank_eff = getattr(self, "_fe_escalation_y_rank_", None)
    if _esc_do_sub:
        _esc_idx = np.sort(np.random.default_rng(seed).choice(n_rows, size=int(_esc_ss_n), replace=False))
        X = X.iloc[_esc_idx].reset_index(drop=True) if hasattr(X, "iloc") else np.asarray(X)[_esc_idx]
        classes_y = np.asarray(classes_y)[_esc_idx]
        if capture_vals:
            capture_vals = {k: np.asarray(v)[_esc_idx] for k, v in capture_vals.items()}
        if admitted_pool:
            admitted_pool = {k: (np.asarray(v)[_esc_idx], m) for k, (v, m) in (admitted_pool or {}).items()}
        if _y_rank_eff is not None and np.asarray(_y_rank_eff).shape[0] == n_rows:
            _y_rank_eff = np.asarray(_y_rank_eff)[_esc_idx]
        n_rows = int(_esc_ss_n)

    # Target for the supervised warp fits. PREFER the rank-transformed raw y stashed
    # by ``_fit_impl`` (``_fe_escalation_y_rank_``): the FE step's ``classes_y`` are
    # LABEL codes from the internal target quantisation -- NOT guaranteed ordinal /
    # monotone in y (measured 37 unordered codes on a heavy-tailed regression y) --
    # which silently destroys a Pearson-validated ALS / periodogram fit (held-out
    # corr 0.42 on the genuine (c,d) term with rank-y vs ~0 with the label codes).
    # The rank is monotone-equivalent to y and heavy-tail-robust; fall back to the
    # codes when the stash is unavailable (multi-output / non-numeric y).
    _y_rank = _y_rank_eff
    if _y_rank is not None and np.asarray(_y_rank).shape[0] == n_rows:
        y_f = np.ascontiguousarray(_y_rank, dtype=np.float64)
    else:
        y_f = np.ascontiguousarray(np.asarray(classes_y), dtype=np.float64)
    y_arr = np.asarray(classes_y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)

    # RAW-RAW pairs only, bounded by the pair budget. Ranking key: RESCUE pairs first,
    # then by joint MI. A prevalence-failed-synergy rescue pair (a genuine SMOOTH ratio
    # interaction the raw-MI ratio under-rates) has LOW raw joint MI BY CONSTRUCTION, so a
    # plain joint-MI sort buries it below the zero-admission cross-mix pairs and the
    # ``max_pairs`` cap drops it (measured on F2: the genuine (a,b) at joint MI 0.028 was
    # squeezed out by 6 higher-MI cross pairs). Rescue pairs are exactly the ones the
    # escalation EXISTS to recover, so they get first claim on the budget; the held-out
    # ALS pairness guard + the full admission gates still decide whether each is admitted.
    _rescue = {tuple(p) for p in (rescue_pairs or set())}
    eligible = []
    for pair, pair_mi in failed_pairs:
        try:
            na, nb = cols[pair[0]], cols[pair[1]]
        except Exception:
            continue
        if na in raw_names and nb in raw_names:
            eligible.append((pair, float(pair_mi), na, nb))
    eligible.sort(key=lambda e: (tuple(e[0]) in _rescue, e[1]), reverse=True)
    eligible = eligible[:max_pairs]
    info["eligible_pairs"] = [(na, nb) for _, _, na, nb in eligible]
    # Raw cols-space index tuples of the processed pairs -- the caller's per-fit
    # dedup ledger key (stable across FE steps: engineered columns append at the end).
    info["eligible_idx"] = [tuple(pair) for pair, _, _, _ in eligible]
    if not eligible:
        return []

    existing_names = set(cols) | set(admitted_pool)
    candidates: list[dict] = []
    for pair, pair_mi, na, nb in eligible:
        x_a = _resolve_operand(X, na, eng_cont)
        x_b = _resolve_operand(X, nb, eng_cont)
        if x_a is None or x_b is None or x_a.size != n_rows or x_b.size != n_rows:
            continue
        pair_cands: list[dict] = []
        # RESIDUAL fitting target for UNDERDELIVERY-triggered pairs (see docstring):
        # remove EVERYTHING a function of the pair's already-admitted capture can
        # explain, so the proposers hunt for the MISSING part only. The removal is the
        # per-column BINNED CONDITIONAL MEAN (not lstsq): the supervised target is
        # rank-y while the capture is raw-valued, so a LINEAR residual leaves the
        # monotone remap of the capture itself in the residual and a proposer happily
        # "recovers" that deterministic function of the existing capture (measured on
        # He3(a)*b: laguerre val_corr 0.72 on the lstsq residual, +0.0008 held-out R^2
        # -- pure remap; the binned-mean removal kills it while the sin(3.7a)*b inner
        # frequency, genuinely absent from its envelope capture, survives).
        # Zero-admission pairs (no capture) fit the full target as before.
        y_pair = y_f
        _cv = (capture_vals or {}).get(tuple(pair))
        if _cv is not None:
            try:
                _A = np.asarray(_cv, dtype=np.float64).reshape(n_rows, -1)[:, :8]
                _r = np.asarray(y_f, dtype=np.float64).copy()
                _nb_res = int(min(32, max(8, n_rows // 64)))
                for _j in range(_A.shape[1]):
                    _cb = _quantile_bin(np.nan_to_num(_A[:, _j], nan=0.0, posinf=0.0, neginf=0.0), nbins=_nb_res).astype(np.int64)
                    _cnt = np.maximum(np.bincount(_cb, minlength=int(_cb.max()) + 1), 1)
                    _means = np.bincount(_cb, weights=_r, minlength=int(_cb.max()) + 1) / _cnt
                    _r = _r - _means[_cb]
                if float(np.std(_r)) > 1e-9:
                    y_pair = _r
            except Exception:
                # No silent swallow: a failure here means we fall back to the FULL target instead of the
                # residual, which defeats residualisation (the proposer re-proposes already-captured signal).
                logger.debug("fe-escalation residualisation failed; using full target", exc_info=True)
        # 1) Signal-adaptive orth-poly ALS warp (higher degree + 4-basis routing).
        poly = _propose_poly(
            x_a, x_b, y_pair, degree=poly_degree, min_val_corr=min_val_corr,
            pairness_margin=float(getattr(self, "fe_escalation_pairness_margin", 1.15)),
        )
        if poly is not None:
            sa, sb, basis, vcorr = poly
            vals = _candidate_values(x_a, sa, x_b, sb)
            if vals is not None:
                pair_cands.append({
                    "name": f"esc_poly_{basis}_mul({na},{nb})",
                    "values": vals, "spec_a": sa, "spec_b": sb,
                    "src_a": na, "src_b": nb, "kind": f"poly_{basis}",
                    "pair": (na, nb), "val_corr": float(vcorr),
                })
        # 2) Demodulated adaptive-frequency Fourier / chirp, both warp directions.
        for x_w, x_m, nw, nm in ((x_a, x_b, na, nb), (x_b, x_a, nb, na)):
            for prop in _propose_fourier(x_w, x_m, y_pair, min_val_corr=min_val_corr, max_freqs=max_freqs, chirp=True):
                spec_m = _identity_prewarp_spec(x_m)
                if spec_m is None:
                    continue
                vals = _candidate_values(x_w, prop["spec_w"], x_m, spec_m)
                if vals is None:
                    continue
                pair_cands.append({
                    "name": f"esc_{prop['kind']}_mul({nw},{nm})",
                    "values": vals, "spec_a": prop["spec_w"], "spec_b": spec_m,
                    "src_a": nw, "src_b": nm, "kind": prop["kind"],
                    "pair": (na, nb), "freqs": prop["freqs"],
                })
        # Score by the SAME MM-debiased plug-in MI the gates use; cap per pair.
        for c in pair_cands:
            vb = _quantile_bin(np.asarray(c["values"], dtype=np.float64), nbins=nbins)
            c["_binned"] = vb
            c["mi"] = float(_cmi_from_binned(vb, y_dense, None))
        pair_cands.sort(key=lambda c: c["mi"], reverse=True)
        candidates.extend(pair_cands[: max(1, per_pair_cap)])

    info["proposed"] = len(candidates)
    if not candidates:
        return []

    # Deduplicate names defensively (two pairs sharing operands cannot collide on the
    # name template, but an operand name containing "," could).
    seen: set = set()
    for c in candidates:
        base = c["name"]
        k = 2
        while c["name"] in existing_names or c["name"] in seen:
            c["name"] = f"{base}_{k}"
            k += 1
        seen.add(c["name"])

    # GATE 2: order-2 maxT permutation floor (MM-debiased MI scale on BOTH sides --
    # the floor was computed with miller_madow=True, ``_cmi_from_binned`` debiases too).
    # GATE 3: marginal-permutation floor (same primitive the S5 gate's significance leg
    # uses) -- protects the degenerate single-candidate path where the S5 gate would
    # otherwise admit on marginal significance alone.
    survivors: list[dict] = []
    for c in candidates:
        if pair_maxt_floor > 0.0 and c["mi"] < float(pair_maxt_floor):
            info["rejected"][c["name"]] = f"below_maxt_floor (mi={c['mi']:.5f} < {pair_maxt_floor:.5f})"
            continue
        floor_m, _null_mean = _conditional_perm_null(c["_binned"], y_dense, None, seed=seed)
        if c["mi"] <= floor_m:
            info["rejected"][c["name"]] = f"below_marginal_perm_floor (mi={c['mi']:.5f} <= {floor_m:.5f})"
            continue
        survivors.append(c)
    if not survivors:
        if verbose:
            logger.info(
                "MRMR FE auto-escalation: %d candidate(s) proposed for %d pair(s), 0 cleared "
                "the maxT/permutation floors (gates decide; noise control held).",
                info["proposed"], len(eligible),
            )
        return []

    # GATE 4: S5 conditional-MI redundancy gate over admitted support + survivors.
    # Verdicts are applied to ESCALATION candidates only.
    pool: dict = {}
    for nm, (vals, marg) in (admitted_pool or {}).items():
        pool[nm] = (np.asarray(vals, dtype=np.float64), float(marg))
    for c in survivors:
        pool[c["name"]] = (np.asarray(c["values"], dtype=np.float64), c["mi"])
    accepted, _diag = apply_cmi_redundancy_gate(
        pool, y_dense, nbins=nbins,
        retain_frac=float(getattr(self, "fe_engineered_cmi_retain_frac", 0.15)),
        seed=seed, verbose=int(bool(verbose)),
    )
    admitted: list[dict] = []
    for c in survivors:
        if c["name"] not in accepted:
            info["rejected"][c["name"]] = "redundant_under_cmi_gate"
            continue
        recipe = build_unary_binary_recipe(
            name=c["name"],
            src_a_name=c["src_a"], src_b_name=c["src_b"],
            unary_a_name="prewarp", unary_b_name="prewarp",
            binary_name="mul",
            unary_preset=str(getattr(self, "fe_unary_preset", "medium")),
            binary_preset=str(getattr(self, "fe_binary_preset", "minimal")),
            quantization_nbins=self.quantization_nbins,
            quantization_method=self.quantization_method,
            quantization_dtype=self.quantization_dtype,
            fit_values_for_edges=np.asarray(c["values"], dtype=np.float64),
            prewarp_a=c["spec_a"], prewarp_b=c["spec_b"],
        )
        c.pop("_binned", None)
        c["recipe"] = recipe
        admitted.append(c)
        info["admitted"].append(c["name"])
    # FULL-n OUTPUT: when the DECISION ran on a subsample, the candidate ``values`` are
    # subsample-length -- rebuild each admitted candidate's column on the full X via its
    # closed-form recipe so the caller materialises the full-n column (output equals a
    # full-data fit given the same admitted set). A candidate whose full replay fails is
    # dropped (it would otherwise inject a wrong-length column).
    #
    # SELECTION-EQUIVALENCE NOTE (P1-5/P1-6): the orth-poly proposers gate on subsample values computed via
    # polyeval_dispatch at the SMALL subsample n (njit/Horner), while this replay rebuilds at full n where
    # the dispatch may pick the CUDA recurrence -- which differs from njit-Horner by ~1e-12 for cheb/leg/herme
    # (see _gpu_resident_fe P2-2 note; laguerre is forward on both). So a near-FLOOR esc-poly admit decided on
    # Horner values ships a column whose binned MI can differ by that ~1e-12. This is far below the gate's
    # effective resolution (min_val_corr / pairness_margin), and escalation admits ~nothing at the canonical
    # fit anyway (the interleaved A/B above records the same eligible pairs + 0 proposed); the decide->replay
    # set is unchanged. Pinning one polyeval backend across decide+replay is a FUTURE change, unneeded here.
    if _esc_do_sub and admitted:
        from .engineered_recipes import apply_recipe
        _rebuilt: list[dict] = []
        for c in admitted:
            try:
                c["values"] = np.asarray(apply_recipe(c["recipe"], _X_full), dtype=np.float64)
                _rebuilt.append(c)
            except Exception:
                logger.warning(
                    "MRMR FE auto-escalation: full-n replay failed for %r; dropping.",
                    c.get("name"),
                )
        admitted = _rebuilt
    if verbose and (admitted or info["proposed"]):
        logger.info(
            "MRMR FE auto-escalation: %d pair(s) had 0 admitted engineered features after the "
            "unary/binary search; proposed %d richer-basis candidate(s) (orth-poly ALS x4 bases "
            "+ demodulated adaptive Fourier/chirp), gates admitted %d: %s",
            len(eligible), info["proposed"], len(admitted),
            [f"{c['name']} (mi={c['mi']:.4f})" for c in admitted],
        )
    return admitted
