"""Extra-basis univariate FE: B-spline + (adaptive / chirp) Fourier columns.

Complementary to the orthogonal-polynomial univariate path: each source column
can emit cubic B-spline basis columns (sharp local threshold rules) and Fourier
sin/cos columns (periodic / chirp patterns), with an optional held-out-validated
adaptive-frequency + quadratic-argument-chirp detector. The shared univariate
scaffolding (``_dedup_collinear_source_cols``, ``score_features_by_mi_uplift``)
lives in the parent module ``_orthogonal_univariate_fe`` and is lazy-imported
in-body to avoid a cycle.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "generate_extra_basis_features",
    "hybrid_orth_extra_basis_fe_with_recipes",
]


# Backlog #13 (2026-06-09): ``"wavelet"`` adds the Haar / localized
# multiresolution basis alongside the global Fourier + fixed-knot spline. Its
# legs are held-out-scale-selected in ``_wavelet_basis_fe`` and emitted here so
# the extra-basis path (``fe_hybrid_orth_enable`` / ``extra_bases``) can route
# them through the same MI-uplift gate. The standalone default-on stage
# (``fe_wavelet_enable``) reuses the same generator + recipe builder.
_EXTRA_BASIS_KINDS = ("spline", "fourier", "wavelet")


from ._orth_extra_basis_fe import (
    _ADAPTIVE_FE_RAW_USABILITY_CAP,
    _chirp_axis,
    _detect_fourier_freqs_for_col,
    _fit_chirp_warp_for_col,
    _fit_fourier_for_col,
    _fit_spline_for_col,
    _heldout_smooth_r2,
    _heldout_smooth_r2_fast,
    _heldout_smooth_r2_prep,
    _is_int_as_cat_axis,
)


def generate_extra_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    fourier_powers: Sequence[int] = (1, 2),
    spline_knots: int = 5,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    y: Optional[np.ndarray] = None,
    fourier_adaptive: bool = False,
    fourier_adaptive_min_val_corr: float = 0.15,
    fourier_chirp: bool = False,
    fourier_chirp_min_val_corr: float = 0.15,
    max_adaptive_cols: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """For each column in cols and each requested extra basis, emit the basis
    columns and return them alongside the per-column fit metadata (knot
    vectors, lo/hi, fourier (lo, span)) needed to build recipes.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric are
        silently skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    extra_bases : tuple of {'spline', 'fourier'}
        Which extra bases to emit. Empty tuple => returns empty frame.
    fourier_freqs : sequence of float
        Frequencies for the Fourier basis. One sin and one cos column per
        frequency per source column.
    spline_knots : int
        Number of inner quantile knots for the cubic B-spline basis.
        Emits ``spline_knots + 3`` basis columns per source column (cubic
        B-spline has K+degree basis functions on K inner knots).
    dedup_collinear_sources : bool, default True
        Drop near-duplicate source columns before basis enumeration
        (mirrors the polynomial univariate path).
    y : array-like, optional
        Target. Only consulted when ``fourier_adaptive`` is True (and only by
        the ADAPTIVE-FREQUENCY detector). Never read for the fixed-grid
        emission, so the legacy path stays leakage-free / y-independent.
    fourier_adaptive : bool, default False
        When True and ``y`` is given, run :func:`_detect_fourier_freqs_for_col`
        on each source column's z (power==1 only) and -- for each held-out-
        validated dominant frequency found (multitone: several peaks via
        residual deflation) -- ADD it to that column's Fourier frequency set.
        The emitted sin/cos meta entries for adaptive frequencies are tagged
        ``"adaptive": True`` so MRMR can protect them past screening. Covers
        arbitrary-period oscillations and their superpositions
        (``sin(3.7*x) + sin(5.3*x) + sin(6.8*x)``) the fixed grid {1, 2} misses.
    fourier_adaptive_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the adaptive detector.
    fourier_chirp : bool, default False
        ADAPTIVE-CHIRP path (2026-06-03). When True and ``y`` is given, run the
        SAME held-out-validated detector on the QUADRATIC-ARGUMENT warp
        ``u = sign(z) * z**2`` (z standardised on the column) for each source
        column. A chirp ``y ~ sin(2*pi*f*z**2)`` -- whose frequency GROWS with z
        -- is STATIONARY in u, so the detector locks its frequency and the
        emitted ``sin(2*pi*f*u)`` / ``cos(2*pi*f*u)`` reconstruct it; a Fourier on
        the LINEAR argument cannot express a frequency that grows with z. The
        emitted sin/cos meta entries carry ``"arg": "quadratic"`` (the warp the
        recipe replays) AND ``"adaptive": True`` (so MRMR protects them past the
        screen, identical to the linear adaptive legs). This is an ADDITIVE
        second path alongside the linear adaptive one -- both fire; on a plain
        linear target the chirp legs are harmless (Ridge regularises them to ~0,
        Phase-0 bench: combined R^2 == linear-only on linear targets, +0.3-0.5
        R^2 on fast chirps). N-gated at >= 800 MI rows like the linear path.
    fourier_chirp_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the chirp detector.
    max_adaptive_cols : int, optional
        2026-07-09 fix: cap on how many columns run the EXPENSIVE adaptive-frequency /
        chirp DETECTION (``_detect_fourier_freqs_for_col``) -- profiled as the dominant
        cost of this whole family (~34% of a wide-p fit's pre-categorize wall, roughly
        linear in column count since each call does its own held-out frequency sweep
        regardless of row count). ``None`` (default) = unlimited, byte-identical legacy
        behaviour. When set and ``len(cols) > max_adaptive_cols``, only the first
        ``max_adaptive_cols`` columns (in ``cols`` order) get adaptive/chirp detection;
        the remaining columns still get the cheap FIXED-GRID Fourier basis (unaffected
        by this cap) -- only the expensive per-column detector is bounded, not basis
        emission itself.

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns with naming
            ``"{col}__sp{i}"`` (spline) and ``"{col}__sin{f}"`` /
            ``"{col}__cos{f}"`` (fourier).
        meta : dict mapping each emitted column name to a dict with the
            metadata required to build the matching recipe. Keys depend
            on basis kind: spline -> {"basis": "spline", "src": ..., "knots":
            ndarray, "idx": int, "lo": float, "hi": float}; fourier ->
            {"basis": "fourier", "src": ..., "kind": "sin"/"cos", "freq":
            float, "lo": float, "span": float, "power": int[, "adaptive": True]}.

    Notes
    -----
    bench-rejected (2026-06-03): a per-column "poly-vs-Fourier COMPETITION gate"
    -- emit only the better of {orth-poly basis, this Fourier path} per column to
    cut the redundant cross-family features that co-occur in the support -- was
    benchmarked and REJECTED. The co-occurrence is genuine COMPLEMENTARITY, not
    redundancy: on kink/step/bump targets (e.g. y=|x|) the Fourier legs carry
    independent residual R^2 0.16-0.60 that a degree<=4 poly under-fits, so a
    winner-takes-all gate HURT the |x| target OOS by -0.06; no OOS win on a tree
    downstream (deltas +/-0.008, inconsistent sign); on a mixed frame the gate
    declines to fire (different columns have different winners). The existing
    Fleuret redundancy + Spearman cross-stage dedup already remove the only real
    redundancy. Don't add a competition gate. (D:/Temp/item3_poly_fourier_findings.md)
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    extra_bases = tuple(b for b in extra_bases if b in _EXTRA_BASIS_KINDS)
    if not extra_bases:
        return pd.DataFrame(index=X.index), {}
    if dedup_collinear_sources:
        from . import _dedup_collinear_source_cols
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    from ..engineered_recipes import _bspline_basis_values  # local import
    out_cols: dict = {}
    meta: dict = {}
    fourier_freqs = tuple(float(f) for f in fourier_freqs)
    spline_knots = max(2, int(spline_knots))
    # Adaptive-frequency detection runs only when requested AND y is supplied.
    # The y array is coerced to float once here (Pearson on y is all the
    # phase-invariant periodogram needs); detection is gated per-column below.
    _y_adapt = None
    _y_adapt_prep = None
    if (fourier_adaptive or fourier_chirp) and y is not None:
        _y_adapt = np.asarray(y, dtype=np.float64).ravel()
        if _y_adapt.size != len(X) or not np.all(np.isfinite(_y_adapt)):
            _y_adapt = None
        else:
            # Hoist the y-only (x-independent) pieces of the per-column auto-gate ONCE for this fixed
            # _y_adapt (cProfile-driven, 2026-07-16: _heldout_smooth_r2 cost 4.6s/50 calls -- same train/val
            # split + val-side sums recomputed every column despite depending only on y). None on degenerate
            # y -> the loop below falls back to the exact original per-column function unchanged.
            _y_adapt_prep = _heldout_smooth_r2_prep(_y_adapt)
    # Coarse z-space frequency sweep grid for the adaptive detector. Covers a
    # wide period range (0.5 .. 8.0) at 0.5 stride; local refinement then
    # snaps to the true non-integer frequency. The set of frequencies the
    # detector may ADD is disjoint from the fixed grid by construction (a
    # fixed freq that already recovers the signal needs no adaptive twin).
    _adaptive_f_grid = tuple(0.5 * k for k in range(1, 17))  # 0.5 .. 8.0
    # The CHIRP warp (u = sign(z)*z**2) concentrates a growing-frequency signal
    # at a HIGHER z-space frequency than the linear axis, so the chirp detector
    # sweeps a WIDER grid (0.5 .. 24.0). Phase-0: fast chirps land at u-space
    # peaks up to ~12 and the multitone deflation needs headroom above them.
    _chirp_f_grid = tuple(0.5 * k for k in range(1, 49))  # 0.5 .. 24.0
    from .._fe_deadline import fe_deadline_passed
    for _col_idx, col in enumerate(cols):
        # Optional-enrichment wall-clock budget: stop the per-column extra-basis scan (spline / fourier / adaptive /
        # chirp + their pair-MI scoring) once MRMR.fit's deadline passes; return the partial output. No-op without a budget.
        if fe_deadline_passed():
            break
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        from .._fe_usability_signal import _crit_np_dtype
        x = np.asarray(X[col].to_numpy(), dtype=_crit_np_dtype())  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            continue
        if not finite_mask.all():
            # A Fourier/spline/chirp basis over a NaN-containing column is unsound: the nanmean-imputed basis becomes a MISSINGNESS PROXY (its binned
            # MI ties / beats the genuine missingness-FE columns and displaces them from MRMR selection) AND the recipe replay does not impute
            # (transform() emits all-NaN). Skip; the missingness signal belongs to the dedicated missingness-FE family.
            # TODO(imputation): skipping the WHOLE column on a single NaN forfeits a genuine periodic/spline signal
            # whenever the column is only lightly missing. A proper fix is a fit-time imputation (e.g. median, or a
            # model-based fill) BAKED INTO the recipe so transform() replays the SAME fill -- then the basis is a
            # genuine signal, not an all-NaN replay, and it no longer doubles as a missingness proxy (pair it with an
            # explicit missing-indicator column so the missingness signal still lands in its dedicated family). Until
            # then the conservative skip stays (correctness over coverage).
            continue
        # Auto-gate: only let the adaptive Fourier/chirp operators fire where the raw column is NOT already a strong smooth predictor of y (see _ADAPTIVE_FE_RAW_USABILITY_CAP).
        _adaptive_fe_ok = True
        if _y_adapt is not None and (fourier_adaptive or fourier_chirp):
            _r2 = _heldout_smooth_r2_fast(x, _y_adapt_prep) if _y_adapt_prep is not None else _heldout_smooth_r2(x, _y_adapt)
            _adaptive_fe_ok = _r2 < _ADAPTIVE_FE_RAW_USABILITY_CAP
            # Column-count cap on the expensive detector itself (2026-07-09 fix, see max_adaptive_cols
            # docstring) -- columns beyond the cap still get the cheap fixed-grid Fourier basis below,
            # only the held-out frequency-sweep detection is skipped for them.
            if max_adaptive_cols is not None and _col_idx >= int(max_adaptive_cols):
                _adaptive_fe_ok = False
        if "spline" in extra_bases:
            try:
                knots, lo, hi, n_basis = _fit_spline_for_col(x, spline_knots)
                span = max(hi - lo, 1e-12)
                z = np.clip((x - lo) / span, 0.0, 1.0)
                for i in range(n_basis):
                    vals = _bspline_basis_values(z, knots, i, degree=3)
                    # Skip near-constant columns -- the boundary cubic
                    # B-splines occasionally collapse to ~0 on quantile-
                    # placed knots when ties pile at the edge.
                    if float(np.std(vals)) <= 1e-12:
                        continue
                    name = f"{col}__sp{i}"
                    out_cols[name] = vals
                    meta[name] = {
                        "basis": "spline", "src": col,
                        "knots": knots, "idx": i,
                        "lo": float(lo), "hi": float(hi),
                    }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: spline on col=%r raised " "%r; skipping spline for that column.",
                    col,
                    exc,
                )
        # Skip Fourier on integer-valued low-cardinality categorical group keys: sin/cos of an arbitrary label code (region 0..9)
        # is spurious periodicity that floods the support and displaces the genuinely useful grouped aggregates of that key.
        if "fourier" in extra_bases and not _is_int_as_cat_axis(x):
            try:
                # POWER-ARGUMENT Fourier (2026-06-03): build the Fourier on x**p for
                # p in fourier_powers, as a SELF-CONTAINED replayable recipe (raw x ->
                # x**p -> Fourier; 1-deep, no nesting). p=2 captures even-argument
                # CHIRPS like ``sin(a**2)`` (freq~1 on the a**2 argument reproduces it
                # exactly) that a Fourier on the linear argument cannot. p=1 keeps the
                # original ``{col}__sin{freq}`` name (back-compat with prior recipes).
                for pwr in fourier_powers:
                    _p = int(pwr)
                    _xp = x if _p == 1 else np.power(x, _p)
                    if not np.all(np.isfinite(_xp)) or float(np.std(_xp)) <= 1e-12:
                        continue
                    lo_f, span_f = _fit_fourier_for_col(_xp)
                    z = (_xp - lo_f) / max(span_f, 1e-12)
                    _pfx = "" if _p == 1 else f"p{_p}"
                    # ADAPTIVE-FREQUENCY (2026-06-03): for the linear argument
                    # (power==1) detect the column's dominant z-space frequency
                    # from a coarse sweep + local refine, held-out validated.
                    # The detected freq is ADDED to this column's freq set and
                    # its sin/cos meta is tagged adaptive=True so MRMR protects
                    # it past screening. Disjoint-by-detection from the fixed
                    # grid: a fixed freq that already recovers the signal makes
                    # the periodogram peak land near it, so the detector's
                    # held-out gate is satisfied by the fixed twin too -- but
                    # we still tag/add the refined freq because the fixed grid
                    # cannot express a non-integer period.
                    _adaptive_freqs: list[float] = []
                    if _p == 1 and _y_adapt is not None and _adaptive_fe_ok:
                        # max_freqs=6: a multitone superposition (3-4 genuine
                        # tones) needs enough sin/cos pairs to SPAN the signal
                        # subspace after the per-iteration deflation leaves a
                        # residual -- 4 pairs recovered the 3-tone gate-A signal
                        # at OOS R^2 ~0.95 but 6 pairs lift it to ~0.985, a far
                        # safer margin above the 0.9 bar. Each extra freq still
                        # passes the held-out 0.30 floor, so noise never inflates
                        # the count (a pure-noise column stops at the first peak).
                        _adaptive_freqs = _detect_fourier_freqs_for_col(
                            z, _y_adapt,
                            f_grid=_adaptive_f_grid,
                            min_val_corr=float(fourier_adaptive_min_val_corr),
                            min_rows=800,
                            max_freqs=6,
                        )
                    _freqs_for_col = list(fourier_freqs)
                    _adaptive_set: set[float] = set()
                    for _af in _adaptive_freqs:
                        if not any(abs(_af - f) < 1e-9 for f in _freqs_for_col):
                            _freqs_for_col.append(_af)
                            _adaptive_set.add(_af)
                    for freq in _freqs_for_col:
                        _is_adaptive = freq in _adaptive_set
                        ang = 2.0 * np.pi * freq * z
                        s_vals = np.sin(ang)
                        c_vals = np.cos(ang)
                        if float(np.std(s_vals)) > 1e-12:
                            name_s = f"{col}__{_pfx}sin{freq:g}"
                            out_cols[name_s] = s_vals
                            meta[name_s] = {
                                "basis": "fourier", "src": col,
                                "kind": "sin", "freq": float(freq),
                                "lo": float(lo_f), "span": float(span_f),
                                "power": _p, "adaptive": _is_adaptive,
                            }
                        if float(np.std(c_vals)) > 1e-12:
                            name_c = f"{col}__{_pfx}cos{freq:g}"
                            out_cols[name_c] = c_vals
                            meta[name_c] = {
                                "basis": "fourier", "src": col,
                                "kind": "cos", "freq": float(freq),
                                "lo": float(lo_f), "span": float(span_f),
                                "power": _p, "adaptive": _is_adaptive,
                            }
                # ADAPTIVE-CHIRP (2026-06-03): a SECOND argument-warp alongside
                # the linear-adaptive path above. The chirp axis u = sign(z)*z**2
                # (z standardised on the column) makes a growing-frequency
                # oscillation ``y ~ sin(2*pi*f*z**2)`` STATIONARY in u, so the
                # SAME held-out-validated multitone detector locks its frequency
                # and the emitted sin/cos on u reconstruct it -- which a Fourier
                # on the linear argument cannot (Phase-0: linear R^2 0.07-0.53 vs
                # chirp 0.88 on a fast chirp). Emitted legs carry arg="quadratic"
                # (the warp the recipe replays) + adaptive=True (so MRMR protects
                # them past the screen, exactly like the linear adaptive legs).
                # Disjoint by name (``__qsin``/``__qcos``) from the linear legs;
                # additive (on a plain linear target the chirp legs are harmless,
                # Ridge regularises them to ~0). N-gated identically (>= 800 rows
                # inside the detector); a pure-noise column admits none.
                if fourier_chirp and _y_adapt is not None and _adaptive_fe_ok:
                    _c_mean, _c_std, _c_lo, _c_span = _fit_chirp_warp_for_col(x)
                    if _c_span > 1e-12 and _c_std > 1e-12:
                        u_axis = _chirp_axis(x, _c_mean, _c_std, _c_lo, _c_span)
                        if np.all(np.isfinite(u_axis)) and float(np.std(u_axis)) > 1e-12:
                            _chirp_freqs = _detect_fourier_freqs_for_col(
                                u_axis, _y_adapt,
                                f_grid=_chirp_f_grid,
                                min_val_corr=float(fourier_chirp_min_val_corr),
                                min_rows=800,
                                max_freqs=6,
                            )
                            for _cf in _chirp_freqs:
                                ang_c = 2.0 * np.pi * _cf * u_axis
                                sc_vals = np.sin(ang_c)
                                cc_vals = np.cos(ang_c)
                                if float(np.std(sc_vals)) > 1e-12:
                                    name_qs = f"{col}__qsin{_cf:g}"
                                    out_cols[name_qs] = sc_vals
                                    meta[name_qs] = {
                                        "basis": "fourier", "src": col,
                                        "kind": "sin", "freq": float(_cf),
                                        "arg": "quadratic",
                                        "mean": float(_c_mean), "std": float(_c_std),
                                        "lo": float(_c_lo), "span": float(_c_span),
                                        "power": 1, "adaptive": True,
                                    }
                                if float(np.std(cc_vals)) > 1e-12:
                                    name_qc = f"{col}__qcos{_cf:g}"
                                    out_cols[name_qc] = cc_vals
                                    meta[name_qc] = {
                                        "basis": "fourier", "src": col,
                                        "kind": "cos", "freq": float(_cf),
                                        "arg": "quadratic",
                                        "mean": float(_c_mean), "std": float(_c_std),
                                        "lo": float(_c_lo), "span": float(_c_span),
                                        "power": 1, "adaptive": True,
                                    }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: fourier on col=%r raised " "%r; skipping fourier for that column.",
                    col,
                    exc,
                )
        # Backlog #13 (2026-06-09): Haar wavelet / localized multiresolution
        # legs. The per-column held-out scale-selection lives in the standalone
        # ``_wavelet_basis_fe`` module (candidate-count control via the noise-aware
        # held-out MAD floor + max_legs cap); here we only emit the selected legs
        # so the extra-basis MI-uplift gate can screen them like spline/Fourier.
        if "wavelet" in extra_bases and y is not None:
            try:
                from .._wavelet_basis_fe import (
                    _dyadic_haar_leg,
                    _select_wavelet_legs,
                )
                _yv = np.asarray(y).ravel()
                if _yv.size == x.size:
                    xf = x[np.isfinite(x)]
                    _w_lo = float(xf.min())
                    _w_hi = float(xf.max())
                    _w_span = max(_w_hi - _w_lo, 1e-12)
                    _w_legs = _select_wavelet_legs(x, _yv, _w_lo, _w_span)
                    if _w_legs:
                        _w_z = np.clip((x - _w_lo) / _w_span, 0.0, 1.0)
                        for _wj, _wk in _w_legs:
                            _w_leg = _dyadic_haar_leg(_w_z, _wj, _wk)
                            if float(np.std(_w_leg)) <= 1e-12:
                                continue
                            name = f"{col}__haar_j{_wj}k{_wk}"
                            out_cols[name] = _w_leg
                            meta[name] = {
                                "basis": "wavelet", "src": col,
                                "j": int(_wj), "k": int(_wk),
                                "lo": float(_w_lo), "span": float(_w_span),
                            }
            except Exception as exc:
                logger.warning(
                    "generate_extra_basis_features: wavelet on col=%r raised " "%r; skipping wavelet for that column.",
                    col,
                    exc,
                )
    return pd.DataFrame(out_cols, index=X.index), meta


def _build_recipe_from_meta(name: str, meta_entry: dict):
    """Materialise an ``EngineeredRecipe`` from one ``generate_extra_basis_features``
    meta entry. Returns None for unknown basis kinds (defensive)."""
    from ..engineered_recipes import (
        build_orth_fourier_recipe,
        build_orth_spline_recipe,
    )
    basis = meta_entry["basis"]
    if basis == "spline":
        return build_orth_spline_recipe(
            name=name, src_name=str(meta_entry["src"]),
            knots=np.asarray(meta_entry["knots"], dtype=np.float64),
            idx=int(meta_entry["idx"]),
            lo=float(meta_entry["lo"]), hi=float(meta_entry["hi"]),
        )
    if basis == "wavelet":
        # Backlog #13 (2026-06-09): Haar wavelet leg recipe (orth_wavelet).
        from .._wavelet_basis_fe import build_orth_wavelet_recipe
        return build_orth_wavelet_recipe(
            name=name, src_name=str(meta_entry["src"]),
            j=int(meta_entry["j"]), k=int(meta_entry["k"]),
            lo=float(meta_entry["lo"]), span=float(meta_entry["span"]),
        )
    if basis == "fourier":
        return build_orth_fourier_recipe(
            name=name, src_name=str(meta_entry["src"]),
            kind=str(meta_entry["kind"]),
            freq=float(meta_entry["freq"]),
            lo=float(meta_entry["lo"]),
            span=float(meta_entry["span"]),
            power=int(meta_entry.get("power", 1)),
            adaptive=bool(meta_entry.get("adaptive", False)),
            arg=str(meta_entry.get("arg", "linear")),
            mean=(None if meta_entry.get("mean") is None else float(meta_entry["mean"])),
            std=(None if meta_entry.get("std") is None else float(meta_entry["std"])),
        )
    return None


def hybrid_orth_extra_basis_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    extra_bases: Sequence[str] = ("spline", "fourier"),
    fourier_freqs: Sequence[float] = (1.0, 2.0),
    fourier_powers: Sequence[int] = (1, 2),
    spline_knots: int = 5,
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    fourier_adaptive: bool = False,
    fourier_adaptive_min_val_corr: float = 0.15,
    fourier_chirp: bool = False,
    fourier_chirp_min_val_corr: float = 0.15,
    subsample_n: int = 0,
    subsample_seed: int = 42,
    max_adaptive_cols: Optional[int] = None,
):
    """Layer 32 hybrid: spline + Fourier univariate basis FE + MI-greedy
    selection. Mirrors :func:`hybrid_orth_mi_fe_with_recipes` for the
    polynomial path but emits extra-basis columns (B-spline, Fourier)
    instead. Returns (X_augmented, scores, recipes).

    SUBSAMPLED DECISION (2026-06-21). When ``subsample_n`` > 0 and the frame is
    larger, BOTH the expensive adaptive-frequency DETECTION and the MI ranking run
    on a seeded row SUBSAMPLE (the pair-search pattern), and the winning columns are
    REPLAYED at full n via ``apply_recipe`` (the recipe carries the detected
    frequency + axis params). This removes the ~n/subsample redundant rows from the
    periodogram detector (the dominant orth-FE CPU cost) and the plug-in-MI sweep,
    and aligns the family with the pair-search + univariate orth-FE. Default 0 =
    legacy full-data decision (byte-for-byte unchanged).

    The selection rule is the same TWO-GATE chain as the polynomial path:
    relative uplift >= min_uplift AND engineered_mi >= max(legacy floor,
    noise-aware floor). See :func:`hybrid_orth_mi_fe` for the rationale.

    ``fourier_adaptive`` (default False) forwards to
    :func:`generate_extra_basis_features` -- when True, each source column's
    dominant z-space frequency is detected (held-out validated) and added to
    its Fourier set, with the emitted sin/cos recipes tagged ``adaptive=True``.

    ``fourier_chirp`` (default False) likewise forwards the ADAPTIVE-CHIRP path:
    the same held-out detector run on the quadratic-argument warp
    ``u = sign(z)*z**2``, emitting ``__qsin``/``__qcos`` legs tagged
    ``arg="quadratic"`` + ``adaptive=True`` (force-admitted past the uplift gate
    and MRMR-protected identically to the linear adaptive legs).
    """
    _full_n = len(X)
    _do_sub = isinstance(subsample_n, int) and 0 < subsample_n < _full_n
    if _do_sub:
        _sub_idx = np.sort(
            np.random.default_rng(int(subsample_seed)).choice(
                _full_n, size=int(subsample_n), replace=False,
            )
        )
        _Xd = X.iloc[_sub_idx].reset_index(drop=True)
        _yd = np.asarray(y)[_sub_idx]
    else:
        _Xd, _yd = X, y
    # DECISION (adaptive-frequency detection + MI ranking) on the (subsampled) fit frame.
    engineered, meta = generate_extra_basis_features(
        _Xd, cols=cols, extra_bases=extra_bases,
        fourier_freqs=fourier_freqs, fourier_powers=fourier_powers,
        spline_knots=spline_knots,
        y=_yd, fourier_adaptive=fourier_adaptive,
        fourier_adaptive_min_val_corr=fourier_adaptive_min_val_corr,
        fourier_chirp=fourier_chirp,
        fourier_chirp_min_val_corr=fourier_chirp_min_val_corr,
        max_adaptive_cols=max_adaptive_cols,
    )
    if engineered.empty:
        empty_scores = pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift",
        ])
        return X, empty_scores, []
    from . import score_features_by_mi_uplift
    raw_X = _Xd[[c for c in (cols or _Xd.columns) if c in _Xd.columns and pd.api.types.is_numeric_dtype(_Xd[c])]]
    # Pass the per-column fit ``meta`` so the ENGINEERED extra-basis matrix is rebuilt DEVICE-BORN from the
    # resident raw operands (SF1c :311 collapse) instead of the host matrix uploading; None-safe host fallback.
    scores = score_features_by_mi_uplift(raw_X, engineered, _yd, nbins=nbins, meta=meta)
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[(scores["uplift"] >= float(min_uplift)) & (scores["engineered_mi"] >= abs_floor)]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    # FORCE-ADMIT adaptive Fourier columns: a single adaptive sin OR cos has a
    # LOW marginal |corr| / MI for a phase-shifted oscillation (the phase is
    # split between the two), so the per-column MI-uplift gate would drop them
    # even though the sin+cos PAIR recovers the signal. The adaptive detector
    # already validated the frequency on a held-out slice, so both legs are
    # admitted unconditionally here; the downstream MRMR adaptive-protection
    # block then keeps them past screening. Append in deterministic name order.
    _adaptive_names = [nm for nm, m in meta.items() if m.get("basis") == "fourier" and m.get("adaptive", False)]
    _keep_set = set(keep)
    for nm in _adaptive_names:
        if nm not in _keep_set and nm in engineered.columns:
            keep.append(nm)
            _keep_set.add(nm)
    recipes = []
    for name in keep:
        if name not in meta:
            continue
        r = _build_recipe_from_meta(name, meta[name])
        if r is not None:
            recipes.append(r)
    # OUTPUT at FULL n. Without subsampling ``engineered`` already holds full-n columns.
    # With subsampling, REPLAY each winner's recipe on the full X (the recipe carries the
    # detected frequency + axis params) so the appended columns are full length -- output
    # equals a full-data fit GIVEN the same winners. A winner without a replayable recipe is
    # dropped (it could not be reproduced at transform time anyway).
    if _do_sub:
        from ..engineered_recipes import apply_recipe
        _full_cols: dict = {}
        for r in recipes:
            try:
                _full_cols[r.name] = np.asarray(apply_recipe(r, X))
            except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                logger.warning("extra-basis subsample replay failed for %r; dropping.", r.name)
        X_aug = pd.concat([X, pd.DataFrame(_full_cols, index=X.index)], axis=1) if _full_cols else X.copy()
    else:
        X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores, recipes
