"""Full docstring of ``generate_extra_basis_features``, kept out of the function module so the function itself stays within the function-length gate."""

from __future__ import annotations

GENERATE_EXTRA_BASIS_FEATURES_DOC = """For each column in cols and each requested extra basis, emit the basis
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
        on each source column's z (power==1 only) and - for each held-out-
        validated dominant frequency found (multitone: several peaks via
        residual deflation) - ADD it to that column's Fourier frequency set.
        The emitted sin/cos meta entries for adaptive frequencies are tagged
        ``"adaptive": True`` so MRMR can protect them past screening. Covers
        arbitrary-period oscillations and their superpositions
        (``sin(3.7*x) + sin(5.3*x) + sin(6.8*x)``) the fixed grid {1, 2} misses.
    fourier_adaptive_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the adaptive detector.
    fourier_chirp : bool, default False
        ADAPTIVE-CHIRP path. When True and ``y`` is given, run the
        SAME held-out-validated detector on the QUADRATIC-ARGUMENT warp
        ``u = sign(z) * z**2`` (z standardised on the column) for each source
        column. A chirp ``y ~ sin(2*pi*f*z**2)`` - whose frequency GROWS with z
        - is STATIONARY in u, so the detector locks its frequency and the
        emitted ``sin(2*pi*f*u)`` / ``cos(2*pi*f*u)`` reconstruct it; a Fourier on
        the LINEAR argument cannot express a frequency that grows with z. The
        emitted sin/cos meta entries carry ``"arg": "quadratic"`` (the warp the
        recipe replays) AND ``"adaptive": True`` (so MRMR protects them past the
        screen, identical to the linear adaptive legs). This is an ADDITIVE
        second path alongside the linear adaptive one - both fire; on a plain
        linear target the chirp legs are harmless (Ridge regularises them to ~0,
        Phase-0 bench: combined R^2 == linear-only on linear targets, +0.3-0.5
        R^2 on fast chirps). N-gated at >= 800 MI rows like the linear path.
    fourier_chirp_min_val_corr : float, default 0.15
        Held-out validation effective-|corr| floor for the chirp detector.
    max_adaptive_cols : int, optional
        2026-07-09 fix: cap on how many columns run the EXPENSIVE adaptive-frequency /
        chirp DETECTION (``_detect_fourier_freqs_for_col``) - profiled as the dominant
        cost of this whole family (~34% of a wide-p fit's pre-categorize wall, roughly
        linear in column count since each call does its own held-out frequency sweep
        regardless of row count). ``None`` (default) = unlimited, byte-identical legacy
        behaviour. When set and ``len(cols) > max_adaptive_cols``, only the first
        ``max_adaptive_cols`` columns (in ``cols`` order) get adaptive/chirp detection;
        the remaining columns still get the cheap FIXED-GRID Fourier basis (unaffected
        by this cap) - only the expensive per-column detector is bounded, not basis
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
    - emit only the better of {orth-poly basis, this Fourier path} per column to
    cut the redundant cross-family features that co-occur in the support - was
    benchmarked and REJECTED. The co-occurrence is genuine COMPLEMENTARITY, not
    redundancy: on kink/step/bump targets (e.g. y=|x|) the Fourier legs carry
    independent residual R^2 0.16-0.60 that a degree<=4 poly under-fits, so a
    winner-takes-all gate HURT the |x| target OOS by -0.06; no OOS win on a tree
    downstream (deltas +/-0.008, inconsistent sign); on a mixed frame the gate
    declines to fire (different columns have different winners). The existing
    Fleuret redundancy + Spearman cross-stage dedup already remove the only real
    redundancy. Don't add a competition gate. (D:/Temp/item3_poly_fourier_findings.md)
    """
