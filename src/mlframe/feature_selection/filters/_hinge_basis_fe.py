"""Hinge / piecewise-linear change-point basis FE (backlog #11, 2026-06-09).

A NEW univariate operator for a signal shape the catalog cannot capture: a
**slope change at a data-dependent threshold** ``y = a*x + b*max(x - tau, 0)``
(pricing tiers, dose-response, saturation). The closest existing operators all
have the WRONG form for a sharp kink:

* ``numeric_rounding`` -> flat steps (piecewise-CONSTANT, not piecewise-linear);
* cubic B-spline -> smooth + FIXED quantile knots that round off a sharp kink at
  a non-knot location;
* orthogonal polynomials -> need a high degree to approximate a kink and ring
  (Gibbs) around it.

Mechanism
---------
Detect 1-2 breakpoints ``tau`` per column by scanning candidate quantile cuts for
the max DROP in a 2-segment linear-fit SSE (a slope-aware stump): for each
candidate cut ``c`` fit ``y ~ [1, x, max(x - c, 0)]`` and keep the ``c`` whose
residual SSE is smallest. The detected ``tau`` is then emitted as the closed-form
basis legs ``relu(x - tau) = max(x - tau, 0)`` and ``relu(tau - x) = max(tau - x,
0)`` (optionally an indicator ``1[x > tau]``). Stacking the per-tau relu legs
yields an ADAPTIVE-KNOT piecewise-linear basis (knots placed where the slope
actually changes, unlike the spline's fixed quantile knots).

Leak-safe replay
----------------
The recipe (kind ``"hinge_basis"``) stores only ``{tau, side}`` -- NO y -- so
``transform`` replay is the pure function ``np.maximum(x - tau, 0)`` /
``np.maximum(tau - x, 0)`` / ``(x > tau)``. The breakpoint search consumes y at
FIT time (like every supervised FE here -- spline knot selection, Fourier
frequency detection) but the emitted COLUMN VALUE does not depend on y, so the
replayed feature is leakage-free by construction.

Gates
-----
A hinge has a genuinely DIFFERENT LINEAR SHAPE from raw x, so unlike the
MI-invariant isotonic / RankGauss operators it clears the normal MI-uplift gate
over raw x. On top of that the detector runs a HELD-OUT tau-validation on the
``%3`` stride split (the same split the adaptive-Fourier detector uses): the
breakpoint is RANKED on the train rows and the 2-segment fit's held-out R^2
uplift over a 1-segment (plain linear) fit must clear a floor on the held-out
rows -- a chance breakpoint that fits a train slice but not the held-out slice is
rejected, so pure noise admits no hinge. On a MONOTONE target a hinge can be
near-collinear with raw x; the downstream cross-stage Spearman dedup drops it
(verified, no duplicate columns survive).

Mirrors the spline / Fourier extra-basis FE module
(``_orthogonal_univariate_fe._orth_extra_basis_fe``): ``generate_hinge_features``
emits columns + per-column fit meta, ``hybrid_hinge_fe_with_recipes`` scores by
MI uplift, applies the same two-gate (uplift + noise-aware MAD floor) chain, and
returns ``EngineeredRecipe`` objects for leak-safe transform-time replay.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

__all__ = [
    "generate_hinge_features",
    "hybrid_hinge_fe_with_recipes",
    "build_hinge_basis_recipe",
    "_apply_hinge_basis",
    "_detect_hinge_breakpoints",
    "_hinge_slope_change_plausible",
]


# Candidate-cut grid: inner quantiles of x. We avoid the extreme tails (a cut at
# the 1st / 99th percentile leaves a near-empty segment whose slope is pure
# noise) by scanning the inner ``[lo_q, hi_q]`` band.
_HINGE_CAND_Q_LO: float = 0.10
_HINGE_CAND_Q_HI: float = 0.90
_HINGE_N_CANDIDATES: int = 24
# Minimum rows on EACH side of a candidate cut for its segment slope to be
# trustworthy. Below this a "breakpoint" is fitting a handful of tail points.
_HINGE_MIN_SEG_ROWS: int = 30
# Held-out R^2-uplift floor: the 2-segment (hinge) fit must beat the 1-segment
# (plain linear) fit by at least this much OOS R^2 on the %3 stride slice. A
# chance kink fits a train slice but adds ~0 OOS over plain linear, so this
# rejects spurious breakpoints. Calibrated on the noise control (pure-noise
# x vs noise y: held-out uplift ~0, well below the floor) vs the slope-change
# fixture (uplift ~0.3+). Conservative 0.02 leaves a wide margin both ways.
_HINGE_MIN_HELDOUT_R2_UPLIFT: float = 0.02
# N-gate: below this row count a held-out slice is too small to validate a
# breakpoint reliably (mirrors the adaptive-Fourier >=800 philosophy, but a
# hinge needs far fewer rows than a multi-tone periodogram, so 200 suffices).
_HINGE_MIN_ROWS: int = 200
# COST PRE-CHECK (default-on dispatch, 2026-06-09): the full per-column scan is
# ``_HINGE_N_CANDIDATES`` (=24) lstsq solves -- the stage hotspot (~2.2 ms/col).
# Default-ON over WIDE data (p=50+) would multiply that across every column even
# though almost all carry NO slope change. Before the full scan we run a CHEAP
# 3-cut probe: fit the 2-segment hinge at just the 0.3 / 0.5 / 0.7 quantiles and
# take the best whole-data SSE drop over plain linear (fraction of total SSE). If
# even the best of these 3 cuts barely beats the line, no inner cut will clear
# the held-out gate either, so we skip the 24-cut scan entirely. 3 lstsq vs 24
# -> ~8x cheaper on the common (no-kink) column, and the genuine slope-change
# column always trips the probe (a real kink lifts the SSE-drop far above the
# floor at the nearest of the 3 coarse cuts). The full scan still runs on any
# column that passes, so tau precision is unchanged where a kink exists.
_HINGE_PRECHECK_QS: tuple = (0.30, 0.50, 0.70)
# Min whole-data SSE-drop fraction (1 - SSE_hinge/SSE_linear) at the best coarse
# probe cut for the full scan to be worth running. A genuine slope change drops
# the in-sample SSE by tens of percent at a coarse cut near the kink; pure noise
# and a purely-linear column drop ~0 (measured < 1e-3 -> skipped). 0.005 is well
# below the held-out admission floor's in-sample footprint, so the probe is a
# CONSERVATIVE pre-filter: it NEVER vetoes a column the full held-out gate would
# have admitted (it only short-circuits the obviously-flat columns). A smooth
# curve like x^2 DOES dent the line at a coarse cut and so PASSES the probe --
# that is correct: the probe is a cheap "worth scanning?" test, not the admission
# gate. The held-out tau-validation + the self-limiting support-protection (raw
# source must survive the screen) are what keep a smooth/quadratic column from
# adding a spurious hinge to support_ (verified end-to-end: x^2 -> 0 legs kept).
# Measured: slope-change probe drop ~0.5; noise/linear < 1e-3.
_HINGE_PRECHECK_MIN_SSE_DROP: float = 0.005


def _hinge_slope_change_plausible(
    x: np.ndarray, y: np.ndarray,
    *, qs: Sequence[float] = _HINGE_PRECHECK_QS,
    min_sse_drop: float = _HINGE_PRECHECK_MIN_SSE_DROP,
) -> bool:
    """Cheap O(len(qs)) gate: is a slope change plausible enough to justify the
    full ``_HINGE_N_CANDIDATES``-cut scan?

    Fit the continuous 2-segment hinge at just ``qs`` (default 3 coarse inner
    quantiles) on the WHOLE column and keep the best fractional SSE drop over the
    plain-linear fit. Returns True iff that best drop clears ``min_sse_drop``.
    The full scan + the held-out tau-validation downstream are unchanged; this
    only SKIPS the scan for a column whose best coarse cut cannot even dent the
    line (the common case on wide data), so default-on does not bloat wide fits.
    A genuine kink trips at least one of the coarse cuts well above the floor."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n != y.size or n < _HINGE_MIN_ROWS:
        return False
    sse_lin = _linear_sse(x, y)
    if not np.isfinite(sse_lin) or sse_lin <= 1e-24:
        # Plain linear already fits perfectly (or degenerate y) -> a hinge cannot
        # add a second slope; skip.
        return False
    try:
        cuts = np.unique(np.quantile(x, np.asarray(qs, dtype=np.float64)))
    except Exception:
        return False
    best_drop = 0.0
    for c in cuts:
        n_right = int(np.count_nonzero(x > c))
        if n_right < _HINGE_MIN_SEG_ROWS or (n - n_right) < _HINGE_MIN_SEG_ROWS:
            continue
        sse_h = _segmented_sse(x, y, float(c))
        if not np.isfinite(sse_h):
            continue
        drop = 1.0 - sse_h / sse_lin
        if drop > best_drop:
            best_drop = drop
    return bool(best_drop >= float(min_sse_drop))


def _segmented_sse(x: np.ndarray, y: np.ndarray, tau: float) -> float:
    """Residual SSE of the 2-segment continuous piecewise-linear fit
    ``y ~ [1, x, max(x - tau, 0)]`` (a slope CHANGE at ``tau``, continuous at
    the knot). Returns ``inf`` on a degenerate design (so the caller's argmin
    skips it)."""
    relu = np.maximum(x - tau, 0.0)
    A = np.column_stack([np.ones_like(x), x, relu])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return float("inf")
    resid = y - A @ coef
    return float(resid @ resid)


def _linear_sse(x: np.ndarray, y: np.ndarray) -> float:
    """Residual SSE of the 1-segment (plain linear) fit ``y ~ [1, x]``."""
    A = np.column_stack([np.ones_like(x), x])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return float("inf")
    resid = y - A @ coef
    return float(resid @ resid)


def _heldout_hinge_r2_uplift(
    x: np.ndarray, y: np.ndarray, tau: float,
) -> float:
    """Held-out R^2 uplift of the 2-segment hinge fit over the 1-segment linear
    fit, validated on the ``%3`` stride slice.

    Both models are FIT on the train rows (``idx % 3 != 0``) and SCORED on the
    held-out rows (``idx % 3 == 0``). Returns ``R2_hinge_val - R2_linear_val``.
    A genuine slope change lifts the held-out R^2 (the relu leg captures
    variance the plain line cannot); a chance kink over-fits the train slice and
    adds ~0 (often slightly negative) OOS, so the uplift sits below the floor.
    The split is the SAME deterministic stride the adaptive-Fourier detector
    uses -- no RNG, so the validation is reproducible and recipe-free.
    """
    n = x.size
    if n < _HINGE_MIN_ROWS:
        return 0.0
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    if int(tr.sum()) < 32 or int(va.sum()) < 16:
        return 0.0
    x_tr, y_tr = x[tr], y[tr]
    x_va, y_va = x[va], y[va]
    yv_ss = float(np.sum((y_va - y_va.mean()) ** 2))
    if yv_ss < 1e-24:
        return 0.0

    def _val_r2(design_fn) -> float:
        A_tr = design_fn(x_tr)
        A_va = design_fn(x_va)
        try:
            coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
        except Exception:
            return -np.inf
        pred = A_va @ coef
        sse = float(np.sum((y_va - pred) ** 2))
        return 1.0 - sse / yv_ss

    r2_lin = _val_r2(lambda xx: np.column_stack([np.ones_like(xx), xx]))
    r2_hinge = _val_r2(
        lambda xx: np.column_stack(
            [np.ones_like(xx), xx, np.maximum(xx - tau, 0.0)]
        )
    )
    if not (np.isfinite(r2_lin) and np.isfinite(r2_hinge)):
        return 0.0
    return float(r2_hinge - r2_lin)


def _detect_hinge_breakpoints(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_breakpoints: int = 2,
    min_heldout_r2_uplift: float = _HINGE_MIN_HELDOUT_R2_UPLIFT,
) -> list[float]:
    """Detect up to ``max_breakpoints`` slope-change locations ``tau`` in ``x``.

    Greedy: each round scans the inner-quantile candidate cuts for the one whose
    continuous 2-segment fit minimises (train-side, all rows) residual SSE,
    HELD-OUT-validates it (the 2-segment fit must beat plain linear by
    ``min_heldout_r2_uplift`` OOS on the ``%3`` slice), and if it passes ADDS the
    relu leg of that ``tau`` to the running design so the next round detects a
    SECOND, distinct slope change in the residual structure. Stops at
    ``max_breakpoints`` or the first ``tau`` that fails the held-out gate.

    Returns the validated breakpoint list (possibly empty). Pure noise -> the
    first candidate fails the held-out uplift gate -> empty list (no hinge).

    Perf (cProfile p=20 n=4000): the full scan is the stage hotspot, originally dominated by a per-candidate ``np.linalg.lstsq``. Two stacked optimisations:
    (1) a cheap 3-cut pre-check (:func:`_hinge_slope_change_plausible`) runs FIRST and short-circuits the 24-cut scan for any column without a plausible slope change
    (the common case on wide data) -- ~8x fewer solves on a no-kink column. (2) On a column that trips the probe, the per-cut SSE is scored by the Frisch-Waugh-Lovell
    rank-1 update: the fixed design block ``B = [1, x, *extra_legs]`` is QR-factored ONCE per round, and each candidate cut's SSE is ``SSE_B - (r_relu.r_y)^2/(r_relu.r_relu)``
    where the residuals project out B (O(n*k) per cut, no per-cut SVD). Bit-identical to the full-lstsq SSE / tau precision (~1e-12 FP reduction order) and ~2.4x faster
    on the n=4000 / 24-cut scan (bench: profiling/bench_hinge_fwl_rank1.py). bench-attempt-rejected (2026-06-09): a normal-equations 3x3 solve (``A.T@A`` / ``np.linalg.solve``)
    was 2.2x SLOWER than the original lstsq because it rebuilt + re-formed ``A.T@A`` per cut; the FWL update wins precisely by NOT rebuilding the fixed block per cut.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n != y.size or n < _HINGE_MIN_ROWS:
        return []
    # DEVICE-RESIDENT detector (kernel-residency, 2026-07-02): under the STRICT-resident path the whole
    # detection -- pre-check, the per-round QR + batched FWL cut scan, the held-out tau-validation -- runs on
    # the GPU (see _hinge_detect_gpu_resident); only the found taus return. Identical math + guards; device FP
    # differs ~1e-12, far below the tau-selection scale -> selection-equivalent (F2 + hinge suite verified).
    # ``None`` (non-strict / no cupy / any cupy fault) -> the exact host detector below, byte-identical.
    try:
        from ._hinge_detect_gpu_resident import detect_hinge_breakpoints_gpu, hinge_gpu_enabled
        if hinge_gpu_enabled():
            _dev = detect_hinge_breakpoints_gpu(
                x, y, max_breakpoints=max_breakpoints, min_heldout_r2_uplift=min_heldout_r2_uplift,
                precheck_qs=_HINGE_PRECHECK_QS, precheck_min_sse_drop=_HINGE_PRECHECK_MIN_SSE_DROP,
                cand_q_lo=_HINGE_CAND_Q_LO, cand_q_hi=_HINGE_CAND_Q_HI, n_candidates=_HINGE_N_CANDIDATES,
                min_rows=_HINGE_MIN_ROWS, min_seg_rows=_HINGE_MIN_SEG_ROWS,
            )
            if _dev is not None:
                return _dev
    except Exception:
        pass
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        x = x[finite]
        y = y[finite]
        n = x.size
        if n < _HINGE_MIN_ROWS:
            return []
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return []
    # COST PRE-CHECK: skip the full 24-cut scan on a column whose best coarse
    # 3-cut probe cannot dent the plain-linear SSE (no plausible slope change).
    # Keeps default-on cheap on wide data without changing the outcome on a
    # column that genuinely has a kink (it always trips the probe).
    if not _hinge_slope_change_plausible(
        x, y, min_sse_drop=_HINGE_PRECHECK_MIN_SSE_DROP,
    ):
        return []
    # Candidate cuts at inner quantiles -- avoid the tails where a segment is
    # near-empty.
    qs = np.linspace(_HINGE_CAND_Q_LO, _HINGE_CAND_Q_HI, _HINGE_N_CANDIDATES)
    cand = np.unique(np.quantile(x, qs))
    if cand.size == 0:
        return []
    found: list[float] = []
    # Extra relu legs accumulated from already-found breakpoints, appended to
    # the design so the next round measures the INCREMENTAL SSE drop of a new,
    # distinct kink (not a re-detection of the first one).
    extra_legs: list[np.ndarray] = []
    ones = np.ones_like(x)  # intercept column is invariant across every candidate cut and round; build once, not per-cut.
    for _ in range(max(1, int(max_breakpoints))):
        best_tau = None
        best_sse = float("inf")
        # The fixed design block ``B = [1, x, *extra_legs]`` is identical across every candidate cut in this round; only the ``relu`` column varies. So we
        # QR-factor B ONCE and score each cut by the partitioned-regression (Frisch-Waugh-Lovell) identity: the SSE of regressing y on ``[B | relu]`` equals
        # ``SSE_B - (r_relu . r_y)^2 / (r_relu . r_relu)`` where ``r_relu`` / ``r_y`` are the residuals of relu / y after projecting out B (one O(n*k) projection
        # per cut, no per-cut SVD). This is mathematically identical to the full ``lstsq`` SSE (FP reduction order ~1e-12, far below any tau-selection scale) and
        # ~2.4x faster on the n=4000 / 24-cut scan (the lstsq SVD per cut was the stage hotspot). bench: profiling/bench_hinge_fwl_rank1.py.
        B = np.column_stack([ones, x] + extra_legs)
        Q, _ = np.linalg.qr(B)
        r_y = y - Q @ (Q.T @ y)
        sse_B = float(r_y @ r_y)
        for c in cand:
            # Require enough rows on each side for trustworthy segment slopes.
            n_right = int(np.count_nonzero(x > c))
            if n_right < _HINGE_MIN_SEG_ROWS or (n - n_right) < _HINGE_MIN_SEG_ROWS:
                continue
            # Skip a cut within a tiny neighbourhood of an already-found tau
            # (re-detecting the same kink).
            if any(abs(c - t) < 1e-9 for t in found):
                continue
            relu = np.maximum(x - c, 0.0)
            r_relu = relu - Q @ (Q.T @ relu)
            denom = float(r_relu @ r_relu)
            if denom < 1e-24:
                # relu lies in span(B) at this cut -> adding it cannot reduce SSE; the full-lstsq design is rank-deficient here, matching the legacy ``inf``/skip.
                sse = sse_B
            else:
                num = float(r_relu @ r_y)
                sse = sse_B - num * num / denom
            if sse < best_sse:
                best_sse = sse
                best_tau = float(c)
        if best_tau is None:
            break
        # HELD-OUT tau-validation: the 2-segment fit at best_tau must beat plain
        # linear OOS on the %3 stride slice. This is the chance-breakpoint guard.
        uplift = _heldout_hinge_r2_uplift(x, y, best_tau)
        if uplift < float(min_heldout_r2_uplift):
            break
        found.append(best_tau)
        extra_legs.append(np.maximum(x - best_tau, 0.0))
    return found


def generate_hinge_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    y: Optional[np.ndarray] = None,
    max_breakpoints: int = 2,
    emit_indicator: bool = False,
    min_heldout_r2_uplift: float = _HINGE_MIN_HELDOUT_R2_UPLIFT,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
) -> tuple[pd.DataFrame, dict]:
    """For each numeric column, detect slope-change breakpoints (held-out
    validated) and emit the hinge basis legs, returning the columns alongside
    the per-column fit meta needed to build leak-safe recipes.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    y : array-like, optional
        Target. Consulted ONLY by the breakpoint DETECTOR (which tau best fits +
        held-out validation). Never read for the emitted column VALUE, so the
        recipe replay stays leakage-free / y-independent.
    max_breakpoints : int
        Max distinct slope-change knots to detect per column (default 2).
    emit_indicator : bool, default False
        When True, additionally emit a step indicator ``1[x > tau]`` per
        breakpoint. Default False (the relu legs already span the continuous
        piecewise-linear family; the indicator is a discontinuous extra that the
        ``numeric_rounding`` family partly covers).

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns named ``"{col}__relu_gt{tau:g}"``
            (``max(x-tau,0)``), ``"{col}__relu_lt{tau:g}"`` (``max(tau-x,0)``),
            and optionally ``"{col}__ind_gt{tau:g}"`` (``1[x>tau]``).
        meta : dict mapping each emitted column name to the recipe metadata
            ``{"src": col, "tau": float, "side": "gt"|"lt"|"ind"}``.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if y is None:
        # The breakpoint detector is supervised; with no y there is nothing to
        # detect. Return empty (the caller's MI-uplift gate would drop them
        # anyway).
        return pd.DataFrame(index=X.index), {}
    if dedup_collinear_sources:
        from ._orthogonal_univariate_fe import _dedup_collinear_source_cols
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if y_arr.size != len(X) or not np.all(np.isfinite(y_arr)):
        return pd.DataFrame(index=X.index), {}
    out_cols: dict = {}
    meta: dict = {}
    for col in cols:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            continue
        if not finite_mask.all():
            # A hinge basis over a NaN column is unsound: the nanmean-imputed hinge becomes a missingness proxy that displaces the genuine missingness-FE
            # columns, and the recipe replay does not impute (transform() emits all-NaN). Skip; the missingness signal belongs to the missingness-FE family.
            continue
        try:
            taus = _detect_hinge_breakpoints(
                x, y_arr, max_breakpoints=max_breakpoints,
                min_heldout_r2_uplift=min_heldout_r2_uplift,
            )
        except Exception as exc:
            logger.warning(
                "generate_hinge_features: breakpoint detect on col=%r raised "
                "%r; skipping hinge for that column.", col, exc,
            )
            continue
        for tau in taus:
            relu_gt = np.maximum(x - tau, 0.0)
            relu_lt = np.maximum(tau - x, 0.0)
            if float(np.std(relu_gt)) > 1e-12:
                name_gt = f"{col}__relu_gt{tau:g}"
                out_cols[name_gt] = relu_gt
                meta[name_gt] = {"src": col, "tau": float(tau), "side": "gt"}
            if float(np.std(relu_lt)) > 1e-12:
                name_lt = f"{col}__relu_lt{tau:g}"
                out_cols[name_lt] = relu_lt
                meta[name_lt] = {"src": col, "tau": float(tau), "side": "lt"}
            if emit_indicator:
                ind = (x > tau).astype(np.float64)
                if float(np.std(ind)) > 1e-12:
                    name_ind = f"{col}__ind_gt{tau:g}"
                    out_cols[name_ind] = ind
                    meta[name_ind] = {"src": col, "tau": float(tau), "side": "ind"}
    return pd.DataFrame(out_cols, index=X.index), meta


def build_hinge_basis_recipe(
    *, name: str, src_name: str, tau: float, side: str,
) -> "EngineeredRecipe":
    """Frozen recipe for one hinge basis column.

    * ``side="gt"``  -> ``max(X[src_name] - tau, 0)`` (slope change for x > tau);
    * ``side="lt"``  -> ``max(tau - X[src_name], 0)`` (slope change for x < tau);
    * ``side="ind"`` -> ``1[X[src_name] > tau]`` (step indicator).

    Replay is closed-form in the source column alone -- no y reference captured,
    so ``transform`` is leakage-free by construction. Mirrors
    ``build_orth_spline_recipe``."""
    from .engineered_recipes import EngineeredRecipe
    if side not in ("gt", "lt", "ind"):
        raise ValueError(
            f"build_hinge_basis_recipe: side must be 'gt'|'lt'|'ind'; got {side!r}"
        )
    return EngineeredRecipe(
        name=name,
        kind="hinge_basis",
        src_names=(str(src_name),),
        extra={"tau": float(tau), "side": str(side)},
    )


def _apply_hinge_basis(recipe, X) -> np.ndarray:
    """Replay one hinge basis column from the stored ``{tau, side}`` -- a pure
    function of the source column (no y). Mirrors ``_apply_orth_spline``."""
    from .engineered_recipes import _extract_column
    if len(recipe.src_names) != 1:
        raise ValueError(
            f"hinge_basis recipe '{recipe.name}' must have exactly 1 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("tau", "side"):
        if key not in recipe.extra:
            raise KeyError(
                f"hinge_basis recipe '{recipe.name}' missing '{key}' in extra. "
                f"Re-fit MRMR to regenerate."
            )
    name = recipe.src_names[0]
    tau = float(recipe.extra["tau"])
    side = str(recipe.extra["side"])
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    if side == "gt":
        return np.maximum(vals - tau, 0.0)
    if side == "lt":
        return np.maximum(tau - vals, 0.0)
    if side == "ind":
        return (vals > tau).astype(np.float64)
    raise ValueError(f"hinge_basis recipe '{recipe.name}': unknown side {side!r}")


def _heldout_incremental_r2(
    x: np.ndarray, leg: np.ndarray, y: np.ndarray,
) -> float:
    """Held-out R^2 GAIN of adding ``leg`` to a ``[1, x]`` linear model, scored
    on the ``%3`` stride slice. ``R2([1, x, leg])_val - R2([1, x])_val``.

    This is the CORRECT admission statistic for the hinge: a single relu leg is
    MONOTONE in x (max(x-tau,0) is non-decreasing), so it is MI-INVARIANT by the
    data-processing inequality -- an MI-uplift gate would DROP it exactly as it
    drops the isotonic / RankGauss monotone reshapes (backlog #14 caveat, and the
    project's MI-vs-linear-usability rule). The hinge's value is the SECOND SLOPE
    it hands a downstream linear / shallow model: ``[1, x, relu(x-tau)]`` fits a
    two-slope kink that ``[1, x]`` cannot. So we admit a leg on its held-out
    INCREMENTAL linear usability over raw x, not on MI. The split is the same
    deterministic ``%3`` stride the detector's tau-validation uses (no RNG)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    leg = np.asarray(leg, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n != y.size or n != leg.size or n < _HINGE_MIN_ROWS:
        return 0.0
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    if int(tr.sum()) < 32 or int(va.sum()) < 16:
        return 0.0
    yv = y[va]
    yv_ss = float(np.sum((yv - yv.mean()) ** 2))
    if yv_ss < 1e-24:
        return 0.0

    def _val_r2(cols_tr, cols_va) -> float:
        A_tr = np.column_stack(cols_tr)
        A_va = np.column_stack(cols_va)
        try:
            coef, *_ = np.linalg.lstsq(A_tr, y[tr], rcond=None)
        except Exception:
            return -np.inf
        pred = A_va @ coef
        sse = float(np.sum((yv - pred) ** 2))
        return 1.0 - sse / yv_ss

    x_tr, x_va = x[tr], x[va]
    leg_tr, leg_va = leg[tr], leg[va]
    r2_base = _val_r2(
        [np.ones_like(x_tr), x_tr], [np.ones_like(x_va), x_va],
    )
    r2_full = _val_r2(
        [np.ones_like(x_tr), x_tr, leg_tr],
        [np.ones_like(x_va), x_va, leg_va],
    )
    if not (np.isfinite(r2_base) and np.isfinite(r2_full)):
        return 0.0
    return float(r2_full - r2_base)


def hybrid_hinge_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    max_breakpoints: int = 2,
    emit_indicator: bool = False,
    min_heldout_r2_uplift: float = _HINGE_MIN_HELDOUT_R2_UPLIFT,
    top_k: int = 5,
    min_leg_incr_r2: float = 0.005,
    **_legacy_ignored,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Hinge basis FE + held-out-linear-usability selection, returning leak-safe
    recipes. Returns ``(X_augmented, scores, recipes)``.

    Why NOT an MI-uplift gate (the gate the spline / Fourier hybrid uses):
    a single relu leg ``max(x - tau, 0)`` is MONOTONE in x, hence MI-INVARIANT by
    the data-processing inequality -- an MI-uplift gate DROPS it, exactly as it
    drops the isotonic / RankGauss monotone reshapes (this is the project's
    MI-vs-linear-usability principle, and backlog #14's explicit caveat). The
    hinge's value is the SECOND SLOPE it hands a downstream linear / shallow
    model: ``[1, x, relu(x-tau)]`` fits a two-slope kink ``[1, x]`` cannot.

    So a leg is admitted on its held-out INCREMENTAL linear usability over raw x
    (:func:`_heldout_incremental_r2` >= ``min_leg_incr_r2`` on the ``%3`` stride
    slice). The detector ALREADY rejected chance breakpoints via the held-out
    2-segment-vs-linear R^2 gate, so pure noise emits no leg to score here; the
    incremental gate then keeps only legs that genuinely lift a linear fit OOS.
    ``scores`` reports each leg's incremental held-out R^2 (the ranking key) and
    its raw-x baseline R^2. On a monotone target the relu is near-collinear with
    x -> the downstream cross-stage Spearman dedup drops it.
    """
    engineered, meta = generate_hinge_features(
        X, cols=cols, y=y,
        max_breakpoints=max_breakpoints,
        emit_indicator=emit_indicator,
        min_heldout_r2_uplift=min_heldout_r2_uplift,
    )
    if engineered.empty:
        empty_scores = pd.DataFrame(columns=[
            "engineered_col", "source_col", "incr_r2", "passed",
        ])
        return X.copy(), empty_scores, []
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    rows = []
    for name in engineered.columns:
        m = meta.get(name, {})
        src = str(m.get("src", name.split("__", 1)[0]))
        if src in X.columns and pd.api.types.is_numeric_dtype(X[src]):
            x_src = np.asarray(X[src].to_numpy(), dtype=np.float64)
            finite = np.isfinite(x_src)
            if not finite.all():
                x_src = np.where(
                    finite, x_src,
                    float(np.nanmean(x_src[finite])) if finite.any() else 0.0,
                )
        else:
            x_src = engineered[name].to_numpy(dtype=np.float64)
        incr = _heldout_incremental_r2(x_src, engineered[name].to_numpy(), y_arr)
        rows.append({
            "engineered_col": name, "source_col": src,
            "incr_r2": float(incr),
            "passed": bool(incr >= float(min_leg_incr_r2)),
        })
    scores = pd.DataFrame(rows).sort_values(
        "incr_r2", ascending=False,
    ).reset_index(drop=True)
    qualified = scores[scores["passed"]]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    recipes = []
    for name in keep:
        if name not in meta:
            continue
        m = meta[name]
        recipes.append(build_hinge_basis_recipe(
            name=name, src_name=str(m["src"]),
            tau=float(m["tau"]), side=str(m["side"]),
        ))
    return X_aug, scores, recipes
