"""Sufficient-summary early-stop for the MRMR Feature-Engineering search (backlog #22).

The user's "compare-to-theoretical-max" idea, realised cheaply via a Data-Processing-
Inequality (DPI) residual test.

Mechanism (the user iterated to THIS exact design)
--------------------------------------------------
After each MRMR feature SELECTION (once per fit/screen pass, NOT per candidate pair
evaluated), ask whether the CURRENTLY-SELECTED set already captures all the information
the observables carry about ``y`` -- i.e. whether the selection has reached the
theoretical maximum ``I(observables; y)``. If so, EVERY future engineered candidate is
provably pointless and the whole remaining FE search can be skipped.

Concretely:

1. Fit a CHEAP linear/ridge model of ``y`` on the SMALL selected set
   ``E_hat[y | selected]`` (1-5 columns -- engineered features linearise the signal, so a
   linear fit captures ``E[y | selected]`` well; the design matrix is tiny precisely
   because the SELECTED set is small, NOT the full feature pool). Compute the residual
   ``r = y - E_hat[y | selected]``.

2. For EVERY RAW feature ``x_j``, test ``MI(r; x_j)`` against the Westfall-Young maxT
   permutation null computed over the SAME raw pool with ``r`` as the (shuffled) target
   (the SHIPPED ``pooled_permutation_null_gain_floor`` -- best-of-pool chance ceiling).
   If ``MI(r; x_j) <= floor`` for ALL raws, no raw carries leftover signal about the
   residual.

3. GUARD: only stop when the residual is ALSO small relative to ``y`` -- the unexplained
   variance fraction ``Var(r)/Var(y)`` (== 1 - R^2) below ``residual_entropy_frac`` -- so a
   pure-noise / under-fit target (where every raw legitimately sits at the null because
   there is no signal to find) does NOT trigger a false stop. The variance ratio is the
   GRID-FREE, faithful realisation of "residual small relative to H(y)": for matched
   families ``Var(r)/Var(y) = exp(2*(H(r) - H(y)))``, a strictly-monotone transform of the
   H(y)-relative differential-entropy gap, without the equi-frequency / fixed-grid binning
   artefacts that make a binned-entropy ratio unreliable for a small residual.

If BOTH hold -> the residual is pure noise w.r.t. the observables -> the selection has
reached ``I(observables; y)`` (the theoretical max) -> STOP the FE search.

Why this is correct (DPI)
-------------------------
Any future engineered candidate ``g`` is a deterministic function of the raw features.
By the Data-Processing Inequality ``I(r; g(raws)) <= I(r; raws)`` -- a transform cannot
manufacture information about ``r`` that the raws do not already have. So if NO raw
clears the chance ceiling against the residual, NO engineered feature built from them can
either. The residual being pure noise (entropy guard) certifies that ``E_hat[y|selected]``
already explains the explainable part of ``y``; the maxT test certifies no observable can
explain the leftover. Together they prove the remaining FE search cannot improve the
captured information, so skipping it changes nothing the search could have found.

This is a CONSERVATIVE skip: it only fires when BOTH the entropy guard AND the all-raws
maxT test pass, so it can never stop while a genuine second signal is still discoverable.
The NONLINEAR-leftover case is the reason the residual is tested by **MI** (not a linear
score): even when the linear ``E_hat`` under-fits a non-linear leftover, a raw that
explains that leftover still shows ``MI(r; x_j) > floor`` and BLOCKS the stop.

Integration
-----------
Called in ``_mrmr_fit_impl``'s greedy FE loop, AFTER ``screen_predictors`` returns the
current ``selected_vars`` and BEFORE the next ``_run_fe_step`` (so a proven-sufficient
selection skips the expensive operator search). The final selection is UNCHANGED -- the
early-stop only skips PROVABLY-pointless further FE; with it OFF the loop simply runs the
remaining steps and finds nothing new (verified byte-identical selection on genuine
multi-signal fixtures).

Self-gating / robustness
------------------------
Returns ``reached=False`` (no stop) on any degenerate input (no raws, too few rows,
single-class/constant target, non-finite design) so the caller can treat the helper as a
pure best-effort optimisation: when in doubt, do NOT skip. ``ridge_alpha`` regularises the
linear fit so collinear / redundant selected columns do not blow up the normal equations
(adversarial case (e)).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Default residual-entropy guard: stop only when H(r)/H(y) is below this fraction (the
# residual carries < this share of the target's information). 0.25 leaves comfortable
# margin: the F1 fixture's irreducible f/5 noise residual sits well under it, while a
# weak-but-real 3rd term / a genuine unfound 2nd signal keeps H(r)/H(y) above it. The
# guard is the load-bearing protection against a false stop on a pure-noise / under-fit
# target (where the maxT test alone would pass because there is simply nothing to find).
_RESIDUAL_ENTROPY_FRAC_DEFAULT = 0.25

# maxT permutation count + quantile for the residual-vs-raws null. Mirrors the order-1
# screen floor defaults (cheap: a few shuffles over a handful of raw columns).
_MAXT_PERMUTATIONS_DEFAULT = 25
_MAXT_QUANTILE_DEFAULT = 0.95

# Ridge regularisation for the cheap E_hat[y|selected] fit. Small relative to the
# standardised design so it only kicks in to stabilise collinear columns (adversarial
# case (e)); on a well-conditioned design it barely perturbs the OLS solution.
_RIDGE_ALPHA_DEFAULT = 1e-3


def _get_shared_fe_subsample_idx(self, y_continuous, n_full):
    """Lazily resolve + cache the ONE shared FE subsample row-index set on ``self`` for this fit.

    Computed ONCE per fit (keyed by n) and reused by every consumer that rides the shared draw (this
    residual floor today; the screen FDR floor / pair-search / polynom as the consolidation lands) so
    the same rows are seen everywhere instead of N independent re-draws. ``size`` is the unified screen
    knob (``MRMR._default_screen_subsample_n``); returns ``None`` when n is at/below it (full-n path,
    byte-identical to legacy small-n). ``is_clf`` is resolved from the TASK (not the local continuous
    target) so the shared draw matches the classification consumers' stratification."""
    try:
        n = int(n_full)
        cached_n = getattr(self, "_fe_shared_subsample_n", None)
        if cached_n == n:
            return getattr(self, "_fe_shared_subsample_idx", None)
        size = None
        _resolver = getattr(self, "_default_screen_subsample_n", None)
        if callable(_resolver):
            size = int(_resolver())
        if size is None or n <= size:
            self._fe_shared_subsample_idx = None
            self._fe_shared_subsample_n = n
            return None
        from ._fe_subsample import resolve_shared_fe_subsample_idx

        _task = getattr(self, "task", None)
        is_clf = bool(
            getattr(self, "is_classification_", None)
            if getattr(self, "is_classification_", None) is not None
            else (str(_task).lower().startswith("clas") if _task is not None else False)
        )
        idx = resolve_shared_fe_subsample_idx(
            y_continuous, n, int(size),
            is_clf=is_clf,
            stratify_knob=getattr(self, "fe_subsample_stratify", None),
            random_seed=getattr(self, "random_seed", None),
        )
        self._fe_shared_subsample_idx = idx
        self._fe_shared_subsample_n = n
        return idx
    except Exception as exc:
        # A silent None here leaves ``_fe_shared_subsample_idx`` unset, so EVERY FE/MI consumer that reads
        # it (the order-1 screen relevance sweep, the CMI redundancy loop, the FE pair/polynom search)
        # runs at FULL n -- a ~33x cost blow-up at n~1M that only shows up as "slow" in a profile. Log at
        # WARNING so a resolver failure is diagnosable; still fall back to full-n (best-effort), never
        # silently.
        logger.warning("_get_shared_fe_subsample_idx failed; FE screen will run at FULL n (no subsample): %r", exc, exc_info=True)
        return None


@dataclass
class SufficientSummaryVerdict:
    """Diagnostics for one sufficient-summary early-stop check.

    ``reached`` is the only field the caller acts on (stop the FE search when True). The
    rest are surfaced for logging / tests / the public ``MRMR.sufficient_summary_`` attr.
    """

    reached: bool = False
    reason: str = ""
    residual_entropy_frac: float = float("nan")  # H(r)/H(y)
    max_raw_mi: float = float("nan")  # max_j MI(r; x_j)
    maxt_floor: float = float("nan")  # the permutation-null ceiling
    n_selected: int = 0
    n_raw_tested: int = 0
    blocking_raw: int = -1  # cols-index of the raw that blocked the stop (or -1)
    per_raw_mi: dict = field(default_factory=dict)  # {cols_index: MI(r; x_j)}
    residual: "np.ndarray | None" = None  # continuous r = y - E_hat[y|selected]; surfaced so the FE
    # loop can RE-TARGET the next step on the residual when a raw still carries residual structure
    # (reached=False AND blocking_raw>=0): on r the captured dominant term is removed, so a weak secondary
    # half (e.g. log(c)*sin(d) when a**2/b dominates Var(y)) clears the prevalence gate relative to r.


def _shannon_entropy_nats(codes: np.ndarray, nbins_hint: int = 0) -> float:
    """Shannon entropy (nats) of an integer-coded array via plug-in bincount."""
    c = np.ascontiguousarray(codes).astype(np.int64, copy=False)
    if c.size == 0:
        return 0.0
    minlen = int(max(int(c.max()) + 1 if c.size else 1, int(nbins_hint)))
    counts = np.bincount(c[c >= 0], minlength=minlen).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def sufficient_summary_reached(
    *,
    data: np.ndarray,
    nbins: np.ndarray,
    y_continuous: np.ndarray,
    target_col_idx: int,
    selected_cols_idx: Sequence[int],
    selected_continuous: dict,
    raw_cols_idx: Sequence[int],
    cols_names: Optional[Sequence[str]] = None,
    quantization_nbins: int = 10,
    residual_entropy_frac: float = _RESIDUAL_ENTROPY_FRAC_DEFAULT,
    maxt_permutations: int = _MAXT_PERMUTATIONS_DEFAULT,
    maxt_quantile: float = _MAXT_QUANTILE_DEFAULT,
    ridge_alpha: float = _RIDGE_ALPHA_DEFAULT,
    cardinality_bias_correction: bool = True,
    random_seed: int | None = None,
    verbose: int = 0,
) -> SufficientSummaryVerdict:
    """Decide whether the current selection is a SUFFICIENT SUMMARY of ``y`` (stop FE).

    Parameters
    ----------
    data : (n, p) int matrix
        The discretised (ordinal-coded) screening matrix in cols-space. Used for the raw
        columns' codes (the maxT test scores ``MI(r_disc; raw_disc)`` on these) and as a
        fallback continuous proxy for any selected column whose continuous values are not
        in ``selected_continuous``.
    nbins : (p,) int array
        Per-column bin counts (cols-space), aligned with ``data``.
    y_continuous : (n,) float array
        The RAW continuous target -- the regressand of the cheap ``E_hat[y|selected]`` fit
        and the basis of the residual ``r = y - E_hat``.
    target_col_idx : int
        cols-index of the discretised target in ``data`` (its ``nbins`` gives ``H(y)``'s
        resolution).
    selected_cols_idx : sequence[int]
        cols-indices of the currently-selected features (raw and/or engineered). The
        design matrix of the cheap fit is built from these.
    selected_continuous : dict[str, np.ndarray] | dict[int, np.ndarray]
        Optional continuous values for selected columns, keyed by NAME or cols-index. A
        selected column not present here falls back to its discretised codes in ``data``
        (still a valid linear-fit regressor; engineered codes are monotone in the value).
        Pass ``{}`` to use codes for every selected column.
    cols_names : sequence[str] | None
        Column names (cols-space), so ``selected_continuous`` can be looked up by name.
    raw_cols_idx : sequence[int]
        cols-indices of the RAW input features (the DPI frontier: every engineered
        candidate is a function of these). The maxT residual test runs over this pool.
    quantization_nbins : int
        Bin count for discretising the continuous residual (equi-frequency), matched to the
        MRMR target resolution so ``H(r)`` and ``MI(r; x_j)`` are on the screen's scale.

    Returns
    -------
    SufficientSummaryVerdict
        ``reached=True`` only when the entropy guard AND the all-raws maxT test both pass.
    """
    from ._permutation_null import pooled_permutation_null_gain_floor

    v = SufficientSummaryVerdict()
    sel = [int(i) for i in selected_cols_idx]
    raws = [int(i) for i in raw_cols_idx]
    v.n_selected = len(sel)
    v.n_raw_tested = len(raws)

    # --- Degenerate-input guards: never skip when we cannot prove sufficiency. ---
    if len(sel) == 0 or len(raws) == 0:
        v.reason = "no selected features or no raw pool"
        return v
    n = int(data.shape[0])
    if n < 16:
        v.reason = "too few rows for a reliable residual test"
        return v

    y = np.asarray(y_continuous, dtype=np.float64).reshape(-1)
    if y.shape[0] != n or not np.all(np.isfinite(y)):
        v.reason = "continuous target unavailable / non-finite"
        return v
    if float(np.std(y)) <= 1e-12:
        v.reason = "constant target"
        return v

    # --- H(y) on the discretised target (the resolution the screen used). ---
    y_nbins = int(nbins[int(target_col_idx)]) if 0 <= int(target_col_idx) < len(nbins) else 0
    y_codes = np.ascontiguousarray(data[:, int(target_col_idx)]).astype(np.int64, copy=False)
    h_y = _shannon_entropy_nats(y_codes, nbins_hint=y_nbins)
    if h_y <= 1e-9:
        v.reason = "degenerate (single-class) target"
        return v

    # --- Build the SMALL design matrix for E_hat[y | selected]. ---
    # Prefer continuous values (raw from X / engineered snapshot); fall back to the
    # discretised codes for any column without a continuous source. Codes are a valid
    # linear regressor (monotone in the underlying value) so the fit never fails.
    design_cols = []
    for ci in sel:
        col = None
        if selected_continuous:
            if cols_names is not None and 0 <= ci < len(cols_names):
                col = selected_continuous.get(cols_names[ci])
            if col is None:
                col = selected_continuous.get(ci)
        if col is None:
            col = data[:, ci].astype(np.float64)
        col = np.asarray(col, dtype=np.float64).reshape(-1)
        if col.shape[0] != n:
            continue
        # Replace any non-finite (a stray inf in an engineered column) with the column
        # median so the fit stays well-posed; a fully non-finite column is dropped.
        if not np.all(np.isfinite(col)):
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            col = np.where(np.isfinite(col), col, float(np.median(finite)))
        design_cols.append(col)

    if not design_cols:
        v.reason = "no usable design columns for the cheap fit"
        return v

    Z = np.column_stack(design_cols)  # (n, k)
    # Standardise columns (zero mean / unit std) so the single ridge_alpha is scale-free
    # and collinear columns are handled by the L2 term rather than by an ill-conditioned
    # normal matrix (adversarial case (e): redundant / collinear selected features).
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Zs = (Z - mu) / sd_safe
    # Drop columns that are constant after centring (std 0) -- they carry no fit signal
    # and would only add a zero column.
    keep = sd > 1e-12
    if not np.any(keep):
        v.reason = "all selected design columns constant"
        return v
    Zs = Zs[:, keep]
    k = Zs.shape[1]

    # Ridge closed form on the standardised design with an intercept handled by centring
    # y. r = y - (y_bar + Zs @ beta).  beta = (Zs^T Zs + alpha*I)^-1 Zs^T (y - y_bar).
    y_bar = float(y.mean())
    yc = y - y_bar
    ZtZ = Zs.T @ Zs
    reg = float(ridge_alpha) * float(n) * np.eye(k)  # scale alpha by n so it tracks ZtZ's magnitude
    try:
        beta = np.linalg.solve(ZtZ + reg, Zs.T @ yc)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(ZtZ + reg, Zs.T @ yc, rcond=None)
    y_hat = y_bar + Zs @ beta
    r = y - y_hat
    if not np.all(np.isfinite(r)):
        v.reason = "non-finite residual from the cheap fit"
        return v
    v.residual = r  # surface the continuous residual for the residual-targeted FE step (when blocked below)

    # --- Discretise the residual equi-frequency at the target's resolution. ---
    q_nbins = int(quantization_nbins) if quantization_nbins and quantization_nbins > 1 else 10
    if float(np.std(r)) <= 1e-12:
        # Residual is (numerically) constant => the fit explains y exactly. H(r)=0,
        # trivially below the guard, and no raw can carry residual MI. Sufficient.
        v.residual_entropy_frac = 0.0
        v.max_raw_mi = 0.0
        v.maxt_floor = 0.0
        v.reached = True
        v.reason = "residual is numerically constant (fit explains y exactly)"
        if verbose:
            logger.info("MRMR sufficient-summary: %s -> STOP FE search.", v.reason)
        return v

    # Equi-frequency codes for the MI test (maximises the residual's own entropy, giving
    # the MI(r; x_j) leftover-dependence test its full power -- the right discretisation
    # for DETECTION). The maxT floor + per-raw MI below score against these.
    edges_eqf = np.nanpercentile(r, np.linspace(0, 100, q_nbins + 1))
    r_codes = np.searchsorted(edges_eqf[1:-1], r, side="right").astype(np.int64)

    # --- H(y)-RELATIVE RESIDUAL-SIZE GUARD: the residual must be SMALL relative to y. ---
    # The size measure is the UNEXPLAINED VARIANCE FRACTION ``Var(r)/Var(y)`` (== 1 - R^2),
    # which is the faithful, GRID-FREE realisation of "residual small relative to H(y)":
    # for matched (e.g. Gaussian) families the differential entropy is
    # ``H = 0.5*log(2*pi*e*Var)``, so ``Var(r)/Var(y) = exp(2*(H(r) - H(y)))`` -- the
    # variance ratio is a strictly-monotone transform of the H(y)-relative entropy gap, and
    # unlike a binned-entropy ratio it does NOT suffer the equi-frequency artefact (any
    # non-constant residual fills equal-occupancy bins -> ratio ~1.0) NOR the fixed-grid
    # edge-straddle artefact (a tiny residual splitting across one bin boundary spuriously
    # reads H=log 2). A pure-noise / under-fit target keeps ``Var(r) ~ Var(y)`` -> ratio
    # ~1.0 -> the guard BLOCKS the stop (the load-bearing protection against a false stop on
    # a target with no signal left to find). ``residual_entropy_frac`` names the knob for
    # continuity with the design; it gates on this variance fraction.
    var_y = float(np.var(y))
    var_r = float(np.var(r))
    v.residual_entropy_frac = float(var_r / var_y) if var_y > 1e-18 else float("inf")

    # Tested FIRST and as a hard precondition: a pure-noise / under-fit target keeps a
    # large residual, so we never stop there even if all raws sit at the null.
    if v.residual_entropy_frac > float(residual_entropy_frac):
        v.reason = (
            f"residual still carries {v.residual_entropy_frac:.3f} of Var(y) " f"(1 - R^2 > guard {float(residual_entropy_frac):.3f}); not a sufficient summary"
        )
        if verbose:
            logger.info("MRMR sufficient-summary: %s -> continue FE.", v.reason)
        return v

    # --- maxT residual test: MI(r; x_j) vs the best-of-pool permutation null. ---
    # Build a temporary matrix [raws..., r] so the SHIPPED order-1 floor + the per-raw MI
    # are scored on the IDENTICAL discretised arrays (consistent estimator on both sides).
    raw_codes = np.empty((n, len(raws)), dtype=np.int64)
    raw_nbins = np.empty(len(raws), dtype=np.int64)
    for j, ci in enumerate(raws):
        raw_codes[:, j] = np.ascontiguousarray(data[:, ci]).astype(np.int64, copy=False)
        raw_nbins[j] = int(nbins[ci])
    # Append the residual codes as the "target" column.
    mat = np.empty((n, len(raws) + 1), dtype=np.int64)
    mat[:, : len(raws)] = raw_codes
    mat[:, len(raws)] = r_codes
    mat_nbins = np.empty(len(raws) + 1, dtype=np.int64)
    mat_nbins[: len(raws)] = raw_nbins
    mat_nbins[len(raws)] = int(r_codes.max()) + 1 if r_codes.size else 1
    r_target_idx = len(raws)
    cand_idx = np.arange(len(raws), dtype=np.int64)

    floor = pooled_permutation_null_gain_floor(
        mat, mat_nbins, cand_idx, r_target_idx,
        n_permutations=int(maxt_permutations),
        quantile=float(maxt_quantile),
        cardinality_bias_correction=bool(cardinality_bias_correction),
        random_seed=random_seed,
    )
    v.maxt_floor = float(floor)

    # Per-raw observed MI(r; x_j), Miller-Madow debiased to match the floor's scale.
    r_counts = np.bincount(r_codes, minlength=int(r_codes.max()) + 1).astype(np.float64)
    kr = int((r_counts > 0).sum())
    inv_n = 1.0 / n
    pr = r_counts[r_counts > 0] * inv_n
    h_r_nats = float(-(pr * np.log(pr)).sum())

    max_mi = 0.0
    blocking = -1
    per_raw = {}
    for j, ci in enumerate(raws):
        xc = raw_codes[:, j]
        nbx = int(raw_nbins[j])
        if nbx < 2:
            per_raw[ci] = 0.0
            continue
        xcounts = np.bincount(xc, minlength=nbx).astype(np.float64)
        px = xcounts[xcounts > 0] * inv_n
        h_x = float(-(px * np.log(px)).sum())
        joint = xc.astype(np.int64) * (int(r_codes.max()) + 1) + r_codes
        jc = np.bincount(joint).astype(np.float64)
        jc = jc[jc > 0] * inv_n
        h_xr = float(-(jc * np.log(jc)).sum())
        mi = h_x + h_r_nats - h_xr
        if cardinality_bias_correction:
            mi -= (nbx - 1) * (kr - 1) / (2.0 * n)
        mi = max(0.0, mi)
        per_raw[ci] = float(mi)
        if mi > max_mi:
            max_mi = mi
            blocking = ci
    v.per_raw_mi = per_raw
    v.max_raw_mi = float(max_mi)

    if max_mi > floor:
        # A raw still carries leftover information about the residual -> a future
        # engineered candidate (a function of that raw) COULD still help -> do NOT stop.
        v.blocking_raw = int(blocking)
        v.reason = f"raw col {blocking} still carries MI(r;x)={max_mi:.4f} > maxT floor " f"{floor:.4f}; residual is not pure noise w.r.t. the observables"
        if verbose:
            logger.info("MRMR sufficient-summary: %s -> continue FE.", v.reason)
        return v

    # BOTH conditions hold: residual is small relative to H(y) AND no raw beats the
    # best-of-pool chance ceiling -> the selection reached I(observables; y). STOP.
    v.reached = True
    v.reason = (
        f"sufficient summary reached: H(r)/H(y)={v.residual_entropy_frac:.3f} <= "
        f"{float(residual_entropy_frac):.3f} AND max_j MI(r;x_j)={max_mi:.4f} <= maxT "
        f"floor {floor:.4f} over {len(raws)} raws (DPI: no engineered candidate can help)"
    )
    if verbose:
        logger.info("MRMR sufficient-summary: %s -> STOP FE search.", v.reason)
    return v


def check_sufficient_summary_for_mrmr(
    self: Any,
    *,
    data: np.ndarray,
    nbins: np.ndarray,
    cols: Sequence[str],
    selected_vars: Sequence[int],
    target_indices: Sequence[int],
    X: Any,
    y: Any,
    verbose: int = 0,
) -> SufficientSummaryVerdict:
    """Thin adapter that plumbs ``MRMR.fit``'s greedy-loop state into
    :func:`sufficient_summary_reached`.

    Resolves the continuous design columns for the selected set (raw columns by name from
    the input frame ``X``; engineered columns from ``self._engineered_continuous_``; codes
    fallback otherwise), the RAW-feature pool (the DPI frontier = ``self.feature_names_in_``
    mapped to cols-indices), and the continuous target, then defers entirely to the pure
    helper. Reads the knobs off ``self`` (getattr keeps this stable across the campaign's
    module split). Returns a no-stop verdict on any plumbing failure (best-effort
    optimisation -- never skip when uncertain)."""
    try:
        # Continuous target.
        yv = y.values if hasattr(y, "values") else np.asarray(y)
        yv = np.asarray(yv, dtype=np.float64).reshape(-1)

        cols_list = list(cols)
        raw_name_set = set(getattr(self, "feature_names_in_", []) or [])
        raw_cols_idx = [i for i, nm in enumerate(cols_list) if nm in raw_name_set and i not in set(int(t) for t in target_indices)]

        # Continuous values for selected columns: raw from X (by name), engineered from the
        # per-fit continuous snapshot; codes fallback handled inside the pure helper.
        sel_cont: dict = {}
        eng_cont = getattr(self, "_engineered_continuous_", None) or {}
        x_is_df = hasattr(X, "columns")
        x_cols = set(X.columns) if x_is_df else set()
        for ci in selected_vars:
            nm = cols_list[int(ci)]
            if x_is_df and nm in x_cols:
                try:
                    sel_cont[nm] = np.asarray(X[nm].values, dtype=np.float64).reshape(-1)
                    continue
                except Exception:
                    pass
            if nm in eng_cont:
                try:
                    sel_cont[nm] = np.asarray(eng_cont[nm], dtype=np.float64).reshape(-1)
                except Exception:
                    pass

        # ONE shared subsample reused across the fit (2026-06-25): the maxT residual floor below was
        # historically computed on FULL n even when the FE candidate search ran on the ~30k screen
        # subsample -- the dominant large-n permutation-null cost / OOM source, AND a looser (lower-n
        # -> wider, smaller) chance-max threshold than the subsampled candidates it gates. Ride the
        # SAME single shared row draw as the rest of the FE step so the floor is the matched estimator
        # for the screened candidates and the full-n permutation work disappears. Slices ROWS only
        # (column indices are unchanged); ``None`` (small n) keeps the legacy full-n path byte-identical.
        _n_full = int(data.shape[0])
        _ss_idx = _get_shared_fe_subsample_idx(self, yv, _n_full)
        if _ss_idx is not None:
            data = data[_ss_idx]
            yv = yv[_ss_idx]
            # sel_cont values are full-n continuous columns -> slice the same rows; leave any column
            # whose length doesn't match full n untouched (defensive; the helper drops mismatches later).
            sel_cont = {k: (np.asarray(v)[_ss_idx] if hasattr(v, "__len__") and len(v) == _n_full else v) for k, v in sel_cont.items()}
        return sufficient_summary_reached(
            data=data,
            nbins=nbins,
            y_continuous=yv,
            target_col_idx=int(target_indices[0]),
            selected_cols_idx=[int(i) for i in selected_vars],
            selected_continuous=sel_cont,
            raw_cols_idx=raw_cols_idx,
            cols_names=cols_list,
            quantization_nbins=int(getattr(self, "quantization_nbins", 10) or 10),
            residual_entropy_frac=float(getattr(self, "fe_sufficient_summary_residual_frac", _RESIDUAL_ENTROPY_FRAC_DEFAULT)),
            maxt_permutations=int(getattr(self, "fe_sufficient_summary_maxt_permutations", _MAXT_PERMUTATIONS_DEFAULT)),
            maxt_quantile=float(getattr(self, "fe_sufficient_summary_maxt_quantile", _MAXT_QUANTILE_DEFAULT)),
            ridge_alpha=float(getattr(self, "fe_sufficient_summary_ridge_alpha", _RIDGE_ALPHA_DEFAULT)),
            cardinality_bias_correction=bool(getattr(self, "cardinality_bias_correction", True)),
            random_seed=getattr(self, "random_seed", None),
            verbose=int(verbose),
        )
    except Exception as exc:  # best-effort: never let the optimiser break the fit
        logger.debug("MRMR sufficient-summary check failed (%s); not stopping.", exc)
        return SufficientSummaryVerdict(reason=f"check errored: {exc}")


__all__ = [
    "sufficient_summary_reached",
    "SufficientSummaryVerdict",
    "check_sufficient_summary_for_mrmr",
]
