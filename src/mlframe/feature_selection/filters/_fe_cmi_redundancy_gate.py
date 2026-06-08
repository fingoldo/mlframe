"""Conditional-MI redundancy gate for engineered FE candidates (strategy S5).

The PRINCIPLED replacement for the hardcoded ``fe_min_engineered_mi_prevalence``
joint-prevalence ratio. Where the ratio gate asks "does the 1-D engineered
column retain >= X% of its operand-pair's 2-D joint MI?" (a constant the user
must hand-tune per dataset), THIS gate asks the constant-free, information-
theoretic question:

    Does the candidate engineered feature carry information about y that
    SURVIVES conditioning on the engineered features already admitted?

A spurious / redundant engineered column (e.g. a ``sub(exp(a), invcbrt(c))``
whose y-information is already wholly carried by the admitted ``div(sqr(a),
abs(b))`` and ``mul(log(c), sin(d))``) collapses to ~0 conditional MI and is
rejected. A genuine engineered column carrying a PRIVATE interaction term that
no admitted feature holds keeps a large CMI and is admitted.

Validated design (S5; won 10/10 vs four failing approaches across 16
(seed, formula) cells in ``D:/Temp/prevalence_proto.py``). The decisive
conditioning is on the OTHER already-selected ENGINEERED features -- NOT the
candidate's own operands (CMI given own operands is ~0 for EVERY feature incl.
genuine ones -- a data-processing-inequality trap), NOT the raw top-k (operand-
coverage collisions kill real features).

Two legs, BOTH load-bearing:
  1. CMI clears a CONDITIONAL-PERMUTATION floor (the significance bar -- reuses
     the production within-stratum permutation null,
     ``_conditional_permutation.conditional_permutation_test``).
  2. CMI retains >= ``retain_frac`` (TAU, default 0.15) of the WEAKEST already-
     admitted feature's CMI (the relative-gap / order-of-magnitude separator
     that the floor alone misses -- a redundant ``sub`` sits a few x above its
     own permutation floor yet ~12-84x below every genuine engineered feature).
     TAU is a SCALE-FREE FRACTION of an in-data quantity, robust over
     [0.084, 1.0); it is NOT an MI-nats constant.

Greedy: seed on the highest-marginal-MI engineered candidate (admitted on its
marginal significance -- nothing to condition on yet), then admit remaining
candidates in MI order subject to the two-leg test, folding each admitted
feature into the conditioning support.

All MI/CMI is computed via the production primitives
(``_cmi_from_binned`` / ``_quantile_bin`` / ``_renumber_joint`` from
``_mi_greedy_cmi_fe``) -- this module does NOT reimplement MI. The function is
pure (no live framework state captured), so a fitted MRMR remains picklable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Default TAU (relative-retention fraction). Scale-free fraction of the weakest
# admitted feature's in-data CMI -- robust window measured [0.084, 1.0) across
# 16 (seed, formula) cells; 0.15 sits in the middle with ~2x margin both sides.
DEFAULT_CMI_RETAIN_FRAC = 0.15

# Conditional-permutation floor: number of within-stratum shuffles and the
# null-quantile used as the significance bar. 25 / 0.95 matches the prototype;
# the floor is the cheap leg (the relative-gap leg does the heavy separation).
_CMI_FLOOR_PERMUTATIONS = 25
_CMI_FLOOR_QUANTILE = 0.95

# Conditioning-support fragmentation cap (chi-squared rule-of-thumb: cells must
# average >= 5 samples for the plug-in CMI to stay reliable). When folding the
# next admitted feature would push the joint support cardinality past
# ``n / _SUPPORT_FRAG_DIVISOR`` the support is FROZEN (the feature is still
# admitted, but later candidates are scored against the previous support so
# their CMI stays measurable). Mirrors ``greedy_cmi_fe_construct``'s frag_cap.
_SUPPORT_FRAG_DIVISOR = 5

# Below this many rows the within-stratum permutation null + the conditional MI
# both become unreliable (strata collapse to <=1 element). Fall back to
# admitting every candidate on its marginal significance.
_MIN_ROWS_FOR_CMI = 500


def _conditional_perm_floor(
    cand_bin: np.ndarray,
    y_bin: np.ndarray,
    z_support: Optional[np.ndarray],
    *,
    n_permutations: int = _CMI_FLOOR_PERMUTATIONS,
    quantile: float = _CMI_FLOOR_QUANTILE,
    seed: int = 0,
) -> float:
    """Conditional-permutation null floor for ``CMI(cand; y | z_support)``.

    Reuses the production within-stratum permutation infrastructure
    (``conditional_permutation_test``): the candidate column is permuted WITHIN
    each support stratum, preserving the ``cand | support`` distribution, so the
    null measures the CMI a candidate of the SAME conditional marginal would show
    by chance. Returns the ``quantile`` of the null distribution -- the
    data-derived significance bar.

    When ``z_support`` is ``None`` (seed step, nothing to condition on) there is
    no conditional null; returns 0.0 (the marginal-significance path handles the
    seed admission separately).
    """
    if z_support is None or z_support.size == 0:
        return 0.0
    # Sparse, renumber-based plug-in CMI -- the SAME estimator used for the
    # observed CMI (``_cmi_from_binned``), so the floor and the point estimate
    # are directly comparable, and the memory stays bounded by n (no dense
    # (K_x, K_y, K_z) contingency allocation when the frozen support's joint
    # cardinality climbs into the thousands).
    from ._mi_greedy_cmi_fe import _cmi_from_binned

    x = np.ascontiguousarray(cand_bin, dtype=np.int64).ravel()
    y = np.ascontiguousarray(y_bin, dtype=np.int64).ravel()
    z = np.ascontiguousarray(z_support, dtype=np.int64).ravel()
    rng = np.random.default_rng(int(seed))
    # Group row indices by support stratum once; permute the CANDIDATE column
    # within each stratum (preserves the ``cand | support`` distribution -- the
    # conditional permutation null of Berrett et al. 2020).
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    boundaries = np.flatnonzero(np.diff(sorted_z)) + 1
    groups = [g for g in np.split(order, boundaries) if g.size > 1]
    if not groups:
        return 0.0
    nulls = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        x_perm = x.copy()
        for g in groups:
            x_perm[g] = x[g[rng.permutation(g.size)]]
        nulls[i] = float(_cmi_from_binned(x_perm, y, z))
    return float(np.quantile(nulls, quantile))


def apply_cmi_redundancy_gate(
    candidates: dict,
    y_bin: np.ndarray,
    *,
    nbins: int = 10,
    retain_frac: float = DEFAULT_CMI_RETAIN_FRAC,
    n_permutations: int = _CMI_FLOOR_PERMUTATIONS,
    quantile: float = _CMI_FLOOR_QUANTILE,
    seed: int = 0,
    verbose: int = 0,
) -> tuple[set, dict]:
    """Greedy CMI-redundancy gate over the surviving engineered candidate pool.

    Parameters
    ----------
    candidates : dict ``{name -> (continuous_values: np.ndarray, marginal_mi: float)}``
        The engineered columns that already cleared the per-pair acceptance
        machinery (joint / prewarp / marginal-uplift). ``continuous_values`` is
        the full-n float column (NOT pre-binned -- binned here so the CMI codes
        match the production quantile binning the prototype validated).
    y_bin : np.ndarray
        Discretised target codes (the same ``classes_y`` the MI sweep scores
        against). Renumbered to dense 0..K-1 internally.
    nbins : int
        Equi-frequency bins per candidate column.
    retain_frac : float
        TAU -- the scale-free relative-retention fraction (default 0.15).
    n_permutations, quantile : int, float
        Conditional-permutation floor config.
    seed : int
        RNG seed for the conditional-permutation floor (deterministic).
    verbose : int
        >0 emits per-candidate accept/reject diagnostics via the module logger.

    Returns
    -------
    (accepted_names, diagnostics)
        ``accepted_names`` is the set of candidate names to KEEP; everything
        else is dropped as redundant. ``diagnostics`` maps each name to a dict
        with ``accept`` / ``cmi`` / ``floor`` / ``rel_bar`` / ``reason``.

    Degenerate fallback: with <2 candidates, or fewer than ``_MIN_ROWS_FOR_CMI``
    rows, there is nothing to condition on (or the conditional estimate is
    unreliable) -- ACCEPT every candidate on its marginal significance rather
    than rejecting everything.
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin, _renumber_joint

    names = list(candidates.keys())
    diagnostics: dict = {}
    if not names:
        return set(), diagnostics

    y_arr = np.asarray(y_bin)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    n_rows = int(y_dense.size)

    # Degenerate: nothing to condition on, or too few rows for a reliable
    # conditional estimate -> admit all on marginal significance.
    if len(names) < 2 or n_rows < _MIN_ROWS_FOR_CMI:
        for nm in names:
            diagnostics[nm] = dict(
                accept=True, cmi=float(candidates[nm][1]), floor=0.0,
                rel_bar=0.0, reason="degenerate_marginal_admit",
            )
        return set(names), diagnostics

    # Bin every candidate once (production quantile binner).
    cand_bins: dict = {}
    for nm in names:
        vals = np.asarray(candidates[nm][0], dtype=np.float64)
        cand_bins[nm] = _quantile_bin(vals, nbins=nbins)
    marg = {nm: float(candidates[nm][1]) for nm in names}

    accepted: list[str] = []          # admitted candidate names, in selection order
    accepted_bins: list[np.ndarray] = []
    admitted_cmis: list[float] = []   # CMI-scale of admitted features (for the rel bar)
    remaining = set(names)
    frag_cap = max(2, n_rows // _SUPPORT_FRAG_DIVISOR)
    z_support: Optional[np.ndarray] = None

    # Seed: highest-marginal-MI candidate, admitted on its marginal significance
    # (nothing to condition on yet). Its marginal MI anchors the relative bar.
    seed_name = max(remaining, key=lambda nm: marg[nm])
    accepted.append(seed_name)
    accepted_bins.append(cand_bins[seed_name])
    admitted_cmis.append(marg[seed_name])
    remaining.discard(seed_name)
    diagnostics[seed_name] = dict(
        accept=True, cmi=marg[seed_name], floor=0.0,
        rel_bar=0.0, reason="seed_marginal",
    )

    while remaining:
        z_support, _ = _renumber_joint(*accepted_bins)
        rel_bar = float(retain_frac) * min(admitted_cmis)
        best_name = None
        best_cmi = -1.0
        scored: dict = {}
        for nm in list(remaining):
            cmi = float(_cmi_from_binned(cand_bins[nm], y_dense, z_support))
            floor = _conditional_perm_floor(
                cand_bins[nm], y_dense, z_support,
                n_permutations=n_permutations, quantile=quantile, seed=seed,
            )
            scored[nm] = (cmi, floor)
            passes_floor = cmi > floor
            passes_rel = cmi >= rel_bar
            passes = passes_floor and passes_rel
            if nm not in diagnostics:
                diagnostics[nm] = {}
            diagnostics[nm].update(
                accept=False, cmi=cmi, floor=floor, rel_bar=rel_bar,
                reason=("redundant_below_floor" if not passes_floor
                        else "redundant_below_rel_bar" if not passes_rel
                        else "pending"),
            )
            if passes and cmi > best_cmi:
                best_cmi = cmi
                best_name = nm
        if best_name is None:
            # No remaining candidate adds enough NEW information -> stop; the
            # rest are redundant given the admitted engineered support.
            break
        diagnostics[best_name].update(accept=True, reason="admitted_cmi")
        # Fold the winner into the conditioning support, respecting the
        # fragmentation cap (freeze support if folding would shatter the strata).
        new_bin = cand_bins[best_name]
        candidate_support, _ = _renumber_joint(*(accepted_bins + [new_bin]))
        if int(np.unique(candidate_support).size) <= frag_cap:
            accepted_bins.append(new_bin)
        # else: keep accepted_bins frozen; the feature is still admitted.
        accepted.append(best_name)
        admitted_cmis.append(best_cmi)
        remaining.discard(best_name)

    if verbose:
        for nm in names:
            d = diagnostics.get(nm, {})
            logger.info(
                "CMI-redundancy gate: %s accept=%s cmi=%.4f floor=%.4f rel_bar=%.4f (%s)",
                nm, d.get("accept"), d.get("cmi", float("nan")),
                d.get("floor", float("nan")), d.get("rel_bar", float("nan")),
                d.get("reason", "-"),
            )
    return set(accepted), diagnostics


__all__ = ["apply_cmi_redundancy_gate", "DEFAULT_CMI_RETAIN_FRAC"]
