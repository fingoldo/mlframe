"""Engineered-vs-engineered subsumption guard for retention-re-attached pure forms.

The usability-aware pure-form retention pass (``_fe_pure_form_retention``) re-attaches a
PURE single-pair engineered form (``mul(log(c),sin(d))``, ``div(sqr(a),sin(b))``) AFTER the
post-FE engineered-vs-engineered CMI redundancy gate (``_fe_cmi_redundancy_gate``) has already
run. The gate therefore never gets to test those re-attached forms against the engineered
survivors that were admitted BEFORE retention. When one of those incumbents is a FUSED compound
that already carries BOTH additive halves of the target -- e.g. the canonical
``add(neg(mul(sqr(a),reciproc(b))),neg(mul(log(c),sin(d))))`` for ``y = a**2/b + log(c)*sin(d)``
-- a re-attached pure half is fully REDUNDANT: given the full compound it carries ~0 additional
information about ``y`` (and a linear model reads the additive term straight off the compound).
The result is the fragmentation regression: one full compound PLUS several pure sub-fragments.

This module supplies the engineered analog of the existing post-retention RAW redundancy drop
(``_fe_raw_redundancy_drop``): before a retention candidate is committed, condition it on the
already-admitted engineered survivors and DROP it when its information collapses given them. The
decision reuses the SAME n-invariant debiased-excess-CMI machinery the S5 gate validated -- a
retention form is subsumed iff its conditional CMI given the incumbent survivors fails the
within-stratum permutation floor OR its debiased conditional excess retains less than
``RETAIN_FRAC`` of its OWN marginal debiased excess. A genuinely COMPLEMENTARY pure form (one the
incumbents do NOT already span -- the case the retention pass exists to rescue) keeps a large
residual and is admitted; only a sub-fragment of an incumbent compound is dropped.

Pure (no live framework state captured) so a fitted MRMR stays picklable.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Scale-free retention fraction (TAU), shared default with the S5 engineered gate and the raw
# redundancy drop. A retention candidate must retain >= this fraction of its OWN marginal
# debiased excess as CONDITIONAL excess (given the incumbent engineered survivors) to count as
# carrying genuinely complementary information. A pure sub-fragment of a fused incumbent compound
# collapses to ~0; a genuinely complementary pure form keeps the bulk of its marginal.
RETAIN_FRAC = 0.15

_BINS = 10
_MIN_ROWS = 500


def retention_form_is_subsumed(
    *,
    cand_continuous: np.ndarray,
    incumbent_continuous: Sequence[np.ndarray],
    y_binned: np.ndarray,
    y_continuous: "np.ndarray | None" = None,
    retain_frac: float = RETAIN_FRAC,
    nbins: int = _BINS,
    seed: int = 0,
) -> bool:
    """True iff the retention candidate carries NO significant information about ``y`` beyond the
    already-admitted incumbent engineered survivors -- i.e. it is a redundant sub-fragment that
    must NOT be re-attached.

    ``cand_continuous`` / ``incumbent_continuous`` are full-n continuous engineered columns
    (binned here with the production quantile binner). With NO incumbent survivor the candidate
    cannot be proven redundant -> returns False (retain). Any estimator failure -> False (the
    conservative retain direction: never silently drop a form the retention pass wanted).

    The verdict is INFORMATION-THEORETIC (two debiased-CMI legs), NOT linear. A linear-usability
    guard was considered -- the retention pass exists to rescue a pure form a LINEAR model needs that
    the MI greedy dropped for a higher-MI nonlinear cross-mix, and a partial-rank-correlation residual
    could in principle distinguish a pure half ADDITIVELY subsumed by a clean compound (``mul(log(c),
    sin(d))`` given ``add(a**2/b, log(c)*sin(d))`` -> DROP) from one whose only incumbents are
    nonlinear cross-mixes (``div(neg(a),sqrt(b))`` -> KEEP). But it was REJECTED: an admitted fused
    compound's nuisance second additive term leaks a spurious partial-linear residual for the pure
    half, so the linear leg re-kept exactly the sub-fragments the strengthened ONE-fused-compound
    contract requires dropping. The CMI legs alone get the F2 cross-mix case right anyway -- a genuine
    cross-mix incumbent does NOT collapse the candidate's conditional CMI, so it stays above the floor
    + relative bar and is retained. ``y_continuous`` is accepted for API symmetry with the raw-
    redundancy linear leg but is not consulted; the CMI verdict stands."""
    try:
        from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin, _renumber_joint
        from ._fe_cmi_redundancy_gate import _conditional_perm_null
    except Exception:
        return False

    inc = [np.asarray(c, dtype=np.float64).ravel() for c in incumbent_continuous if c is not None]
    if not inc:
        return False
    cand = np.asarray(cand_continuous, dtype=np.float64).ravel()
    n = cand.shape[0]
    if n < _MIN_ROWS or any(c.shape[0] != n for c in inc):
        return False
    y_arr = np.ascontiguousarray(np.asarray(y_binned)).ravel()
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    if y_arr.shape[0] != n:
        return False

    try:
        cand = np.nan_to_num(cand, nan=0.0, posinf=0.0, neginf=0.0)
        cand_bin = np.asarray(_quantile_bin(cand, nbins=nbins)).astype(np.int64).ravel()
        inc_bins = [
            np.asarray(_quantile_bin(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0), nbins=nbins)).astype(np.int64).ravel()
            for c in inc
        ]
        z_support, _ = _renumber_joint(*inc_bins)
        # Candidate's OWN marginal debiased excess (the reference scale for the relative bar).
        marg_floor, marg_null_mean = _conditional_perm_null(cand_bin, y_arr, None, seed=seed)
        marg_cmi = float(_cmi_from_binned(cand_bin, y_arr, None))
        marg_excess = max(0.0, marg_cmi - marg_null_mean)
        # Conditional CMI given the incumbent engineered survivors.
        cmi = float(_cmi_from_binned(cand_bin, y_arr, z_support))
        floor, null_mean = _conditional_perm_null(cand_bin, y_arr, z_support, seed=seed)
        excess = max(0.0, cmi - null_mean)
    except Exception:
        return False

    passes_floor = cmi > floor
    passes_rel = excess >= float(retain_frac) * max(0.0, marg_excess)
    # SUBSUMED iff it neither clears the significance floor NOR retains a meaningful fraction of
    # its marginal given the incumbents. Carrying genuinely new information requires BOTH legs to
    # hold -- a redundant sub-fragment of an incumbent compound fails them; a complementary form
    # passes. ``y_continuous`` is accepted for API symmetry with the raw-redundancy linear leg but
    # the verdict is intentionally INFORMATION-theoretic: a pure half whose information is wholly
    # carried by an admitted fused compound is dropped even when a degenerate partial-linear test
    # would (spuriously, via the compound's nuisance second term) read a residual -- the strengthened
    # ONE-fused-compound contract.
    return not (passes_floor and passes_rel)


__all__ = ["retention_form_is_subsumed", "RETAIN_FRAC"]
