"""Raw-vs-engineered conditional-redundancy drop (n-invariant, debiased excess CMI).

The companion of the engineered-vs-engineered S5 gate (``_fe_cmi_redundancy_gate``),
applied to the FINAL MRMR selection. When a raw operand is FULLY subsumed by a
surviving engineered feature built from it -- e.g. ``y = (a**2)/b + noise`` whose
ratio is captured byte-for-byte by ``div(neg(a),sqrt(b))`` (since
``(a/sqrt(b))**2 = a**2/b``) -- the raw operands ``a`` and ``b`` carry NO
information about ``y`` beyond the engineered child and MUST drop. The greedy MRMR
order selects such an operand on its high MARGINAL relevance BEFORE the engineered
ratio is in support, so the redundancy penalty never fires against it; the various
raw-retention / raw-signal-augmentation passes then re-add it. This final sweep
removes those re-admitted subsumed operands using the SAME debiased excess-CMI idea
the S5 gate validated, so the decision is n-invariant (works identically at n=1000
and n=50000) and never drops a raw that carries genuine independent signal.

Decision (per raw operand of >=1 surviving engineered survivor):

    raw is REDUNDANT given its engineered survivors iff BOTH
      (1) CMI(raw; y | engineered survivors) <= conditional-permutation floor
          (the significance leg -- the residual is indistinguishable from the
          within-stratum shuffle null), OR
      (2) the DEBIASED EXCESS  max(0, CMI - null_mean)  is below
          ``retain_frac`` * (the weakest engineered survivor's own debiased
          marginal excess) -- the scale-free relative-gap leg.

A redundant operand's excess collapses to ~0 (its conditional MI given the
engineered child is pure finite-sample / binning-gap bias, which the
within-stratum permutation null reproduces and the excess subtracts). A raw
carrying a PRIVATE additive term keeps a large excess (a substantial fraction of
the engineered anchor) and is KEPT. Both the relative bar and the anchor live on
the debiased-excess scale, so the finite-sample bias cancels and the verdict is
the same at every n (validated ws1 / F2 / F4 on n=1000..50000 in
``D:/Temp/feq_proto2.py`` / ``feq_proto3.py``).

All MI/CMI uses the production primitives (``_cmi_from_binned`` / ``_quantile_bin``
/ ``_renumber_joint``) and the production conditional-permutation null
(``_conditional_perm_null`` from the S5 gate). The function is pure -- no live
framework state captured -- so a fitted MRMR stays picklable.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Scale-free retention fraction (TAU). Shared default with the S5 engineered-vs-
# engineered gate: a raw operand must retain >= this fraction of the weakest
# engineered survivor's own debiased excess to be judged a genuine independent
# term. 0.15 sat with >=2x margin both sides across the validated cells.
DEFAULT_RAW_RETAIN_FRAC = 0.15

# Equi-frequency bins for raw / engineered / target columns. 10 matches the S5
# gate; deliberately NOT finer -- a very fine engineered binning fragments the
# conditioning strata at large n and re-inflates the residual (measured: 32 bins
# made ws1 ``a`` clear the shrunken floor at n=25000). The RELATIVE-excess bar,
# not the bin count, does the separation.
_BINS = 10
_MIN_ROWS = 500

# Conditioning-support fragmentation guard (chi-squared rule-of-thumb: joint cells
# should average >= a few samples for the plug-in CMI to stay reliable). The
# engineered survivors are binned at the target cardinality, but that count is
# capped so the JOINT support of the consuming survivors keeps measurable strata
# at small n. Mirrors the S5 gate's ``_SUPPORT_FRAG_DIVISOR``.
_SUPPORT_FRAG_DIVISOR = 5

# Split an engineered name into identifier tokens (``div(sqr(a),abs(b))`` ->
# {div, sqr, a, abs, b}); the raw operands are the tokens that are raw columns.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _excess_and_floor(cand_bin, y_bin, z_support, *, seed=0):
    """Return ``(cmi, floor, excess)`` for ``CMI(cand; y | z_support)`` using the
    S5 conditional-permutation null (within-stratum shuffle reproduces the same
    finite-sample bias, so ``excess = max(0, cmi - null_mean)`` is n-invariant).
    ``z_support=None`` -> marginal MI / free-shuffle null (used for the anchor)."""
    from ._mi_greedy_cmi_fe import _cmi_from_binned
    from ._fe_cmi_redundancy_gate import _conditional_perm_null

    cmi = float(_cmi_from_binned(cand_bin, y_bin, z_support))
    floor, null_mean = _conditional_perm_null(cand_bin, y_bin, z_support, seed=seed)
    return cmi, floor, max(0.0, cmi - null_mean)


def drop_redundant_raw_operands(
    *,
    data: np.ndarray,
    cols,
    selected_cols_idx,
    raw_name_set,
    y_binned: np.ndarray,
    y_continuous: Optional[np.ndarray] = None,
    engineered_continuous: Optional[dict] = None,
    retain_frac: float = DEFAULT_RAW_RETAIN_FRAC,
    seed: int = 0,
    verbose: int = 0,
) -> tuple[list, list]:
    """Drop selected RAW operands conditionally redundant given a surviving
    engineered child built from them.

    Parameters
    ----------
    data : (n, n_cols) float matrix holding RAW *and* engineered columns (the FE
        step appends engineered columns to ``data`` / ``cols``). NOTE the columns
        here are the LOSSY ~10-code screening bins, not the continuous values.
    cols : list[str] -- column names indexing ``data`` columns.
    selected_cols_idx : iterable[int] -- the FINAL selected column indices into
        ``cols`` (raw + engineered), AFTER all retention / augmentation passes.
    raw_name_set : set[str] -- the raw input feature names.
    y_binned : (n,) int -- the discretised target codes (``classes_y``).
    engineered_continuous : dict ``{name -> continuous float array}`` (optional)
        The CONTINUOUS engineered values (fit-time scratch). When present, the
        engineered survivor is binned FINELY from its continuous values (at the
        target cardinality) so it resolves y as finely as the target codes -- this
        is load-bearing: binning the engineered ratio at the lossy 10-code
        ``data`` column leaves a fully-subsumed DENOMINATOR operand (``b`` in
        ``a**2/b``) a spurious residual CMI and wrongly keeps it. Falls back to the
        ``data`` codes for any survivor missing from the snapshot.
    retain_frac : TAU (scale-free relative-retention fraction).
    seed : RNG seed for the conditional-permutation null (deterministic).
    verbose : >0 logs each kept/dropped operand.

    Returns
    -------
    (kept_idx, dropped_names)
        ``kept_idx`` -- the selected indices with redundant raw operands removed
        (engineered columns and genuine raws preserved, original order kept).
        ``dropped_names`` -- the raw names removed (for logging / provenance).

    Degenerate (too few rows, no engineered survivors, no raw operands): returns
    ``selected_cols_idx`` unchanged.
    """
    from ._mi_greedy_cmi_fe import _quantile_bin, _renumber_joint

    sel = list(selected_cols_idx)
    n_rows = int(np.asarray(y_binned).shape[0])
    if n_rows < _MIN_ROWS or len(sel) < 2:
        return sel, []

    sel_names = [cols[i] for i in sel]
    # Engineered survivors = selected columns whose name is not a raw column.
    eng_idx = [i for i, nm in zip(sel, sel_names) if nm not in raw_name_set]
    raw_sel_idx = [i for i, nm in zip(sel, sel_names) if nm in raw_name_set]
    if not eng_idx or not raw_sel_idx:
        return sel, []

    # raw_name -> list of engineered survivor column indices that consume it.
    eng_consumers: dict[str, list[int]] = {}
    for ei in eng_idx:
        toks = [t for t in _TOKEN_SPLIT.split(cols[ei]) if t]
        # Map ``a__He3``-style warped tokens back to their raw base too.
        bases = set()
        for t in toks:
            if t in raw_name_set:
                bases.add(t)
            elif "__" in t and t.split("__", 1)[0] in raw_name_set:
                bases.add(t.split("__", 1)[0])
        for base in bases:
            eng_consumers.setdefault(base, []).append(ei)

    if not eng_consumers:
        return sel, []

    # Target codes. Prefer an EQUI-FREQUENCY re-binning of the CONTINUOUS target on a
    # skewed regression target: the screening ``classes_y`` is frequently heavily
    # imbalanced (``y=(a**2)/b`` puts ~89% of rows in one bin), which crushes the
    # engineered anchor's MI (measured 0.31 vs the faithful 2.0) and inflates a subsumed
    # operand's apparent residual fraction past the relative bar -- wrongly KEEPING it. An
    # equi-frequency target restores the faithful anchor and the redundant operand's frac
    # collapses to ~0. Only re-bin a genuinely CONTINUOUS target (many distinct values);
    # an already-discrete / classification target keeps its class codes.
    y_arr = np.ascontiguousarray(np.asarray(y_binned)).ravel()
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _target_card = int(np.unique(y_arr).size)
    if y_continuous is not None:
        _yc = np.asarray(y_continuous).reshape(-1)
        if _yc.shape[0] == n_rows and np.issubdtype(_yc.dtype, np.number):
            # Continuous iff far more distinct values than the (possibly collapsed) code
            # cardinality -- avoids re-binning an already-discrete few-class target.
            if int(np.unique(_yc).size) > max(2 * _BINS, 2 * _target_card):
                _nb = int(min(max(_BINS, _target_card),
                             max(2, n_rows // (_BINS * _SUPPORT_FRAG_DIVISOR))))
                y_arr = np.ascontiguousarray(_quantile_bin(_yc.astype(np.float64), nbins=_nb)).astype(np.int64)
                _target_card = int(np.unique(y_arr).size)

    # Bin the engineered survivors at the TARGET cardinality (>= _BINS) FROM THEIR
    # CONTINUOUS VALUES so the engineered column resolves y as finely as the target
    # codes do. Two binning hazards this avoids:
    #   * the ``data`` column is the LOSSY ~10-code screening bin -- re-binning those
    #     codes cannot recover resolution the screen already threw away, so a
    #     fully-subsumed DENOMINATOR operand (``b`` in ``a**2/b``) shows a spurious
    #     residual CMI given the coarse ratio and is wrongly kept (measured ws1
    #     ``b`` at n=2k: CMI(b|eng_codes)=0.05, anchor only 0.28). Binning the
    #     CONTINUOUS engineered value at the target cardinality lifts the anchor to
    #     its true ~2.0 and collapses ``b``'s residual.
    #   * production ``classes_y`` grows with n (~11 bins at n=2k, ~23 at n=25k); a
    #     fixed coarse engineered binning under a finer target re-opens the same
    #     mismatch. Tying the engineered binning to the target cardinality cancels
    #     it at every n, while a GENUINE shared operand keeps its private excess
    #     (F4 ``b`` stays well above the relative bar).
    # The cardinality is capped by a fragmentation guard so the JOINT consuming-
    # survivor support keeps measurable strata at small n.
    _eng_card = int(min(max(_BINS, int(np.unique(y_arr).size)),
                        max(2, n_rows // (_BINS * _SUPPORT_FRAG_DIVISOR))))
    _eng_cont = engineered_continuous or {}
    eng_bin: dict[int, np.ndarray] = {}
    eng_anchor_excess: dict[int, float] = {}
    for ei in eng_idx:
        _ename = cols[ei]
        _cont = _eng_cont.get(_ename)
        if _cont is not None and np.asarray(_cont).shape[0] == n_rows:
            # Fine binning from the continuous engineered values (preferred).
            eb = _quantile_bin(np.asarray(_cont, dtype=np.float64), nbins=_eng_card)
        else:
            # Fallback: the lossy screening codes already in ``data`` (use as-is;
            # re-binning coarse codes gains nothing). Conservative -- a missing
            # continuous value can only make the gate KEEP more (smaller anchor).
            eb = np.asarray(data[:, ei]).astype(np.int64).ravel()
        eng_bin[ei] = eb
        _, _, exc = _excess_and_floor(eb, y_arr, None, seed=seed)
        eng_anchor_excess[ei] = exc

    drop_names: list[str] = []
    drop_idx_set: set[int] = set()
    for ri in raw_sel_idx:
        rname = cols[ri]
        consumers = eng_consumers.get(rname)
        if not consumers:
            continue  # raw not consumed by any survivor -> genuine, keep
        # The raw column is binned at the SCREENING codes the selector itself used
        # (already integer 0..k in ``data``). We do NOT up-resolve the raw from
        # continuous values: the redundancy verdict must judge the raw at the
        # resolution the selector saw, and a finer raw binning would only inflate
        # its residual CMI (bias toward KEEP), the unsafe direction.
        rb = np.asarray(data[:, ri]).astype(np.int64).ravel()
        z_support, _ = _renumber_joint(*(eng_bin[ei] for ei in consumers))
        cmi, floor, excess = _excess_and_floor(rb, y_arr, z_support, seed=seed)
        # Anchor = weakest consuming engineered survivor's own debiased excess.
        anchor = min(eng_anchor_excess[ei] for ei in consumers)
        rel_bar = float(retain_frac) * max(0.0, anchor)
        passes_floor = cmi > floor
        passes_rel = excess >= rel_bar
        if passes_floor and passes_rel:
            if verbose:
                logger.info(
                    "raw-redundancy: KEEP %s (cmi=%.4f floor=%.4f excess=%.5f "
                    "rel_bar=%.5f -- carries independent signal given %s)",
                    rname, cmi, floor, excess, rel_bar, [cols[e] for e in consumers],
                )
            continue
        drop_names.append(rname)
        drop_idx_set.add(ri)
        if verbose:
            logger.info(
                "raw-redundancy: DROP %s (cmi=%.4f floor=%.4f excess=%.5f "
                "rel_bar=%.5f -- redundant given %s)",
                rname, cmi, floor, excess, rel_bar, [cols[e] for e in consumers],
            )

    if not drop_idx_set:
        return sel, []
    kept = [i for i in sel if i not in drop_idx_set]
    return kept, drop_names


__all__ = ["drop_redundant_raw_operands", "DEFAULT_RAW_RETAIN_FRAC"]
