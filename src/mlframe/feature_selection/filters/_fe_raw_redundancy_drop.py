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

  STEP 0 -- DPI-TRAP CONSUMER FILTER (2026-06-10). Restrict the conditioning /
  anchor set to engineered children that are TRUE COMBINATIONS: they draw genuine
  signal from a SECOND signal-bearing raw source besides the raw under test. A
  child whose ONLY signal-bearing parent is the raw itself (``relu_lt(x_a)`` /
  ``exp(x0)`` / ``He2(a)`` and friends -- a monotone/basis self-transform) is the
  data-processing-inequality trap the S5 gate warns about: conditioning a raw on a
  basis of ITSELF drives CMI to ~0 for EVERY raw -- genuine or redundant -- so it
  proves nothing. A "signal-bearing" parent is one whose own marginal debiased
  excess clears its marginal permutation null (a noise operand, e.g. ``x3`` in
  ``add(exp(x0),sign(x3))``, does NOT count -- so that child is a self-transform of
  x0, not a combination). When NO legitimate multi-source consumer survives this
  filter the raw is NOT redundancy-dropped (the protective retention stands).

  STEP 1 -- KEEP iff EITHER leg holds (DROP only when BOTH fail):
      (A) SIGNIFICANT INDEPENDENT RESIDUAL -- CMI(raw; y | combination children)
          clears the within-stratum conditional-permutation floor AND the debiased
          conditional excess retains >= ``RAW_SELF_RETAIN_FRAC`` of the raw's OWN
          marginal debiased excess. A genuine PRIVATE LINEAR term keeps ~6-11% of
          its marginal given the interaction product; a fully-subsumed ``a**2/b``
          ratio operand keeps ~0.3-2% and barely (or never) clears the floor.
      (B) THE STRONGEST CONSUMING CHILD IS NOT A SUPERSET -- its own marginal
          debiased excess is <= ``RAW_SUPERSET_MULT`` x the raw's marginal excess.
          A child that merely RE-EXPRESSES the raw through a (near-)monotone unary
          paired with a noise operand (``add(exp(x0),sign(x3))`` ~1.8x the raw) is
          NOT a superset; the raw keeps a cleaner LINEAR signal than the noise-
          polluted child, so it is retained for downstream linear usability. A true
          combination (the ``a**2/b`` ratio ~6x, the interaction product ~5x) IS a
          superset and does not trip leg B -- those genuinely-subsumed operands rely
          on leg A's failure to drop.

A redundant operand's conditional excess collapses to ~0 (its CMI given the
combination child is pure finite-sample / binning-gap bias, reproduced by the
within-stratum permutation null and subtracted by the excess) AND a strictly-more-
informative combination child captures it -> both legs fail -> DROP. A raw carrying
a private term (linear / additive) or paired only with noise keeps a significant
residual or is not faced with a superset -> KEPT. All scales are debiased excesses
or scale-free multiples, so the verdict is n-stable (validated on the genuine
x_a/x_b/x0 + subsumed a**2/b a/b across n=500..100000 cells, 2026-06-10).

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

# SELF-RETENTION fraction (keep leg A, 2026-06-10). The raw must retain >= this
# fraction of its OWN marginal debiased excess as CONDITIONAL excess (given the
# combination child) to count as a significant independent residual. A genuine
# private LINEAR term keeps ~6-11% of its marginal given the interaction product;
# a fully-subsumed ``a**2/b`` ratio operand keeps ~0.3-2%. 0.05 sits between with
# >=2x margin on both sides across the validated genuine/subsumed cells.
RAW_SELF_RETAIN_FRAC = 0.05

# SUPERSET multiple (keep leg B, 2026-06-10). A consuming engineered child counts
# as a genuine SUPERSET (capable of subsuming the raw) only when its own marginal
# debiased excess exceeds this multiple of the raw's marginal excess. A ratio /
# interaction product is ~5-6x the operand's marginal (superset -> may subsume via
# leg A's failure); a (near-)monotone re-expression paired with a noise operand is
# only ~1.8x (NOT a superset -> the raw is kept for downstream linear usability).
# 3.0 separates the validated re-expression case (x0, 1.8x) from the genuine
# combinations (>=5x) with comfortable margin.
RAW_SUPERSET_MULT = 3.0

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
    replayable_eng_names: Optional[set] = None,
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
    replayable_eng_names : set[str] | None -- names of the engineered survivors
        that have a REPLAYABLE recipe and will therefore survive into the fitted
        ``transform()`` output. A raw operand may only be judged redundant against
        a child that will actually EXIST at predict time; a nested-engineered
        child (parents themselves engineered, no 1-deep recipe) is dropped from
        transform output, so crediting a raw as "subsumed" by it deletes BOTH the
        raw AND the child -> an EMPTY selection (no features reach the downstream
        model). When provided, engineered survivors NOT in this set are excluded
        from the subsumer / anchor set. ``None`` (legacy) trusts every survivor --
        only safe when the caller guarantees all engineered survivors are
        replayable. The fit pipeline always passes the concrete replayable set.
    retain_frac : ACCEPTED FOR BACK-COMPAT but no longer drives the verdict. The
        2026-06-10 redesign replaced the single ``retain_frac * weakest-anchor``
        relative bar (which over-dropped genuine raws whose linear/additive private
        term sits beside a high-MI interaction child) with the two scale-free legs
        described in the module docstring -- ``RAW_SELF_RETAIN_FRAC`` (the raw's own
        marginal-excess self-retention) and ``RAW_SUPERSET_MULT`` (the child-is-a-
        superset test). The kwarg is kept so existing callers
        (``fe_raw_redundancy_retain_frac``) do not break.
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

    # REPLAYABLE-ANCHOR GUARD (2026-06-11). A raw operand can only be conditionally
    # redundant given an engineered child that will SURVIVE into the fitted
    # ``transform()`` output. A nested-engineered survivor (parents themselves
    # engineered -> no 1-deep replayable recipe) is dropped from transform output;
    # if it is allowed to anchor the redundancy verdict it deletes BOTH the raw
    # operands it "subsumes" AND itself -> an EMPTY selection that hands the
    # downstream model zero features (observed on the canonical
    # ``y=a**2/b + log(c)*sin(d)`` fixture where the strongest survivor was the
    # un-replayable ``add(prewarp(div(sqr(a),abs(b))),neg(mul(log(c),sin(d))))``,
    # so a/b/c/d all dropped and the support went empty). Restrict the engineered
    # subsumer/anchor set to replayable survivors. ``None`` keeps the legacy
    # behaviour (trust every survivor) for callers that guarantee replayability.
    if replayable_eng_names is not None:
        _replayable = set(replayable_eng_names)
        eng_idx = [i for i in eng_idx if cols[i] in _replayable]
        if not eng_idx:
            if verbose:
                logger.info(
                    "raw-redundancy: no REPLAYABLE engineered survivor anchors the "
                    "redundancy verdict (all engineered survivors are nested / "
                    "un-replayable and would be dropped from transform output); "
                    "keeping all raw operands."
                )
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

    # DPI-TRAP GUARD (2026-06-10): an engineered child is a LEGITIMATE subsumer of a
    # raw operand only when it draws genuine signal from a SECOND raw source -- i.e.
    # it re-expresses a COMBINATION (the ``a**2/b`` ratio ``div(neg(a),sqrt(b))``,
    # the interaction product ``mul(x_a,x_b)``) rather than being a sole-operand
    # monotone/basis transform of the raw itself (``relu_lt(x_a)`` / ``exp(x0)`` /
    # ``He2(a)`` ...). Conditioning a raw on a basis of ITSELF is the data-processing-
    # inequality trap the S5 gate docstring warns about: CMI(raw; y | own-transform)
    # collapses to ~0 for EVERY raw -- genuine OR redundant -- so it cannot
    # distinguish a fully-subsumed operand from one carrying a private additive term
    # the transform does not span (e.g. the LINEAR ``x_a`` in ``sign(x_a+x_b+2 x_a x_b)``
    # given ``mul(x_a,x_b)`` -- raw retains CMI ~0.32). A "second signal-bearing
    # source" is another raw parent whose OWN marginal debiased excess clears its
    # marginal permutation null (so a child built from the raw + a NOISE column, e.g.
    # ``add(exp(x0),sign(x3))`` where x3 is pure noise, is NOT a legitimate subsumer:
    # x0 is its only signal source). When a raw has NO legitimate multi-source
    # consumer it is never redundancy-dropped here -- the protective retention stands.
    # Compute each raw operand's MARGINAL (cmi, floor, debiased-excess) once (one
    # marginal perm-null per consumed raw); cache the tuple. The marginal excess is
    # the raw's OWN signal scale, reused below as (a) the "signal-bearing source"
    # test and (b) the keep-rule's self-retention reference.
    _raw_marg_cache: dict[str, tuple] = {}

    def _raw_marginal(_rname: str) -> tuple:
        if _rname in _raw_marg_cache:
            return _raw_marg_cache[_rname]
        try:
            _ridx = cols.index(_rname)
        except ValueError:
            _raw_marg_cache[_rname] = (0.0, 0.0, 0.0)
            return _raw_marg_cache[_rname]
        _rb = np.asarray(data[:, _ridx]).astype(np.int64).ravel()
        _res = _excess_and_floor(_rb, y_arr, None, seed=seed)
        _raw_marg_cache[_rname] = _res
        return _res

    def _raw_is_signal_bearing(_rname: str) -> bool:
        _mcmi, _mfloor, _mexc = _raw_marginal(_rname)
        return _mcmi > _mfloor and _mexc > 0.0

    # raw_name -> list of engineered survivor names that consume it (parsed earlier
    # as ``eng_consumers``). Build the set of distinct SIGNAL-BEARING raw parents of
    # each engineered survivor so we can tell a true combination from a self-transform.
    _eng_signal_parents: dict[int, set[str]] = {}
    for ei in eng_idx:
        _parents = set()
        _toks = [t for t in _TOKEN_SPLIT.split(cols[ei]) if t]
        for t in _toks:
            base = t if t in raw_name_set else (t.split("__", 1)[0] if ("__" in t and t.split("__", 1)[0] in raw_name_set) else None)
            if base is not None:
                _parents.add(base)
        _eng_signal_parents[ei] = {p for p in _parents if _raw_is_signal_bearing(p)}

    drop_names: list[str] = []
    drop_idx_set: set[int] = set()
    for ri in raw_sel_idx:
        rname = cols[ri]
        all_consumers = eng_consumers.get(rname)
        if not all_consumers:
            continue  # raw not consumed by any survivor -> genuine, keep
        # DPI-TRAP GUARD: restrict the conditioning / anchor set to engineered children
        # that are TRUE COMBINATIONS -- they draw genuine signal from a SECOND raw source
        # besides ``rname`` (>= 2 signal-bearing raw parents). A child whose only
        # signal-bearing parent is ``rname`` itself is a monotone/basis self-transform;
        # conditioning the raw on it is the data-processing-inequality trap (CMI ~0 for
        # EVERY raw) and cannot prove redundancy, so it is excluded. When NO legitimate
        # multi-source consumer remains, the raw is NOT redundancy-dropped here (the
        # protective retention stands) -- this is what restores the genuine ``x_a``/``x_b``
        # (interaction-product operands carrying a private LINEAR term) and ``x0``
        # (paired with a noise column in ``add(exp(x0),sign(x3))``) the 2026-06-08
        # blanket sweep wrongly dropped, while the true ``a**2/b`` ratio operands -- whose
        # subsumer ``div(neg(a),sqrt(b))`` is a genuine two-source combination -- still drop.
        consumers = [
            ei for ei in all_consumers
            if len(_eng_signal_parents.get(ei, set()) - {rname}) >= 1
        ]
        if not consumers:
            if verbose:
                logger.info(
                    "raw-redundancy: KEEP %s (no multi-source engineered subsumer; "
                    "consumers %s are sole-operand self-transforms -- DPI-trap, cannot "
                    "prove redundancy)",
                    rname, [cols[e] for e in all_consumers],
                )
            continue
        # The raw column is binned at the SCREENING codes the selector itself used
        # (already integer 0..k in ``data``). We do NOT up-resolve the raw from
        # continuous values: the redundancy verdict must judge the raw at the
        # resolution the selector saw, and a finer raw binning would only inflate
        # its residual CMI (bias toward KEEP), the unsafe direction.
        rb = np.asarray(data[:, ri]).astype(np.int64).ravel()
        z_support, _ = _renumber_joint(*(eng_bin[ei] for ei in consumers))
        cmi, floor, excess = _excess_and_floor(rb, y_arr, z_support, seed=seed)
        # Raw's OWN marginal debiased excess -- the reference scale for both keep legs.
        _r_mcmi, _r_mfloor, raw_marg_excess = _raw_marginal(rname)
        # Strongest consuming engineered survivor's own debiased marginal excess.
        max_anchor = max(eng_anchor_excess[ei] for ei in consumers)
        # TWO KEEP LEGS (2026-06-10). The raw survives the redundancy drop iff EITHER:
        #   (A) it carries a SIGNIFICANT INDEPENDENT RESIDUAL given the combination
        #       child(ren): its conditional CMI clears the within-stratum permutation
        #       floor AND its debiased conditional excess retains >= RAW_SELF_RETAIN_FRAC
        #       of its OWN marginal excess. The floor leg alone admits a borderline
        #       noise-crossing (a true ``a**2/b`` operand whose residual just nicks the
        #       floor at one n); the self-retention fraction is the n-stable gate that
        #       a genuine private LINEAR term (``x_a``/``x_b`` keep ~6-11% of their
        #       marginal given the product) clears and a subsumed operand (~1-2%) does
        #       not. OR
        #   (B) the strongest consuming child is NOT a genuine SUPERSET of the raw --
        #       its own marginal excess is <= RAW_SUPERSET_MULT x the raw's marginal
        #       excess. A child that merely RE-EXPRESSES the raw through a (near-)
        #       monotone transform plus a noise operand (``add(exp(x0),sign(x3))``,
        #       x3 noise: child ~1.8x the raw, NOT a superset) leaves a cleaner LINEAR
        #       signal in the raw than in the noise-polluted child, so the raw is kept
        #       for downstream linear usability. A true combination (the ``a**2/b``
        #       ratio is ~6x the operand's marginal; the interaction product ~5x) is a
        #       superset and does NOT trip leg B -- those rely on leg A instead.
        # Fails both -> the raw is genuinely subsumed (no significant residual AND a
        # strictly-more-informative combination child captures it) -> DROP.
        passes_floor = cmi > floor
        keep_leg_a = passes_floor and (excess >= RAW_SELF_RETAIN_FRAC * max(0.0, raw_marg_excess))
        keep_leg_b = max_anchor <= RAW_SUPERSET_MULT * max(0.0, raw_marg_excess)
        if keep_leg_a or keep_leg_b:
            if verbose:
                logger.info(
                    "raw-redundancy: KEEP %s (cmi=%.4f floor=%.4f cond_excess=%.5f "
                    "marg_excess=%.5f max_child_anchor=%.4f legA=%s legB=%s -- carries "
                    "independent signal / child is a re-expression not a superset, given %s)",
                    rname, cmi, floor, excess, raw_marg_excess, max_anchor,
                    keep_leg_a, keep_leg_b, [cols[e] for e in consumers],
                )
            continue
        drop_names.append(rname)
        drop_idx_set.add(ri)
        if verbose:
            logger.info(
                "raw-redundancy: DROP %s (cmi=%.4f floor=%.4f cond_excess=%.5f "
                "marg_excess=%.5f max_child_anchor=%.4f -- no significant residual AND "
                "fully subsumed by a strictly-more-informative combination child %s)",
                rname, cmi, floor, excess, raw_marg_excess, max_anchor,
                [cols[e] for e in consumers],
            )

    if not drop_idx_set:
        return sel, []
    kept = [i for i in sel if i not in drop_idx_set]
    return kept, drop_names


__all__ = ["drop_redundant_raw_operands", "DEFAULT_RAW_RETAIN_FRAC"]
