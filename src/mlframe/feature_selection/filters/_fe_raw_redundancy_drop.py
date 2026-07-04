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

  STEP 1 -- KEEP iff the raw carries a SIGNIFICANT INDEPENDENT RESIDUAL given the
  combination child(ren): CMI(raw; y | combination children) clears the within-
  stratum conditional-permutation floor AND the debiased conditional excess retains
  >= ``RAW_SELF_RETAIN_FRAC`` of the raw's OWN marginal debiased excess. A genuine
  PRIVATE LINEAR term keeps ~6-11% of its marginal given the interaction product; a
  fully-subsumed ``a**2/b`` ratio operand keeps ~0.3-2% and does not clear the bar
  -> DROP.

  (HISTORY 2026-06-12) A second keep leg -- "the strongest consuming child is NOT a
  SUPERSET" (``max_anchor <= RAW_SUPERSET_MULT x raw_marg_excess``) -- was present
  2026-06-10..06-12 and has been REMOVED. It aimed to retain a raw whose child only
  RE-EXPRESSES it through a monotone unary paired with a NOISE operand, but that
  case is already handled by STEP 0's DPI-trap consumer filter (the noise-paired
  child has a single signal-bearing parent and is excluded from ``consumers``), so
  the raw never reaches the keep rule. Leg B's only live effect was a FALSE KEEP of
  a DOMINANT raw operand whose large marginal excess made ``3 x marg_excess`` exceed
  any realistic child anchor -- e.g. ``a`` in ``a**2/b``, kept despite being fully
  subsumed by the ``a**2/b`` child (the BUG1 spurious-raw regression).

A redundant operand's conditional excess collapses to ~0 (its CMI given the
combination child is pure finite-sample / binning-gap bias, reproduced by the
within-stratum permutation null and subtracted by the excess) -> the keep rule
fails -> DROP. A raw carrying a private term (linear / additive) keeps a
significant residual, and a raw paired only with noise is shielded by the DPI-trap
filter -> KEPT. All scales are debiased excesses or scale-free fractions, so the
verdict is n-stable (validated on the genuine x_a/x_b/x0 + subsumed a**2/b a/b
across n=500..100000 cells, 2026-06-10/06-12).

All MI/CMI uses the production primitives (``_cmi_from_binned`` / ``_quantile_bin``
/ ``_renumber_joint``) and the production conditional-permutation null
(``_conditional_perm_null`` from the S5 gate). The function is pure -- no live
framework state captured -- so a fitted MRMR stays picklable.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Sequence

import numpy as np

from ._fe_raw_redundancy_helpers import (
    _is_pseudo_remix_child,
    raw_retains_signal_given_genuine_children,
    _recipe_subexprs,
    _subexpr_continuous,
    _excess_and_floor,
    _rank_transform,  # noqa: F401
    _residualize,  # noqa: F401
    raw_retains_linear_signal_given_children,
    _heldout_ridge_r2,
    RAW_SELF_RETAIN_FRAC,
    _MIN_ROWS,
)
from ._fe_usability_signal import (  # shared leaf |corr| (numpy-only, no cycle)
    abs_pearson as _abs_pearson,
    _crit_np_dtype,  # MLFRAME_CRIT_DTYPE_RELAXED-aware dtype for the usability-corr casts
)

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Scale-free retention fraction (TAU). Shared default with the S5 engineered-vs-
# engineered gate: a raw operand must retain >= this fraction of the weakest
# engineered survivor's own debiased excess to be judged a genuine independent
# term. 0.15 sat with >=2x margin both sides across the validated cells.
DEFAULT_RAW_RETAIN_FRAC = 0.15

# SUPERSET multiple (RETIRED 2026-06-12; formerly "keep leg B", 2026-06-10). Kept
# as a module constant only for back-compat / provenance. Leg B
# (``max_anchor <= RAW_SUPERSET_MULT x raw_marg_excess`` -> KEEP) was removed: the
# DPI-trap consumer filter (step 0) already and correctly protects the case it
# targeted (a raw whose child is a monotone re-expression paired with a NOISE
# operand -- that child is excluded from ``consumers`` because the raw is its only
# signal-bearing parent), so by the time the keep rule runs every consumer is a
# genuine multi-source combination and leg B's premise is false by construction.
# Its only live effect was a FALSE KEEP of a DOMINANT raw operand (large marginal
# excess -> ``3.0 x marg_excess`` exceeds any realistic child anchor), which kept a
# fully-subsumed operand like ``a`` in ``a**2/b`` (the BUG1 regression). No longer
# referenced by the verdict.
RAW_SUPERSET_MULT = 3.0

# Downstream no-harm epsilon for the raw-redundancy DROP outcome guard: the maximum held-out linear (Ridge)
# R^2 the full drop may cost before it is reverted (the child is judged linearly lossy and the raws are kept).
# Well inside the I4b/I5 contract's 0.05 no-harm bar, so a genuine harm reverts while a neutral cosmetic drop
# (the child captures the raw linearly, R^2 unchanged) is never disturbed.
_RAW_DROP_NO_HARM_EPS = 0.01

# The downstream no-harm guard's held-out Ridge R^2 probe is meaningful only for a CONTINUOUS (regression)
# target. For classification the caller passes the discrete class labels as ``y_continuous``, and a Ridge R^2
# on those is unreliable (spuriously reverts a drop, re-breaking the strict-drop invariant). Treat a target
# with fewer than this many distinct values as classification-like and SKIP the guard (a regression target is
# effectively continuous -> ~n distinct values, far above this floor).
_MIN_TARGET_DISTINCT_FOR_GUARD = 20

# Equi-frequency bins for raw / engineered / target columns. 10 matches the S5
# gate; deliberately NOT finer -- a very fine engineered binning fragments the
# conditioning strata at large n and re-inflates the residual (measured: 32 bins
# made ws1 ``a`` clear the shrunken floor at n=25000). The RELATIVE-excess bar,
# not the bin count, does the separation.
_BINS = 10

# Conditioning-support fragmentation guard (chi-squared rule-of-thumb: joint cells
# should average >= a few samples for the plug-in CMI to stay reliable). The
# engineered survivors are binned at the target cardinality, but that count is
# capped so the JOINT support of the consuming survivors keeps measurable strata
# at small n. Mirrors the S5 gate's ``_SUPPORT_FRAG_DIVISOR``.
_SUPPORT_FRAG_DIVISOR = 5

# Split an engineered name into identifier tokens (``div(sqr(a),abs(b))`` ->
# {div, sqr, a, abs, b}); the raw operands are the tokens that are raw columns.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")

# Pseudo-child source-token splitter. Unlike ``_TOKEN_SPLIT`` (which keeps ``_`` as a
# word char so ``gate_mask__a__b__t0`` stays a SINGLE token), the gate / binagg / argmax
# canonical names join their source columns with ``__`` / ``(`` / ``|`` and embed a mode
# prefix + tau suffix, so the raw operands are recovered only by splitting on EVERY
# non-alphanumeric run (``gate_mask__a__b__t0.1`` -> {gate, mask, a, b, t0, 0, 1};
# ``binagg_skew(c|qbin(a))`` -> {binagg, skew, c, qbin, a}). The raw operands are the
# tokens that match a raw column name; the FE-keyword tokens (gate/mask/binagg/qbin/...)
# never collide with a raw name.
_PSEUDO_SRC_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def drop_redundant_raw_operands(
    *,
    data: np.ndarray,
    cols: Sequence[str],
    selected_cols_idx: Iterable[int],
    raw_name_set: set,
    y_binned: np.ndarray,
    y_continuous: Optional[np.ndarray] = None,
    engineered_continuous: Optional[dict] = None,
    replayable_eng_names: Optional[set] = None,
    recipes: Optional[dict] = None,
    raw_X=None,
    retain_frac: float = DEFAULT_RAW_RETAIN_FRAC,
    floor_margin_mult: float = 1.0,
    linear_usability_keep: bool = False,
    tail_subsume_enable: bool = True,
    tail_subsume_min_corr: float = 0.85,
    tail_subsume_rank_frac: float = 0.7,
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
    import os as _os

    from ._mi_greedy_cmi_fe import _quantile_bin, _renumber_joint

    sel = list(selected_cols_idx)
    n_rows = int(np.asarray(y_binned).shape[0])
    if n_rows < _MIN_ROWS or len(sel) < 2:
        return sel, []

    # DEVICE-BORN candidate-code residency (default ON under fe_gpu_strict_resident_enabled; env opt-out
    # MLFRAME_FE_GATE_RESIDENT_CANDS=0), mirroring the CMI-redundancy gate (409d63fe). The raw-redundancy
    # sweep scores each engineered survivor's marginal MI (anchor) and each raw operand's marginal + conditional
    # CMI / perm-null on HOST-binned int64 codes, so the SAME candidate content the gate already device-binned
    # (roles ``qbin_x`` / ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x``) re-crossed H2D at every scoring
    # site here. When on, each SCORED candidate (the engineered anchor ``eb``, the raw operand ``rb`` / ``_rb``)
    # is quantile-binned ONCE on device (``_quantile_bin_gpu_resident``, identical percentile-edge partition to
    # the host ``_quantile_bin``) and its RESIDENT int64 codes are threaded through ``_excess_and_floor`` -->
    # ``_cmi_from_binned`` (cupy resident-input branch) + ``_conditional_perm_null`` (resident-input branch), so
    # the candidate never re-uploads. A byte-identical host copy is retained for the genuinely host-only sites
    # (the ``_renumber_joint`` support build, where the same column serves as a CONDITIONING z / sibling). A
    # non-finite column or any cupy fault falls back per-candidate to the host ``_quantile_bin`` (no resident
    # code -> that candidate's device sites re-upload the host code, exactly as before this change).
    _gate_resident = False
    if _os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
            from ._mi_greedy_cmi_fe import _cmi_gpu_enabled
            _gate_resident = bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled())
        except Exception:
            _gate_resident = False

    def _dev_from_cont(_vals) -> "object | None":
        """RESIDENT int64 codes from CONTINUOUS values via the device equi-frequency binner (identical
        partition to ``_quantile_bin``), or ``None`` when residency is off / the column is non-finite / cupy
        faults -- the caller then keeps the host ``_quantile_bin`` codes."""
        if not _gate_resident:
            return None
        try:
            _v = np.asarray(_vals, dtype=np.float64)
            if not np.isfinite(_v).all():
                return None
            from ._mi_greedy_cmi_fe import _quantile_bin_gpu_resident
            return _quantile_bin_gpu_resident(_v, int(_eng_card))
        except Exception:
            return None

    def _dev_from_codes(_codes) -> "object | None":
        """RESIDENT int64 codes for an ALREADY-BINNED host int column (the lossy ``data`` screening codes),
        uploaded ONCE and kept resident so the scored candidate's device sites read it without re-crossing
        H2D. Routed through the content-keyed resident cache (``cmi_cand_x``), so the SAME content the gate /
        anchor already uploaded HITS that copy. ``None`` when residency is off or cupy faults."""
        if not _gate_resident:
            return None
        try:
            from ._fe_resident_operands import resident_code_operand
            return resident_code_operand(np.asarray(_codes).ravel(), "cmi_cand_x")
        except Exception:
            return None

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
    # Cache the per-engineered raw-base set on this first pass; the identical
    # token-split + raw-base lookup is reused below to build ``_eng_signal_parents``.
    eng_consumers: dict[str, list[int]] = {}
    _eng_base_sets: dict[int, set[str]] = {}
    for ei in eng_idx:
        toks = [t for t in _TOKEN_SPLIT.split(cols[ei]) if t]
        # Map ``a__He3``-style warped tokens back to their raw base too.
        bases = set()
        for t in toks:
            if t in raw_name_set:
                bases.add(t)
            elif "__" in t and t.split("__", 1)[0] in raw_name_set:
                bases.add(t.split("__", 1)[0])
        _eng_base_sets[ei] = bases
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

    # RAW-OPERAND BINNING (2026-06-15 compromise). The prior design binned the raw at the LOSSY fit codes
    # in ``data`` and deliberately did NOT up-resolve (a finer raw binning inflates its residual CMI -> bias
    # toward KEEP, the unsafe direction). That holds in the tuned regime (fit nbins >= the engineered
    # survivors' resolution _eng_card), but at a COARSE fit nbins (e.g. 5) the ~5-code raw binning washes
    # out a genuine INDEPENDENT linear residual (equal-coef sum operands), OVER-dropping them and emptying
    # the raw support (never-empty regression at nbins=5). Compromise: up-resolve the raw from its
    # CONTINUOUS values ONLY when the fit codes are COARSER than _eng_card, and CAP at _eng_card -- i.e.
    # match the engineered survivor's resolution, NEVER finer (so the prior inflation concern is honoured;
    # the "32 bins inflated the residual" failure mode is excluded). In the tuned regime (fit levels >=
    # _eng_card) this is BYTE-IDENTICAL to the prior fit-code path -- the raw is still judged at the
    # resolution the selector saw. Fair-comparison rationale: the raw and the consuming survivor are then
    # binned at the SAME cardinality, so the conditional-excess verdict is not skewed by a resolution gap.
    _raw_codes_cache: dict = {}
    # Parallel RESIDENT-code cache: for each raw ``_ridx`` the device-born int64 codes (byte-identical to the
    # host ``_raw_codes`` partition) fed to the SCORED-candidate device sites. Populated on the first
    # ``_raw_codes`` build; ``None`` for a column that failed the device binner (host fallback).
    _raw_dev_cache: dict = {}
    def _raw_codes(_rname, _ridx):
        if _ridx in _raw_codes_cache:
            return _raw_codes_cache[_ridx]
        _fit = np.asarray(data[:, _ridx]).astype(np.int64).ravel()
        _levels = (int(_fit.max()) + 1) if _fit.size else 0
        _out = _fit
        _dev = None
        if raw_X is not None and 0 < _levels < _eng_card:
            try:
                _rc = None
                if hasattr(raw_X, "columns") and _rname in getattr(raw_X, "columns", []):
                    _rc = np.asarray(raw_X[_rname], dtype=np.float64).ravel()
                if _rc is not None and _rc.shape[0] == n_rows and np.isfinite(_rc).any():
                    _clean = np.nan_to_num(_rc, nan=0.0, posinf=0.0, neginf=0.0)
                    _dev = _dev_from_cont(_clean)   # device-born codes = same percentile-edge partition
                    if _dev is not None:
                        import cupy as _cp
                        _out = _cp.asnumpy(_dev).astype(np.int64)   # host copy = D2H of the SAME resident partition
                    else:
                        _out = np.ascontiguousarray(
                            _quantile_bin(_clean, nbins=_eng_card)
                        ).astype(np.int64)
            except Exception:
                _out = _fit
                _dev = None
        if _dev is None:
            # The lossy ``data`` fit codes (or a host-fallback bin): upload once, kept resident so the scored
            # candidate reads it without re-crossing H2D (host copy stays the source of truth for _renumber_joint).
            _dev = _dev_from_codes(_out)
        _raw_codes_cache[_ridx] = _out
        _raw_dev_cache[_ridx] = _dev
        return _out

    def _raw_dev(_rname, _ridx):
        """RESIDENT int64 codes for the raw operand scored as the CANDIDATE, byte-identical to ``_raw_codes``;
        ``None`` -> the caller passes the host codes (which the device sites then upload, as before)."""
        if _ridx not in _raw_codes_cache:
            _raw_codes(_rname, _ridx)
        return _raw_dev_cache.get(_ridx)
    eng_bin: dict[int, np.ndarray] = {}
    # RESIDENT (device) twin of each engineered conditioning code (None where it fell back to host binning). Used
    # to build the round conditioning support DEVICE-BORN (``_renumber_joint_gpu``) so it never crosses H2D.
    eng_bin_dev: dict = {}
    eng_anchor_excess: dict[int, float] = {}
    for ei in eng_idx:
        _ename = cols[ei]
        _cont = _eng_cont.get(_ename)
        _eb_dev = None
        if _cont is not None and np.asarray(_cont).shape[0] == n_rows:
            # Fine binning from the continuous engineered values (preferred). Device-bin ONCE (resident codes
            # for the scored anchor MI); the host copy = D2H of the SAME partition, kept for the ``_renumber_joint``
            # conditioning-support build (``eng_bin[ei]`` feeds ``z_support`` below). Host fallback on cupy fault.
            _cvals = np.asarray(_cont, dtype=np.float64)
            _eb_dev = _dev_from_cont(_cvals)
            if _eb_dev is not None:
                import cupy as _cp
                eb = _cp.asnumpy(_eb_dev).astype(np.int64)
            else:
                eb = _quantile_bin(_cvals, nbins=_eng_card)
        else:
            # Fallback: the lossy screening codes already in ``data`` (use as-is;
            # re-binning coarse codes gains nothing). Conservative -- a missing
            # continuous value can only make the gate KEEP more (smaller anchor).
            eb = np.asarray(data[:, ei]).astype(np.int64).ravel()
            _eb_dev = _dev_from_codes(eb)
        eng_bin[ei] = eb
        eng_bin_dev[ei] = _eb_dev
        # Score the anchor MI from the RESIDENT codes when available so the candidate never re-crosses H2D at
        # the ``cmi_cand_x`` / ``permnull_cand_x`` sites; host codes otherwise.
        _, _, exc = _excess_and_floor(_eb_dev if _eb_dev is not None else eb, y_arr, None, seed=seed,
                                      kx=(int(eb.max()) + 1 if getattr(eb, "size", 0) else 1))
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
        _rb = _raw_codes(_rname, _ridx)
        _rb_dev = _raw_dev(_rname, _ridx)   # RESIDENT candidate code (scored marginal) -> no re-upload; host otherwise
        _res = _excess_and_floor(_rb_dev if _rb_dev is not None else _rb, y_arr, None, seed=seed,
                                 kx=(int(_rb.max()) + 1 if getattr(_rb, "size", 0) else 1))
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
        # Reuse the raw-base set cached during the ``eng_consumers`` pass above
        # (identical token-split + raw-base extraction) instead of recomputing it.
        _parents = _eng_base_sets.get(ei)
        if _parents is None:
            _parents = set()
            for t in (t for t in _TOKEN_SPLIT.split(cols[ei]) if t):
                base = t if t in raw_name_set else (t.split("__", 1)[0] if ("__" in t and t.split("__", 1)[0] in raw_name_set) else None)
                if base is not None:
                    _parents.add(base)
        _eng_signal_parents[ei] = {p for p in _parents if _raw_is_signal_bearing(p)}

    # NESTED-OPERAND CLEAN-SUBEXPRESSION ANCHOR (BUG1, 2026-06-12). A selected
    # survivor may be a FUSED composite that combines two independent signal terms
    # into one feature -- e.g. ``add(div(log(c),reciproc(d)),abs(div(sqr(a),abs(b))))``
    # fuses the ``a**2/b`` ratio with the ``log(c)*sin(d)`` product. Conditioning a
    # raw operand ``a`` on the WHOLE fused composite does NOT cleanly isolate a's
    # capture: the second term acts as nuisance variation across the conditioning
    # strata, so ``CMI(a; y | fused_composite)`` retains a spurious residual and the
    # raw is wrongly KEPT (the BUG1 seed-dependent failure: the composite collapses
    # the whole selection on some seeds -> never-empty path drops a, but on seeds
    # where ``a`` is selected ALONGSIDE the composite the main path here ran and kept
    # it). The fix walks the consumer's recipe operand TREE (the dataclass
    # nested-parent structure, not str()) for the CLEANEST ``rname``-containing
    # sub-expression -- the tightest sub-recipe that still contains ``rname`` plus a
    # SECOND signal-bearing raw (a true combination, not a self-transform) -- replays
    # it to its own continuous values, and conditions ``rname`` on THAT. On the user
    # fixture the cleanest sub-expression of ``a`` is ``div(sqr(a),abs(b))`` = a**2/b,
    # which captures ``a`` fully -> conditional excess collapses -> DROP. A GENUINE
    # private term (``y += 3*a``) keeps a large residual given the same clean ratio
    # sub-expression -> KEPT (the over-drop control). Conservative: any failure to
    # parse / replay falls back to the fused composite bin (the KEEP direction).
    _recipes = recipes or {}
    # (rname, ei) -> binned clean sub-expression conditioning column (or None).
    _clean_subexpr_bin: dict[tuple, Optional[np.ndarray]] = {}
    # (rname, ei) -> RESIDENT (device) twin of the clean sub-expression code (None where host-binned), so a
    # conditioning support that uses the clean sub-expression can still be built DEVICE-BORN.
    _clean_subexpr_bin_dev: dict = {}
    if _recipes and raw_X is not None:
        # Pre-extract each consumer's sub-recipe tree once.
        _consumer_subtrees: dict[int, dict] = {}
        for ei in eng_idx:
            _r = _recipes.get(cols[ei])
            if _r is not None:
                _consumer_subtrees[ei] = _recipe_subexprs(_r)

        def _subexpr_signal_parents(_sub) -> set:
            _p = set()
            for _t in _TOKEN_SPLIT.split(getattr(_sub, "name", "") or ""):
                if not _t:
                    continue
                _base = _t if _t in raw_name_set else (
                    _t.split("__", 1)[0] if ("__" in _t and _t.split("__", 1)[0] in raw_name_set) else None
                )
                if _base is not None:
                    _p.add(_base)
            return {p for p in _p if _raw_is_signal_bearing(p)}

        for ei, _subtree in _consumer_subtrees.items():
            for _rn in raw_name_set:
                # Candidate sub-expressions: contain ``_rn`` AND a second signal-bearing
                # raw (true combination -> not a DPI-trap self-transform).
                _cands = []
                for _sname, _sub in _subtree.items():
                    _sp = _subexpr_signal_parents(_sub)
                    if _rn in _sp and len(_sp - {_rn}) >= 1:
                        _cands.append((len(_sp), _sname, _sub))
                if not _cands:
                    continue
                # CLEANEST = fewest signal-bearing parents (tightest isolation of the
                # raw's capture). Ties broken by the SHORTER name (the inner node).
                _cands.sort(key=lambda t: (t[0], len(t[1])))
                _, _best_name, _best_sub = _cands[0]
                # Only worth a separate anchor if it is a PROPER sub-expression of the
                # whole survivor (fewer signal-bearing parents than the survivor) --
                # otherwise the survivor bin already isolates the raw and no
                # nuisance-fusion problem exists.
                if len(_subexpr_signal_parents(_best_sub)) >= len(_eng_signal_parents.get(ei, set())):
                    continue
                _vals = _subexpr_continuous(_best_sub, raw_X)
                if _vals is None or _vals.shape[0] != n_rows:
                    continue
                # Device-bin the clean sub-expression ONCE (resident twin for the device-born support build); the
                # host copy is the D2H of the SAME percentile-edge partition (byte-identical to _quantile_bin).
                _cvals_clean = np.asarray(_vals, dtype=np.float64)
                _cs_dev = _dev_from_cont(_cvals_clean)
                if _cs_dev is not None:
                    import cupy as _cp
                    _clean_subexpr_bin[(_rn, ei)] = _cp.asnumpy(_cs_dev).astype(np.int64)
                    _clean_subexpr_bin_dev[(_rn, ei)] = _cs_dev
                else:
                    _clean_subexpr_bin[(_rn, ei)] = _quantile_bin(_cvals_clean, nbins=_eng_card)
                    _clean_subexpr_bin_dev[(_rn, ei)] = None
                if verbose:
                    logger.info(
                        "raw-redundancy: conditioning raw %s on CLEAN nested sub-expression "
                        "%s (isolated from fused composite %s) instead of the whole composite.",
                        _rn, _best_name, cols[ei],
                    )

    drop_names: list[str] = []
    def _join_dev(*dev_codes):
        """DEVICE-BORN conditioning-support join of the resident conditioning codes (``_renumber_joint_gpu``), so
        the support never crosses H2D (the ``cmi_z`` + perm-null order/z_rank uploads). Returns the resident
        joint OR None when any code lacks a resident twin (resident path off / host-fallback binning) / on any
        cupy fault -> the caller scores the host support (byte path unchanged). Same partition as the host
        ``_renumber_joint`` -> selection-identical CMI."""
        if not dev_codes or any(_d is None for _d in dev_codes):
            return None
        try:
            from ._mi_greedy_cmi_fe import _renumber_joint_gpu
            return _renumber_joint_gpu(*dev_codes)[0]
        except Exception:
            return None

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
        # GATE / BINAGG / ARGMAX pseudo-remix EXCLUSION (2026-06-13). A conditional-gate /
        # binned-numeric-agg / row-argmax child is a lossy THRESHOLD/BINNING re-mix of the raw, not a
        # combination that can SUBSUME it: conditioning the raw on such a re-mix of ITSELF is the
        # data-processing-inequality trap (CMI collapses for every raw, genuine or redundant), so it
        # MASKS a genuine private term and must NOT participate in the drop/keep verdict. Drop pseudo
        # consumers from the conditioning/anchor set; only GENUINE elementary composites (ratio /
        # product / poly), which can truly subsume the raw, remain. Byte-identical when no pseudo child
        # consumes the raw (the exclusion set is empty). A raw FULLY subsumed by a genuine ``a**2/b``
        # child still drops (that child stays); a raw whose ONLY consumers are pseudo re-mixes is no
        # longer spuriously dropped (no genuine subsumer remains -> the DPI-empty guard below keeps it).
        consumers = [
            ei for ei in all_consumers
            if (len(_eng_signal_parents.get(ei, set()) - {rname}) >= 1)
            and not _is_pseudo_remix_child(cols[ei])
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
        # The raw column is binned via ``_raw_codes`` (see its definition): the selector's lossy fit codes
        # in the tuned regime (fit levels >= _eng_card, BYTE-IDENTICAL to the prior fit-code path -- judge
        # the raw at the resolution the selector saw, no residual inflation), up-resolved from continuous to
        # _eng_card ONLY when the fit binning is COARSER than the survivors' resolution (the coarse-nbins
        # washout fix), never finer than _eng_card (the prior finer-binning inflation concern is honoured).
        rb = _raw_codes(cols[ri], ri)
        # RESIDENT candidate code for the raw scored below (byte-identical to ``rb``): threaded into every
        # ``_excess_and_floor(rb, ...)`` scoring call so the raw operand never re-crosses H2D at the
        # ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x`` sites; the host ``rb`` still builds the
        # ``_renumber_joint`` conditioning support (siblings). ``None`` -> host codes are scored (as before).
        rb_dev = _raw_dev(cols[ri], ri)
        rb_cand = rb_dev if rb_dev is not None else rb
        # Per-consumer conditioning column: prefer the CLEAN nested ``rname``-containing
        # sub-expression (BUG1) when it was successfully isolated/replayed above, else the
        # whole consuming survivor's bin. The clean sub-expression isolates the raw's
        # capture from the fused composite's second signal term so a fully-subsumed
        # operand's conditional excess collapses (DROP) while a genuine private term keeps
        # its residual (KEEP).
        _cond_bins = [
            _clean_subexpr_bin.get((rname, ei), eng_bin[ei])
            for ei in consumers
        ]
        # RESIDENT twin of each conditioning column (clean sub-expr dev if that column used one, else eng dev),
        # for the device-born support join. None-safe: any missing twin -> host support scored.
        _cond_bins_dev = [
            (_clean_subexpr_bin_dev.get((rname, ei)) if (rname, ei) in _clean_subexpr_bin else eng_bin_dev.get(ei))
            for ei in consumers
        ]
        z_support, _zcard = _renumber_joint(*_cond_bins)   # _renumber_joint returns the occupied cardinality
        z_support_dev = _join_dev(*_cond_bins_dev)
        cmi, floor, excess = _excess_and_floor(rb_cand, y_arr, z_support, seed=seed, z_support_dev=z_support_dev,
                                               kx=(int(rb.max()) + 1 if getattr(rb, "size", 0) else 1), kz=int(_zcard))
        # SIBLING-OPERAND CONDITIONING (BUG1 non-invertible-fusion subsumer, 2026-06-16). A
        # consuming composite can FUSE ``rname`` with a SECOND signal-bearing operand in a
        # form that is not invertible from the composite alone -- e.g. ``add(a, sin(c))``
        # carries ``a`` LINEARLY plus a ``sin(c)`` nuisance term. Conditioning ``a`` on the
        # fused sum ALONE leaves the ``sin(c)`` variation un-held across the strata, so ``a``
        # retains a spurious finite-sample residual (measured s909: cond-excess frac 6.7% >
        # the 5% self-retention bar -> wrongly KEPT) even though ``a`` is FULLY recoverable
        # once its sibling operand is known (``a = add(a,sin(c)) - sin(c)``). The clean-
        # subexpr anchor cannot help: ``add(a,sin(c))`` has no tighter sub-expression that
        # still pairs ``a`` with a second signal source. So ALSO measure the residual with
        # each consumer's OTHER signal-bearing raw operands (the siblings) added to the
        # conditioning -- which HOLDS the nuisance term fixed -- and take the SMALLEST
        # debiased excess across {base, +siblings} as the verdict: "is the raw subsumed
        # under the BEST available conditioning?". A linearly-fused operand collapses with
        # its sibling held (s909 ``a``: 6.7% -> 0.57%, cmi below floor -> DROP). Taking the
        # MIN never over-drops a GENUINE PRIVATE term: its residual is high under EVERY
        # conditioning (siblings -- other raws -- cannot manufacture independence from a
        # term the composite+siblings do not span). It also defuses the converse hazard the
        # naive "always add siblings" form created: for an ALREADY-collapsed operand whose
        # composite is invertible without the sibling (``b`` in ``div(sqr(a),exp(b))``:
        # ``e**b = a**2/div``, base frac 2%), adding the sibling ``a`` only FRAGMENTS the
        # strata and INFLATES ``b``'s plug-in residual to 11% -- a false KEEP. The MIN keeps
        # the un-fragmented base verdict there (2% -> DROP). Siblings are added one at a time
        # only while the realised joint cell count stays within the fragmentation budget
        # (avg rows/cell >= _SUPPORT_FRAG_DIVISOR), strongest-marginal first. Byte-identical
        # when a raw has no signal-bearing sibling operand (the candidate set is empty).
        # FULL-COMPOSITE FALLBACK CONDITIONING (2026-06-20). The clean nested sub-expression anchor
        # (BUG1) REPLACES the whole-composite bin in ``_cond_bins`` to isolate the raw's capture from a
        # fused composite's second additive term. But that replacement can be the WRONG direction: when
        # the WHOLE composite already cleanly subsumes the raw (its high-resolution fused codes leave the
        # raw ZERO residual) while the coarser clean SUB-expression leaves a spurious finite-sample
        # residual, conditioning ONLY on the sub-expression KEEPS a genuinely subsumed operand (the
        # canonical ``c`` in ``add(a**2/b, log(c)*sin(d))``: excess GIVEN the full compound 0.000, GIVEN
        # the clean ``mul(log(c),sin(d))`` sub-expr 0.0092 -> wrongly KEPT). The clean sub-expr must
        # AUGMENT the verdict, not REPLACE it: also measure the residual GIVEN the whole consuming
        # composite and take the SMALLEST debiased excess across {clean-subexpr, full-composite}. Like the
        # sibling-conditioning MIN below, taking the strongest subsumption evidence never over-drops a
        # GENUINE private term (its residual is high under EVERY conditioning) but collapses an operand the
        # full composite already captures. Byte-identical when no clean sub-expr was substituted (the
        # full-composite bin IS ``_cond_bins`` then).
        for ei in consumers:
            # SELECTION-EXACT short-circuit (2026-07-03). The debiased ``excess`` is the MIN across conditionings
            # and ``_excess_and_floor`` clamps it >= 0. The blocks below refine it ONLY on a strict ``_excess_f <
            # excess``, so once a CHEAPER conditioning (the base clean-subexpr / low-card support) has already
            # driven ``excess`` to 0 no further conditioning can lower it and no update can fire -- the remaining
            # full-composite / sibling perm-nulls (which land on a near-unique HIGH-cardinality support, kz~n,
            # the degenerate df<=0 case) are pure wasted compute. Skipping them is bit-for-bit identical to the
            # verdict (MIN(0, x>=0) == 0, same accompanying cmi/floor), not just selection-equivalent.
            if excess <= 0.0:
                break
            _clean = _clean_subexpr_bin.get((rname, ei))
            if _clean is not None:
                _z_full, _ = _renumber_joint(eng_bin[ei])
                _z_full_dev = _join_dev(eng_bin_dev.get(ei))
                _cmi_f, _floor_f, _excess_f = _excess_and_floor(
                    rb_cand, y_arr, _z_full, seed=seed, z_support_dev=_z_full_dev)
                if _excess_f < excess:
                    cmi, floor, excess = _cmi_f, _floor_f, _excess_f
        _sibling_names: list = []
        for ei in consumers:
            for _sp in (_eng_signal_parents.get(ei, set()) - {rname}):
                if _sp not in _sibling_names:
                    _sibling_names.append(_sp)
        if _sibling_names and excess > 0.0:   # same selection-exact short-circuit: excess already 0 -> no update possible
            _sibling_names.sort(key=lambda nm: -_raw_marginal(nm)[2])
            _budget = max(1, n_rows // _SUPPORT_FRAG_DIVISOR)
            _sib_cond = list(_cond_bins)
            _sib_cond_dev = list(_cond_bins_dev)   # parallel resident twins for the device-born support join
            _added = False
            for _sn in _sibling_names:
                try:
                    _sidx = cols.index(_sn)
                except ValueError:
                    continue
                _sb = _raw_codes(_sn, _sidx)
                _trial, _ = _renumber_joint(*_sib_cond, _sb)
                if (int(np.unique(_trial).size) if _trial.size else 1) > _budget:
                    continue  # adding this sibling would over-fragment the joint strata
                _sib_cond.append(_sb)
                _sib_cond_dev.append(_raw_dev(_sn, _sidx))   # resident twin (None if host-fallback -> host z)
                _added = True
            if _added:
                _z_sib, _ = _renumber_joint(*_sib_cond)
                _z_sib_dev = _join_dev(*_sib_cond_dev)
                _cmi_s, _floor_s, _excess_s = _excess_and_floor(
                    rb_cand, y_arr, _z_sib, seed=seed, z_support_dev=_z_sib_dev)
                # Take the conditioning that gives the SMALLEST debiased excess -- the
                # strongest evidence of subsumption -- carrying its own (cmi, floor) so the
                # floor check below stays consistent with the chosen conditioning.
                if _excess_s < excess:
                    cmi, floor, excess = _cmi_s, _floor_s, _excess_s
        # Raw's OWN marginal debiased excess -- the reference scale for both keep legs.
        _r_mcmi, _r_mfloor, raw_marg_excess = _raw_marginal(rname)
        # Strongest consuming engineered survivor's own debiased marginal excess.
        max_anchor = max(eng_anchor_excess[ei] for ei in consumers)
        # KEEP RULE (2026-06-12 simplification of the 2026-06-10 two-leg form). The raw
        # survives the redundancy drop iff it carries a SIGNIFICANT INDEPENDENT RESIDUAL
        # given the combination child(ren): its conditional CMI clears the within-stratum
        # permutation floor AND its debiased conditional excess retains >=
        # RAW_SELF_RETAIN_FRAC of its OWN marginal excess. A genuine private LINEAR term
        # (``x_a``/``x_b`` keep ~6-11% of their marginal given the x_a*x_b product) clears
        # this; a fully-subsumed ratio operand (``a`` / ``b`` in ``a**2/b``, ~0.3-2%) does
        # not, so it DROPS.
        #
        # The former leg B (``max_anchor <= RAW_SUPERSET_MULT x raw_marg_excess`` ->
        # KEEP "the child is only a re-expression, not a superset") is REMOVED. It was
        # introduced (2026-06-10) to protect a raw whose engineered child merely
        # RE-EXPRESSES it through a monotone unary paired with a NOISE operand
        # (``add(exp(x0),sign(x3))``, x3 noise). But that protection is ALREADY supplied,
        # and supplied CORRECTLY, by the DPI-TRAP CONSUMER FILTER (step 0): a child whose
        # only signal-bearing parent is the raw itself is dropped from ``consumers``, so
        # such a raw never even reaches the keep legs (``test_redundancy_drop_keeps_signal_
        # raw_paired_with_noise_operand`` is satisfied by the DPI guard, not leg B --
        # verified). By the time the keep rule runs, EVERY consumer is a genuine
        # multi-source combination, so leg B's premise ("the child is not a superset") is
        # false by construction. Its only live effect was a FALSE KEEP of a DOMINANT raw
        # operand: when the raw's marginal excess is large (``a`` in ``a**2/b`` carries
        # the bulk of the target's variance -> marg_excess ~0.54), ``3.0 x marg_excess``
        # exceeds any realistic child anchor, so leg B rescued ``a`` even though it is
        # FULLY subsumed by the ``a**2/b`` child (leg A correctly failed at ~1.4%
        # retention). That is the BUG1 spurious-raw-kept regression. Dropping leg B lets a
        # conditionally-subsumed dominant operand drop while leg A + the DPI guard keep
        # every genuine private-term raw.
        # ``floor_margin_mult`` (>1.0) tightens the significance leg: the conditional CMI must
        # clear the within-stratum permutation floor by that multiple, not merely exceed it. The
        # default 1.0 is the historical bare ``cmi > floor`` (byte-identical for every existing
        # caller). A caller running the sweep on the FINAL selection (post-retention) passes a
        # margin > 1.0 to separate a genuine private residual (clears the floor robustly, ratio
        # >> 1) from a WEAK operand whose tiny conditional excess merely grazes the floor -- the
        # latter is a finite-sample / non-invertible-unary-binning artifact, not private signal,
        # and is the operand a multi-operand survivor structurally subsumes (I4b: ``b`` inside
        # ``sin(b)`` of ``div(qubed(a),sin(b))``, cmi 0.0023 vs floor 0.0018 -> ratio 1.28).
        passes_floor = cmi > floor * float(floor_margin_mult)
        keep = passes_floor and (excess >= RAW_SELF_RETAIN_FRAC * max(0.0, raw_marg_excess))
        # LINEAR-USABILITY KEEP-LEG (variant-3, 2026-06-20). The CMI legs above DROP a raw whose
        # conditional excess collapses given the engineered children -- correct in FULL FE mode
        # (the caller opted into replacing subsumed raws with engineered survivors: I4b drops
        # ``a`` in ``a**2/b``), but WRONG in SIMPLE mode where the user wants a robust raw set and
        # the engineered children are spurious nonlinear nestings of a fundamentally linear signal
        # (``s0`` in ``y=2*s0-1.3*s1+0.8*s2`` is info-subsumed by a complex child yet still the
        # right feature for a downstream model). Statistically the two cases are indistinguishable
        # per-raw (a subsumed monotone operand is as linearly usable as a genuine linear term), so
        # the override is gated on the mode: keep a linearly-usable raw ONLY in simple mode. Uses a
        # permutation-floored partial rank-correlation given the children (n-invariant); a pure
        # noise raw has ~0 residual -> stays dropped even in simple mode.
        if not keep and linear_usability_keep:
            try:
                _lin_raw = None
                if raw_X is not None and rname in getattr(raw_X, "columns", []):
                    _lin_raw = np.asarray(raw_X[rname], dtype=np.float64).ravel()
                if _lin_raw is None:
                    _lin_raw = np.asarray(data[:, ri], dtype=np.float64).ravel()
                _lin_y = (np.asarray(y_continuous, dtype=np.float64).ravel()
                          if y_continuous is not None else np.asarray(y_arr, dtype=np.float64).ravel())
                _lin_children = []
                for _ei in consumers:
                    _enm = cols[_ei]
                    if engineered_continuous and _enm in engineered_continuous:
                        _lin_children.append(np.asarray(engineered_continuous[_enm], dtype=np.float64).ravel())
                    else:
                        _lin_children.append(np.asarray(data[:, _ei], dtype=np.float64).ravel())
                if raw_retains_linear_signal_given_children(_lin_raw, _lin_y, _lin_children, seed=seed):
                    keep = True
                    if verbose:
                        logger.info(
                            "raw-redundancy: KEEP %s via LINEAR-USABILITY leg (CMI collapsed "
                            "cond_excess=%.5f but raw retains significant private linear signal "
                            "given %s -- nonlinear child is not a linear equivalent)",
                            rname, excess, [cols[e] for e in consumers],
                        )
            except Exception:
                pass
        # TAIL-CONCENTRATION CONTINUOUS-SUBSUMPTION DROP (2026-07-03). The binned-CMI keep legs above KEEP a
        # raw whose conditional excess given its engineered children does NOT collapse -- but under heavy
        # outliers a ratio operand (``a`` inside ``div(sqr(a),abs(b))`` of the selected compound) is
        # TAIL-CONCENTRATED: its rank association with y collapses in the bulk, so binned CMI(a; y | compound)
        # sees PHANTOM private signal and keeps it, even though the compound CONTINUOUSLY subsumes it (the
        # compound ~= y, |corr(continuous y)| ~0.99). Same rank-vs-linear blindness as the upstream gates, now
        # at the raw-drop stage. DROP such a raw when: it is a source token of a REPLAYABLE selected survivor
        # (guaranteed -- ``consumers`` are already filtered to ``replayable_eng_names``) whose |corr(continuous
        # y)| is HIGH (>= ``tail_subsume_min_corr``), the raw is linearly WEAKER than that survivor (adds no
        # linear signal it lacks), AND the raw's OWN rank association with y has COLLAPSED relative to its linear
        # |corr| (rank <= ``tail_subsume_rank_frac`` x linear -- the tail-concentration signature). Gated on the
        # rank-collapse leg -> FALSE for BALANCED canonical / the 4 passing F2 profiles (there the raw's rank and
        # linear AGREE, so this never fires and the existing binned-CMI verdict stands byte-identically; those
        # profiles already drop ``a`` via CMI anyway). Best-effort: any error keeps the binned-CMI verdict.
        if keep and tail_subsume_enable and y_continuous is not None:
            try:
                _uc_dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default) -- |corr| is wide-margin
                _yc = np.asarray(y_continuous, dtype=_uc_dt).ravel()
                _rc = None
                if raw_X is not None and hasattr(raw_X, "columns") and rname in getattr(raw_X, "columns", []):
                    _rc = np.asarray(raw_X[rname], dtype=_uc_dt).ravel()
                if _rc is None:
                    _rc = np.asarray(data[:, ri], dtype=_uc_dt).ravel()
                if _rc.shape[0] == _yc.shape[0] and _yc.shape[0] >= 3:
                    # raw's best single-operand LINEAR usability (raw and its square) ...
                    _r_lin = max(_abs_pearson(_yc, _rc), _abs_pearson(_yc, _rc * _rc))
                    # ... and its RANK association (max over the same two forms -- conservative: a strong rank
                    # association on EITHER form blocks the drop, so only a genuine tail collapse fires it).
                    _ry = _rank_transform(_yc)
                    _r_rank = max(
                        _abs_pearson(_ry, _rank_transform(_rc)),
                        _abs_pearson(_ry, _rank_transform(_rc * _rc)),
                    )
                    # strongest subsuming (replayable, selected) survivor's continuous |corr(y)|.
                    _s_lin = 0.0
                    for _ei in consumers:
                        _enm = cols[_ei]
                        _sv = (np.asarray(engineered_continuous[_enm], dtype=_uc_dt).ravel()
                               if (engineered_continuous and _enm in engineered_continuous) else None)
                        if _sv is not None and _sv.shape[0] == _yc.shape[0]:
                            _s_lin = max(_s_lin, _abs_pearson(_yc, _sv))
                    if (
                        _s_lin >= float(tail_subsume_min_corr)
                        and _r_lin < _s_lin
                        and _r_rank <= float(tail_subsume_rank_frac) * _r_lin
                    ):
                        keep = False
                        if verbose:
                            logger.info(
                                "raw-redundancy: DROP %s via TAIL-CONCENTRATION continuous-subsumption "
                                "(raw rank|corr(y)|=%.3f collapsed vs raw linear|corr|=%.3f while the subsuming "
                                "survivor's |corr(y)|=%.3f -- binned CMI kept it on phantom tail signal): %s",
                                rname, _r_rank, _r_lin, _s_lin, [cols[e] for e in consumers],
                            )
            except Exception:
                pass
        if keep:
            if verbose:
                logger.info(
                    "raw-redundancy: KEEP %s (cmi=%.4f floor=%.4f cond_excess=%.5f "
                    "marg_excess=%.5f max_child_anchor=%.4f -- carries significant "
                    "independent residual given %s)",
                    rname, cmi, floor, excess, raw_marg_excess, max_anchor,
                    [cols[e] for e in consumers],
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

    # bench-attempt-rejected (2026-07-02): a NON-OPERAND redundant-raw pass here (test each still-selected
    # non-operand raw against the best-MI survivor subexpression via raw_retains_signal_given_genuine_children)
    # did NOT fix the F6_decoy ab_log survival and showed no other benefit. Root cause of ab_log surviving is
    # NOT this operand-redundancy sweep: (a) the sweep's own drops here are reverted by the downstream no-harm
    # Ridge guard because the nonlinear compound is LINEARLY LOSSY (kept-set held-out R2 0.89 << raw-only 0.99,
    # so the raw operands carry linear signal the child loses); (b) ab_log is re-attached AFTERWARDS by the
    # usability-aware raw-retention device (it is linearly usable and statistically indistinguishable from a
    # genuine linear term). Dropping it would risk genuine linearly-usable raws + violate the no-harm contract.
    # So the redundancy sweep is the wrong lever; the F6_decoy cell stays a documented class-4 xfail.
    if not drop_idx_set:
        return sel, []

    # DOWNSTREAM NO-HARM GUARD (2026-07-01). The per-raw CMI/rank-MI verdict can DROP a raw whose engineered
    # child is LINEARLY LOSSY on skewed terrain: a prewarp/product ENTANGLES its operands so a linear (or tree)
    # model cannot recover the raw's private contribution -- e.g. dropping ``b`` beside ``mul(sqr(a),prewarp(b))``
    # on lognormal costs ~0.23 held-out R^2 (the I4b/I5 no-harm violation). A per-raw linear-usability probe
    # cannot separate this from a genuinely-subsumed operand (the product masks b's per-raw residual), so verify
    # the drop at the OUTCOME level against the SAME reference the contract measures: does the KEPT set's HELD-OUT
    # linear fit (StandardScaler+Ridge) fall materially below the RAW-ONLY baseline (all raw features)? If so,
    # revert the whole drop. The baseline is ALL RAWS, NOT kept+dropped -- the latter is over-sensitive (adding
    # any column rarely lowers held-out Ridge, so it reverts even a delta-neutral cosmetic drop and re-breaks the
    # strict-drop check on uniform terrain, measured). On well-behaved terrain the child captures the raw linearly
    # so the kept set matches raw-only and the drop stands; only a genuinely lossy child drops the kept set below
    # raw-only and reverts. Regression-only (needs continuous y); best-effort. ``_RAW_DROP_NO_HARM_EPS`` is the
    # held-out-R^2 shortfall below raw-only tolerated before reverting (well inside the contract's 0.05 bar).
    _yv = np.asarray(y_continuous, dtype=np.float64).ravel() if y_continuous is not None else None
    _guard_on = (
        _yv is not None
        and _yv.shape[0] == n_rows
        and len(np.unique(_yv)) >= _MIN_TARGET_DISTINCT_FOR_GUARD  # regression only (see constant)
        and _os.environ.get("MLFRAME_FE_DROP_NO_HARM", "1").strip().lower() in ("1", "true", "on", "yes")
    )
    if _guard_on:
        try:
            def _cont_of(i):
                nm = cols[i]
                if nm in raw_name_set and raw_X is not None and hasattr(raw_X, "columns") and nm in raw_X.columns:
                    return np.asarray(raw_X[nm], dtype=np.float64).ravel()
                if engineered_continuous and nm in engineered_continuous:
                    return np.asarray(engineered_continuous[nm], dtype=np.float64).ravel()
                return np.asarray(data[:, i], dtype=np.float64).ravel()

            _kept_idx = [i for i in sel if i not in drop_idx_set]
            _raw_names = ([c for c in raw_X.columns if c in raw_name_set]
                          if (raw_X is not None and hasattr(raw_X, "columns")) else [])
            if _kept_idx and _raw_names and _yv.shape[0] == n_rows:
                _X_kept = np.column_stack([_cont_of(i) for i in _kept_idx])
                _X_rawonly = np.column_stack([np.asarray(raw_X[c], dtype=np.float64).ravel() for c in _raw_names])
                _r_kept = _heldout_ridge_r2(_X_kept, _yv)
                _r_rawonly = _heldout_ridge_r2(_X_rawonly, _yv)
                if _r_kept is not None and _r_rawonly is not None and _r_kept < _r_rawonly - _RAW_DROP_NO_HARM_EPS:
                    if verbose:
                        logger.info(
                            "raw-redundancy: REVERT drop of %s -- kept-set held-out Ridge R2 %.4f is below raw-only "
                            "%.4f by %.4f > %.4f eps (the engineered child is linearly lossy); keep the raws.",
                            drop_names, _r_kept, _r_rawonly, _r_rawonly - _r_kept, _RAW_DROP_NO_HARM_EPS,
                        )
                    return sel, []
        except Exception:
            pass

    kept = [i for i in sel if i not in drop_idx_set]
    return kept, drop_names


__all__ = [
    "drop_redundant_raw_operands",
    "DEFAULT_RAW_RETAIN_FRAC",
    "raw_retains_signal_given_genuine_children",
    "_is_pseudo_remix_child",
]
