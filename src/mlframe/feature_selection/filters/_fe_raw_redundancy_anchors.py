"""Anchor-building phase of ``drop_redundant_raw_operands`` (``_fe_raw_redundancy_drop.py``).

Carved out to keep the parent module under the 1k LOC ceiling. Holds everything the per-raw
DPI-trap / redundancy verdict loop needs BEFORE it can score a single raw operand: the
raw-consumer index, the equi-frequency target/engineered binning (with optional GPU-resident
device twins), the per-raw marginal-signal cache/tests, the "which engineered survivors are true
multi-source combinations" (DPI-trap) classification, and the BUG1 nested clean-sub-expression
conditioning anchors. Returns a ``SimpleNamespace`` bundling both the resolved data AND the
closures (``_raw_codes`` / ``_raw_dev`` / ``_raw_marginal`` / ...) the caller's per-raw loop still
needs to call, since the residency cache + binning caches they close over are private to this
build phase.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

from ._fe_raw_redundancy_helpers import _excess_and_floor, _recipe_subexprs, _subexpr_continuous

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Split an engineered name into identifier tokens (``div(sqr(a),abs(b))`` ->
# {div, sqr, a, abs, b}); the raw operands are the tokens that are raw columns.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")

# Equi-frequency bins for raw / engineered / target columns (mirrors ``_BINS`` in the parent module).
_BINS = 10

# Conditioning-support fragmentation guard divisor (mirrors ``_SUPPORT_FRAG_DIVISOR`` in the parent module).
_SUPPORT_FRAG_DIVISOR = 5


def build_raw_redundancy_anchors(
    *,
    data: np.ndarray,
    cols: Sequence[str],
    sel: list,
    raw_name_set: set,
    y_binned: np.ndarray,
    y_continuous: Optional[np.ndarray],
    engineered_continuous: Optional[dict],
    replayable_eng_names: Optional[set],
    recipes: Optional[dict],
    raw_X: Any,
    seed: int,
    verbose: int,
    n_rows: int,
    gate_resident: bool,
) -> SimpleNamespace:
    """Build the anchor/consumer/binning context ``drop_redundant_raw_operands`` scores each raw against.

    Returns a namespace with ``early_return`` set to ``(sel, [])`` when the degenerate/guard paths short-circuit
    (no engineered or raw survivors, no replayable anchor, no consumer map) -- the caller must check it first and
    return immediately when non-``None``. Otherwise every field the per-raw loop needs is populated.
    """
    _gate_resident = gate_resident

    def _dev_from_cont(_vals, _eng_card_local: int) -> Any:
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
            return _quantile_bin_gpu_resident(_v, int(_eng_card_local))
        except Exception as exc:
            # Hot per-candidate path: debug-only. Caller falls back to host _quantile_bin codes.
            logger.debug("GPU-resident quantile-bin failed, falling back to host binning: %s", exc)
            return None

    def _dev_from_codes(_codes) -> Any:
        """RESIDENT int64 codes for an ALREADY-BINNED host int column (the lossy ``data`` screening codes),
        uploaded ONCE and kept resident so the scored candidate's device sites read it without re-crossing
        H2D. Routed through the content-keyed resident cache (``cmi_cand_x``), so the SAME content the gate /
        anchor already uploaded HITS that copy. ``None`` when residency is off or cupy faults."""
        if not _gate_resident:
            return None
        try:
            from ._fe_resident_operands import resident_code_operand
            return resident_code_operand(np.asarray(_codes).ravel(), "cmi_cand_x")
        except Exception as exc:
            # Hot per-candidate path: debug-only. Caller re-uploads host codes as before.
            logger.debug("resident code-operand upload failed, falling back to host codes: %s", exc)
            return None

    sel_names = [cols[i] for i in sel]
    eng_idx = [i for i, nm in zip(sel, sel_names) if nm not in raw_name_set]
    raw_sel_idx = [i for i, nm in zip(sel, sel_names) if nm in raw_name_set]
    if not eng_idx or not raw_sel_idx:
        return SimpleNamespace(early_return=(sel, []))

    # REPLAYABLE-ANCHOR GUARD (2026-06-11): restrict the engineered subsumer/anchor set to survivors that will
    # actually survive into the fitted ``transform()`` output (see the parent module docstring for the full
    # nested-engineered-anchor rationale).
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
            return SimpleNamespace(early_return=(sel, []))

    # raw_name -> list of engineered survivor column indices that consume it.
    eng_consumers: dict[str, list[int]] = {}
    _eng_base_sets: dict[int, set[str]] = {}
    for ei in eng_idx:
        toks = [t for t in _TOKEN_SPLIT.split(cols[ei]) if t]
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
        return SimpleNamespace(early_return=(sel, []))

    # Target codes (equi-frequency re-binning of a skewed continuous target; see the parent module docstring).
    y_arr = np.ascontiguousarray(np.asarray(y_binned)).ravel()
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _target_card = int(np.unique(y_arr).size)
    if y_continuous is not None:
        from ._mi_greedy_cmi_fe import _quantile_bin

        _yc = np.asarray(y_continuous).reshape(-1)
        if _yc.shape[0] == n_rows and np.issubdtype(_yc.dtype, np.number):
            if int(np.unique(_yc).size) > max(2 * _BINS, 2 * _target_card):
                _nb = int(min(max(_BINS, _target_card), max(2, n_rows // (_BINS * _SUPPORT_FRAG_DIVISOR))))
                y_arr = np.ascontiguousarray(_quantile_bin(_yc.astype(np.float64), nbins=_nb)).astype(np.int64)
                _target_card = int(np.unique(y_arr).size)

    _eng_card = int(min(max(_BINS, int(np.unique(y_arr).size)), max(2, n_rows // (_BINS * _SUPPORT_FRAG_DIVISOR))))
    _eng_cont = engineered_continuous or {}

    from ._mi_greedy_cmi_fe import _quantile_bin

    _raw_codes_cache: dict = {}
    _raw_dev_cache: dict = {}

    def _raw_codes(_rname, _ridx):
        """Cached binned codes for raw column ``_ridx`` at the fair-comparison resolution (up-resolved from
        continuous values to ``_eng_card`` only when the fit codes in ``data`` are coarser, never finer -- see
        the 2026-06-15 compromise note in the parent module). Also populates the parallel resident-code cache."""
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
                    _dev = _dev_from_cont(_clean, _eng_card)
                    if _dev is not None:
                        import cupy as _cp
                        _out = _cp.asnumpy(_dev).astype(np.int64)
                    else:
                        _out = np.ascontiguousarray(_quantile_bin(_clean, nbins=_eng_card)).astype(np.int64)
            except Exception as exc:
                # Hot per-candidate path: debug-only. Falls back to the coarser fit codes.
                logger.debug("raw-operand up-resolve failed, keeping fit-resolution codes: %s", exc)
                _out = _fit
                _dev = None
        if _dev is None:
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
    eng_bin_dev: dict = {}
    eng_anchor_excess: dict[int, float] = {}
    for ei in eng_idx:
        _ename = cols[ei]
        _cont = _eng_cont.get(_ename)
        _eb_dev = None
        if _cont is not None and np.asarray(_cont).shape[0] == n_rows:
            _cvals = np.asarray(_cont, dtype=np.float64)
            _eb_dev = _dev_from_cont(_cvals, _eng_card)
            if _eb_dev is not None:
                import cupy as _cp
                eb = _cp.asnumpy(_eb_dev).astype(np.int64)
            else:
                eb = _quantile_bin(_cvals, nbins=_eng_card)
        else:
            eb = np.asarray(data[:, ei]).astype(np.int64).ravel()
            _eb_dev = _dev_from_codes(eb)
        eng_bin[ei] = eb
        eng_bin_dev[ei] = _eb_dev
        _, _, exc = _excess_and_floor(_eb_dev if _eb_dev is not None else eb, y_arr, None, seed=seed, kx=(int(eb.max()) + 1 if getattr(eb, "size", 0) else 1))
        eng_anchor_excess[ei] = exc

    _raw_marg_cache: dict[str, tuple] = {}

    def _raw_marginal(_rname: str) -> tuple:
        """Cached ``(cmi, floor, debiased_excess)`` for the raw's UNCONDITIONAL relationship with ``y`` (see the
        parent module docstring's DPI-TRAP GUARD section)."""
        if _rname in _raw_marg_cache:
            return _raw_marg_cache[_rname]
        try:
            _ridx = cols.index(_rname)
        except ValueError:
            _raw_marg_cache[_rname] = (0.0, 0.0, 0.0)
            return _raw_marg_cache[_rname]
        _rb = _raw_codes(_rname, _ridx)
        _rb_dev = _raw_dev(_rname, _ridx)
        _res = _excess_and_floor(_rb_dev if _rb_dev is not None else _rb, y_arr, None, seed=seed, kx=(int(_rb.max()) + 1 if getattr(_rb, "size", 0) else 1))
        _raw_marg_cache[_rname] = _res
        return _res  # type: ignore[no-any-return]

    def _raw_is_signal_bearing(_rname: str) -> bool:
        """True iff the raw's marginal CMI clears its own permutation floor with positive debiased excess."""
        _mcmi, _mfloor, _mexc = _raw_marginal(_rname)
        return bool(_mcmi > _mfloor and _mexc > 0.0)

    _eng_signal_parents: dict[int, set[str]] = {}
    for ei in eng_idx:
        _parents = _eng_base_sets.get(ei)
        if _parents is None:
            _parents = set()
            for t in (t for t in _TOKEN_SPLIT.split(cols[ei]) if t):
                base = t if t in raw_name_set else (t.split("__", 1)[0] if ("__" in t and t.split("__", 1)[0] in raw_name_set) else None)
                if base is not None:
                    _parents.add(base)
        _eng_signal_parents[ei] = {p for p in _parents if _raw_is_signal_bearing(p)}

    # NESTED-OPERAND CLEAN-SUBEXPRESSION ANCHOR (BUG1, 2026-06-12); see the parent module docstring.
    _recipes = recipes or {}
    _clean_subexpr_bin: dict[tuple, np.ndarray] = {}
    _clean_subexpr_bin_dev: dict = {}
    if _recipes and raw_X is not None:
        _consumer_subtrees: dict[int, dict] = {}
        for ei in eng_idx:
            _r = _recipes.get(cols[ei])
            if _r is not None:
                _consumer_subtrees[ei] = _recipe_subexprs(_r)

        def _subexpr_signal_parents(_sub) -> set:
            """Signal-bearing raw parents referenced by a recipe sub-expression node's name."""
            _p = set()
            for _t in _TOKEN_SPLIT.split(getattr(_sub, "name", "") or ""):
                if not _t:
                    continue
                _base = _t if _t in raw_name_set else (_t.split("__", 1)[0] if ("__" in _t and _t.split("__", 1)[0] in raw_name_set) else None)
                if _base is not None:
                    _p.add(_base)
            return {p for p in _p if _raw_is_signal_bearing(p)}

        for ei, _subtree in _consumer_subtrees.items():
            # ``_subexpr_signal_parents(_sub)`` depends only on ``_sub`` (fixed per ``(ei, _sname)``), NOT on
            # ``_rn`` -- the old code recomputed it once per ``_rn in raw_name_set`` (hundreds of raw columns)
            # for the SAME sub-expression set, an O(|raw_name_set|) redundant recompute. Hoisted to run ONCE
            # per ``ei`` (56980 calls / 0.97s tottime on a 99401x~519 wellbore-shaped profile), reused across
            # every ``_rn`` below via a plain dict lookup. Selection-identical (same values, just computed once).
            _sub_parents = {_sname: _subexpr_signal_parents(_sub) for _sname, _sub in _subtree.items()}
            # FE_REDUNDANCY_SYNERGY-2 fix: this loop used to iterate the CALLER'S
            # FULL raw input-feature-name set (P in the tens of thousands on a wide production frame), not
            # just the raw names actually referenced by this ei's sub-expressions -- an O(P) trivial-but-
            # non-zero pass (a set-membership + length check per raw name per sub-expression) for a decision
            # the recipe already narrows to 2-4 relevant raw names. Iterate only the (bounded) union of
            # actually-referenced parents instead; bit-identical result (excluded raw names never had a
            # candidate anyway, since `_rn in _sp` would always be False for them).
            _all_relevant_raws = set().union(*_sub_parents.values()) if _sub_parents else set()
            for _rn in _all_relevant_raws:
                _cands = []
                for _sname, _sub in _subtree.items():
                    _sp = _sub_parents[_sname]
                    if _rn in _sp and len(_sp - {_rn}) >= 1:
                        _cands.append((len(_sp), _sname, _sub))
                if not _cands:
                    continue
                _cands.sort(key=lambda t: (t[0], len(t[1])))
                _, _best_name, _best_sub = _cands[0]
                if len(_sub_parents[_best_name]) >= len(_eng_signal_parents.get(ei, set())):
                    continue
                _vals = _subexpr_continuous(_best_sub, raw_X)
                if _vals is None or _vals.shape[0] != n_rows:
                    continue
                _cvals_clean = np.asarray(_vals, dtype=np.float64)
                _cs_dev = _dev_from_cont(_cvals_clean, _eng_card)
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

    def _join_dev(*dev_codes):
        """DEVICE-BORN conditioning-support join of the resident conditioning codes (``_renumber_joint_gpu``),
        so the support never crosses H2D. ``None`` when any code lacks a resident twin or on any cupy fault."""
        if not dev_codes or any(_d is None for _d in dev_codes):
            return None
        try:
            from ._mi_greedy_cmi_fe import _renumber_joint_gpu
            return _renumber_joint_gpu(*dev_codes)[0]
        except Exception as exc:
            # Hot per-candidate path: debug-only. Caller falls back to host-side join.
            logger.debug("device-resident conditioning-support join failed, falling back to host join: %s", exc)
            return None

    return SimpleNamespace(
        early_return=None,
        sel_names=sel_names,
        eng_idx=eng_idx,
        raw_sel_idx=raw_sel_idx,
        eng_consumers=eng_consumers,
        y_arr=y_arr,
        eng_bin=eng_bin,
        eng_bin_dev=eng_bin_dev,
        eng_anchor_excess=eng_anchor_excess,
        eng_signal_parents=_eng_signal_parents,
        clean_subexpr_bin=_clean_subexpr_bin,
        clean_subexpr_bin_dev=_clean_subexpr_bin_dev,
        raw_marginal=_raw_marginal,
        raw_is_signal_bearing=_raw_is_signal_bearing,
        raw_codes=_raw_codes,
        raw_dev=_raw_dev,
        join_dev=_join_dev,
    )
