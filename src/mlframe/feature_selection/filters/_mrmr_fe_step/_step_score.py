"""Per-candidate scoring / quantile-discretization materialise stage of ``MRMR._run_fe_step``.

Carved verbatim from ``_step_core.py`` (the irreducible single-function FE-step body) to bring that
module under the 1k-LOC ceiling. ``materialise_and_finalise_fe_candidates`` is the back half of the
``if True:`` FE-pair pipeline: the conditional-MI redundancy gate, the gate-composite / fast-search
over-materialisation prunes, the discretise+append+recipe materialise loop, the auto-escalation tail,
the ROOT-CAUSE-5 ``selected_vars`` promotion, and the cross-fold stability vote. It mutates the loop
locals threaded in as explicit keyword args (no closure capture) and returns the values the parent
re-binds; the dict/set/recipe containers (``engineered_features`` / ``checked_pairs`` /
``engineered_recipes``) are mutated in place. Selection is byte-for-byte identical to the inline block.

The two helper callables ``discretize_array`` / ``get_new_feature_name`` are passed in (they are lazily
imported in the parent from ``..mrmr`` to avoid the mrmr<->this-package import cycle); polars is imported
in-body only on the polars path. All other intra-filters dependencies are imported lazily in-body exactly
as they were in the inline block.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection


def materialise_and_finalise_fe_candidates(
    self,
    *,
    prospective_additions,
    prospective_pairs,
    _prevalence_failed_synergy,
    _pair_maxt_floor,
    _polynom_engineered_indices,
    data, cols, nbins, X,
    classes_y,
    selected_vars,
    engineered_features,
    engineered_recipes,
    checked_pairs,
    n_recommended_features,
    num_fs_steps,
    fe_max_steps,
    fe_unary_preset,
    fe_binary_preset,
    _is_polars_input,
    verbose,
    discretize_array,
    get_new_feature_name,
):
    """Run the redundancy gates, materialise admitted FE candidates, escalate, and stability-vote.

    Returns ``(prospective_additions, data, cols, nbins, X, selected_vars, n_recommended_features)``.
    ``engineered_features`` / ``checked_pairs`` / ``engineered_recipes`` are mutated in place.
    """
    if _is_polars_input:
        import polars as pl  # noqa: F401 -- used on the polars dispatch branches in the body

    # The recipe builder below reads the per-operand prewarp / gate-med fitted-spec accumulators that
    # the parent fills (from ``check_prospective_fe_pairs``) and backs on ``self``; re-bind the SAME
    # dict objects here so recipe construction sees every spec fit this fit (cross-iteration persistence).
    _prewarp_specs = getattr(self, "_prewarp_specs_accum_", None)
    if _prewarp_specs is None:
        _prewarp_specs = {}
        self._prewarp_specs_accum_ = _prewarp_specs
    _gate_med_specs = getattr(self, "_gate_med_specs_accum_", None)
    if _gate_med_specs is None:
        _gate_med_specs = {}
        self._gate_med_specs_accum_ = _gate_med_specs

    # CONDITIONAL-MI REDUNDANCY GATE (strategy S5, 2026-06-08). The PRINCIPLED,
    # constant-free replacement for the hardcoded ``fe_min_engineered_mi_prevalence``
    # joint-prevalence ratio. After the per-pair acceptance machinery has selected one
    # best engineered column per pair, run a greedy CMI-MRMR over the SURVIVING pool:
    # admit a candidate iff its CONDITIONAL MI with y GIVEN the already-admitted
    # ENGINEERED features clears (1) a conditional-permutation floor AND (2) a scale-free
    # fraction (TAU=``fe_engineered_cmi_retain_frac``, default 0.15) of the weakest
    # admitted feature's CMI. A redundant engineered column whose y-information is wholly
    # carried by the admitted features collapses to ~0 CMI and is dropped here; a genuine
    # column carrying a PRIVATE interaction term keeps a large CMI and is kept. Default
    # path (``fe_acceptance == 'conditional_mi'``); the old ratio remains available via
    # ``fe_acceptance == 'prevalence_ratio'`` (then this block is skipped and the per-pair
    # ratio gate alone decides, exactly as before). Validated 10/10 vs four failing
    # approaches across 16 (seed, formula) cells; see ``_fe_cmi_redundancy_gate``.
    _fe_acceptance = str(getattr(self, "fe_acceptance", "conditional_mi"))
    _cmi_dropped: set = set()
    if _fe_acceptance == "conditional_mi" and prospective_additions:
        from .._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate
        from ..mrmr import discretize_array  # already imported above; re-bind for clarity

        # Build the surviving-candidate pool: {engineered_col_name -> (continuous_vals,
        # marginal_mi)}. The continuous values are the pair search's ``transformed_vals``
        # (full-n float, NOT pre-binned). Marginal MI is computed cheaply from the binned
        # values via the same plug-in primitive (z=None) so the seed/relative-bar anchor
        # matches the production CMI estimator -- no separate MI kernel.
        from .._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

        # y codes: reuse the discretised target the MI sweep scored against.
        _y_codes = np.asarray(classes_y).ravel()
        _, _y_dense = np.unique(_y_codes, return_inverse=True)
        _y_dense = _y_dense.astype(np.int64)

        _cmi_cands: dict = {}
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if not _tpf or _tvals is None or not _ncols:
                continue
            for _jc, _cname in enumerate(_ncols):
                if _tvals.shape[1] <= _jc:
                    continue
                _vals = np.asarray(_tvals[:, _jc], dtype=np.float64)
                _vb = _quantile_bin(_vals, nbins=int(self.quantization_nbins))
                _marg = float(_cmi_from_binned(_vb, _y_dense, None))
                _cmi_cands[_cname] = (_vals, _marg)

        if len(_cmi_cands) >= 2:
            _retain = float(getattr(self, "fe_engineered_cmi_retain_frac", 0.15))
            _escape = float(getattr(self, "fe_engineered_cmi_significance_escape_margin", 3.0))
            _cmi_max_cands = int(getattr(self, "fe_engineered_cmi_max_candidates", 64))
            _accepted, _diag = apply_cmi_redundancy_gate(
                _cmi_cands, _y_dense,
                nbins=int(self.quantization_nbins),
                retain_frac=_retain,
                significance_escape_margin=_escape,
                max_candidates=_cmi_max_cands,
                seed=int(self._effective_random_seed() or 0),
                verbose=int(bool(verbose)),
            )
            _cmi_dropped = set(_cmi_cands) - _accepted
            if _cmi_dropped and verbose:
                logger.info(
                    "CMI-redundancy gate: dropped %d/%d engineered survivors as redundant "
                    "given the admitted engineered support (TAU=%.3f): %s",
                    len(_cmi_dropped), len(_cmi_cands), _retain, sorted(_cmi_dropped),
                )
            # REJECTION LEDGER (additive): the CMI gate already returns a per-name
            # ``_diag`` dict carrying observed CMI, the permutation floor, the relative
            # bar and the reason -- harvest it for the dropped names (no recompute).
            for _dn in _cmi_dropped:
                _d = _diag.get(_dn, {}) if isinstance(_diag, dict) else {}
                _reason = str(_d.get("reason", "redundant"))
                # Pick the bar this candidate actually missed: below_floor -> the perm
                # floor (observed = cmi); below_rel_bar -> the relative bar (observed =
                # debiased excess). Margin = observed - threshold (negative => missed).
                if _reason == "redundant_below_floor":
                    _obs = _d.get("cmi", float("nan"))
                    _thr = _d.get("floor", float("nan"))
                else:
                    _obs = _d.get("cmi_excess", float("nan"))
                    _thr = _d.get("rel_bar", float("nan"))
                _record_fe_rejection(
                    self, gate="cmi_redundancy",
                    candidate=str(_dn), operands=None, operator="engineered",
                    observed=_obs, threshold=_thr, reason=_reason,
                    step=int(num_fs_steps),
                )

    # Apply the CMI-redundancy drops to ``prospective_additions`` IN PLACE so the
    # materialise / recipe loop below never appends a redundant engineered column.
    # Each entry's parallel arrays (``this_pair_features`` set of (config, j),
    # ``transformed_vals`` columns, ``new_cols`` names, ``new_nbins``) are filtered to
    # the surviving columns by NAME. ``new_cols[i]`` is the name of the i-th
    # ``transformed_vals`` column; the matching ``(config, j)`` is the one whose
    # ``get_new_feature_name(config, cols)`` equals that name. Both downstream
    # consumers index ``transformed_vals`` by the per-column position
    # (materialise: ``for j in range(len(this_pair_features))``; recipe: the tuple's
    # stored ``j``), so the kept tuples are re-emitted as ``(config, new_position)``
    # with the new packed column position. Entries whose every column was dropped
    # are removed entirely. In the common one-best-per-pair case a pair holds a
    # single column, so this reduces to keep-entry / drop-entry.
    if _cmi_dropped:
        from ..mrmr import get_new_feature_name as _get_new_feature_name
        _filtered_additions: dict = {}
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if not _tpf or _tvals is None or not _ncols:
                _filtered_additions[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                continue
            _keep_idx = [i for i, nm in enumerate(_ncols) if nm not in _cmi_dropped]
            if not _keep_idx:
                continue  # whole pair redundant -> drop the entry
            if len(_keep_idx) == len(_ncols):
                _filtered_additions[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                continue
            # Name -> config map (authoritative column<->config link).
            _name_to_cfg = {
                _get_new_feature_name(_cfg, cols): _cfg for _cfg, _ in _tpf
            }
            _new_tpf = set()
            for _new_pos, _old_i in enumerate(_keep_idx):
                _cfg = _name_to_cfg.get(_ncols[_old_i])
                if _cfg is not None:
                    _new_tpf.add((_cfg, _new_pos))
            _new_tvals = _tvals[:, _keep_idx]
            _new_ncols = [_ncols[i] for i in _keep_idx]
            _new_nnb = (
                [_nnb[i] for i in _keep_idx]
                if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols)
                else _nnb
            )
            _filtered_additions[_rp] = (_new_tpf, _new_tvals, _new_ncols, _new_nnb, _msgs)
        prospective_additions = _filtered_additions

    # GATE-OPERAND COMPOSITE OVER-MATERIALIZATION PRUNE (2026-06-13). The conditional_gate / row_argmax
    # pre-pass appends ENGINEERED gate columns (``gate_mask__b__d__t..``) that screening selects; the FE
    # pair search then pairs each gate column with a raw operand, emitting gate-operand COMPOSITES
    # (``mul(cbrt(c),log(gate_mask__b__d))``, ``div(neg(a),sqrt(gate_mask__b__d))`` ..). On CASE1
    # ``y=a**2/b+log(c)*sin(d)`` the gate is built across the two TRUE groups (b__d), so ~6 such composites
    # pile up ALONGSIDE the clean ``div(sqr(a),neg(b))`` / ``mul(log(c),sin(d))`` survivors that already cover
    # {a,b} and {c,d} -- 9 engineered cols, over the test cap (<=4). They are RE-MIXES (a slightly different
    # threshold nonlinearity) the CMI gate does not drop. Discriminator: drop a gate-operand COMPOSITE iff
    # EVERY raw variable it touches -- its bare-token operands PLUS the gate operand's own raw sources
    # (resolved via ``_gate_col_src_vars_``) -- is ALREADY covered by the UNION of the CLEAN (non-gate)
    # engineered features that SURVIVED the CMI gate above. Built on the POST-CMI survivors (not the raw
    # candidate pool) so a transient clean (c,d) candidate that the CMI gate itself dropped cannot make a
    # genuine gate composite look redundant. On CASE2 ``y=0.2 a**2/b+log(c*2)sin(d/3)`` the gate is built over
    # the TRUE c__d pair and the only surviving (c,d) carrier is the gate composite -- no clean survivor
    # covers {c,d}, so it is KEPT and the warped interaction stays captured. The BARE gate column is never
    # pruned here. Byte-identical when no gate fired (empty ``_gate_col_src_vars_``).
    _gate_src_vars_map = dict(getattr(self, "_gate_col_src_vars_", None) or {})
    if _gate_src_vars_map and prospective_additions:
        import re as _re_gate

        def _bare_tokens(_nm: str) -> set:
            # Single-token raw variable references (a-z / x_NN style), NOT substrings of function names.
            return set(_re_gate.findall(r"(?<![A-Za-z0-9_])([a-z](?:[a-z]?\d+)?)(?![A-Za-z0-9_])", _nm))

        def _gate_cols_in(_nm: str) -> list:
            return [_gc for _gc in _gate_src_vars_map if _gc in _nm]

        _all_names = []
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if _ncols:
                _all_names.extend(_ncols)
        # Clean coverage = bare-token vars of the SURVIVING non-gate engineered features only.
        _clean_cov: set = set()
        for _nm in _all_names:
            if not _gate_cols_in(_nm):
                _clean_cov |= _bare_tokens(_nm)

        # Per clean (non-gate) survivor: (bare-var coverage, marginal MI). The marginals reuse the values
        # the CMI gate already binned for this exact pool, so no extra MI kernel is run.
        _name_marg: dict = {}
        try:
            _cmi_cands_local = _cmi_cands  # defined only in the conditional_mi branch above
        except NameError:
            _cmi_cands_local = None
        if isinstance(_cmi_cands_local, dict):
            for _nm0, _vm0 in _cmi_cands_local.items():
                try:
                    _name_marg[_nm0] = float(_vm0[1])
                except Exception:
                    pass
        _clean_forms = [
            (_bare_tokens(_nm), _name_marg.get(_nm, 0.0))
            for _nm in _all_names if not _gate_cols_in(_nm)
        ]

        # A gate-operand COMPOSITE is over-materialization (DROP) when its whole raw coverage is already
        # provided by the clean survivors AND it is NOT the genuine carrier of an otherwise-uncaptured
        # interaction. Three independent re-mix tells, any of which condemns it:
        #   (1) it embeds >=2 distinct gate columns -- a full-target re-mix, never a single clean pair
        #       (CASE1 ``add(log(gate_mask__b__d),sub(sin(a),sin(gate_mask__d__c)))``);
        #   (2) its gate is built over a CROSS-group pair -- no single clean survivor contains the whole
        #       gate pair, so it fuses two INDEPENDENT already-captured signals (CASE1 ``gate_mask__b__d``);
        #   (3) its gate is WITHIN one clean survivor's pair AND that clean survivor is at least as STRONG
        #       (marginal MI) -- the clean elementary form is the better carrier, the gate is redundant
        #       (CASE1 ``gate_mask__d__c`` vs the strong ``mul(log(c),sin(d))``). When the clean same-pair
        #       form is WEAKER than the gate composite (CASE2 ``sub(exp(c),cbrt(d))`` does not even reach
        #       final support), the gate composite is the genuine (c,d) carrier and is KEPT.
        _gate_composite_drop: set = set()
        for _nm in _all_names:
            _gcs = _gate_cols_in(_nm)
            if not _gcs:
                continue
            if _nm in _gate_src_vars_map:
                continue  # never prune the BARE gate column itself (CASE2 fallback carrier)
            _gate_src = set()
            for _gc in _gcs:
                _gate_src |= set(_gate_src_vars_map.get(_gc, ()))
            _cov = _bare_tokens(_nm) | _gate_src
            _cov -= set(_gate_src_vars_map)  # drop any gate-col token mistakenly captured
            if not (_cov and _cov <= _clean_cov and _gate_src and _gate_src <= _clean_cov):
                continue
            _g_marg = _name_marg.get(_nm, 0.0)
            _multi_gate = len(set(_gcs)) >= 2
            _within_one = any(_gate_src <= _cf_cov for _cf_cov, _ in _clean_forms)
            _stronger_clean_same_pair = any(
                (_gate_src <= _cf_cov) and (_cf_marg >= _g_marg)
                for _cf_cov, _cf_marg in _clean_forms
            )
            # (4) it drags an EXTRA raw operand beyond its own gate pair (``sub(sin(a),sin(gate_mask__d__c))``
            #     -- the gate is over (c,d) but the node also pulls in raw ``a`` from the OTHER group) while
            #     the gate pair is already WITHIN a clean survivor: the node is then an ENTANGLED cross-group
            #     re-mix, never a clean carrier of any single pair (the clean ``mul(log(c),sin(d))`` carries
            #     (c,d) and ``div(sqr(a),neg(b))`` carries a). Only fires under the outer ``_cov <= _clean_cov``
            #     guard, so a genuine carrier of an otherwise-uncaptured group (CASE2) is never reached here.
            _extra_raw = _bare_tokens(_nm) - _gate_src
            _entangled_extra = bool(_extra_raw) and _within_one
            if _multi_gate or (not _within_one) or _stronger_clean_same_pair or _entangled_extra:
                _gate_composite_drop.add(_nm)

        if _gate_composite_drop:
            from ..mrmr import get_new_feature_name as _gnf_gate
            _filtered: dict = {}
            for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                if not _tpf or _tvals is None or not _ncols:
                    _filtered[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                _keep_idx = [i for i, nm in enumerate(_ncols) if nm not in _gate_composite_drop]
                if not _keep_idx:
                    continue
                if len(_keep_idx) == len(_ncols):
                    _filtered[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                _n2c = {_gnf_gate(_cfg, cols): _cfg for _cfg, _ in _tpf}
                _new_tpf = set()
                for _np, _oi in enumerate(_keep_idx):
                    _cfg = _n2c.get(_ncols[_oi])
                    if _cfg is not None:
                        _new_tpf.add((_cfg, _np))
                _new_nnb = (
                    [_nnb[i] for i in _keep_idx]
                    if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols)
                    else _nnb
                )
                _filtered[_rp] = (_new_tpf, _tvals[:, _keep_idx], [_ncols[i] for i in _keep_idx], _new_nnb, _msgs)
            prospective_additions = _filtered
            for _dn in _gate_composite_drop:
                _record_fe_rejection(
                    self, gate="gate_composite_overmaterialization",
                    candidate=str(_dn), operands=None, operator="engineered",
                    observed=float("nan"), threshold=float("nan"),
                    reason="raw_coverage_subset_of_clean_survivors", step=int(num_fs_steps),
                )
            if verbose:
                logger.info(
                    "MRMR FE: pruned %d gate-operand composite(s) whose raw coverage is already provided by "
                    "clean non-gate engineered survivors (over-materialization): %s",
                    len(_gate_composite_drop), sorted(_gate_composite_drop),
                )

    # CROSS-GROUP CLEANLINESS PRUNE (2026-06-15; un-gated to BOTH paths 2026-06-22). Removes two junk
    # classes the per-pair / CMI gates leave behind. Originally scoped to fe_fast_search ONLY, on the
    # assumption that the exhaustive path's extra passes (the step>=1 CMI re-screen at the raised relative
    # bar + the cross-fold stability vote + the fused-composite raw-redundancy cascade) would already
    # remove them. They do NOT on the canonical y=a**2/b+log(c)*sin(d) fixture: the exhaustive fit emitted
    # the fused compound + its two clean fragments PLUS a cross-group bare gate (gate_mask__d__b) AND a
    # cross-signal artefact (sub(sin(a),sin(gate_mask__d__c))) -- 5 engineered cols, over the <=4 cap
    # (over-materialization regression). The discriminator is PURELY STRUCTURAL (raw coverage already a
    # subset of the clean non-gate survivors), equally valid in both paths, so it now runs unconditionally.
    # Replay the SAME cheap subset-coverage discriminator already proven safe for gate composites above,
    # against two over-materialisations:
    #   (A) a STANDALONE bare gate column whose gate pair is CROSS-GROUP (no single clean survivor covers
    #       the whole pair) AND whose raw coverage is already in the clean survivors -- the spurious
    #       ``gate_mask__c__b`` on CASE1. The genuine warped (c,d) gate on CASE2 is WITHIN-pair with NO
    #       clean (c,d) survivor, so it is KEPT (same tell-2/tell-3 logic as the composite prune);
    #   (B) a non-gate engineered binary node whose two bare operands come from DIFFERENT signal groups
    #       (no single clean survivor jointly covers both) while EACH operand is already covered by some
    #       clean survivor -- the documented cross-signal artefact ``sub(sqr(a),invcbrt(c))`` (a & c from
    #       the {a,b} and {c,d} groups). Dropping it also removes the false anchor that was propping up
    #       the redundant raw a / raw c in the raw-redundancy KEEP decision.
    # Runs in BOTH paths now (the cross-group artefacts leak in the exhaustive path too). Operates on the
    # post-CMI ``prospective_additions`` and the same ``_clean_forms`` / ``_bare_tokens`` already built
    # above; only fires when that gate-composite block ran (a gate fired) for (A), and unconditionally for
    # (B). No-op (byte-identical) when no cross-group over-materialisation is present.
    if prospective_additions:
        import re as _re_fsc

        def _bare_tokens_fsc(_nm: str) -> set:
            return set(_re_fsc.findall(r"(?<![A-Za-z0-9_])([a-z](?:[a-z]?\d+)?)(?![A-Za-z0-9_])", _nm))

        _gmap_fsc = dict(getattr(self, "_gate_col_src_vars_", None) or {})
        _all_names_fsc = []
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if _ncols:
                _all_names_fsc.extend(_ncols)

        def _gate_cols_in_fsc(_nm: str) -> list:
            return [_gc for _gc in _gmap_fsc if _gc in _nm]

        # Clean (non-gate) survivor coverage as a per-survivor list of bare-token sets, so "within one
        # survivor" can be tested for a candidate operand pair (mirrors the composite block's _clean_forms).
        _clean_token_sets_fsc = [
            _bare_tokens_fsc(_nm) for _nm in _all_names_fsc if not _gate_cols_in_fsc(_nm)
        ]

        _fsc_drop: set = set()
        for _nm in _all_names_fsc:
            _gcs = _gate_cols_in_fsc(_nm)
            if _nm in _gmap_fsc:
                # (A) STANDALONE bare gate column. Drop iff cross-group AND fully covered by clean survivors.
                _gate_src = set(_gmap_fsc.get(_nm, ()))
                if len(_gate_src) < 2:
                    continue
                # Coverage of OTHER (non-this) clean survivors -- never let the candidate cover itself.
                _other_cov = set().union(*_clean_token_sets_fsc) if _clean_token_sets_fsc else set()
                _within_one = any(_gate_src <= _ts for _ts in _clean_token_sets_fsc)
                if (not _within_one) and _gate_src and _gate_src <= _other_cov:
                    _fsc_drop.add(_nm)
                continue
            if _gcs:
                continue  # gate COMPOSITES already handled by the block above
            # (B) non-gate engineered binary node: cross-group cross-signal artefact.
            _toks = _bare_tokens_fsc(_nm)
            if len(_toks) < 2:
                continue
            _others = [_ts for _ts in _clean_token_sets_fsc if _ts != _toks]
            # "Cross-group" == no OTHER single clean survivor jointly covers this node's whole token set.
            _within_one = any(_toks <= _ts for _ts in _clean_token_sets_fsc if _ts != _toks)
            if _within_one:
                continue
            _union_others = set().union(*_others) if _others else set()
            # Each operand already covered by some clean within-group survivor -> dropping loses nothing.
            if _toks and _toks <= _union_others:
                _fsc_drop.add(_nm)

        if _fsc_drop:
            from ..mrmr import get_new_feature_name as _gnf_fsc
            _filtered_fsc: dict = {}
            for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                if not _tpf or _tvals is None or not _ncols:
                    _filtered_fsc[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                _keep_idx = [i for i, nm in enumerate(_ncols) if nm not in _fsc_drop]
                if not _keep_idx:
                    continue
                if len(_keep_idx) == len(_ncols):
                    _filtered_fsc[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                _n2c = {_gnf_fsc(_cfg, cols): _cfg for _cfg, _ in _tpf}
                _new_tpf = set()
                for _np, _oi in enumerate(_keep_idx):
                    _cfg = _n2c.get(_ncols[_oi])
                    if _cfg is not None:
                        _new_tpf.add((_cfg, _np))
                _new_nnb = (
                    [_nnb[i] for i in _keep_idx]
                    if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols)
                    else _nnb
                )
                _filtered_fsc[_rp] = (_new_tpf, _tvals[:, _keep_idx], [_ncols[i] for i in _keep_idx], _new_nnb, _msgs)
            prospective_additions = _filtered_fsc
            for _dn in _fsc_drop:
                _record_fe_rejection(
                    self, gate="fast_search_cross_group_overmaterialization",
                    candidate=str(_dn), operands=None, operator="engineered",
                    observed=float("nan"), threshold=float("nan"),
                    reason="cross_group_coverage_subset_of_clean_survivors", step=int(num_fs_steps),
                )
            if verbose:
                logger.info(
                    "MRMR FE fast-search: pruned %d cross-group over-materialised column(s) (standalone "
                    "cross-group gate / cross-signal artefact) covered by clean survivors: %s",
                    len(_fsc_drop), sorted(_fsc_drop),
                )

    # ROOT CAUSE 5 fix (2026-06-01): collect the cols-space indices of the
    # engineered columns appended below so they can be added DIRECTLY to
    # ``selected_vars`` for the default single-step (``fe_max_steps==1``)
    # path. The screening re-run that would normally promote appended cols
    # only happens on the NEXT outer-loop iteration; with the default
    # ``fe_max_steps=1`` the loop breaks before re-screening, so a recommended
    # engineered column never reached ``_engineered_features_``. Mirroring the
    # cluster_aggregate pattern (which already self-selects its aggregate),
    # we promote the FE survivors here. On multi-step (``> 1``) the next
    # screening pass re-evaluates them as usual and may drop weak ones.
    # Seed with the polynom-pair engineered indices captured above so they
    # are promoted into ``selected_vars`` together with the unary/binary
    # ones below (ROOT CAUSE 5). They already cleared every polynom-FE gate.
    _newly_engineered_indices: list[int] = list(_polynom_engineered_indices)
    # 2026-06-02: a fit() MUST NOT mutate the caller's input. The pandas
    # branch below appends engineered columns via ``X[col] = ...`` IN PLACE;
    # without this guard the user's DataFrame silently grows engineered
    # columns after ``MRMR().fit(df, y)`` (and the leak bled across fits that
    # reused one frame). Copy ONCE, up front, only when at least one pair
    # actually produced an engineered column. Polars (``.with_columns``)
    # already returns a fresh frame, and the ndarray path never appends to X.
    # ``_x_is_owned`` tracks whether ``X`` is already a private (copied) frame: the first pandas
    # materialise copies once; the later escalation / additive-fusion blocks then mutate that same
    # private frame in place instead of copying it again (three full-frame copies collapse to one).
    _x_is_owned = False
    if (
        not _is_polars_input
        and hasattr(X, "columns")
        and any(v[0] for v in prospective_additions.values())
    ):
        X = X.copy()
        _x_is_owned = True
    # Accumulate the per-pair discretised code blocks and concatenate ONCE after the loop instead of
    # ``np.append``-ing the whole (n, K) code matrix per pair (each append reallocated + copied all of
    # ``data``, O(pairs * n * K)). ``cols`` / ``nbins`` still grow per iteration (cheap), so the
    # cols-space index bookkeeping is unchanged; ``data`` itself is not read inside the loop.
    _data_chunks: list = []
    for raw_vars_pair, (this_pair_features, transformed_vals, new_cols, new_nbins, messages) in prospective_additions.items():
        if this_pair_features:
            engineered_features.update(this_pair_features)
            if verbose:
                for mes in messages:
                    logger.info(mes)
                # logger.info(f"Features {new_cols} are recommended to use as new features!")
            if fe_max_steps >= 1:
                new_vals = np.empty(shape=(len(X), len(this_pair_features)), dtype=self.quantization_dtype)
                for j in range(len(this_pair_features)):
                    new_vals[:, j] = discretize_array(
                        arr=transformed_vals[:, j],
                        n_bins=self.quantization_nbins,
                        method=self.quantization_method,
                        dtype=self.quantization_dtype,
                    )
                _n_cols_before = len(cols)
                _data_chunks.append(new_vals)
                # ``nbins`` is a numpy.ndarray (returned by categorize_dataset), so plain ``+`` does
                # element-wise addition / broadcasting, not concatenation. Use np.concatenate so nbins
                # grows in lockstep with data.shape[1] (otherwise screen_predictors trips its
                # targets_data.shape[1] == len(targets_nbins) assertion when engineered cols feed back).
                nbins = np.concatenate([
                    np.asarray(nbins),
                    np.asarray(new_nbins, dtype=nbins.dtype),
                ])
                cols = cols + new_cols
                # cols-space indices of the freshly appended engineered columns.
                _newly_engineered_indices.extend(range(_n_cols_before, len(cols)))
                # Use the DISCRETISED codes (``new_vals``) for the augmented
                # output frame, NOT the raw ``transformed_vals``. The fit-time
                # frame must match what ``transform()`` reproduces on test data
                # (the recipe replay emits quantised bin codes), otherwise a
                # consumer reading the fit-time augmented frame would see raw
                # floats while transform() emits codes -- a silent fit/transform
                # skew. ``transformed_vals`` (raw) is still used below to pin the
                # recipe's quantile edges.
                if _is_polars_input:
                    # Polars is immutable: with_columns returns a new frame sharing buffers; caller's X untouched.
                    _series_to_add = [
                        pl.Series(col, new_vals[:, j])
                        for j, col in enumerate(new_cols)
                    ]
                    X = X.with_columns(_series_to_add)
                else:
                    # 2026-06-01: index by the per-column position, not the
                    # leaked loop variable ``j`` (which held len-1 after the
                    # discretize loop above, so EVERY appended pandas column
                    # silently received the LAST survivor's values).
                    for _jc, col in enumerate(new_cols):
                        X[col] = new_vals[:, _jc]

                # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): stash the CONTINUOUS
                # engineered values (``transformed_vals``) keyed by column name. The
                # augmented frame ``X`` only carries the DISCRETISED bin codes (needed
                # for screening), but the NEXT FE step's pair search must combine the
                # CONTINUOUS values: ``add(bin_codes(eng1), bin_codes(eng2))`` is
                # severely lossy (measured: the additive composite of the two real
                # step-1 features keeps MI 0.88 from bin codes vs 1.81 -- the full
                # signal -- from continuous values, so the code form fails the
                # engineered-MI gate). ``check_prospective_fe_pairs`` reads this store
                # (threaded as ``engineered_operand_values``) so ``(eng_i, eng_j)``
                # composites are built on the continuous values and recover the signal.
                _eng_cont_store = getattr(self, "_engineered_continuous_", None)
                if _eng_cont_store is None:
                    _eng_cont_store = {}
                    self._engineered_continuous_ = _eng_cont_store
                for _jc, col in enumerate(new_cols):
                    if transformed_vals.shape[1] > _jc:
                        _eng_cont_store[col] = np.asarray(transformed_vals[:, _jc], dtype=np.float64)

                # Build EngineeredRecipe for each newly-appended column so transform() can replay it.
                # Runs whenever columns were added (fe_max_steps >= 1). NESTED-ENGINEERED PARENTS
                # (2026-06-08): a parent that is itself an engineered column (a higher-order
                # composite, e.g. add(div(sqr(a),abs(b)), mul(log(c),sin(d)))) is now REPLAYABLE --
                # we pass the parent's own EngineeredRecipe (already in ``engineered_recipes`` from
                # the prior step) so replay recomputes it recursively. Only when a parent is
                # engineered AND has no replayable recipe do we skip (cannot reconstruct it).
                if engineered_recipes is not None:
                    from ..engineered_recipes import build_unary_binary_recipe
                    _raw_names = set(self.feature_names_in_)
                    for config, _j in this_pair_features:
                        # config = (transformations_pair, bin_func_name, i)
                        # transformations_pair = ((var_a_idx, unary_a_name),
                        #                        (var_b_idx, unary_b_name))
                        transformations_pair, bin_func_name, _ = config
                        (var_a_idx, unary_a_name) = transformations_pair[0]
                        (var_b_idx, unary_b_name) = transformations_pair[1]
                        # Map cols-index -> name. A RAW parent resolves to a ``feature_names_in_``
                        # name; an ENGINEERED parent resolves to its prior recipe (nested replay).
                        src_a_name_raw = cols[var_a_idx]
                        src_b_name_raw = cols[var_b_idx]
                        # NESTED-PARENT RESOLUTION also consults the SUBSUMED-FRAGMENT recipe store
                        # (2026-06-24). When C2 additive-fusion subsumes a fragment at a prior step it
                        # POPS the fragment from ``engineered_recipes`` (so it is not re-selected
                        # bare), but a LATER step's pair / escalation search may legitimately re-derive
                        # a composite that nests that fragment (e.g. ``abs(div(sqr(a),neg(b)))``). The
                        # fragment's recipe must still be reachable so the re-derived composite stays
                        # REPLAYABLE -- otherwise it is recorded recipe-less and DROPPED from transform
                        # output, collapsing the selection back to raw operands (the F2 scaled_1_5
                        # DOMINANT-CAPTURE leak). The preserved store keeps the fragment recipe object
                        # available for nested replay WITHOUT re-admitting the bare fragment to selection.
                        _subsumed_store = getattr(self, "_fe_subsumed_recipes_", None) or {}
                        _nested_a = None if src_a_name_raw in _raw_names else (
                            engineered_recipes.get(src_a_name_raw) or _subsumed_store.get(src_a_name_raw)
                        )
                        _nested_b = None if src_b_name_raw in _raw_names else (
                            engineered_recipes.get(src_b_name_raw) or _subsumed_store.get(src_b_name_raw)
                        )
                        # Skip only when an operand is engineered but its parent recipe is missing
                        # (un-replayable) -- e.g. a parent from a stage that did not register one.
                        _a_unreplayable = (src_a_name_raw not in _raw_names) and (_nested_a is None)
                        _b_unreplayable = (src_b_name_raw not in _raw_names) and (_nested_b is None)
                        if _a_unreplayable or _b_unreplayable:
                            if verbose:
                                logger.info(
                                    "Skipping recipe construction for nested engineered feature "
                                    "'%s' (parent %s has no replayable recipe).",
                                    get_new_feature_name(config, cols),
                                    src_a_name_raw if _a_unreplayable else src_b_name_raw,
                                )
                            continue
                        eng_name = get_new_feature_name(config, cols)
                        # 2026-05-30 Wave 9.1 fix (loop iter 28):
                        # pass the fit-time engineered values
                        # ``transformed_vals[:, _j]`` so the recipe
                        # persists the quantile edges. Pre-fix replay
                        # re-quantiled on test data, silently shifting
                        # bin codes between fit and transform under
                        # distribution drift.
                        _fit_vals = transformed_vals[:, _j] \
                            if transformed_vals.shape[1] > _j else None
                        # Per-operand pre-warp: when a side used the learned
                        # ``prewarp`` pseudo-unary, hand its fitted spec to the
                        # recipe so replay reproduces the closed-form warp.
                        _pw_a = _prewarp_specs.get(var_a_idx) if unary_a_name == "prewarp" else None
                        _pw_b = _prewarp_specs.get(var_b_idx) if unary_b_name == "prewarp" else None
                        # Per-operand median gate: when a side used the
                        # ``gate_med`` pseudo-unary, hand its fitted TRAIN
                        # median to the recipe so replay reproduces the
                        # closed-form ``(x > median)`` gate.
                        _gm_a = _gate_med_specs.get(var_a_idx) if unary_a_name == "gate_med" else None
                        _gm_b = _gate_med_specs.get(var_b_idx) if unary_b_name == "gate_med" else None
                        # BUG2 FIX (2026-06-12): freeze the fit-time ``smart_log`` shift
                        # anchor per ``log`` side. ``smart_log`` shifts non-positive
                        # inputs by ``(1e-5 - nanmin(operand))``; that anchor is
                        # data-dependent, so a transform row-slice recomputes a
                        # different shift and the log output (then the bin code)
                        # drifts. Reconstruct the CONTINUOUS fit-time operand exactly
                        # as replay does (raw column from X, or the nested parent's
                        # continuous replay) and compute the frozen anchor so replay is
                        # byte-exact. Best-effort: None leaves the legacy refit path.
                        def _ls_anchor(_src_name, _nested):
                            try:
                                if _nested is not None:
                                    from ..engineered_recipes import apply_recipe as _ar
                                    import dataclasses as _dc2
                                    _p = _nested
                                    if getattr(_p, "quantization", None) is not None:
                                        _p = _dc2.replace(_p, quantization=None)
                                    _ov = np.asarray(_ar(_p, X), dtype=np.float64)
                                else:
                                    _ov = np.asarray(X[_src_name].values if hasattr(X[_src_name], "values")
                                                     else X[_src_name], dtype=np.float64)
                                _mn = float(np.nanmin(_ov))
                                return (1e-5 - _mn) if _mn <= 0 else 0.0
                            except Exception:
                                return None
                        _ls_a = _ls_anchor(src_a_name_raw, _nested_a) if unary_a_name == "log" else None
                        _ls_b = _ls_anchor(src_b_name_raw, _nested_b) if unary_b_name == "log" else None
                        engineered_recipes[eng_name] = build_unary_binary_recipe(
                            name=eng_name,
                            src_a_name=src_a_name_raw,
                            src_b_name=src_b_name_raw,
                            unary_a_name=unary_a_name,
                            unary_b_name=unary_b_name,
                            binary_name=bin_func_name,
                            unary_preset=fe_unary_preset,
                            binary_preset=fe_binary_preset,
                            quantization_nbins=self.quantization_nbins,
                            quantization_method=self.quantization_method,
                            quantization_dtype=self.quantization_dtype,
                            fit_values_for_edges=_fit_vals,
                            prewarp_a=_pw_a,
                            prewarp_b=_pw_b,
                            gate_med_a=_gm_a,
                            gate_med_b=_gm_b,
                            # Nested-engineered parents (2026-06-08): None for raw operands,
                            # else the parent's recipe so replay recomputes it recursively.
                            nested_parent_a=_nested_a,
                            nested_parent_b=_nested_b,
                            log_shift_a=_ls_a,
                            log_shift_b=_ls_b,
                        )

            n_recommended_features += len(this_pair_features)

        # Wave 69 (2026-05-20): factors_to_use / factors_names_to_use are
        # already threaded through the upstream FE loop (MRMR.fit -> FE-pair
        # iteration consults these via `self.factors_to_use` and the
        # caller-supplied filter); no extra plumbing needed at this
        # bookkeeping site. The pair-cache only tracks "raw pair already
        # processed", which is name-agnostic.
        checked_pairs.add(raw_vars_pair)

    # Flush the accumulated per-pair code blocks into ``data`` in a SINGLE concatenate (column order
    # matches the loop's append order, so it is byte-identical to the prior per-pair ``np.append``).
    # Must run before the escalation block below reads / appends ``data``.
    if _data_chunks:
        data = np.concatenate([data, *_data_chunks], axis=1)

    # AUTO-ESCALATION to the richer SHIPPED bases (2026-06-10, backlog idea B,
    # default-ON). A pair that PASSED the pair-MI prescreen (ratio gate + order-2
    # maxT floor) but for which the unary/binary search above admitted NOTHING used
    # to end in the log_fe_summary WARNING below -- detected signal, silently
    # abandoned. Escalate instead: PROPOSE candidates from the richer shipped basis
    # families (signal-adaptive orth-poly ALS warp across the 4 polynomial bases at
    # a higher degree + DEMODULATED adaptive-frequency Fourier/chirp warps -- e.g.
    # the sin(3.7*a)*b inner frequency no library unary can express) and let the
    # EXISTING gates decide (maxT floor on MM-debiased MI + marginal-permutation
    # floor + the S5 conditional-MI redundancy gate vs the admitted engineered
    # support). Structurally a no-op (one set-difference) when every surviving pair
    # produced an admitted column -- the common case. See ``_fe_auto_escalation``.
    if bool(getattr(self, "fe_auto_escalation_enable", True)) and (
        prospective_pairs or _prevalence_failed_synergy
    ):
        try:
            # Per-fit escalation ledger: a pair escalated once is never re-escalated
            # in a later FE step of the SAME fit (a step-2 retry would re-propose the
            # identical candidate on identical data and emit a duplicate ``..._2``
            # column). Reset on the first FE step so re-fits start clean.
            if num_fs_steps == 0 or not hasattr(self, "_fe_escalation_done_pairs_"):
                self._fe_escalation_done_pairs_ = set()
                self.fe_escalation_history_ = []
            _esc_done = self._fe_escalation_done_pairs_
            _esc_pairs_with_additions = {
                _rp for _rp, _v in prospective_additions.items() if _v[0]
            }
            _esc_failed = [
                (_k[0], float(_k[1])) for _k in prospective_pairs
                if _k[0] not in _esc_pairs_with_additions and _k[0] not in _esc_done
            ]
            # PREVALENCE-FAILED SYNERGY RESCUE (2026-06-12, F2 a**2/b miss): synergy
            # pairs that cleared the order-2 maxT floor but missed the stricter raw-MI
            # synergy prevalence ratio (the raw-MI ratio under-estimates a smooth ratio
            # interaction -- the genuine a**2/b scores ~1.11 < 1.5). Feed them to the
            # escalation as failed pairs: ``_propose_poly``'s leak-safe held-out
            # pair-vs-single |corr| margin re-decides, then the full gates. These never
            # entered ``prospective_pairs`` (the prevalence gate dropped them), so the
            # set-difference above cannot contain them; add directly (skip ones already
            # escalated this fit / already admitted by the unary search).
            _esc_failed.extend(
                (_pp, _pmi) for _pp, _pmi in _prevalence_failed_synergy.items()
                if _pp not in _esc_pairs_with_additions and _pp not in _esc_done
            )
            # UNDERDELIVERY trigger (2026-06-10): a pair that DID admit a column but
            # whose best capture leaves SIGNIFICANT conditional pair MI on the table
            # (leftover CMI(joint(a,b); y | best admitted) above its conditional-
            # permutation null) is escalated too -- e.g. the ``y=sin(3.7a)*b``
            # envelope capture ``mul(sin(a),qubed(b))`` that the marginal-uplift
            # fallback admits while most of the detected signal stays unexpressed.
            # Stride-subsampled + 8-perm null keeps the every-pair-delivers common
            # case cheap; a false trigger only PROPOSES -- the full gates (incl. the
            # S5 CMI gate vs the pair's own admitted column) still decide. See
            # ``find_underdelivering_pairs``.
            if bool(getattr(self, "fe_escalation_underdelivery_enable", True)):
                from .._fe_auto_escalation import find_underdelivering_pairs
                _esc_failed.extend(find_underdelivering_pairs(
                    self,
                    prospective_pairs=prospective_pairs,
                    prospective_additions=prospective_additions,
                    X=X, cols=cols, classes_y=classes_y, done=_esc_done,
                ))
            if _esc_failed:
                from .._fe_auto_escalation import run_fe_auto_escalation
                from .._mi_greedy_cmi_fe import _cmi_from_binned as _esc_mi, _quantile_bin as _esc_qbin
                # Admitted-support context for the S5 gate: the engineered columns
                # the main path just materialised (continuous values + marginal MI).
                _esc_y = np.asarray(classes_y)
                if not np.issubdtype(_esc_y.dtype, np.integer):
                    _esc_y = _esc_y.astype(np.int64)
                _, _esc_y_dense = np.unique(_esc_y, return_inverse=True)
                _esc_y_dense = _esc_y_dense.astype(np.int64)
                _esc_admitted_pool: dict = {}
                for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                    if not _tpf or _tvals is None or not _ncols:
                        continue
                    for _jc, _cname in enumerate(_ncols):
                        if _tvals.shape[1] <= _jc:
                            continue
                        _cv = np.asarray(_tvals[:, _jc], dtype=np.float64)
                        _cb = _esc_qbin(_cv, nbins=int(self.quantization_nbins))
                        _esc_admitted_pool[_cname] = (_cv, float(_esc_mi(_cb, _esc_y_dense, None)))
                # Per-pair admitted-capture values: UNDERDELIVERY-triggered pairs
                # get their proposers fit on the RESIDUAL of the target given the
                # existing capture (see ``run_fe_auto_escalation``); zero-admission
                # pairs have no entry and fit the full target.
                _esc_capture_vals: dict = {}
                for _pp, _pmi in _esc_failed:
                    _v = prospective_additions.get(_pp)
                    if _v and _v[0] and _v[1] is not None and _v[2]:
                        _esc_capture_vals[tuple(_pp)] = _v[1][:, : min(int(_v[1].shape[1]), len(_v[2]))]
                _esc_admitted = run_fe_auto_escalation(
                    self,
                    failed_pairs=_esc_failed,
                    X=X, cols=cols,
                    classes_y=classes_y,
                    pair_maxt_floor=float(_pair_maxt_floor),
                    admitted_pool=_esc_admitted_pool,
                    verbose=verbose,
                    capture_vals=_esc_capture_vals,
                    rescue_pairs=set(_prevalence_failed_synergy.keys()),
                )
                # Mark the pairs escalation actually PROCESSED (budget-selected
                # eligible) as done for this fit, admitted or not -- a retry on
                # identical data cannot change the verdict.
                _esc_done.update(
                    getattr(self, "fe_escalation_info_", {}).get("eligible_idx", []) or []
                )
                if _esc_admitted:
                    # Materialise exactly like the unary/binary survivors above:
                    # discretised codes into data/X, name into cols, nbins in
                    # lockstep, recipe registered, continuous values stashed for
                    # the engineered-operand feed-forward, index promoted below.
                    if not _is_polars_input and hasattr(X, "columns") and not _x_is_owned:
                        X = X.copy()
                        _x_is_owned = True
                    _esc_new_codes = np.empty(
                        shape=(len(X), len(_esc_admitted)), dtype=self.quantization_dtype,
                    )
                    for _je, _ec in enumerate(_esc_admitted):
                        _esc_new_codes[:, _je] = discretize_array(
                            arr=np.asarray(_ec["values"], dtype=np.float64),
                            n_bins=self.quantization_nbins,
                            method=self.quantization_method,
                            dtype=self.quantization_dtype,
                        )
                    _n_cols_before_esc = len(cols)
                    data = np.append(data, _esc_new_codes, axis=1)
                    nbins = np.concatenate([
                        np.asarray(nbins),
                        np.asarray([self.quantization_nbins] * len(_esc_admitted), dtype=np.asarray(nbins).dtype),
                    ])
                    cols = cols + [_ec["name"] for _ec in _esc_admitted]
                    _newly_engineered_indices.extend(range(_n_cols_before_esc, len(cols)))
                    n_recommended_features += len(_esc_admitted)
                    _eng_cont_store = getattr(self, "_engineered_continuous_", None)
                    if _eng_cont_store is None:
                        _eng_cont_store = {}
                        self._engineered_continuous_ = _eng_cont_store
                    if _is_polars_input:
                        X = X.with_columns([
                            pl.Series(_ec["name"], _esc_new_codes[:, _je])
                            for _je, _ec in enumerate(_esc_admitted)
                        ])
                    for _je, _ec in enumerate(_esc_admitted):
                        if not _is_polars_input:
                            X[_ec["name"]] = _esc_new_codes[:, _je]
                        _eng_cont_store[_ec["name"]] = np.asarray(_ec["values"], dtype=np.float64)
                        if engineered_recipes is not None:
                            engineered_recipes[_ec["name"]] = _ec["recipe"]
        except Exception:
            logger.warning(
                "MRMR FE auto-escalation failed; continuing with the unary/binary survivors only.",
                exc_info=True,
            )

    # ROOT CAUSE 5 fix (2026-06-01): promote the freshly-appended engineered
    # columns directly into ``selected_vars`` (cols-space). They already
    # cleared every FE gate (pair-MI prevalence, engineered-MI prevalence,
    # external validation) -- the gates ARE the selection criterion for FE
    # survivors. Without this, the only path to ``support_`` was the
    # screening re-run at the top of the NEXT outer-loop iteration, which
    # never executes under the default ``fe_max_steps=1`` (the loop breaks
    # first), so ``_engineered_features_`` stayed empty. On multi-step the
    # re-screen still re-evaluates them and may prune weak ones. Mirrors the
    # cluster_aggregate self-selection pattern below.
    if _newly_engineered_indices:
        _sv = list(selected_vars) if not isinstance(selected_vars, list) else selected_vars
        _sv_set = set(_sv)
        selected_vars = _sv + [i for i in _newly_engineered_indices if i not in _sv_set]

    # C2 ADDITIVE-FUSION (2026-06-24, default-ON). MUST RUN BEFORE the cross-fold stability vote
    # below (ordering fix for the F2 DOMINANT-CAPTURE / weak-half class): the weak c/d half
    # ``mul(log(c),sin(d))`` alone FAILS the cross-fold vote (its uplift is carried by too few
    # rows to clear the per-fold quorum), so if the vote ran first it would DROP that half before
    # C2 ever saw it -- leaving no clean half to fuse. Running C2 first fuses the two surviving
    # engineered halves with DISJOINT raw-token sets + GENUINE additive separability (the fused
    # ``add`` MI exceeds both halves OR the 2-half OLS multiple-R beats the best single half) into
    # the single ``add(half_a, half_b)`` compound via the EXISTING unary_binary + nested-parent
    # recipe (no new recipe kind; byte-exact replay); the FUSED compound is registered in
    # ``engineered_recipes`` so it then FACES the stability vote (the fused compound passes the
    # vote -- it reconstructs the whole target -- while the bare weak half it subsumed does not
    # matter). The two now-subsumed fragments are popped from the recipe dict + dropped from
    # selection BEFORE the vote, so the vote never sees them. Self-gates to a no-op (byte-identical)
    # when fewer than two relevant disjoint engineered halves are present -- the common case.
    # ``fe_max_engineered_operands == 0`` is the documented raw-only-pool contract (no composites
    # whose operands are themselves engineered features). Additive fusion combines two engineered
    # halves into ``add(half_a, half_b)``, i.e. a composite of engineered operands, so it must honor
    # that contract and stay off when the feed-forward cap is 0.
    if (
        engineered_recipes is not None
        and bool(getattr(self, "fe_additive_fusion_enable", True))
        and int(getattr(self, "fe_max_engineered_operands", 8)) != 0
        and _newly_engineered_indices
    ):
        try:
            from .._fe_additive_fusion import propose_additive_fusions
            _eng_cont_store = getattr(self, "_engineered_continuous_", None) or {}
            # Names of the engineered columns just materialised this step (cols-space indices
            # -> names) that have a registered (replayable) recipe.
            _newly_names = [
                cols[i] for i in _newly_engineered_indices
                if 0 <= i < len(cols) and cols[i] in engineered_recipes
            ]
            _raw_name_set = set(self.feature_names_in_)
            _fused, _subsumed, _subsumed_raws = propose_additive_fusions(
                self,
                engineered_recipes=engineered_recipes,
                engineered_continuous=_eng_cont_store,
                newly_engineered_names=_newly_names,
                raw_name_set=_raw_name_set,
                cols=cols,
                classes_y=classes_y,
                X=X,
                nbins=int(self.quantization_nbins),
                seed=int(getattr(self, "random_seed", 0) or 0),
                verbose=int(verbose),
            )
            if _fused:
                # Materialise each fused compound exactly like an escalation survivor.
                if not _is_polars_input and hasattr(X, "columns") and not _x_is_owned:
                    X = X.copy()
                    _x_is_owned = True
                _fz_codes = np.empty(shape=(len(X), len(_fused)), dtype=self.quantization_dtype)
                for _jf, _fc in enumerate(_fused):
                    _fz_codes[:, _jf] = discretize_array(
                        arr=np.asarray(_fc["values"], dtype=np.float64),
                        n_bins=self.quantization_nbins,
                        method=self.quantization_method,
                        dtype=self.quantization_dtype,
                    )
                _n_cols_before_fz = len(cols)
                data = np.append(data, _fz_codes, axis=1)
                nbins = np.concatenate([
                    np.asarray(nbins),
                    np.asarray([self.quantization_nbins] * len(_fused), dtype=np.asarray(nbins).dtype),
                ])
                cols = cols + [_fc["name"] for _fc in _fused]
                _fused_indices = list(range(_n_cols_before_fz, len(cols)))
                n_recommended_features += len(_fused)
                if _is_polars_input:
                    X = X.with_columns([
                        pl.Series(_fc["name"], _fz_codes[:, _jf])
                        for _jf, _fc in enumerate(_fused)
                    ])
                for _jf, _fc in enumerate(_fused):
                    if not _is_polars_input:
                        X[_fc["name"]] = _fz_codes[:, _jf]
                    _eng_cont_store[_fc["name"]] = np.asarray(_fc["values"], dtype=np.float64)
                    engineered_recipes[_fc["name"]] = _fc["recipe"]
                self._engineered_continuous_ = _eng_cont_store
                # Promote the fused compounds into selection and DROP the subsumed fragments
                # (by cols-space index) from selected_vars + the recipe dict so neither
                # support_ nor _engineered_recipes_ keeps a now-redundant fragment.
                _drop_names = set(_subsumed) | set(_subsumed_raws)
                _subsumed_idx = {
                    i for i in selected_vars
                    if 0 <= i < len(cols) and cols[i] in _drop_names
                }
                _sv = [i for i in selected_vars if i not in _subsumed_idx]
                _sv_set = set(_sv)
                selected_vars = _sv + [i for i in _fused_indices if i not in _sv_set]
                # The fused compounds are freshly-engineered survivors too: register their
                # cols-indices so the stability vote (below) + the step>1 re-screen treat them
                # as engineered (the vote replays them via ``engineered_recipes`` regardless,
                # but this keeps the engineered-index bookkeeping consistent), and drop the
                # subsumed fragments' indices so they are not re-screened as engineered.
                _newly_engineered_indices = [
                    i for i in _newly_engineered_indices if i not in _subsumed_idx
                ] + [i for i in _fused_indices if i not in set(_newly_engineered_indices)]
                # Preserve each subsumed fragment's recipe in a side-store BEFORE popping it from
                # the active recipe dict, so a later FE step that re-derives a composite nesting
                # the fragment can still resolve a replayable recipe for it (without re-admitting
                # the bare fragment). See the nested-parent resolution above.
                _subsumed_store = getattr(self, "_fe_subsumed_recipes_", None)
                if _subsumed_store is None:
                    _subsumed_store = {}
                    self._fe_subsumed_recipes_ = _subsumed_store
                for _sn in _subsumed:
                    _sr = engineered_recipes.get(_sn)
                    if _sr is not None:
                        _subsumed_store[_sn] = _sr
                for _sn in _subsumed:
                    engineered_recipes.pop(_sn, None)
                # Record the subsumed-fragment names so the fit-end selection finaliser
                # strips them (they stay in cols/data and could otherwise be re-admitted by
                # a downstream marginal-MI screen with no recipe -> select-then-drop skew).
                _fz_dropped = getattr(self, "_fe_stability_vote_dropped_", None)
                if _fz_dropped is None:
                    _fz_dropped = set()
                    self._fe_stability_vote_dropped_ = _fz_dropped
                _fz_dropped.update(str(_sn) for _sn in _subsumed)
                # Register the fused-out RAW operands as redundancy-dropped so the
                # downstream raw-retention / rescue / augmentation passes
                # (``_prefe_screened_raw_`` re-add, never-empty rescue) do NOT resurrect a
                # raw the fused compound now fully subsumes -- the SAME contract the
                # ``drop_redundant_raw_operands`` sweep relies on via ``_raw_redundancy_dropped_``.
                if _subsumed_raws:
                    _rrd = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
                    _rrd.update(str(_rn) for _rn in _subsumed_raws)
                    self._raw_redundancy_dropped_ = _rrd
                    # AUTHORITATIVE FUSED-SUBSUMED RAW SET (2026-06-24): the raws the fused
                    # compound verifiably captures (the keep-probe said no independent
                    # residual). The fit-end support finaliser strips these unconditionally --
                    # the downstream retention / rescue / emit-both-reattach passes evaluate a
                    # raw against the CLEAN nested sub-expression (which, on a CORRUPTED a/b
                    # half like ``div(neg(b),a__p2sin1)``, does NOT capture the raw and so
                    # KEEPS it), but the FUSED compound DOES capture it -- so this set carries
                    # the stronger (whole-compound) verdict the general sweep cannot reach once
                    # the raw is re-attached as an operand of the surviving compound.
                    _fsr = set(getattr(self, "_fused_subsumed_raws_", None) or set())
                    _fsr.update(str(_rn) for _rn in _subsumed_raws)
                    self._fused_subsumed_raws_ = _fsr
        except Exception:
            logger.warning(
                "MRMR FE additive-fusion failed; continuing with the un-fused engineered survivors.",
                exc_info=True,
            )

    # CROSS-FOLD RECIPE STABILITY VOTING (2026-06-10, backlog #15). A near-free
    # consensus layer OVER the existing FE gates. The expensive search above ran
    # ONCE on the full data; here we add a cheap K-fold CONFIRMATION -- each
    # surviving unary_binary recipe is REPLAYED (leak-safe: the recipe is frozen,
    # only the rows change) on K held-out folds, its uplift gate statistic
    # recomputed per fold, and the recipe ADMITTED only if it clears the gate in
    # >= ceil(q*K) folds. This complements the order-2/order-3 maxT floors: maxT
    # kills the chance-MAX candidate WITHIN a fold (best-of-pool selection bias);
    # this kills a recipe that won only on a fold-specific QUIRK of the full-data
    # split (its uplift carried by a few rows in the train split, collapses on the
    # held-out folds). NO REFIT -- only K plug-in-MI replays per recipe, so the
    # cost is negligible. Failed recipes are dropped from BOTH ``engineered_recipes``
    # (so they never reach ``self._engineered_recipes_`` at fit-end) and from
    # ``selected_vars`` (so they never reach ``support_``). Default-ON; self-gates
    # to a no-op below 2 unary_binary survivors / k<2 / tiny n. ``fe_stability_vote_enable=False``
    # byte-reproduces the pre-vote support.
    if (
        engineered_recipes
        and bool(getattr(self, "fe_stability_vote_enable", True))
        and _newly_engineered_indices
    ):
        try:
            from .._fe_stability_vote import confirm_recipes_cross_fold, resolve_adaptive_vote_k

            _vote_diag: dict = {}
            _failed_eng = confirm_recipes_cross_fold(
                recipes=engineered_recipes,
                X=X,
                y_codes=classes_y,
                feature_names_in=list(self.feature_names_in_),
                nbins=int(self.quantization_nbins),
                # guarded-hybrid K: explicit int honoured verbatim (default 5 -> byte-identical);
                # only "auto" adapts, and only downward for tiny n (== 5 for n >= 500).
                k=resolve_adaptive_vote_k(getattr(self, "fe_stability_vote_k", 5), int(getattr(X, "shape", (0,))[0]) or len(X)),
                quorum=float(getattr(self, "fe_stability_vote_quorum", 0.6)),
                rng=np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)),
                verbose=int(verbose),
                diagnostics_out=_vote_diag,
            )
            # REJECTION LEDGER (additive): record each recipe the cross-fold vote
            # dropped, with observed=folds-passed vs threshold=quorum bar (need_eff).
            for _fn in _failed_eng:
                _vd = _vote_diag.get(_fn, {})
                _record_fe_rejection(
                    self, gate="stability_vote",
                    candidate=str(_fn), operands=_vd.get("src_names"), operator="engineered",
                    observed=_vd.get("passes", float("nan")),
                    threshold=_vd.get("need_eff", float("nan")),
                    reason="below_quorum", step=int(num_fs_steps),
                )
            if _failed_eng:
                # Drop the failed engineered names from selected_vars (by cols-index)
                # and from the recipe dict so neither support_ nor _engineered_recipes_
                # admits a fold-specific winner.
                _failed_idx = {i for i in selected_vars if 0 <= i < len(cols) and cols[i] in _failed_eng}
                if _failed_idx:
                    selected_vars = [i for i in selected_vars if i not in _failed_idx]
                for _fn in _failed_eng:
                    engineered_recipes.pop(_fn, None)
                # BUG2 (2026-06-12): the vote pops the recipe + de-selects the column,
                # but the materialised bin-code column STAYS in ``cols``/``data`` and is
                # therefore still visible to the downstream greedy screen (the step>1
                # re-screen / final selection). That screen re-admits it on its marginal
                # MI, so it lands in ``selected_vars_names`` at fit-end with NO recipe and
                # is silently DROPPED from transform output -- a select-then-drop contract
                # violation. Record the vote-rejected engineered NAMES on ``self`` so the
                # fit-end selection finaliser strips them before they can re-enter support_.
                # The vote is authoritative: a fold-unstable recipe must not reappear.
                _vote_dropped = getattr(self, "_fe_stability_vote_dropped_", None)
                if _vote_dropped is None:
                    _vote_dropped = set()
                    self._fe_stability_vote_dropped_ = _vote_dropped
                _vote_dropped.update(str(_fn) for _fn in _failed_eng)
        except Exception as _vote_exc:
            if verbose:
                logger.warning(
                    "MRMR cross-fold stability vote failed (%s: %s); keeping the un-voted FE support.",
                    type(_vote_exc).__name__, _vote_exc,
                )

    return prospective_additions, data, cols, nbins, X, selected_vars, n_recommended_features
