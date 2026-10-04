"""Helpers carved out of ``_step_score`` to keep that module under its size budget."""
from __future__ import annotations

import logging
import os


from mlframe.feature_selection.filters._mrmr_fe_step._step_name_tokens import build_candidate_provenance

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
from types import SimpleNamespace as _SimpleNamespace


from ._step_score_parts2 import (  # noqa: F401  -- carved helpers
    _fe_tail_budget_spent,
    _materialise_and_fina_step3_gate_composite_drop,
    _materialise_and_fina_step3_cols_space_index,
    _materialise_and_fina_step1_try,
    _materialise_and_fina_step1_column_reset_first,
    _materialise_and_fina_step2_esc_failed,
    _materialise_and_fina_step4_contract_stay_off,
    _materialise_and_fina_step5_byte_reproduces_pre,
)


def _materialise_and_fina_step1_st_simplenamespace_long(self, _is_polars_input, X, cols):
    """Step 1 of materialise_and_finalise_fe_candidates: lines starting at ``st = _SimpleNamespace() # long-lived locals of this function (see the ``."""
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    if _is_polars_input:
        pass

    # The recipe builder below reads the per-operand prewarp / gate-med fitted-spec accumulators that
    # the parent fills (from ``check_prospective_fe_pairs``) and backs on ``self``; re-bind the SAME
    # dict objects here so recipe construction sees every spec fit this fit (cross-iteration persistence).
    st._prewarp_specs = getattr(self, "_prewarp_specs_accum_", None)
    if st._prewarp_specs is None:
        st._prewarp_specs = {}
        self._prewarp_specs_accum_ = st._prewarp_specs
    st._gate_med_specs = getattr(self, "_gate_med_specs_accum_", None)
    # One frozen smart_log anchor per (source, nested parent) for this call; sibling candidates share parents.
    st._ls_anchor_memo = {}
    if st._gate_med_specs is None:
        st._gate_med_specs = {}
        self._gate_med_specs_accum_ = st._gate_med_specs

    # DEVICE-BORN resident-gate flag shared by the CMI redundancy gate AND the auto-escalation admitted-pool
    # build below. Computed once at function scope so both paths see it regardless of which gates run - the
    # escalation block executes even when the conditional-MI gate is skipped (fe_acceptance != 'conditional_mi'
    # or no prospective_additions at gate time), so a block-local definition left it unbound there.
    st._gate_resident = False
    if os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from mlframe.feature_selection.filters._gpu_strict_fe import fe_gpu_strict_resident_enabled
            from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_gpu_enabled
            st._gate_resident = bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled(n=int(X.shape[0]), p=len(cols)))
        except Exception as e:
            logger.debug("fe_gpu_strict_resident_enabled/_cmi_gpu_enabled check failed, defaulting to non-resident: %s", e)
            st._gate_resident = False

    # CONDITIONAL-MI REDUNDANCY GATE (strategy S5). The PRINCIPLED,
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
    st._fe_acceptance = str(getattr(self, "fe_acceptance", "conditional_mi"))
    st._cmi_dropped = set()
    return st


def _materialise_and_fina_step3_single_column_reduces(self, st, prospective_additions, cols, engineered_recipes, num_fs_steps, verbose, _polynom_engineered_indices):
    """Step 3 of materialise_and_finalise_fe_candidates: lines starting at ``if st._cmi_dropped:``."""
    if st._cmi_dropped:
        from mlframe.feature_selection.filters.mrmr import get_new_feature_name as _get_new_feature_name
        _filtered_additions: dict = {}
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if not _tpf or _tvals is None or not _ncols:
                _filtered_additions[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                continue
            st._keep_idx = [i for i, nm in enumerate(_ncols) if nm not in st._cmi_dropped]
            if not st._keep_idx:
                continue  # whole pair redundant -> drop the entry
            if len(st._keep_idx) == len(_ncols):
                _filtered_additions[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                continue
            # Name -> config map (authoritative column<->config link).
            _name_to_cfg = {_get_new_feature_name(_cfg, cols): _cfg for _cfg, _ in _tpf}
            st._new_tpf = set()
            for _new_pos, _old_i in enumerate(st._keep_idx):
                _cfg = _name_to_cfg.get(_ncols[_old_i])
                if _cfg is not None:
                    st._new_tpf.add((_cfg, _new_pos))
            _new_tvals = _tvals[:, st._keep_idx]
            _new_ncols = [_ncols[i] for i in st._keep_idx]
            # a length-mismatched _nnb used to pass through
            # UNFILTERED (not narrowed to _keep_idx), silently reintroducing the nbins-vs-cols length
            # mismatch this filtering exists to prevent. Log so a genuine producer-shape bug is never
            # indistinguishable from the expected None case.
            if _nnb is not None and (not hasattr(_nnb, "__len__") or len(_nnb) != len(_ncols)):
                log_throttle(
                    logger, "step_score_nnb_length_mismatch_additions", logging.WARNING,
                    "mrmr: _nnb length mismatch (expected %d columns) while filtering to kept columns; "
                    "passing it through UNFILTERED -- a downstream nbins/cols length assertion may now fire.",
                    len(_ncols),
                )
            st._new_nnb = [_nnb[i] for i in st._keep_idx] if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols) else _nnb
            _filtered_additions[_rp] = (st._new_tpf, _new_tvals, _new_ncols, st._new_nnb, _msgs)
        prospective_additions = _filtered_additions

    # GATE-OPERAND COMPOSITE OVER-MATERIALIZATION PRUNE. The conditional_gate / row_argmax
    # pre-pass appends ENGINEERED gate columns (``gate_mask__b__d__t..``) that screening selects; the FE
    # pair search then pairs each gate column with a raw operand, emitting gate-operand COMPOSITES
    # (``mul(cbrt(c),log(gate_mask__b__d))``, ``div(neg(a),sqrt(gate_mask__b__d))`` ..). On CASE1
    # ``y=a**2/b+log(c)*sin(d)`` the gate is built across the two TRUE groups (b__d), so ~6 such composites
    # pile up ALONGSIDE the clean ``div(sqr(a),neg(b))`` / ``mul(log(c),sin(d))`` survivors that already cover
    # {a,b} and {c,d} - 9 engineered cols, over the test cap (<=4). They are RE-MIXES (a slightly different
    # threshold nonlinearity) the CMI gate does not drop. Discriminator: drop a gate-operand COMPOSITE iff
    # EVERY raw variable it touches - its bare-token operands PLUS the gate operand's own raw sources
    # (resolved via ``_gate_col_src_vars_``) - is ALREADY covered by the UNION of the CLEAN (non-gate)
    # engineered features that SURVIVED the CMI gate above. Built on the POST-CMI survivors (not the raw
    # candidate pool) so a transient clean (c,d) candidate that the CMI gate itself dropped cannot make a
    # genuine gate composite look redundant. On CASE2 ``y=0.2 a**2/b+log(c*2)sin(d/3)`` the gate is built over
    # the TRUE c__d pair and the only surviving (c,d) carrier is the gate composite - no clean survivor
    # covers {c,d}, so it is KEPT and the warped interaction stays captured. The BARE gate column is never
    # pruned here. Byte-identical when no gate fired (empty ``_gate_col_src_vars_``).
    st._gate_src_vars_map = dict(getattr(self, "_gate_col_src_vars_", None) or {})
    prospective_additions = _materialise_and_fina_step2_pruned_here_byte(self, st, prospective_additions, cols, engineered_recipes, num_fs_steps, verbose)

    # CROSS-GROUP CLEANLINESS PRUNE (un-gated to BOTH paths). Removes two junk
    # classes the per-pair / CMI gates leave behind. Originally scoped to fe_fast_search ONLY, on the
    # assumption that the exhaustive path's extra passes (the step>=1 CMI re-screen at the raised relative
    # bar + the cross-fold stability vote + the fused-composite raw-redundancy cascade) would already
    # remove them. They do NOT on the canonical y=a**2/b+log(c)*sin(d) fixture: the exhaustive fit emitted
    # the fused compound + its two clean fragments PLUS a cross-group bare gate (gate_mask__d__b) AND a
    # cross-signal artefact (sub(sin(a),sin(gate_mask__d__c))) - 5 engineered cols, over the <=4 cap
    # (over-materialization regression). The discriminator is PURELY STRUCTURAL (raw coverage already a
    # subset of the clean non-gate survivors), equally valid in both paths, so it now runs unconditionally.
    # Replay the SAME cheap subset-coverage discriminator already proven safe for gate composites above,
    # against two over-materialisations:
    #   (A) a STANDALONE bare gate column whose gate pair is CROSS-GROUP (no single clean survivor covers
    #       the whole pair) AND whose raw coverage is already in the clean survivors - the spurious
    #       ``gate_mask__c__b`` on CASE1. The genuine warped (c,d) gate on CASE2 is WITHIN-pair with NO
    #       clean (c,d) survivor, so it is KEPT (same tell-2/tell-3 logic as the composite prune);
    #   (B) a non-gate engineered binary node whose two bare operands come from DIFFERENT signal groups
    #       (no single clean survivor jointly covers both) while EACH operand is already covered by some
    #       clean survivor - the documented cross-signal artefact ``sub(sqr(a),invcbrt(c))`` (a & c from
    #       the {a,b} and {c,d} groups). Dropping it also removes the false anchor that was propping up
    #       the redundant raw a / raw c in the raw-redundancy KEEP decision.
    # Runs in BOTH paths now (the cross-group artefacts leak in the exhaustive path too). Operates on the
    # post-CMI ``prospective_additions`` and the same ``_clean_forms`` / provenance sets already built
    # above; only fires when that gate-composite block ran (a gate fired) for (A), and unconditionally for
    # (B). No-op (byte-identical) when no cross-group over-materialisation is present.
    prospective_additions = _materialise_and_fina_step1_no_op_byte(self, prospective_additions, cols, engineered_recipes, st, num_fs_steps, verbose)

    # Collect the cols-space indices of the
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
    # ones below. They already cleared every polynom-FE gate.
    st._newly_engineered_indices = list(_polynom_engineered_indices)
    # a fit() MUST NOT mutate the caller's input. The pandas
    # branch below appends engineered columns via ``X[col] = ...`` IN PLACE;
    # without this guard the user's DataFrame silently grows engineered
    # columns after ``MRMR().fit(df, y)`` (and the leak bled across fits that
    # reused one frame). Copy ONCE, up front, only when at least one pair
    # actually produced an engineered column. Polars (``.with_columns``)
    # already returns a fresh frame, and the ndarray path never appends to X.
    # ``_x_is_owned`` tracks whether ``X`` is already a private (copied) frame: the first pandas
    # materialise copies once; the later escalation / additive-fusion blocks then mutate that same
    # private frame in place instead of copying it again (three full-frame copies collapse to one).
    # SHALLOW: every write below assigns a NEW engineered column name, never an existing one, so the private frame only needs its own column
    # index. A deep copy duplicated the caller's whole frame, which on the 100GB frames this package is built for is the one thing it must not
    # do; sharing the existing columns' buffers and adding to the copy leaves the caller's frame untouched just the same.
    st._x_is_owned = False
    return prospective_additions


def _materialise_and_fina_step2_pruned_here_byte(self, st, prospective_additions, cols, engineered_recipes, num_fs_steps, verbose):
    """Step 2 of materialise_and_finalise_fe_candidates: lines starting at ``if st._gate_src_vars_map and prospective_additions:``."""
    if st._gate_src_vars_map and prospective_additions:
        _all_names = []
        for _tpf, _tvals, _ncols, _nnb, _msgs in prospective_additions.values():
            if _ncols:
                _all_names.extend(_ncols)
        # Each candidate's raw coverage comes from the operand indices it was built from, resolved through any engineered parent's own
        # src_names, rather than from its rendered name.
        _raw_src_of, _gates_of = build_candidate_provenance(prospective_additions, cols, engineered_recipes, st._gate_src_vars_map)
        # Clean coverage = raw vars of the SURVIVING non-gate engineered features only.
        _clean_cov: set = set()
        for _nm in _all_names:
            if not _gates_of.get(_nm, ()):
                _clean_cov |= _raw_src_of.get(_nm, frozenset())

        # Per clean (non-gate) survivor: (bare-var coverage, marginal MI). The marginals reuse the values
        # the CMI gate already binned for this exact pool, so no extra MI kernel is run.
        _name_marg: dict = {}
        _cmi_cands_local = getattr(st, "_cmi_cands", None)  # set only in the conditional_mi branch above
        _materialise_and_fina_step1_isinstance_cmi_cands(_cmi_cands_local, _name_marg)
        _clean_forms = [(_raw_src_of.get(_nm, frozenset()), _name_marg.get(_nm, 0.0)) for _nm in _all_names if not _gates_of.get(_nm, ())]

        # A gate-operand COMPOSITE is over-materialization (DROP) when its whole raw coverage is already
        # provided by the clean survivors AND it is NOT the genuine carrier of an otherwise-uncaptured
        # interaction. Three independent re-mix tells, any of which condemns it:
        #   (1) it embeds >=2 distinct gate columns - a full-target re-mix, never a single clean pair
        #       (CASE1 ``add(log(gate_mask__b__d),sub(sin(a),sin(gate_mask__d__c)))``);
        #   (2) its gate is built over a CROSS-group pair - no single clean survivor contains the whole
        #       gate pair, so it fuses two INDEPENDENT already-captured signals (CASE1 ``gate_mask__b__d``);
        #   (3) its gate is WITHIN one clean survivor's pair AND that clean survivor is at least as STRONG
        #       (marginal MI) - the clean elementary form is the better carrier, the gate is redundant
        #       (CASE1 ``gate_mask__d__c`` vs the strong ``mul(log(c),sin(d))``). When the clean same-pair
        #       form is WEAKER than the gate composite (CASE2 ``sub(exp(c),cbrt(d))`` does not even reach
        #       final support), the gate composite is the genuine (c,d) carrier and is KEPT.
        _gate_composite_drop: set = set()
        _materialise_and_fina_step2_final_support_gate(_all_names, _gates_of, st, _raw_src_of, _clean_cov, _name_marg, _clean_forms, _gate_composite_drop)

        prospective_additions = _materialise_and_fina_step3_gate_composite_drop(self, _gate_composite_drop, prospective_additions, st, cols, num_fs_steps, verbose)
    return prospective_additions


def _materialise_and_fina_step1_isinstance_cmi_cands(_cmi_cands_local, _name_marg):
    """Step 1 of _materialise_and_fina_step2_pruned_here_byte: lines starting at ``if isinstance(_cmi_cands_local, dict):``."""
    if isinstance(_cmi_cands_local, dict):
        for _nm0, _vm0 in _cmi_cands_local.items():
            try:
                _name_marg[_nm0] = float(_vm0[1])
            except Exception as e:  # noqa: PERF203 - per-iteration fault isolation is intentional, not a hoisting candidate
                # The candidate then counts as 0.0 marginal MI when gate composites are compared against clean survivors.
                log_throttle(
                    logger, "fe_step_marginal_mi_unreadable", logging.WARNING,
                    "fe step: marginal MI unreadable for candidate %r (%s: %s); it counts as 0.0 in gate-composite pruning", _nm0, type(e).__name__, e,
                )


def _materialise_and_fina_step2_final_support_gate(_all_names, _gates_of, st, _raw_src_of, _clean_cov, _name_marg, _clean_forms, _gate_composite_drop):
    """Step 2 of _materialise_and_fina_step2_pruned_here_byte: lines starting at ``for _nm in _all_names:``."""
    for _nm in _all_names:
        st._gcs = _gates_of.get(_nm, ())
        if not st._gcs:
            continue
        if _nm in st._gate_src_vars_map:
            continue  # never prune the BARE gate column itself (CASE2 fallback carrier)
        st._gate_src = set()
        for _gc in st._gcs:
            st._gate_src |= set(st._gate_src_vars_map.get(_gc, ()))
        _cov = set(_raw_src_of.get(_nm, frozenset())) | st._gate_src
        _cov -= set(st._gate_src_vars_map)  # drop any gate-col token mistakenly captured
        if not (_cov and _cov <= _clean_cov and st._gate_src and st._gate_src <= _clean_cov):
            continue
        _g_marg = _name_marg.get(_nm, 0.0)
        _multi_gate = len(set(st._gcs)) >= 2
        st._within_one = any(st._gate_src <= _cf_cov for _cf_cov, _ in _clean_forms)
        _stronger_clean_same_pair = any((st._gate_src <= _cf_cov) and (_cf_marg >= _g_marg) for _cf_cov, _cf_marg in _clean_forms)
        # (4) it drags an EXTRA raw operand beyond its own gate pair (``sub(sin(a),sin(gate_mask__d__c))``
        #     - the gate is over (c,d) but the node also pulls in raw ``a`` from the OTHER group) while
        #     the gate pair is already WITHIN a clean survivor: the node is then an ENTANGLED cross-group
        #     re-mix, never a clean carrier of any single pair (the clean ``mul(log(c),sin(d))`` carries
        #     (c,d) and ``div(sqr(a),neg(b))`` carries a). Only fires under the outer ``_cov <= _clean_cov``
        #     guard, so a genuine carrier of an otherwise-uncaptured group (CASE2) is never reached here.
        _extra_raw = set(_raw_src_of.get(_nm, frozenset())) - st._gate_src
        _entangled_extra = bool(_extra_raw) and st._within_one
        if _multi_gate or (not st._within_one) or _stronger_clean_same_pair or _entangled_extra:
            _gate_composite_drop.add(_nm)


def _materialise_and_fina_step1_no_op_byte(self, prospective_additions, cols, engineered_recipes, st, num_fs_steps, verbose):
    """Step 1 of materialise_and_finalise_fe_candidates: lines starting at ``if prospective_additions:``."""
    if prospective_additions:
        _gmap_fsc = dict(getattr(self, "_gate_col_src_vars_", None) or {})
        _raw_src_fsc, _gates_fsc = build_candidate_provenance(prospective_additions, cols, engineered_recipes, _gmap_fsc)
        _all_names_fsc = []
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if _ncols:
                _all_names_fsc.extend(_ncols)

        # Clean (non-gate) survivor coverage as a per-survivor list of bare-token sets, so "within one
        # survivor" can be tested for a candidate operand pair (mirrors the composite block's _clean_forms).
        _clean_token_sets_fsc = [_raw_src_fsc.get(_nm, frozenset()) for _nm in _all_names_fsc if not _gates_fsc.get(_nm, ())]

        _fsc_drop: set = set()
        for _nm in _all_names_fsc:
            st._gcs = _gates_fsc.get(_nm, ())
            if _nm in _gmap_fsc:
                # (A) STANDALONE bare gate column. Drop iff cross-group AND fully covered by clean survivors.
                st._gate_src = set(_gmap_fsc.get(_nm, ()))
                if len(st._gate_src) < 2:
                    continue
                # Coverage of OTHER (non-this) clean survivors - never let the candidate cover itself.
                _other_cov = set().union(*_clean_token_sets_fsc) if _clean_token_sets_fsc else set()
                st._within_one = any(st._gate_src <= _ts for _ts in _clean_token_sets_fsc)
                if (not st._within_one) and st._gate_src and st._gate_src <= _other_cov:
                    _fsc_drop.add(_nm)
                continue
            if st._gcs:
                continue  # gate COMPOSITES already handled by the block above
            # (B) non-gate engineered binary node: cross-group cross-signal artefact.
            _toks = set(_raw_src_fsc.get(_nm, frozenset()))
            if len(_toks) < 2:
                continue
            _others = [_ts for _ts in _clean_token_sets_fsc if _ts != _toks]
            # "Cross-group" == no OTHER single clean survivor jointly covers this node's whole token set.
            st._within_one = any(_toks <= _ts for _ts in _clean_token_sets_fsc if _ts != _toks)
            if st._within_one:
                continue
            _union_others = set().union(*_others) if _others else set()
            # Each operand already covered by some clean within-group survivor -> dropping loses nothing.
            if _toks and _toks <= _union_others:
                _fsc_drop.add(_nm)

        if _fsc_drop:
            from mlframe.feature_selection.filters.mrmr import get_new_feature_name as _gnf_fsc
            _filtered_fsc: dict = {}
            for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
                if not _tpf or _tvals is None or not _ncols:
                    _filtered_fsc[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                st._keep_idx = [i for i, nm in enumerate(_ncols) if nm not in _fsc_drop]
                if not st._keep_idx:
                    continue
                if len(st._keep_idx) == len(_ncols):
                    _filtered_fsc[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                    continue
                st._n2c = {_gnf_fsc(_cfg, cols): _cfg for _cfg, _ in _tpf}
                st._new_tpf = set()
                for _np, _oi in enumerate(st._keep_idx):
                    _cfg = st._n2c.get(_ncols[_oi])
                    if _cfg is not None:
                        st._new_tpf.add((_cfg, _np))
                # a length-mismatched _nnb used to pass through
                # UNFILTERED (not narrowed to _keep_idx), silently reintroducing the nbins-vs-cols length
                # mismatch this filtering exists to prevent. Log so a genuine producer-shape bug is never
                # indistinguishable from the expected None case.
                if _nnb is not None and (not hasattr(_nnb, "__len__") or len(_nnb) != len(_ncols)):
                    log_throttle(
                        logger, "step_score_nnb_length_mismatch_fsc", logging.WARNING,
                        "mrmr: _nnb length mismatch (expected %d columns) while filtering to kept columns; "
                        "passing it through UNFILTERED -- a downstream nbins/cols length assertion may now fire.",
                        len(_ncols),
                    )
                st._new_nnb = [_nnb[i] for i in st._keep_idx] if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols) else _nnb
                _filtered_fsc[_rp] = (st._new_tpf, _tvals[:, st._keep_idx], [_ncols[i] for i in st._keep_idx], st._new_nnb, _msgs)
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
    return prospective_additions


def _materialise_and_fina_step2_produced_admitted_column(self, prospective_pairs, _prevalence_failed_synergy, verbose, num_fs_steps, prospective_additions, X, cols, classes_y, st, _pair_maxt_floor, _is_polars_input, discretize_array, data, nbins, n_recommended_features, engineered_recipes):
    """Step 2 of materialise_and_finalise_fe_candidates: lines starting at ``if bool(getattr(self, "fe_auto_escalation_enable", True)) and (prospec``."""

    if bool(getattr(self, "fe_auto_escalation_enable", True)) and (prospective_pairs or _prevalence_failed_synergy) and not _fe_tail_budget_spent("auto-escalation", verbose):
        X, cols, data, n_recommended_features, nbins = _materialise_and_fina_step1_try(self, num_fs_steps, prospective_additions, prospective_pairs, _prevalence_failed_synergy, X, cols, classes_y, st, _pair_maxt_floor, verbose, _is_polars_input, discretize_array, data, nbins, n_recommended_features, engineered_recipes)
    return X, cols, data, n_recommended_features, nbins


def _materialise_and_fina_step3_cluster_aggregate_self(st, selected_vars):
    """Step 3 of materialise_and_finalise_fe_candidates: lines starting at ``if st._newly_engineered_indices:``."""
    if st._newly_engineered_indices:
        st._sv = list(selected_vars) if not isinstance(selected_vars, list) else selected_vars
        st._sv_set = set(st._sv)
        selected_vars = st._sv + [i for i in st._newly_engineered_indices if i not in st._sv_set]
    return selected_vars
