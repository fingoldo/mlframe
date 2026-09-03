"""Sibling of ``_friend_graph_and_redundancy/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
friend-graph-and-redundancy post-screen block, itself further split for the 1k-LOC module-size gate).

Holds passes: friend-graph, cluster-aggregate-removal, standalone-gate-prune, interactions-order-2-drop, prefe-raw-reconsider, adaptive-fourier-readd, missingness-indicator-readd. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``(selected_vars, cols, data, nbins)`` threading contract
(all four are BOTH incoming parameters AND part of the return value -- mirrors the parent's own).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _friend_graph_and_redundancy_passes_group1(
    self,
    *,
    X,
    classes_y,
    cols,
    data,
    nbins,
    target_indices,
    y,
    verbose,
    cached_MIs,
    engineered_recipes,
    _eng_continuous_snapshot,
    selected_vars,
    _effective_min_relevance_gain,
    _hinge_deferred_recipes,
    _hinge_deferred_values,
    _hybrid_orth_pre_recipes,
    _miss_ind_pre_recipes,
    _persisted_dcd_state,
    _y_np,
    fe_to_pandas,
    _fe_family_on,
):
    """Run the friend-graph, cluster-aggregate-removal, standalone-gate-prune, interactions-order-2-drop, prefe-raw-reconsider, adaptive-fourier-readd, missingness-indicator-readd pass(es) and return ``(selected_vars, cols, data, nbins)``.
    See the package docstring for the full section this carves out."""
    self.friend_graph_ = None
    # ``len(...)`` not truthiness: by this point ``selected_vars`` may be a numpy array (the empty-screen
    # FE fallback rebinds it), and ``and <array>`` raises "truth value ... ambiguous". Empty list AND empty
    # array both give len 0, so the guard reads "build the graph only when something was selected".
    # build_friend_graph defaults OFF (diagnostic-display only); friend_graph_prune REQUIRES the graph, so auto-build
    # it whenever pruning is on even if the diagnostic build was left off.
    if (getattr(self, "build_friend_graph", False) or getattr(self, "friend_graph_prune", False)) and len(selected_vars) > 0:
        try:
            from ...friend_graph import build_friend_graph as _build_fg, prune_by_friend_graph as _prune_fg

            _fg = _build_fg(
                selected_vars=selected_vars,
                factors_data=data,
                factors_nbins=nbins,
                target_indices=target_indices,
                feature_names=cols,
                mi_eps=self.friend_graph_mi_eps,
                edge_significance=self.friend_graph_edge_significance,
                garbage_min_degree=self.friend_graph_garbage_min_degree,
                garbage_unique_ratio=self.friend_graph_unique_ratio,
                unique_max_degree=self.friend_graph_unique_max_degree,
                max_nodes=self.friend_graph_max_nodes,
                seed=self.random_seed,
                gpu_backend=getattr(self, "friend_graph_gpu_backend", None),
            )
            if self.friend_graph_prune:
                # Protect cluster-aggregate columns from pruning: they are correlated with all their
                # members by construction, so the sink classifier could mis-flag them.
                _ca_protect = [v for v in selected_vars if getattr(engineered_recipes.get(cols[v]), "kind", None) == "cluster_aggregate"]
                _pruned, _reasons = _prune_fg(_fg, selected_vars, protect_indices=_ca_protect)
                if _reasons:
                    if verbose:
                        logger.info(
                            "MRMR friend-graph pruned %d suspected-sink feature(s): %s",
                            len(_fg.pruned), _fg.pruned,
                        )
                    selected_vars = _pruned
            self.friend_graph_ = _fg
        except Exception as _fg_exc:
            logger.warning(
                "MRMR friend-graph post-analysis failed (%s: %s); continuing without it.",
                type(_fg_exc).__name__, _fg_exc,
            )

    # Clustered-feature aggregation, replace mode: drop the aggregated cluster MEMBERS from
    # selected_vars (cols-space) so only the denoised aggregate survives into support_. Idempotent
    # set-difference (composes with the friend-graph prune above). The aggregate itself is an
    # engineered name and is routed into _engineered_recipes_ by the remap below.
    _ca_removed = getattr(self, "_cluster_aggregate_removals_", None)
    if _ca_removed:
        _removed_set = set(_ca_removed)
        selected_vars = [v for v in selected_vars if cols[v] not in _removed_set]

    # STANDALONE CROSS-GROUP GATE PRUNE (2026-06-15; un-gated to BOTH paths 2026-06-22). The conditional-gate
    # pre-pass appends gate_mask columns into the screening pool and the greedy can select a STANDALONE one;
    # the FE-step cross-group prune cannot reach it (it filters prospective_additions, not selected_vars).
    # Drop a selected standalone gate column iff its gate pair is CROSS-GROUP (no single clean ENGINEERED
    # survivor jointly covers its raw sources) AND those sources are already covered by the clean survivors'
    # union - the spurious gate_mask__c__b / gate_mask__b__d on y=a**2/b+log(c)*sin(d) (b,d from the two
    # different groups, both already in div(sqr(a),neg(b))+mul(log(c),sin(d))). A genuine warped (c,d) carrier
    # is WITHIN-pair (or embedded in a composite, not a standalone gate name) so it is KEPT. Originally scoped
    # to fe_fast_search on the assumption the exhaustive path's extra passes would remove these; they do NOT
    # (the canonical exhaustive fit selected TWO such standalone cross-group gates -> over the <=4 cap). The
    # discriminator is purely structural and never empties the support, so it now runs in BOTH paths.
    if len(selected_vars) and getattr(self, "_gate_col_src_vars_", None):
        try:
            import re as _re_sg
            _gmap_sg = dict(self._gate_col_src_vars_)
            _tok_sg = _re_sg.compile(r"(?<![A-Za-z0-9_])([a-z](?:[a-z]?\d+)?)(?![A-Za-z0-9_])")
            _sel_names_sg = [cols[v] for v in selected_vars]
            # Clean engineered survivors = selected composites that are NOT a bare gate column and NOT raw.
            _clean_tok_sets_sg = [set(_tok_sg.findall(nm)) for nm in _sel_names_sg if nm not in _gmap_sg and ("(" in nm) and ("gate_mask" not in nm)]
            _clean_union_sg = set().union(*_clean_tok_sets_sg) if _clean_tok_sets_sg else set()
            # "Genuine single-pair carrier" anchors for the within-one test are PURE-PAIR survivors only
            # (<= 2 distinct raw vars). A FULL-TARGET fused compound spans every var ({a,b,c,d}) and would
            # otherwise make EVERY gate look within-one, defeating the cross-group test - so the canonical
            # ``add(sqrt(div(sqr(a),neg(b))),sin(mul(log(c),sin(d))))`` is excluded as an anchor; the clean
            # pure pairs ``div(sqr(a),neg(b))`` / ``mul(log(c),sin(d))`` are the real carriers and a
            # cross-group gate over {b,d} (whose pair no single pure survivor covers) is correctly dropped.
            _pair_tok_sets_sg = [_ts for _ts in _clean_tok_sets_sg if len(_ts) <= 2]
            _drop_sg = set()
            for nm in _sel_names_sg:
                if nm not in _gmap_sg:
                    continue  # only standalone bare gate columns
                _src = set(str(s) for s in _gmap_sg.get(nm, ()))
                if len(_src) < 2:
                    continue
                _within_one = any(_src <= _ts for _ts in _pair_tok_sets_sg)
                if (not _within_one) and _src and _src <= _clean_union_sg:
                    _drop_sg.add(nm)
            if _drop_sg:
                selected_vars = [v for v in selected_vars if cols[v] not in _drop_sg]
                if getattr(self, "verbose", 0):
                    logger.info(
                        "MRMR FE fast-search: pruned %d standalone cross-group gate column(s) covered by "
                        "clean engineered survivors: %s", len(_drop_sg), sorted(_drop_sg),
                    )
        except Exception as _sg_exc:
            logger.warning("MRMR fast-search standalone-gate prune skipped (%s); continuing.", type(_sg_exc).__name__)

    # N-WAY SYNERGY SEEDING. The greedy screen assembles features one-at-a-time by
    # CONDITIONAL gain, which cannot climb a PURE-synergy gradient: on a 3-way XOR every operand has
    # ~0 marginal AND ~0 conditional gain until ALL members are present, so the genuine {x0,x1,x2}
    # interaction is never assembled (test_3way_screening, documented tracked-red whose specced fix is
    # exactly this - evaluate the n-way JOINT directly and surface it). When the user opted into n-way
    # interactions (interactions_max_order>=2), evaluate candidate raw COMBOS by their MM-corrected
    # JOINT MI vs the SUM of member marginals and SEED the members of combos showing strong synergy +
    # clearing an absolute joint-MI floor. The Miller-Madow correction keeps a noise combo's joint MI
    # ~0 (verified: on 3-way XOR among 5 noise vars ONLY {x0,x1,x2} fires), so noise is never seeded.
    # Off when interactions_max_order<2 (the default) -> byte-identical there. Bounded combo
    # enumeration (candidate + order caps) so wide-p fits stay tractable.
    # NOTE (2026-07-21, test_3way_screening tracked-red follow-up): candidate columns MUST be
    # re-quantile-binned here, NOT sourced from ``data`` (the fit's MDLP-binned matrix).
    # MDLP is supervised on each column's OWN marginal relevance to y, and a pure-synergy operand
    # has ~0 marginal MI BY CONSTRUCTION - MDLP collapses it to 1-2 bins (measured: x0 of a 3-way
    # XOR gets nbins=2), which then guts the JOINT-MI resolution this exact detector needs to see
    # the interaction. Fixed-width equi-frequency quantile bins depend only on rank order, not on
    # marginal relevance, so a zero-marginal operand keeps its full joint-MI resolution.
    _iac_max_order = getattr(self, "interactions_max_order", None)
    if int(_iac_max_order if _iac_max_order is not None else 1) >= 2 and len(selected_vars) >= 0:
        try:
            from ..._fe_synergy_screen import detect_synergy_combos
            from ..._mi_greedy_cmi_fe import _quantile_bin

            _raw_set_syn = set(self.feature_names_in_)
            _cand_syn = [i for i, _nm in enumerate(cols) if _nm in _raw_set_syn]
            if 2 <= len(_cand_syn) <= 60:
                _yc_syn = np.asarray(classes_y).astype(np.int64).ravel()
                _X_pd_syn = fe_to_pandas(X)
                _order_syn = int(_iac_max_order if _iac_max_order is not None else 3)
                # ADAPTIVE NBINS: detect_synergy_combos rejects any combo whose joint
                # cell count leaves fewer than min_rows_per_cell(=5.0 default) rows/cell - with the
                # fit's own quantization_nbins (10) an order-3 combo needs 10**3=1000 cells, i.e.
                # n>=5000, so at n=2000 EVERY order-3 combo was silently skipped regardless of binning
                # source (measured: quantile-rebin alone did not change the outcome). Size nbins so a
                # max-order combo clears the floor: nbins = floor((n / (5*min_rows_per_cell))^(1/order)),
                # clamped to [2, quantization_nbins] - never coarser than 2 bins, never finer than the
                # fit's own resolution.
                _mrpc_syn = 5.0
                _nbins_syn = int((float(_yc_syn.shape[0]) / (_mrpc_syn * 5.0)) ** (1.0 / max(1, _order_syn)))
                _nbins_syn = max(2, min(int(self.quantization_nbins), _nbins_syn))
                _code_cols_syn = {i: _quantile_bin(_X_pd_syn[cols[i]].to_numpy(dtype=np.float64), _nbins_syn).astype(np.int64) for i in _cand_syn}
                _iac_min_order = getattr(self, "interactions_min_order", None)
                _combos_syn = detect_synergy_combos(
                    _code_cols_syn, _yc_syn, _cand_syn,
                    max_order=_order_syn,
                    min_order=max(2, int(_iac_min_order if _iac_min_order is not None else 2)),
                    min_rows_per_cell=_mrpc_syn,
                )
                _sv_syn = set(selected_vars)
                _seed_syn = []
                for _combo_syn, _jmi_syn in _combos_syn:
                    for _ci_syn in _combo_syn:
                        if _ci_syn not in _sv_syn:
                            _seed_syn.append(_ci_syn)
                            _sv_syn.add(_ci_syn)
                if _seed_syn:
                    selected_vars = list(selected_vars) + _seed_syn
                    if verbose:
                        logger.info(
                            "MRMR n-way synergy seeding: added %d raw operand(s) of synergy combo(s) the "
                            "greedy could not assemble (joint MI >> sum of marginals): %s",
                            len(_seed_syn), [cols[i] for i in _seed_syn],
                        )
        except Exception as _syn_exc:
            logger.warning("MRMR n-way synergy seeding failed: %s; keeping support.", _syn_exc)

    # RAW-RETENTION: re-add SCREENING-confirmed genuine raw features
    # that the post-FE re-selection dropped, UNLESS a SINGLE-PARENT engineered child
    # substitutes them (the prefer-engineered raw->transform swap, which is a
    # legitimate, intended replacement). Screening permutation-validated these raw
    # columns as genuine; at small n an engineered feature can absorb a weak genuine
    # one as a redundant near-duplicate and the re-selection then drops the clean raw
    # signal entirely (measured: a genuine X5 at n=500, and both operands of a
    # pair-interaction target, dropped from support_). A raw feature only legitimately
    # leaves the support when a sole-parent transform of it survives.
    _prefe_raw = getattr(self, "_prefe_screened_raw_", None)
    if _prefe_raw and len(selected_vars):
        from ..._confirm_predictor import _extract_single_raw_parent
        _raw_names_set = set(self.feature_names_in_)
        _cur_names = set(np.asarray(cols)[np.asarray(selected_vars, dtype=np.intp)])
        # Raw parents already represented by a SOLE-parent engineered survivor:
        _substituted = set()
        for _v in selected_vars:
            if cols[_v] in _raw_names_set:
                continue
            _p = _extract_single_raw_parent([_v], cols, _raw_names_set)
            if _p is not None:
                _substituted.add(_p)
        # Cluster members folded into a denoised MULTI-parent aggregate (cluster_aggregate 'replace' mode -> _cluster_aggregate_removals_, or a DCD PC1/mean_z swap -> cluster_members_) are
        # ALREADY represented by that aggregate. _extract_single_raw_parent only recognises a SOLE-parent transform substitute, so without this exclusion raw-retention would resurrect the
        # very members 'replace' mode just removed and re-inject the redundancy the aggregation collapsed. Same exclusion the additional-RFECV rescue pool applies below.
        for _ca_member in getattr(self, "_cluster_aggregate_removals_", None) or []:
            _substituted.add(_ca_member)
        _cm_for_raw_retention = getattr(self, "cluster_members_", None)
        if isinstance(_cm_for_raw_retention, dict):
            for _anchor, _members in _cm_for_raw_retention.items():
                _substituted.add(_anchor)
                if isinstance(_members, (list, tuple, set)):
                    _substituted.update(_members)
        # MULTI-PARENT OPERAND SCOPE (guards a past regression): a raw feature that
        # is an OPERAND of a SURVIVING multi-parent engineered feature is NOT covered
        # by the sole-parent ``_substituted`` exclusion above, so the original blanket
        # re-add resurrected EVERY such operand - including ones whose entire signal
        # flowed into the engineered child (e.g. ``y = a**2/b + log(c)*sin(d)``: raw
        # ``a, c, d`` carry NO information about ``y`` beyond ``div(sqr(a),abs(b))`` and
        # ``mul(log(c),sin(d))``, yet were re-added with ``support_rank -1`` and no gain,
        # padding the support with three redundant columns). The post-FE re-selection
        # ALREADY judged them redundant via the Fleuret conditional-MI redundancy term.
        # We restore the OLD CORRECT behaviour by deferring to that verdict for such
        # operands at large n (where the conditional-MI estimate is reliable), while
        # keeping the protective unconditional re-add at small n (the regime the
        # protection was built and validated for) and for raws NOT consumed by any
        # surviving engineered feature (the originally-intended absorbed-by-unrelated case).
        from ..._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _RR_TOK_SPLIT
        # Map each raw-operand name -> list of surviving ENGINEERED survivor column indices that consume it.
        _eng_operands_of: dict = {}  # raw_name -> list[engineered survivor col idx]
        for _v in selected_vars:
            _vname = cols[_v]
            if _vname in _raw_names_set:
                continue
            for _tok in _RR_TOK_SPLIT.split(_vname):
                if not _tok:
                    continue
                _base = _tok if _tok in _raw_names_set else (_tok.split("__", 1)[0] if "__" in _tok else None)
                if _base in _raw_names_set:
                    _eng_operands_of.setdefault(_base, []).append(_v)
        # Sample-size scope: the small-n regime the protection was BUILT and
        # validated for (n=500 / 2000 / 3000 fixtures). At large n the post-FE re-selection's
        # conditional-MI redundancy term is statistically reliable - its drop of a redundant
        # operand IS the OLD CORRECT behaviour - so we do NOT override it there. ``_RR_PROTECT_MAX_N``
        # sits well above the largest validated fixture (3000) and far below the regression case (1e5).
        _RR_PROTECT_MAX_N = int(getattr(self, "fe_raw_retention_max_n", 20000) or 0)
        _n_rows_rr = int(data.shape[0])

        def _rr_raw_is_relevant_given_engineered(_raw_idx, _eng_cols):
            """Whether a raw operand of a surviving engineered child carries signal the
            engineered set does NOT capture, so raw-retention should OVERRIDE the re-selection's
            redundancy drop. Two regimes:

            * small n (``n <= _RR_PROTECT_MAX_N``): the conditional-MI redundancy estimate the
              re-selection used is unreliable at small n (the protection's whole reason to exist),
              so keep the protective re-add unconditionally - preserves the n<=3000 contracts.
            * large n: the re-selection's conditional-MI redundancy verdict is trustworthy, so we
              DEFER to it - an operand it dropped is genuinely redundant given the engineered
              child (``a`` in ``div(sqr(a),abs(b))`` for ``y=a**2/b`` carries no signal about ``y``
              beyond the ratio). We do NOT re-add it, restoring the pre-2026-06-03 selection. Note a
              bare ``CMI >= relevance_floor`` check does NOT work here: a coarsely-binned (~10-bin)
              engineered child leaves a small but above-floor residual conditional MI on its operand
              purely from the binning gap (measured: redundant ``a/c/d`` sit at CMI 0.002-0.023, the
              floor is ~0.0013), so the absolute floor cannot separate residual-binning-noise from a
              real independent term - only the re-selection's RELATIVE redundancy criterion can, and
              it already ran.

            CMI-estimator import/edge failures fall back to the protective re-add (never drop a
            screening-confirmed raw on an estimator error)."""
            if not _eng_cols:
                return True  # absorbed by an UNRELATED engineered feature -> original intent
            if _n_rows_rr <= _RR_PROTECT_MAX_N:
                return True  # small-n protective regime: keep the unconditional re-add
            # large n: defer to the re-selection's redundancy drop for engineered operands.
            return False

        # PERMUTATION-SIGNIFICANCE GATE on the re-add: a raw column the
        # screen flagged as ``_prefe_screened_raw_`` can be a small-n FALSE POSITIVE -
        # the coarse-binning plug-in MI is upward-biased, so a PURE-NOISE column (one
        # NOT in the target equation, e.g. CC4's ``e`` in ``y=log(a)*c+0.4*f``) can leave
        # a tiny residual debiased MI that the screen confirms and retention then
        # re-injects, padding the support with noise. Gate the re-add on the SAME
        # within-data permutation-significance test the empty-RAW rescue uses (computed on
        # the screen's own ``data`` / ``nbins`` so it matches ``cached_MIs``): a candidate
        # that sits WITHIN its own null (p >= alpha) is genuine-screen noise and is NOT
        # re-added. A genuinely weak-BUT-real raw (above its null) still passes. Best-
        # effort: a kernel failure falls through to the permissive re-add (never drop a
        # screening-confirmed raw on an estimator error).
        try:
            from ...permutation import mi_direct as _mi_direct_rr
        except Exception as exc:
            logger.debug("mrmr: mi_direct import/binding failed for the raw-redundancy significance probe; probe disabled: %r", exc, exc_info=True)
            _mi_direct_rr = None  # type: ignore[assignment]
        _rr_signif_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
        _rr_q_dtype = getattr(self, "quantization_dtype", np.int32)

        def _rr_raw_is_significant(_idx):
            """True iff the raw column at cols-index ``_idx`` sits ABOVE its permutation
            null against y (genuine signal). Pure-screen-noise sits within (p>=alpha)."""
            if _mi_direct_rr is None:
                return True
            try:
                _sig = _mi_direct_rr(
                    data, x=np.array([int(_idx)], dtype=np.int64), y=target_indices,  # type: ignore[arg-type]
                    factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                    return_null_mean=True, parallelism="none", dtype=_rr_q_dtype, prefer_gpu=False,
                )
                return float(_sig[3]) < _rr_signif_alpha
            except Exception as e:
                # The permissive policy is deliberate -- never drop a screening-confirmed raw because the
                # estimator failed -- but it silently reverts this gate to its pre-fix behaviour, and the gate
                # exists precisely because coarse-binning plug-in MI upward-biases pure-noise columns. One
                # throttled warning per fit, so a SYSTEMATIC estimator failure is visible rather than a debug
                # line per candidate.
                log_throttle(
                    logger,
                    "mrmr_readd_significance_probe_failed",
                    logging.WARNING,
                    "Marginal-MI significance re-add probe failed (%s: %s); re-adding screening-confirmed raws UNTESTED for the rest of this fit.",
                    type(e).__name__,
                    e,
                )
                return True  # significance unavailable -> permissive re-add

        _sv_set = set(selected_vars)
        # C2 ADDITIVE-FUSION EXCLUSION: a raw operand the FE step's
        # additive-fusion proposer judged FULLY subsumed by the fused ``add(...)`` compound
        # (recorded in ``_raw_redundancy_dropped_`` via the production keep-probe) must NOT
        # be resurrected here - the fused compound carries its additive term, so re-adding
        # it would re-inject a redundant single-group fragment beside the clean compound
        # (the FUSION-blocked goal's leftover raw). The fusion ran the same n-invariant
        # conditional-excess verdict ``drop_redundant_raw_operands`` uses, so this is the
        # authoritative drop. Byte-identical when no fusion fired (the set is empty).
        _fused_dropped_raw = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
        _readd = []
        _dropped_redundant = []
        _dropped_insignificant = []
        # name -> index map built once (O(F)) instead of a ``.index()`` rescan of ``cols`` per
        # ``_rn`` (O(F) each) - turns the O(K*F) loop below into O(K+F).
        _prefe_cols_idx = {nm: i for i, nm in enumerate(cols)}
        for _rn in _prefe_raw:
            if _rn in _cur_names or _rn in _substituted:
                continue
            if _rn in _fused_dropped_raw:
                _dropped_redundant.append(_rn)
                continue
            _idx = _prefe_cols_idx.get(_rn)
            if _idx is None:
                continue
            if _idx in _sv_set:
                continue
            _eng_cols = _eng_operands_of.get(_rn)
            if _eng_cols and not _rr_raw_is_relevant_given_engineered(_idx, _eng_cols):
                # Fully captured by a surviving engineered child -> respect the
                # re-selection's redundancy verdict (the OLD CORRECT behaviour).
                _dropped_redundant.append(_rn)
                continue
            if not _rr_raw_is_significant(_idx):
                # Screen false positive (pure noise within its own null) -> do not re-add.
                _dropped_insignificant.append(_rn)
                continue
            _readd.append(_idx)
            _sv_set.add(_idx)
        if _dropped_insignificant and verbose:
            logger.info(
                "MRMR raw-retention: withheld %d screening-flagged raw feature(s) that "
                "sit WITHIN their permutation null (p>=%.2f -- genuine-screen noise, not "
                "re-added): %s",
                len(_dropped_insignificant), _rr_signif_alpha, _dropped_insignificant,
            )
        if _readd:
            selected_vars = list(selected_vars) + _readd
            if verbose:
                logger.info(
                    "MRMR raw-retention: re-added %d screening-confirmed raw feature(s) "
                    "dropped by the post-FE re-selection (carry conditional signal beyond "
                    "their engineered children): %s",
                    len(_readd), [cols[i] for i in _readd],
                )
        if _dropped_redundant and verbose:
            logger.info(
                "MRMR raw-retention: kept %d raw feature(s) DROPPED -- fully captured by a "
                "surviving engineered child (conditional MI given the engineered set below "
                "the relevance floor): %s",
                len(_dropped_redundant), _dropped_redundant,
            )

    # ADAPTIVE-FOURIER PROTECTION: re-add held-out-validated
    # ADAPTIVE Fourier columns the MRMR screen dropped. The adaptive detector
    # already confirmed the column's dominant frequency on a held-out slice;
    # the screen drops it anyway because a SINGLE sin OR cos has low marginal MI
    # (the phase is split across the two legs, so neither alone clears the
    # relevance floor and the screen prefers a lower-MI fixed-freq twin). We
    # re-add the index of every adaptive name that is a column in ``cols`` but
    # absent from ``selected_vars``; its recipe is already in
    # ``engineered_recipes`` (merged from ``_hybrid_orth_pre_recipes`` above)
    # and survives into ``self._engineered_recipes_`` via the remap below, so
    # transform() replays the fit-time column byte-for-byte. Runs BEFORE the
    # ``selected_vars_names`` remap so the re-added index is routed correctly.
    _adaptive_fourier = getattr(self, "_adaptive_fourier_features_", None)
    if _adaptive_fourier and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _readd_adaptive = []
        for _an in _adaptive_fourier:
            _idx = _cols_index.get(_an)
            if _idx is None:
                continue
            if _idx not in _sv_set:
                _readd_adaptive.append(_idx)
                _sv_set.add(_idx)
        if _readd_adaptive:
            selected_vars = list(selected_vars) + _readd_adaptive
            if verbose:
                logger.info(
                    "MRMR adaptive-fourier protection: re-added %d held-out-" "validated adaptive Fourier feature(s) dropped by the screen: %s",
                    len(_readd_adaptive),
                    [cols[i] for i in _readd_adaptive],
                )

    # MISSINGNESS-INDICATOR PROTECTION: re-add the clean ``is_missing__{col}`` indicator the MRMR screen dropped IN FAVOUR OF its raw source. Under ``nan_strategy='separate_bin'``
    # the raw column's NaN bin already encodes the MNAR pattern, so the binned MI of the indicator and the raw source are near-identical (a true tie); the greedy screen keeps the raw column
    # and discards the indicator as redundant. But the raw column is mostly NaN - the downstream model cannot consume the missingness signal from it, only from the standalone numeric
    # indicator (the whole point of Layer 37). When the raw source IS selected, the indicator carries the SAME signal in a clean, model-ready form, so we re-add it. Gating on "the raw source
    # survived the screen" keeps a pure-noise indicator (MAR column the screen never selects) out of support. The count / pattern encoders have no single raw source and are screened normally.
    _miss_indicators = list(getattr(self, "missingness_indicator_features_", None) or [])
    if _miss_indicators and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _sel_names_now = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        _readd_miss = []
        for _mn in _miss_indicators:
            _idx = _cols_index.get(_mn)
            if _idx is None or _idx in _sv_set:
                continue
            _rec_mi = _miss_ind_pre_recipes.get(_mn)
            _src_mi = tuple(getattr(_rec_mi, "src_names", ()) or ())
            # Re-add only when the indicator's raw source survived the screen (i.e. the signal is real and the screen kept the redundant raw twin in its place).
            if _src_mi and _src_mi[0] in _sel_names_now:
                _readd_miss.append(_idx)
                _sv_set.add(_idx)
        if _readd_miss:
            selected_vars = list(selected_vars) + _readd_miss
            if verbose:
                logger.info(
                    "MRMR missingness-indicator protection: re-added %d clean "
                    "is_missing__ indicator(s) the screen dropped in favour of "
                    "the redundant raw NaN-bin source: %s",
                    len(_readd_miss), [cols[i] for i in _readd_miss],
                )

    # HINGE / CHANGE-POINT DEFERRED MATERIALISATION: the hinge stage
    # ran BEFORE the pair-FE loop (it needs the raw source columns) but DEFERRED
    # appending its legs so they could not perturb composite recovery. Now that the
    # FE loop has settled (composites recovered untouched), materialise the buffered
    # legs into the candidate matrix (``data`` bin-codes / ``cols`` / ``nbins``),
    # the augmented frame ``X``, and the recipe registry, then let the protection
    # block below re-add the deserving ones into ``selected_vars``. Skipped wholesale
    # when nothing was detected (legacy / no-kink path: the buffer is empty).

    return selected_vars, cols, data, nbins
