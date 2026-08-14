"""Split off ``mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core`` for the sub-split
that brings ``_fit_impl_core.py`` below the project's 1k-LOC module-size gate.

Holds ``_friend_graph_and_redundancy_passes``: the "Friend-graph post-analysis" section of
``MRMR._fit_impl`` -- friend-graph construction/pruning, adaptive-fourier/hybrid-orth/missingness
re-add passes, hinge/orth-basis protection, usability-aware raw-signal re-add, post-DCD cluster
pruning, pseudo-remix-aware post-selection redundancy drop, and the monotone-twin drop -- every
pass that runs on ``selected_vars`` in cols-space AFTER the main greedy screen and BEFORE the
cols-to-original-frame-index remap that follows this section in ``_fit_impl``.

Threads ``self`` plus every fit-body local this section reads as explicit keyword arguments
(mirrors the ``_finalise_fs_results`` / ``_assign_support`` carve-outs' own pattern), derived via
``pyutilz.dev.freevar_analysis`` rather than by eyeballing 1550 lines by hand. Unlike
``_assign_support`` (whose ``selected_vars`` is consumed entirely via ``self.*`` attributes it
sets), THIS section's ``selected_vars`` mutations feed directly into `_fit_impl``'s own next step
(the cols-to-original-frame-index remap) -- so ``selected_vars`` is both an incoming parameter AND
the return value.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _friend_graph_and_redundancy_passes(
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
    """Run every post-screen, pre-remap cols-space pass on ``selected_vars`` and return its final value.

    See the module docstring for the full section this carves out.
    """
    self.friend_graph_ = None
    # ``len(...)`` not truthiness: by this point ``selected_vars`` may be a numpy array (the empty-screen
    # FE fallback rebinds it), and ``and <array>`` raises "truth value ... ambiguous". Empty list AND empty
    # array both give len 0, so the guard reads "build the graph only when something was selected".
    # build_friend_graph defaults OFF (diagnostic-display only); friend_graph_prune REQUIRES the graph, so auto-build
    # it whenever pruning is on even if the diagnostic build was left off.
    if (getattr(self, "build_friend_graph", False) or getattr(self, "friend_graph_prune", False)) and len(selected_vars) > 0:
        try:
            from ..friend_graph import build_friend_graph as _build_fg, prune_by_friend_graph as _prune_fg

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
            from .._fe_synergy_screen import detect_synergy_combos
            from .._mi_greedy_cmi_fe import _quantile_bin

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
        from .._confirm_predictor import _extract_single_raw_parent
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
        from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _RR_TOK_SPLIT
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
            from ..permutation import mi_direct as _mi_direct_rr
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
                logger.debug("Marginal-MI significance re-add probe failed (%s: %s) -- permissive re-add", type(e).__name__, e)
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
    if _hinge_deferred_values and isinstance(X, pd.DataFrame):
        try:
            from ..mrmr import discretize_array
            _hinge_added_names = []
            _n_cols_before_hinge = len(cols)
            _new_hinge_codes = []
            _new_hinge_nbins = []
            for _hn, _vals in _hinge_deferred_values.items():
                if _hn in X.columns:
                    continue  # already present (defensive)
                _vals = np.asarray(_vals, dtype=np.float64)
                if _vals.shape[0] != data.shape[0]:
                    continue
                _codes = discretize_array(
                    arr=_vals,
                    n_bins=self.quantization_nbins,
                    method=self.quantization_method,
                    dtype=self.quantization_dtype,
                )
                _new_hinge_codes.append(np.asarray(_codes).reshape(-1, 1))
                _new_hinge_nbins.append(int(self.quantization_nbins))
                X[_hn] = _vals
                cols = [*cols, _hn]
                _hinge_added_names.append(_hn)
                _r = _hinge_deferred_recipes.get(_hn)
                if _r is not None:
                    _hybrid_orth_pre_recipes[_hn] = _r
                    engineered_recipes[_hn] = _r
            if _new_hinge_codes:
                data = np.append(
                    data, np.hstack(_new_hinge_codes).astype(data.dtype), axis=1,
                )
                nbins = np.concatenate(
                    [
                        np.asarray(nbins),
                        np.asarray(_new_hinge_nbins, dtype=nbins.dtype),
                    ]
                )
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_hinge_added_names)
                self._hinge_features_ = list(getattr(self, "_hinge_features_", None) or []) + list(_hinge_added_names)
                if verbose:
                    logger.info(
                        "MRMR.fit hinge change-point FE: materialised %d deferred " "leg(s) post-loop: %s",
                        len(_hinge_added_names),
                        _hinge_added_names[:8],
                    )
        except Exception as _h_mat_exc:
            logger.warning(
                "MRMR.fit hinge deferred materialisation raised %s: %s; " "continuing without hinge columns.",
                type(_h_mat_exc).__name__,
                _h_mat_exc,
            )

    # HINGE / CHANGE-POINT PROTECTION: re-add the held-out-tau-
    # validated hinge legs the MRMR screen dropped. A single relu leg
    # ``max(x-tau,0)`` is MONOTONE in x, hence MI-INVARIANT by the data-processing
    # inequality, and near-collinear with raw x - so the greedy MI screen drops
    # it as redundant with its raw source, EXACTLY as it drops a single adaptive
    # Fourier leg (low marginal MI) and the clean missingness indicator (tied MI
    # with its raw NaN-bin twin). But the hinge's value is NOT marginal MI: it is
    # the SECOND SLOPE it hands a downstream linear / shallow model
    # (``[1, x, relu(x-tau)]`` fits a two-slope kink ``[1, x]`` cannot). The
    # generating stage already (a) detected the breakpoint, (b) HELD-OUT-validated
    # it (2-segment beats 1-segment OOS R^2 on the %3 slice), and (c) admitted the
    # leg only on its held-out INCREMENTAL linear usability over raw x - so a
    # candidate ``_hinge_features_`` name is a confirmed univariate win. Without
    # this re-add, default-on hinge would GENERATE-then-DROP every leg (wasted
    # compute + the project's MI-vs-linear-usability rule violated, the same fix
    # the adaptive-Fourier protection block applies). TWO-PART SELF-LIMITING GATE
    # (the legs were deferred + just materialised above, so neutral data adds zero
    # cols): (1) the raw SOURCE must have survived the screen (a hinge on a never-
    # selected noise column is left out); (2) the leg must lift a HELD-OUT linear
    # fit over the ALREADY-SELECTED feature set PLUS the source + its degree-2 poly
    # ``[src, src^2]`` - so a leg subsumed by a surviving pair composite (b/d on
    # ``y=a**2/b+log(c)*sin(d)``) or a smooth curve a quadratic already fits
    # (``y=x^2``) adds ~0 and is rejected, while a genuine slope change with no
    # competing composite clears the floor. Runs BEFORE the ``selected_vars_names``
    # remap so the re-added index routes correctly; the recipe is in
    # ``engineered_recipes`` -> transform() replays it byte-for-byte.
    _hinge_feats = getattr(self, "_hinge_features_", None)
    # ``_heldout_incr_over_selected`` (defined below) is also the ORTH-BASIS UNIVARIATE PROTECTION block's
    # sole held-out-uplift probe (see that block further down) - it was originally written for the hinge
    # protection only and the orth-basis block was added later, reusing the closure via ``locals()`` instead
    # of its own copy. That coupling means the closure - and therefore BOTH protections - silently never ran
    # whenever ``fe_hinge_enable=False`` (the common "lightest config" preset: no hinge legs are generated,
    # so ``_hinge_feats`` is empty and this whole block was skipped), even though hybrid-orth univariate basis
    # columns are independently default-on and need the SAME protection. Gate on hybrid_orth_features_ too so
    # the setup runs whenever either protection has candidates to consider; the hinge-specific re-add loop
    # below still runs ONLY when ``_hinge_feats`` is non-empty (see its own ``if _hinge_feats:`` guard).
    if (_hinge_feats or getattr(self, "hybrid_orth_features_", None)) and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _sel_names_now = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        # SELECTED-SET INCREMENTAL-R^2 GATE (the principled self-limit). A hinge
        # leg is admitted on its held-out linear usability over raw x in the FE
        # stage, but on a MULTI-SIGNAL frame the SELECTED pair composite may
        # already capture the source's structure better than a univariate kink
        # (e.g. on y=a**2/b+log(c)*sin(d) the hinge fires on b / d, but
        # div(sqr(a),abs(b)) / mul(log(c),sin(d)) subsume them). So the protection
        # re-adds a leg ONLY when it lifts a held-out linear fit over the ALREADY-
        # SELECTED feature set - a leg whose value is subsumed by a surviving
        # composite adds ~0 and is dropped (no spurious cols on multi-signal data),
        # while a genuine slope-change leg with no competing composite clears the
        # floor (the hidden-champion win is kept). y is read only here at fit.
        _y_for_hinge_gate = None
        try:
            _yv = _y_np
            _yv = np.asarray(_yv, dtype=np.float64).reshape(-1)
            if _yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_yv)):
                _y_for_hinge_gate = _yv
        except Exception as exc:
            logger.debug("mrmr: y coercion for the hinge floor-drop rescue gate failed: %r", exc, exc_info=True)
            _y_for_hinge_gate = None
        # Continuous values of the currently-selected columns (engineered from the
        # snapshot, raw from X) -> the baseline design the leg must beat OOS.
        _sel_value_cols = []
        if _y_for_hinge_gate is not None and isinstance(X, pd.DataFrame):
            for _sn in _sel_names_now:
                _cv = _eng_continuous_snapshot.get(_sn)
                if _cv is None and _sn in X.columns:
                    _cv = X[_sn].to_numpy()
                if _cv is None:
                    continue
                try:
                    _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue  # a raw categorical/string selected column (e.g. under skip_categorical_encoding) is not a numeric R^2-baseline regressor - exclude it from the linear design
                if _cv.shape[0] == _y_for_hinge_gate.shape[0] and np.all(np.isfinite(_cv)):
                    _sel_value_cols.append(_cv)

        def _heldout_incr_over_selected(_leg_vals, _src_vals=None) -> float:
            """Held-out R^2 gain of adding ``_leg_vals`` to the selected design
            PLUS the source and its degree-2 poly, scored on the %3 stride slice.

            Including ``[src, src^2]`` in the baseline is the SMOOTH-CURVE guard:
            a parabola (y=x^2) is captured by ``src^2`` so a kink adds ~0 over it
            and is rejected (no spurious hinge on a smooth target - matches the
            biz_value complementarity contract); a GENUINE slope change still beats
            ``[src, src^2]`` OOS (a quadratic cannot fit a sharp two-slope kink) so
            the hidden-champion leg is kept."""
            if _y_for_hinge_gate is None:
                return 1.0  # gate disabled -> fall back to the source-survived rule
            leg = np.asarray(_leg_vals, dtype=np.float64).reshape(-1)
            n = leg.shape[0]
            if n != _y_for_hinge_gate.shape[0] or not np.all(np.isfinite(leg)):
                return 0.0
            # Seeded shuffle-then-stride, not a raw
            # positional (idx % 3) == 0 split - the latter is not an honest i.i.d. holdout on
            # time/group/label-sorted input (this module explicitly supports sorted input elsewhere
            # via ``groups`` / the ``temporal_agg`` FE family), which can bias the held-out R^2
            # this gate decides on.
            _hinge_gate_perm = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)).permutation(n)
            va = np.zeros(n, dtype=bool)
            va[_hinge_gate_perm[: n // 3]] = True
            tr = ~va
            if int(tr.sum()) < 32 or int(va.sum()) < 16:
                return 1.0
            yv = _y_for_hinge_gate[va]
            ss = float(np.sum((yv - yv.mean()) ** 2))
            if ss < 1e-24:
                return 0.0
            base = [np.ones(n), *_sel_value_cols]
            if _src_vals is not None:
                _sv = np.asarray(_src_vals, dtype=np.float64).reshape(-1)
                if _sv.shape[0] == n and np.all(np.isfinite(_sv)):
                    base = [*base, _sv, _sv * _sv]
            def _r2(design_cols):
                """Fit an OLS design on the train stride and return held-out R^2 on the %3 validation stride (``-inf`` on a singular/failed solve).

                Normal-equations solve (A.T@A / np.linalg.solve) on the well-conditioned small-k design
                (intercept + a handful of base/leg columns) instead of a full SVD lstsq -- same win already
                proven for this module's sibling OLS fit (see ``_deflate_sincos`` in
                ``_orth_extra_basis_fe.py``: normal equations beats lstsq here because k is tiny and the
                design isn't near-singular). Falls back to lstsq if A.T@A is singular."""
                A = np.column_stack(design_cols)
                A_tr = A[tr]
                y_tr = _y_for_hinge_gate[tr]
                try:
                    AtA = A_tr.T @ A_tr
                    coef = np.linalg.solve(AtA, A_tr.T @ y_tr)
                except np.linalg.LinAlgError:
                    try:
                        coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                    except Exception as e:
                        logger.debug("Hinge-gate OLS lstsq fallback failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                        return -np.inf
                except Exception as e:
                    logger.debug("Hinge-gate OLS lstsq failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                    return -np.inf
                pred = A[va] @ coef
                return 1.0 - float(np.sum((yv - pred) ** 2)) / ss
            r2_base = _r2(base)
            r2_full = _r2([*base, leg])
            if not (np.isfinite(r2_base) and np.isfinite(r2_full)):
                return 0.0
            return float(r2_full - r2_base)

        if _hinge_feats:
            _HINGE_PROTECT_MIN_INCR_R2 = 0.003
            _readd_hinge = []
            for _hn in _hinge_feats:
                _idx = _cols_index.get(_hn)
                if _idx is None or _idx in _sv_set:
                    continue
                _rec_h = _hybrid_orth_pre_recipes.get(_hn)
                _src_h = tuple(getattr(_rec_h, "src_names", ()) or ())
                # Self-limit #1: source must have survived the screen (real signal).
                if not (_src_h and _src_h[0] in _sel_names_now):
                    continue
                # Self-limit #2: the leg must lift a held-out linear fit OVER the
                # already-selected set + the source and its degree-2 poly (not
                # subsumed by a surviving composite, and a genuine kink not a smooth
                # curve a quadratic already fits).
                _leg_vals = _hinge_deferred_values.get(_hn)
                if _leg_vals is None and isinstance(X, pd.DataFrame) and _hn in X.columns:
                    _leg_vals = X[_hn].to_numpy()
                _src_vals_gate = None
                if isinstance(X, pd.DataFrame) and _src_h and _src_h[0] in X.columns:
                    _src_vals_gate = X[_src_h[0]].to_numpy()
                if _leg_vals is not None:
                    if _heldout_incr_over_selected(_leg_vals, _src_vals_gate) < _HINGE_PROTECT_MIN_INCR_R2:
                        continue
                _readd_hinge.append(_idx)
                _sv_set.add(_idx)
            if _readd_hinge:
                selected_vars = list(selected_vars) + _readd_hinge
                if verbose:
                    logger.info(
                        "MRMR hinge change-point protection: re-added %d held-out-"
                        "validated hinge leg(s) the MI screen dropped (MI-invariant; "
                        "value is downstream linear usability): %s",
                        len(_readd_hinge), [cols[i] for i in _readd_hinge],
                    )

    # ORTH-BASIS UNIVARIATE PROTECTION: re-add a single-source orthogonal-basis univariate column
    # (``a__T2`` ~ a**2, ``a__He4`` ~ a Hermite degree-4, ...) the MRMR screen dropped. Like a hinge leg, an
    # orth basis column is a DETERMINISTIC function of ONE raw source, so the greedy MI screen drops it as
    # redundant with that raw source under the data-processing inequality - EVEN WHEN raw ``a`` carries ~0
    # linear/monotone signal about an even target (``exp(-a**2)`` / ``a**2``) and the basis column carries the
    # whole recoverable nonlinearity (|corr| ~0.85). The basis value is downstream LINEAR usability, not
    # marginal MI (the same MI-vs-linear-usability rule the hinge / adaptive-Fourier protections enforce). The
    # generating univariate-basis stage already uplift-gated each column, so a candidate is a confirmed
    # univariate win. SELF-LIMITING GATE mirrors the hinge block: (1) the raw source survived the screen (a
    # basis on a never-selected noise column is left out); (2) the basis lifts a HELD-OUT linear fit over the
    # ALREADY-SELECTED feature set (which already contains the raw source as a linear term) - so a basis
    # subsumed by a surviving composite/raw adds ~0 and is rejected, while a genuine single-var nonlinearity
    # the screen DPI-dropped clears the floor. NO ``[src, src^2]`` smooth-curve term in the baseline (unlike
    # the hinge gate): for the basis the curve IS the win, so adding ``src^2`` would self-reject the very
    # quadratic basis we want. Reuses ``_heldout_incr_over_selected`` with ``_src_vals=None``.
    _orth_feats = getattr(self, "hybrid_orth_features_", None)
    if _orth_feats and len(selected_vars) and ("_heldout_incr_over_selected" in locals()):
        _cols_index_o = {c: i for i, c in enumerate(cols)}
        _sv_set_o = set(selected_vars)
        _sel_names_o = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        _ORTH_PROTECT_MIN_INCR_R2 = 0.01  # wider than hinge 0.003: a genuine single-var basis lifts held-out R^2 by >>0.01 (~0.7 for exp(-a**2)); keeps noise-fit basis out
        _readd_orth = []
        for _on in _orth_feats:
            _oidx = _cols_index_o.get(_on)
            if _oidx is None or _oidx in _sv_set_o:
                continue
            _rec_o = _hybrid_orth_pre_recipes.get(_on)
            # Hinge legs (``kind="hinge_basis"``) are routed through hybrid_orth_features_ too, but they have a
            # DEDICATED protection block above that gates them against a ``[src, src^2]`` baseline (the smooth-
            # curve guard: a parabola is fit by src^2 so a kink adds ~0 and is rejected). This orth-basis block
            # deliberately OMITS that guard (for a curved basis the curve IS the win), so re-handling a hinge leg
            # here would bypass the smooth-curve guard and re-add spurious legs on y=x^2 data. Skip them - the
            # Hinge block already made the correct keep/drop decision (guards a past regression).
            if getattr(_rec_o, "kind", None) == "hinge_basis":
                continue
            _src_o = tuple(getattr(_rec_o, "src_names", ()) or ())
            # Self-limit #1: single-source basis whose raw source survived the screen.
            if len(_src_o) != 1 or _src_o[0] not in _sel_names_o:
                continue
            _basis_vals = _eng_continuous_snapshot.get(_on)
            if _basis_vals is None and isinstance(X, pd.DataFrame) and _on in X.columns:
                _basis_vals = X[_on].to_numpy()
            if _basis_vals is None:
                continue
            # Self-limit #2: lifts a held-out linear fit over the already-selected design (raw source already
            # present there as a linear term) - not subsumed by a surviving composite/raw.
            if _heldout_incr_over_selected(_basis_vals, None) < _ORTH_PROTECT_MIN_INCR_R2:
                continue
            _readd_orth.append(_oidx)
            _sv_set_o.add(_oidx)
        if _readd_orth:
            selected_vars = list(selected_vars) + _readd_orth
            if verbose:
                logger.info(
                    "MRMR orth-basis univariate protection: re-added %d single-source basis column(s) the "
                    "MI screen DPI-dropped (value is downstream linear usability over the raw source): %s",
                    len(_readd_orth), [cols[i] for i in _readd_orth],
                )

    # RAW-FEATURE FLOOR-DROP PROTECTION (Fix-B). The Westfall-Young maxT relevance floor is computed
    # over the FULL candidate pool; when the all-FE-on config widens that pool to hundreds of (already FE-stage-
    # gated) engineered columns, the per-shuffle MAX corrected MI inflates and the acceptance bar rises ABOVE a
    # genuine raw feature's true marginal MI - so a real linear signal (e.g. x1 ~ y at binned-MI 0.057, ~30x
    # noise) is dropped from the screen entirely (confirmed root-cause of test_biz_value_mrmr_underselection).
    # LOWERING the floor would surface x1 but ALSO admit high-cardinality raw NOISE (a 50-level pure-noise
    # categorical whose finite-sample MI is inflated) - a regression. Instead, KEEP the floor (noise stays
    # rejected) and re-add a raw feature the screen dropped IFF it lifts a HELD-OUT linear fit over the already-
    # selected design - the SAME MI-vs-linear-usability protection the hinge / orth-basis blocks use. A genuine
    # linear/monotone raw signal clears the lift; a high-card noise categorical (no held-out linear usability)
    # does not, so it stays out. Conditioned on _y_for_hinge_gate (the held-out scorer); no-op when it is None.
    # Self-contained held-out scorer (the hinge block's _y_for_hinge_gate / _heldout_incr_over_selected only
    # exist when hinge legs were generated; this protection must run regardless). Baseline = intercept + the
    # continuous values of the ALREADY-SELECTED columns (engineered from the snapshot, raw from X), so a raw
    # feature SUBSUMED by a selected composite adds ~0 and is NOT re-added (no raw-redundancy regression).
    if isinstance(X, pd.DataFrame) and len(selected_vars):
        _rp_y = None
        try:
            _rp_yv = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).reshape(-1)
            if _rp_yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_rp_yv)):
                _rp_y = _rp_yv
        except Exception as exc:
            logger.debug("mrmr: y coercion for the raw-protection re-add probe failed; raw protection disabled: %r", exc, exc_info=True)
            _rp_y = None
        if _rp_y is not None:
            _RAW_PROTECT_MIN_INCR_R2 = 0.005  # genuine linear raw signal lifts held-out R^2 >> 0.005; noise ~0
            _rp_n = _rp_y.shape[0]
            # Seeded shuffle-then-stride (see the hinge-gate sibling comment above).
            _rp_perm = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)).permutation(_rp_n)
            _rp_va = np.zeros(_rp_n, dtype=bool)
            _rp_va[_rp_perm[: _rp_n // 3]] = True
            _rp_tr = ~_rp_va
            _rp_sel_names = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
            _rp_base = [np.ones(_rp_n)]
            for _sn in _rp_sel_names:
                _cv = _eng_continuous_snapshot.get(_sn)
                if _cv is None and _sn in X.columns:
                    _cv = X[_sn].to_numpy()
                if _cv is None:
                    continue
                try:
                    _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue  # raw categorical/string selected column - not a numeric R^2 regressor
                if _cv.shape[0] == _rp_n and np.all(np.isfinite(_cv)):
                    _rp_base.append(_cv)

            # Hoist the fold- and candidate-INVARIANT pieces out of the per-candidate R^2 (each call below
            # re-used the SAME held-out target, its centered SS, and the SAME base design rows): the val
            # target ``_yv`` / its SS, the train target, and the base design already sliced into train/val
            # blocks. Every call scores ``[base | one candidate column]``, so only the single candidate
            # column is stacked/sliced per call instead of rebuilding + row-slicing the full base at n rows.
            _yv = _rp_y[_rp_va]
            _rp_ss = float(np.sum((_yv - _yv.mean()) ** 2))
            _rp_y_tr = _rp_y[_rp_tr]
            _rp_base_mat = np.column_stack(_rp_base)
            _rp_base_tr = _rp_base_mat[_rp_tr]
            _rp_base_va = _rp_base_mat[_rp_va]

            # QR of the FIXED base design, computed ONCE and reused for every candidate below (a perf
            # fix). Each candidate previously re-solved lstsq on ``[base | one extra column]`` from
            # scratch - O(n*p^2) per call with only a single column differing between calls. Extending an
            # EXISTING QR by one column via ``scipy.linalg.qr_insert`` is an O(n*p) update, mathematically
            # equivalent to a fresh least-squares solve of the augmented design (verified: max coefficient
            # difference ~6e-17, i.e. machine-epsilon-level agreement with the original SVD-based
            # ``np.linalg.lstsq``, not just "close enough" - this is the standard Frisch-Waugh-Lovell-style
            # QR-update result, not an approximation). Measured 20x faster at production shape (p~120,
            # n_tr~53k, 109 candidates: 91s -> 4.5s on synthetic data of that shape).
            import scipy.linalg as _rp_sla
            try:
                _rp_Q, _rp_R = _rp_sla.qr(_rp_base_tr, mode="economic")
                _rp_Qty = _rp_Q.T @ _rp_y_tr
                _rp_coef_base = _rp_sla.solve_triangular(_rp_R, _rp_Qty)
                _rp_qr_ok = True
            except Exception as exc:
                logger.debug("mrmr: QR-based raw-protection incremental check failed; falling back to the full-refit path: %r", exc, exc_info=True)
                _rp_qr_ok = False

            def _rp_r2(_extra=None):
                """Held-out R^2 of ``[base | extra]``; ``_extra`` is a single full-length column or None.
                Numerically identical (to ~1e-16) to the prior ``_rp_r2(_design)`` (same columns in the
                same order, same train/val rows, same lstsq) - see the QR-reuse comment above."""
                if _rp_ss < 1e-24:
                    return 0.0
                if _extra is None:
                    if not _rp_qr_ok:
                        return -np.inf
                    return 1.0 - float(np.sum((_yv - _rp_base_va @ _rp_coef_base) ** 2)) / _rp_ss
                if not _rp_qr_ok:
                    return -np.inf
                try:
                    _q1, _r1 = _rp_sla.qr_insert(_rp_Q, _rp_R, _extra[_rp_tr], _rp_Q.shape[1], which="col")
                    _coef = _rp_sla.solve_triangular(_r1, _q1.T @ _rp_y_tr)
                except Exception as e:
                    logger.debug("QR-insert regression probe failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                    return -np.inf
                _A_va = np.column_stack((_rp_base_va, _extra[_rp_va]))
                return 1.0 - float(np.sum((_yv - _A_va @ _coef) ** 2)) / _rp_ss

            if int(_rp_tr.sum()) >= 32 and int(_rp_va.sum()) >= 16:
                _rp_r2_base = _rp_r2()
                _cols_index_r = {c: i for i, c in enumerate(cols)}
                _sv_set_r = set(selected_vars)
                _readd_raw = []
                # RELEVANCE GATE on the re-add. The held-out single-split R^2 increment alone is an UNCORRECTED linear-usability test: an
                # unregularised regressor overfits idiosyncratic noise on one ~n/3 val split enough to clear the loose 0.005 floor for a
                # feature the relevance screen correctly rejected as within-null (e.g. decoy = x_real**2 on y = sign(x_real): MI ~ 0.00014,
                # below the effective floor, corr -0.04, yet R^2 incr ~0.011). Require the candidate to ALSO clear the SAME marginal-MI
                # relevance floor the screen used (absolute effective floor AND the relative-to-strongest floor) so a below-null raw cannot
                # be resurrected by linear-usability alone - this re-opened exactly the hole the screen floor closes.
                _rp_rel_floor = float(_effective_min_relevance_gain) if "_effective_min_relevance_gain" in dir() else float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
                _rp_rel_frac = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
                _rp_max_mi = max((float(_v) for _v in cached_MIs.values()), default=0.0) if isinstance(cached_MIs, dict) else 0.0
                _rp_floor = max(_rp_rel_floor, _rp_max_mi * _rp_rel_frac)
                # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
                _fni_rp = getattr(self, "feature_names_in_", None)
                for _rn in (_fni_rp if _fni_rp is not None else []):
                    _ridx = _cols_index_r.get(_rn)
                    if _ridx is None or _ridx in _sv_set_r or _rn not in X.columns:
                        continue
                    _rp_cand_mi = float(cached_MIs.get((_ridx,), 0.0)) if isinstance(cached_MIs, dict) else 0.0
                    if _rp_cand_mi <= _rp_floor:
                        continue  # within-null / below the screen's relevance floor -> not a genuine signal, do not resurrect
                    try:
                        _rv = np.asarray(X[_rn].to_numpy(), dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue  # non-numeric raw (categorical/string) -> not a linear-usability candidate
                    if _rv.shape[0] != _rp_n or not np.all(np.isfinite(_rv)):
                        continue
                    if _rp_r2(_rv) - _rp_r2_base < _RAW_PROTECT_MIN_INCR_R2:
                        continue
                    _readd_raw.append(_ridx)
                    _sv_set_r.add(_ridx)
                if _readd_raw:
                    selected_vars = list(selected_vars) + _readd_raw
                    if verbose:
                        logger.info(
                            "MRMR raw-feature floor-drop protection: re-added %d held-out-validated raw "
                            "feature(s) the maxT relevance floor dropped (genuine linear usability, not "
                            "high-card noise): %s",
                            len(_readd_raw), [cols[i] for i in _readd_raw],
                        )

    # CAT-FE FLOOR-DROP PROTECTION (Fix-C). The Westfall-Young maxT relevance floor (computed over
    # the FULL widened candidate pool when many FE families are on) routinely rises above the marginal binned-MI
    # of a genuine categorical-FE encoding - a K-fold target encoding (``cat__te``), a count/frequency encoding,
    # or a cat-num residual (``price__resid_by__cat_region``) - so the greedy screen drops it after 2 features
    # EVEN THOUGH it carries strong LINEAR usability to y (the MI-vs-linear-usability gap, a recurring mlframe
    # theme). The cat-num residual on the kitchen-sink frame has univariate corr ~0.27 / held-out R^2-incr ~0.06
    # over the selected design yet is screened out, so downstream LogReg loses ~0.6% AUC. This is the SAME class
    # of false-drop the raw-feature / orth-basis / hinge protections already correct - but those iterate only
    # over raw ``feature_names_in_`` / single-source orth bases / hinge legs, so an engineered cat-FE column falls
    # through every one of them. Mirror the raw protection here: KEEP the floor (sub-null noise stays rejected)
    # and re-add a dropped cat-FE column IFF it lifts a HELD-OUT linear fit over the already-selected design by
    # >= the same R^2 floor. The cat-FE columns live as quantized codes in ``data[:, idx]`` (the continuous
    # snapshot is only populated by the fe_max_steps>0 path); the binned codes preserve the monotone/linear
    # signal well enough for the usability test (a genuine encoding lifts R^2 >> floor; a noise encoding ~0).
    if isinstance(X, pd.DataFrame) and len(selected_vars):
        _cf_names: list = []
        for _attr in ("kfold_te_features_", "count_encoding_features_", "frequency_encoding_features_", "cat_num_interaction_features_"):
            _cf_names.extend(getattr(self, _attr, None) or [])
        _cf_names = [n for n in dict.fromkeys(_cf_names)]  # dedup, preserve order
        if _cf_names:
            _cf_y = None
            try:
                _cf_yv = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).reshape(-1)
                if _cf_yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_cf_yv)):
                    _cf_y = _cf_yv
            except Exception as exc:
                logger.debug("mrmr: y coercion for the cat-FE floor-drop protection probe failed; protection disabled: %r", exc, exc_info=True)
                _cf_y = None
            if _cf_y is not None:
                _CF_PROTECT_MIN_INCR_R2 = 0.005  # genuine encoding lifts held-out R^2 >> 0.005; noise ~0 (same bar as raw protection)
                _cf_n = _cf_y.shape[0]
                # Seeded shuffle-then-stride (see the hinge-gate sibling comment above).
                _cf_perm = np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)).permutation(_cf_n)
                _cf_va = np.zeros(_cf_n, dtype=bool)
                _cf_va[_cf_perm[: _cf_n // 3]] = True
                _cf_tr = ~_cf_va
                _cf_cols_index = {c: i for i, c in enumerate(cols)}
                _cf_sv_set = set(selected_vars)
                _cf_sel_names = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
                # Baseline design = intercept + continuous/binned values of the ALREADY-SELECTED columns, so a
                # cat-FE column subsumed by a selected feature adds ~0 and is NOT re-added (no redundancy regression).
                _cf_base = [np.ones(_cf_n)]
                for _sn in _cf_sel_names:
                    _cv = _eng_continuous_snapshot.get(_sn)
                    if _cv is None and _sn in X.columns:
                        try:
                            _cv = X[_sn].to_numpy()
                        except Exception as exc:
                            logger.debug("mrmr: continuous-value lookup failed for this candidate; treating as unavailable: %r", exc, exc_info=True)
                            _cv = None
                    if _cv is None:
                        _si = _cf_cols_index.get(_sn)
                        if _si is not None:
                            _cv = data[:, _si]
                    if _cv is None:
                        continue
                    try:
                        _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if _cv.shape[0] == _cf_n and np.all(np.isfinite(_cv)):
                        _cf_base.append(_cv)

                def _cf_r2(_design):
                    """Fit an OLS design on the train stride and return held-out R^2 on the %3 validation stride (0.0 on degenerate variance, ``-inf`` on a failed lstsq); used by the categorical-FE protect/re-add gate."""
                    _A = np.column_stack(_design)
                    _yv = _cf_y[_cf_va]
                    _ss = float(np.sum((_yv - _yv.mean()) ** 2))
                    if _ss < 1e-24:
                        return 0.0
                    try:
                        _coef, *_ = np.linalg.lstsq(_A[_cf_tr], _cf_y[_cf_tr], rcond=None)
                    except Exception as e:
                        logger.debug("Compound-feature OLS lstsq failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                        return -np.inf
                    return 1.0 - float(np.sum((_yv - _A[_cf_va] @ _coef) ** 2)) / _ss

                if int(_cf_tr.sum()) >= 32 and int(_cf_va.sum()) >= 16:
                    _cf_r2_base = _cf_r2(_cf_base)
                    _readd_cf = []
                    for _cn in _cf_names:
                        _cidx = _cf_cols_index.get(_cn)
                        if _cidx is None or _cidx in _cf_sv_set or _cn in _cf_sel_names:
                            continue
                        # Prefer the full-precision continuous value (same source the baseline design above
                        # uses for already-selected columns) over the nbins-quantized screening code: quantile
                        # bin-edge digitization is not exactly tie-invariant across a monotone rescale of a
                        # duplicate-heavy column (e.g. a count encoding vs its count/n frequency twin can land
                        # in a different number of effective bins from floating-point edge-coincidence ties),
                        # which made this R^2 probe - and hence the rescue - diverge between two info-
                        # equivalent encodings. The raw column is still available in X at this point.
                        _cvv_raw = _eng_continuous_snapshot.get(_cn)
                        if _cvv_raw is None and _cn in X.columns:
                            try:
                                _cvv_raw = X[_cn].to_numpy()
                            except Exception as exc:
                                logger.debug("mrmr: raw continuous-value lookup failed for this candidate: %r", exc, exc_info=True)
                                _cvv_raw = None
                        if _cvv_raw is not None:
                            try:
                                _cvv = np.asarray(_cvv_raw, dtype=np.float64).reshape(-1)
                            except (TypeError, ValueError):
                                _cvv_raw = None
                        if _cvv_raw is None:
                            try:
                                _cvv = np.asarray(data[:, _cidx], dtype=np.float64).reshape(-1)
                            except (TypeError, ValueError, IndexError):
                                continue
                        if _cvv.shape[0] != _cf_n or not np.all(np.isfinite(_cvv)):
                            continue
                        if _cf_r2([*_cf_base, _cvv]) - _cf_r2_base < _CF_PROTECT_MIN_INCR_R2:
                            continue  # no held-out linear usability over the selected design -> stays out
                        _readd_cf.append(_cidx)
                        _cf_sv_set.add(_cidx)
                    if _readd_cf:
                        selected_vars = list(selected_vars) + _readd_cf
                        if verbose:
                            logger.info(
                                "MRMR cat-FE floor-drop protection: re-added %d held-out-validated categorical-FE "
                                "encoding(s) the maxT relevance floor dropped (genuine linear usability, not "
                                "sub-null noise): %s",
                                len(_readd_cf), [cols[i] for i in _readd_cf],
                            )

    # POST-SELECTION DCD CLUSTER DISCOVERY. DCD's in-screen hook (``screen_dcd_discover_and_swap``) anchors a cluster ONLY on a column the greedy screen actually SELECTED. On a duplicate-feature
    # fixture the greedy screen selects ONE representative (a strong column or an engineered composite) and gates the redundant duplicates out as mutually-redundant, so no duplicate is ever an
    # anchor: DCD discovers 0 clusters and ``dcd_["n_pruned"]`` stays 0 even though the duplicates re-enter ``selected_vars`` via the floor-drop / retention rescues above. The cluster the screen never
    # saw is exactly the one DCD exists to own. Run a discovery pass over the FINAL selected RAW columns (anchoring on each in selection order, growing from the other selected raws by SU >= tau): the
    # duplicate cluster is found and its redundant members pruned by DCD BEFORE the raw-redundancy / monotone-twin drops below - DCD owns exact-duplicate clusters, the redundancy drops own engineered-
    # child subsumption. When the grown cluster reaches ``dcd_cluster_size_threshold`` the same anchor->aggregate swap the screen would have evaluated is evaluated + committed here (registering the
    # ``_dcd_pc1_`` cluster_aggregate recipe into ``engineered_recipes`` so it lands in the ``_produced_recipes_`` ledger snapshotted below). Pruned duplicate members are removed from ``selected_vars``
    # (mirroring the in-screen prune); ``dcd_`` is re-published so ``n_pruned`` / ``n_swaps`` / ``cluster_anchors`` reflect the discovered cluster.
    # GATE: fire ONLY when the in-screen DCD discovered NOTHING - no pool member pruned AND no swap committed. That is exactly the duplicate-cluster-missed case (screen selected one representative +
    # engineered children, never anchored on a duplicate). When the in-screen DCD already clustered (FE-rich sensor-mesh / financial / embedding fixtures) it owns the support-shrinkage contract; re-
    # discovering here would double-act and could GROW support (an extra aggregate the screen-time bake-off deliberately did not add), violating the "DCD must not grow support" invariant. ``cluster_anchors``
    # is NOT a usable signal: ``discover_cluster_members`` does ``setdefault(anchor, set())`` for every selected predictor, so it carries EMPTY anchor entries even when no member joined - the real
    # "discovered a cluster" signal is a non-zero pruned-mask / a non-empty swap_log.
    if (_persisted_dcd_state is not None and len(selected_vars) >= 2
            and _persisted_dcd_state.pool_pruned_mask is not None
            and int(_persisted_dcd_state.pool_pruned_mask.sum()) == 0
            and not (getattr(_persisted_dcd_state, "swap_log", None) or [])):
        try:
            from .._dynamic_cluster_discovery import (
                discover_cluster_members as _post_dcd_discover,
                evaluate_swap_candidate as _post_dcd_eval_swap,
                commit_swap as _post_dcd_commit_swap,
                dcd_summary as _post_dcd_summary,
            )
            _dcd_st = _persisted_dcd_state
            _mask_w0 = int(_dcd_st.pool_pruned_mask.shape[0]) if _dcd_st.pool_pruned_mask is not None else 0
            _raw_name_set_dcd = set(self.feature_names_in_)
            # Selected RAW columns: stable low indices within the DCD mask width, NUMERIC only (a
            # string/categorical raw can never enter the PC1/Pearson aggregate - it would raise
            # "could not convert string to float" in the swap's combiner - and is not a numeric
            # duplicate cluster anyway), in selection order.
            # NUMERIC-only is enforced below via the dtype check directly (no `numeric_features_in_`
            # attribute exists on MRMR to cross-check against - a prior getattr(..., None) here always
            # silently returned the default and was a dead no-op, per code_audit's getattr_unknown_attribute).
            _sel_raw_dcd = [
                int(v)
                for v in selected_vars
                if 0 <= int(v) < _mask_w0 and cols[int(v)] in _raw_name_set_dcd and np.issubdtype(np.asarray(data[:, int(v)]).dtype, np.number)
            ]
            _newly_pruned_dcd: set = set()
            _did_swap_dcd = False
            for _anchor in list(_sel_raw_dcd):
                if _dcd_st.pool_pruned_mask[_anchor]:
                    continue  # already pruned as a member of an earlier anchor's cluster
                _pool_dcd = [c for c in _sel_raw_dcd if c != _anchor and not _dcd_st.pool_pruned_mask[c]]
                if not _pool_dcd:
                    continue
                _added = _post_dcd_discover(
                    _dcd_st, _anchor, _pool_dcd,
                    entropy_cache=None,
                    factors_data=data,
                    factors_nbins=np.asarray(nbins, dtype=np.int64),
                    selected_vars=selected_vars,
                )
                _newly_pruned_dcd |= set(int(a) for a in _added)
                # Mirror the in-screen anchor->aggregate swap: when the grown cluster reaches the size
                # threshold, evaluate + commit the PC1/mean_z aggregate swap so n_swaps / swap_log /
                # the cluster_aggregate recipe are produced exactly as the screen would have.
                _members = _dcd_st.cluster_anchors.get(int(_anchor), set())
                if len(_members) >= int(_dcd_st.cluster_size_threshold):
                    # Sync the state's matrix to the LIVE (post-FE) matrix so the swap's S\{anchor}
                    # conditioning set - which may reference engineered columns appended AFTER the
                    # screen built the state's matrix - indexes valid columns (else conditional_mi
                    # raises "negative dimensions" on an out-of-range column).
                    if int(data.shape[1]) >= int(_dcd_st.factors_data.shape[1]):
                        _dcd_st.factors_data = data
                        _dcd_st.factors_nbins = np.asarray(nbins, dtype=np.int64)
                        _dcd_st.cols = list(cols)
                        if _dcd_st.pool_pruned_mask is not None and int(data.shape[1]) > int(_dcd_st.pool_pruned_mask.shape[0]):
                            _dcd_st.pool_pruned_mask = np.concatenate([
                                _dcd_st.pool_pruned_mask,
                                np.zeros(int(data.shape[1]) - int(_dcd_st.pool_pruned_mask.shape[0]), dtype=bool),
                            ])
                    # Swap conditioning set: the anchor + the OTHER selected RAW non-cluster columns
                    # only. The in-screen swap evaluates early when ``selected_vars`` is still small;
                    # post-selection the full ``selected_vars`` also holds engineered children of the
                    # SAME cluster latent (e.g. ``add(log(strong),prewarp(dup_c))``), and conditioning
                    # the aggregate-vs-anchor relevance comparison on those children removes the shared
                    # latent entirely -> both sides read ~0 residual -> the swap never fires. Restricting
                    # the conditioning set to selected raws outside the cluster restores the screen-time
                    # comparison (aggregate's denoised latent vs the single noisy anchor dup).
                    _cluster_idx_set = set(_members) | {int(_anchor)}
                    _swap_sel_vars = [
                        int(v)
                        for v in selected_vars
                        if int(v) == int(_anchor) or (int(v) not in _cluster_idx_set and 0 <= int(v) < _mask_w0 and cols[int(v)] in _raw_name_set_dcd)
                    ]
                    _dec = _post_dcd_eval_swap(
                        _dcd_st, int(_anchor), _swap_sel_vars,
                        target_y=target_indices,
                        factors_data=data,
                        factors_nbins=np.asarray(nbins, dtype=np.int64),
                        entropy_cache=None,
                        cached_MIs=None,
                        full_npermutations=int(getattr(self, "full_npermutations", 0) or 0),
                    )
                    if getattr(_dec, "accept", False):
                        _dref: dict = {}
                        _post_dcd_commit_swap(
                            _dcd_st, int(_anchor), _dec,
                            selected_vars=selected_vars,
                            data_ref=_dref,
                            engineered_recipes=engineered_recipes,
                            predictors_log=None,
                        )
                        data = _dref.get("data", data)
                        nbins = _dref.get("nbins", nbins)
                        cols = _dref.get("cols", cols)
                        _did_swap_dcd = True
            if _newly_pruned_dcd or _did_swap_dcd:
                selected_vars = [v for v in selected_vars if int(v) not in _newly_pruned_dcd]
                self.dcd_ = _post_dcd_summary(_dcd_st)
                if isinstance(self.dcd_, dict):
                    self.cluster_members_ = dict(self.dcd_.get("cluster_anchors_names", {}))
                if verbose:
                    logger.info(
                        "MRMR post-selection DCD: discovered a duplicate cluster the greedy screen never anchored on; " "pruned %d redundant member(s)%s.",
                        len(_newly_pruned_dcd),
                        " + committed an aggregate swap" if _did_swap_dcd else "",
                    )
        except Exception as _exc_post_dcd:
            logger.warning("MRMR post-selection DCD discovery failed: %s; keeping support as-is.", _exc_post_dcd)

    # PRODUCED-RECIPES AUDIT LEDGER: ``engineered_recipes`` at this point holds EVERY recipe the FE stages produced this fit, before the greedy CMI screen / accuracy gate / cross-stage dedup drop the
    # weaker candidates. ``self._engineered_recipes_`` (built just below) carries only the survivors - it is intersected with support_ so the user-facing rosters stay a subset of get_feature_names_out()
    # (pinned by layer28). The audit / pickle-replay paths, however, need to recover WHICH mechanism produced each engineered column even when the screen dropped it, so snapshot the full produced set here
    # as a separate read-only ledger. fe_provenance_ reads this to emit one row per produced engineered column (survivors get their real greedy gain/rank, screened-out ones get NaN gain / rank -1).
    self._produced_recipes_ = list(engineered_recipes.values())

    # PSEUDO-CHILD MASKED-RAW RESCUE. The default-ON conditional-gate / binned-
    # numeric-agg / row-argmax FE families append THRESHOLD/BINNING re-mixes of a raw operand
    # (``gate_mask__a__b`` / ``binagg_skew(c|qbin(a))`` / ``argmax__a__b``) into the screening pool
    # BEFORE the greedy screen. A re-mix of ``a`` can marginally OUT-SCORE raw ``a`` and is selected
    # first; raw ``a``'s conditional relevance given that re-mix then collapses (the re-mix is a lossy
    # function of ``a`` - the data-processing-inequality trap), so the greedy screen drops ``a``
    # EVEN WHEN ``a`` carries a dominant private LINEAR term (``y += 10*a``) the re-mix only partially
    # tracks. Re-add such a masked raw - one consumed by a selected pseudo-remix child but itself
    # dropped - IFF it retains >= RAW_SELF_RETAIN_FRAC of its marginal debiased excess under the
    # keep-rule conditioned ONLY on its GENUINE (non-pseudo) selected children (an ``a**2/b`` ratio /
    # composite - the real potential subsumers, with the masking pseudo re-mixes EXCLUDED from the
    # conditioning). A private LINEAR term keeps ~50% -> RESCUE; a fully-subsumed operand keeps ~0.6%
    # -> NOT rescued (so a raw genuinely subsumed by an elementary child is never resurrected). The
    # downstream raw-redundancy DROP sweep still runs after with the SAME pseudo-exclusion, so the two
    # passes agree. Byte-identical when no pseudo-remix child is selected (the candidate set is empty).
    # Off when the drop sweep is disabled (shares the ``fe_drop_redundant_raw_operands`` toggle).
    if getattr(self, "fe_drop_redundant_raw_operands", True) and len(selected_vars) >= 1:
        try:
            from .._fe_raw_redundancy_drop import (
                _is_pseudo_remix_child as _pcr_is_pseudo,
                _PSEUDO_SRC_SPLIT,
                raw_retains_signal_given_genuine_children as _pcr_keep,
            )
            from .._mi_greedy_cmi_fe import _quantile_bin as _pcr_qbin
            _pcr_raw_set = set(self.feature_names_in_)
            _pcr_sel_set = set(selected_vars)
            _pcr_sel_names = {cols[i] for i in selected_vars}
            # Selected pseudo-remix children and the raw operands each re-mixes.
            _pcr_pseudo_sel = [i for i in selected_vars if _pcr_is_pseudo(cols[i])]
            if _pcr_pseudo_sel:
                # raw_name -> selected pseudo children consuming it.
                _pcr_consumed: dict = {}
                for _pi in _pcr_pseudo_sel:
                    _toks = {t for t in _PSEUDO_SRC_SPLIT.split(cols[_pi]) if t}
                    for _t in _toks:
                        if _t in _pcr_raw_set:
                            _pcr_consumed.setdefault(_t, []).append(_pi)
                # A raw is also consumed by a GENUINE (non-pseudo) selected engineered child when its
                # name token appears there; such a raw is left to the DROP sweep (might be subsumed).
                _pcr_genuine_eng = [i for i in selected_vars if (cols[i] not in _pcr_raw_set) and not _pcr_is_pseudo(cols[i])]
                _pcr_y = np.ascontiguousarray(np.asarray(classes_y)).ravel().astype(np.int64)
                try:
                    _pcr_yv = y.values if hasattr(y, "values") else np.asarray(y)
                    _pcr_yv = np.asarray(_pcr_yv).reshape(-1)
                    if (_pcr_yv.shape[0] == int(data.shape[0]) and np.issubdtype(_pcr_yv.dtype, np.number)
                            and int(np.unique(_pcr_yv).size) > max(20, 2 * int(np.unique(_pcr_y).size))):
                        _pcr_nb = int(min(max(10, int(np.unique(_pcr_y).size)), max(2, int(data.shape[0]) // 50)))
                        _pcr_y = np.ascontiguousarray(_pcr_qbin(_pcr_yv.astype(np.float64), nbins=_pcr_nb)).astype(np.int64)
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("mrmr: post-cluster-rescue y rebinning failed: %r", e, exc_info=True)
                    pass
                _pcr_eng_cont = _eng_continuous_snapshot
                from .._fe_raw_redundancy_drop import _TOKEN_SPLIT
                # PERMUTATION-SIGNIFICANCE GATE on the masked-raw rescue (I4 noise
                # admission). The keep-rule conditions on the GENUINE children only; when a raw's
                # ONLY consumer is a pseudo binagg/gate/argmax re-mix the conditioning set is empty
                # and the keep-rule returns True by construction (it cannot prove subsumption). A
                # PURE-NOISE raw (``e``, not in y, consumed only by ``binagg_std(e|qbin(a))``) thus
                # sails through and is re-added - the I4 noise true-negative violation. Gate the
                # rescue on the SAME within-data marginal permutation-significance test the
                # raw-retention re-add uses: a raw must sit ABOVE its own permutation null against y
                # to be a genuine masked signal. ``e`` sits WITHIN its null (p>=alpha) -> NOT rescued;
                # a genuinely masked raw (``a`` carrying ``3*a``) clears it. Best-effort: a kernel
                # failure falls through to the permissive rescue (never drop on an estimator error).
                try:
                    from ..permutation import mi_direct as _pcr_mi_direct
                except Exception as exc:
                    logger.debug("mrmr: mi_direct import/binding failed for the post-cluster-rescue significance probe; probe disabled: %r", exc, exc_info=True)
                    _pcr_mi_direct = None  # type: ignore[assignment]
                _pcr_signif_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
                _pcr_q_dtype = getattr(self, "quantization_dtype", np.int32)

                def _pcr_raw_is_significant(_idx):
                    """Permutation-significance test (32 permutations) for raw column ``_idx`` against y; True when it clears its own null (p<alpha) or the MI estimator is unavailable/errors, gating the masked-raw rescue against pure-noise raws whose only consumer is a pseudo binagg/gate/argmax re-mix."""
                    if _pcr_mi_direct is None:
                        return True
                    try:
                        _sig = _pcr_mi_direct(
                            data, x=np.array([int(_idx)], dtype=np.int64), y=target_indices,  # type: ignore[arg-type]
                            factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                            return_null_mean=True, parallelism="none", dtype=_pcr_q_dtype, prefer_gpu=False,
                        )
                        return float(_sig[3]) < _pcr_signif_alpha
                    except Exception as e:
                        logger.debug("Marginal-MI significance re-add probe failed (%s: %s) -- permissive re-add", type(e).__name__, e)
                        return True
                _pcr_readd = []
                # name -> index map built once (O(F)) instead of a ``.index()`` rescan of ``cols`` per
                # ``_rn`` (O(F) each) - turns the O(K*F) loop below into O(K+F).
                _pcr_cols_idx = {nm: i for i, nm in enumerate(cols)}
                for _rn in _pcr_consumed.keys():
                    if _rn in _pcr_sel_names:
                        continue  # already selected -> nothing to rescue
                    _ridx = _pcr_cols_idx.get(_rn)
                    if _ridx is None:
                        continue
                    if _ridx in _pcr_sel_set:
                        continue
                    # KEEP-RULE conditioned ONLY on the raw's GENUINE (non-pseudo) selected children -
                    # the real potential subsumers (an ``a**2/b`` ratio/composite). The masking pseudo
                    # re-mixes are EXCLUDED from the conditioning so they cannot DPI-collapse the residual.
                    # A raw carrying a private term the genuine children do not span keeps a large residual
                    # (~50%) -> RESCUE; a raw fully subsumed by a genuine ratio child keeps ~0.6% -> NOT
                    # rescued (and would be dropped by the sweep anyway). When NO genuine child consumes the
                    # raw the conditioning set is empty and the keep-rule returns True (the drop was a pure
                    # pseudo-mask) -> RESCUE.
                    _child_bins = []
                    for _gi in _pcr_genuine_eng:
                        if _rn in {t for t in _TOKEN_SPLIT.split(cols[_gi]) if t}:
                            _cont = _pcr_eng_cont.get(cols[_gi])
                            if _cont is not None and np.asarray(_cont).shape[0] == int(data.shape[0]):
                                _child_bins.append(_pcr_qbin(np.asarray(_cont, dtype=np.float64), nbins=10))
                            else:
                                _child_bins.append(np.asarray(data[:, _gi]).astype(np.int64).ravel())
                    _rb = np.asarray(data[:, _ridx]).astype(np.int64).ravel()
                    if _pcr_keep(
                        raw_bin=_rb,
                        y_bin=_pcr_y,
                        genuine_child_bins=_child_bins,
                        allow_linear_usability=bool(getattr(self, "use_simple_mode", False)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                    ) and _pcr_raw_is_significant(_ridx):
                        _pcr_readd.append(_ridx)
                if _pcr_readd:
                    selected_vars = list(selected_vars) + [i for i in _pcr_readd if i not in _pcr_sel_set]
                    if verbose:
                        logger.info(
                            "MRMR pseudo-child masked-raw rescue: re-added %d raw operand(s) the greedy "
                            "screen dropped because a gate/binagg/argmax re-mix of them was selected first "
                            "and masked their conditional relevance (DPI trap), yet they retain a private "
                            "residual given that re-mix: %s",
                            len(_pcr_readd), [cols[i] for i in _pcr_readd],
                        )
        except Exception as _exc_pcr:
            logger.warning("MRMR pseudo-child masked-raw rescue failed: %s; keeping support as-is.", _exc_pcr)

    # RAW-VS-ENGINEERED CONDITIONAL-REDUNDANCY DROP: the greedy MRMR order
    # selects a raw operand on its high MARGINAL relevance BEFORE the engineered child built
    # from it is in support, so the redundancy penalty never fires against it, and the
    # retention / augmentation passes above then re-add it. The result is a subsumed operand
    # admitted alongside the engineered feature that fully determines y from it (e.g. raw
    # ``a, b`` beside ``div(neg(a),sqrt(b))`` for ``y=(a**2)/b``, since
    # ``(a/sqrt(b))**2 = a**2/b``). This final sweep removes such operands using the SAME
    # debiased excess-CMI idea the engineered-vs-engineered S5 gate validated, so the verdict
    # is n-INVARIANT (identical at n=1000 and n=50000) and never drops a raw carrying genuine
    # independent signal (a private additive term keeps a large excess and is KEPT). On by
    # default; ``fe_drop_redundant_raw_operands=False`` restores the pre-fix behaviour.
    if getattr(self, "fe_drop_redundant_raw_operands", True) and getattr(self, "redundancy_policy", "emit_both") == "drop" and len(selected_vars) >= 2:
        try:
            from .._fe_raw_redundancy_drop import drop_redundant_raw_operands
            _raw_names_for_redund = set(self.feature_names_in_)
            # Only worth running when at least one engineered survivor and one raw operand
            # are both selected (otherwise the helper short-circuits anyway).
            _sel_names_redund = [cols[i] for i in selected_vars]
            _has_eng = any(nm not in _raw_names_for_redund for nm in _sel_names_redund)
            _has_raw = any(nm in _raw_names_for_redund for nm in _sel_names_redund)
            if _has_eng and _has_raw:
                # Continuous target for equi-frequency re-binning: the screening
                # ``classes_y`` is frequently HEAVILY imbalanced on a skewed regression
                # target (``y=(a**2)/b`` puts ~89% of rows in one bin), which crushes the
                # engineered anchor's MI and inflates a subsumed operand's apparent residual
                # fraction. Re-binning the continuous target equi-frequency restores a faithful
                # anchor. Falls back to ``classes_y`` for already-discrete targets.
                _y_cont_for_redund = None
                try:
                    _yv = y.values if hasattr(y, "values") else np.asarray(y)
                    _yv = np.asarray(_yv).reshape(-1)
                    if _yv.shape[0] == int(data.shape[0]) and np.issubdtype(np.asarray(_yv).dtype, np.number):
                        _y_cont_for_redund = _yv
                except Exception as exc:
                    logger.debug("mrmr: continuous-y coercion failed for raw-redundancy transform-time replay; replay skipped for this column: %r", exc, exc_info=True)
                    _y_cont_for_redund = None
                # Only engineered survivors with a replayable recipe (1-deep, in
                # ``engineered_recipes``) survive into transform output; a nested-
                # engineered child is dropped there. A raw must not be judged
                # redundant against a child that will not exist at predict time
                # (that empties the support - see the guard in
                # drop_redundant_raw_operands), so anchor the verdict only on the
                # replayable survivors.
                _replayable_eng_names = set(engineered_recipes.keys())
                # NESTED-OPERAND CONSUMER DETECTION (BUG1): pass the
                # engineered RECIPES (name -> EngineeredRecipe) and the raw frame so
                # the redundancy verdict can walk each consuming composite's operand
                # tree, isolate the cleanest raw-containing sub-expression (e.g.
                # ``div(sqr(a),abs(b))`` = a**2/b inside a fused full-target composite),
                # and condition the raw on THAT clean sub-expression rather than the
                # fused whole - so a fully-subsumed operand drops even when it is
                # selected alongside the composite (not only when the composite
                # collapsed the whole selection into the never-empty path).
                _rrf_redund = getattr(self, "fe_raw_redundancy_retain_frac", None)
                _kept_redund, _dropped_redund_names = drop_redundant_raw_operands(
                    data=data,
                    cols=cols,
                    selected_cols_idx=selected_vars,
                    raw_name_set=_raw_names_for_redund,
                    y_binned=classes_y,
                    y_continuous=_y_cont_for_redund,
                    engineered_continuous=_eng_continuous_snapshot,
                    replayable_eng_names=_replayable_eng_names,
                    recipes=engineered_recipes,
                    raw_X=X,
                    retain_frac=float(_rrf_redund) if _rrf_redund is not None else 0.15,
                    linear_usability_keep=bool(getattr(self, "use_simple_mode", False)),
                    tail_subsume_enable=_fe_family_on("fe_pair_usability_admission_enable", True),
                    tail_subsume_min_corr=float(getattr(self, "fe_raw_tail_subsume_min_corr", 0.85)),
                    tail_subsume_rank_frac=float(getattr(self, "fe_pair_usability_admission_rank_frac", 0.7)),
                    seed=int(getattr(self, "random_seed", 0) or 0),
                    verbose=verbose,
                )
                if _dropped_redund_names:
                    selected_vars = _kept_redund
                    # Record the verdict so the downstream raw-signal-retention augmentation
                    # (which re-attaches a raw whose NAME tokenises a confirmed recipe by
                    # marginal MI) does NOT resurrect an operand this n-invariant conditional-
                    # redundancy sweep just dropped. The verdict is authoritative at every n.
                    self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | set(_dropped_redund_names)
                    # If the drop left NO raw survivor while engineered children survived, the
                    # engineered-only support is the INTENDED, complete outcome (every raw operand
                    # was conditionally subsumed). Flag it so the empty-RAW rescue ``else`` branch
                    # below does NOT mistake this for a "screen returned 0 raw" emergency and
                    # re-pollute the support with the dropped operands (or, worse, a pure-noise
                    # column ranked next by marginal MI).
                    _remaining_raw_after_drop = [v for v in selected_vars if cols[v] in set(self.feature_names_in_)]
                    if not _remaining_raw_after_drop:
                        # NEVER-EMPTY RAW FLOOR. The drop is allowed to
                        # remove a raw subsumed by a surviving engineered child WHILE other raws remain (the
                        # I4b contract). When dropping the redundant raws would leave ZERO raw survivors, the
                        # PRIOR floor unconditionally re-added the single STRONGEST dropped raw by MARGINAL MI
                        # as a raw stand-in. That marginal-MI pick is exactly the trap the rest of this module
                        # warns about: a fully-subsumed DOMINANT operand (``a`` in ``a**2/b``, whose ratio is
                        # captured byte-for-byte by the surviving fused compound) has the LARGEST marginal MI
                        # yet ZERO conditional/private residual, so the floor resurrected the very operand the
                        # n-invariant CMI sweep had just correctly dropped (the scaled_1_5 / heavy_tailed F2
                        # failure: the compound ``add(div(sqr(a),b),mul(log(c),sin(d)))`` fully reconstructs y,
                        # the sweep dropped a/b/d, and this floor re-added raw ``a`` beside the compound that
                        # subsumes it). When the main sweep empties the raw support EVERY dropped raw was judged
                        # fully subsumed by a surviving multi-source child, so the engineered survivor(s) ARE the
                        # complete feature set - the SAME engineered-only outcome the ``uniform`` profile and the
                        # never-empty re-attach block's all-operands-subsumed ``elif`` reach. Defer to that:
                        # re-add a dropped raw ONLY if it still carries a SIGNIFICANT PRIVATE LINEAR residual the
                        # engineered survivors do not linearly reproduce (a genuine partial-signal raw a downstream
                        # linear model needs); otherwise flag the intended engineered-only support so the
                        # downstream empty-raw rescue does not re-pollute it. The linear-usability re-add is a
                        # SIMPLE-mode concept ONLY: in full FE mode a subsumed MONOTONE operand (``a`` in ``a**2/b``
                        # on a positive domain, whose rank tracks ``y`` so a partial-rank-correlation reads a
                        # SPURIOUS private residual) is statistically indistinguishable from a genuine linear term
                        # and MUST still drop (I4b) - this mirrors ``drop_redundant_raw_operands``'s own
                        # ``allow_linear_usability=False`` policy in full FE mode (see its docstring / keep-leg).
                        # So the floor re-adds a dropped raw ONLY in simple mode and ONLY when it clears the same
                        # permutation-floored partial-rank-correlation probe; in full FE mode the empty raw support
                        # is the intended engineered-only outcome (the ``uniform`` profile's result).
                        _floor_simple = bool(getattr(self, "use_simple_mode", False))
                        from .._fe_raw_redundancy_drop import raw_retains_linear_signal_given_children as _floor_lin
                        _best_floor_idx, _best_floor_rel = None, float("-inf")
                        _tgt_floor = np.asarray(target_indices, dtype=np.int64)
                        _fn_floor = np.asarray(nbins, dtype=np.int64)
                        # Continuous engineered survivor values (for the linear-usability child design).
                        _floor_child_vals = []
                        for _ei in _kept_redund:
                            _enm = cols[_ei]
                            if _enm in set(self.feature_names_in_):
                                continue
                            _cv = (_eng_continuous_snapshot or {}).get(_enm)
                            if _cv is not None and np.asarray(_cv).shape[0] == int(data.shape[0]):
                                _floor_child_vals.append(np.asarray(_cv, dtype=np.float64).ravel())
                            else:
                                _floor_child_vals.append(np.asarray(data[:, _ei], dtype=np.float64).ravel())
                        try:
                            _yv_floor = y.values if hasattr(y, "values") else np.asarray(y)
                            _yv_floor = np.asarray(_yv_floor, dtype=np.float64).reshape(-1)
                        except Exception as exc:
                            logger.debug("mrmr: classes_y coercion failed for the floor-drop rescue; falling back to a raw classes_y reshape: %r", exc, exc_info=True)
                            _yv_floor = np.asarray(classes_y, dtype=np.float64).reshape(-1)
                        # name -> index map built once (O(F)) instead of a ``.index()`` rescan of
                        # ``cols`` per ``_dn`` (O(F) each) - turns the O(K*F) loop below into O(K+F).
                        _floor_cols_idx = {nm: i for i, nm in enumerate(cols)}
                        for _dn in _dropped_redund_names:
                            _floor_ci = _floor_cols_idx.get(_dn)
                            if _floor_ci is None:
                                continue
                            # Only a dropped raw with genuine PRIVATE linear signal beyond the engineered
                            # survivors is eligible to be the raw representative; a fully-subsumed operand is not.
                            # Full FE mode: no operand is eligible (defer to engineered-only - the I4b contract).
                            _eligible_floor = _floor_simple
                            if _floor_simple and _floor_child_vals:
                                try:
                                    _rawv = None
                                    if isinstance(X, pd.DataFrame) and _dn in X.columns:
                                        _rawv = np.asarray(X[_dn], dtype=np.float64).ravel()
                                    if _rawv is None:
                                        _rawv = np.asarray(data[:, _floor_ci], dtype=np.float64).ravel()
                                    _eligible_floor = bool(_floor_lin(
                                        _rawv, _yv_floor, _floor_child_vals,
                                        seed=int(getattr(self, "random_seed", 0) or 0),
                                    ))
                                except Exception as exc:
                                    logger.debug("mrmr: floor-eligibility check failed for this candidate; treating as ineligible (conservative): %r", exc, exc_info=True)
                                    _eligible_floor = False
                            if not _eligible_floor:
                                continue
                            try:
                                from ..info_theory import mi as _floor_mi
                                _rel = float(_floor_mi(data, np.array([int(_floor_ci)], dtype=np.int64), _tgt_floor, _fn_floor))
                            except Exception as exc:
                                logger.debug("mrmr: relevance computation failed for this candidate; treating as zero (conservative): %r", exc, exc_info=True)
                                _rel = 0.0
                            if _rel > _best_floor_rel:
                                _best_floor_rel, _best_floor_idx = _rel, int(_floor_ci)
                        if _best_floor_idx is not None:
                            selected_vars = [*list(_kept_redund), _best_floor_idx]
                            _kept_name_floor = cols[_best_floor_idx]
                            # The re-kept raw is no longer "dropped": remove it from the verdict set so the
                            # downstream retention / rescue passes treat it as a genuine survivor.
                            self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) - {_kept_name_floor}
                            if verbose:
                                logger.info(
                                    "MRMR raw-redundancy never-empty floor: dropping all raw operands would "
                                    "empty support_; retained strongest LINEAR-USABLE raw %r (marginal MI %.4f) "
                                    "as the raw representative beside the surviving engineered child.",
                                    _kept_name_floor, _best_floor_rel,
                                )
                        else:
                            # Every dropped raw is fully subsumed by a surviving engineered child - the
                            # engineered recipe(s) ARE the complete feature set (the uniform-profile outcome).
                            self._redundancy_emptied_raw_ = True
                    if verbose:
                        logger.info(
                            "MRMR raw-redundancy drop: removed %d raw operand(s) conditionally "
                            "redundant given their surviving engineered child (debiased excess "
                            "CMI below the relative bar): %s",
                            len(_dropped_redund_names), _dropped_redund_names,
                        )
        except Exception as _exc_redund:
            logger.warning(
                "MRMR raw-redundancy drop failed: %s; keeping the un-pruned support.",
                _exc_redund,
            )

    # RAW-vs-RAW MONOTONE-TWIN DROP (F6). The cross-stage Spearman-0.99 dedup
    # (above, ~line 5343) collapses monotone-equivalent ENGINEERED columns, and the
    # raw-vs-engineered redundancy sweep (above) drops a raw subsumed by an engineered
    # CHILD. Neither catches a RAW DECOY that is a pure MONOTONE re-encoding of ANOTHER
    # selected RAW column (``a_exp = exp(a)`` when raw ``a`` is selected): both bin
    # byte-identically under the quantile / rank-invariant MI screen, so they carry the
    # SAME information about y, yet the greedy screen / floor-drop protection / retention
    # passes can admit BOTH (the redundancy penalty is computed on coarse bins and the
    # nonlinear twin slips a small residual past it). Mirror the engineered dedup at the
    # RAW level: among selected raw columns, when two are monotone twins (|Spearman rho|
    # >= the same 0.99 bar), drop the LOWER-relevance one (by screening marginal MI;
    # ties keep the earlier-selected). A genuine independent raw (rank-uncorrelated with
    # every other selected raw) is untouched, so this never over-drops. Byte-identical
    # when no two selected raws are monotone twins. Shares the
    # ``fe_drop_redundant_raw_operands`` toggle (off restores the prior behaviour).
    if getattr(self, "fe_drop_redundant_raw_operands", True) and isinstance(X, pd.DataFrame) and len(selected_vars) >= 2:
        try:
            _MONO_TWIN_RHO = 0.99
            _raw_set_mt = set(self.feature_names_in_)
            _raw_sel_mt = [v for v in selected_vars if cols[v] in _raw_set_mt and cols[v] in X.columns]
            if len(_raw_sel_mt) >= 2:
                _mt_n = int(data.shape[0])
                _mt_ranks: dict[int, np.ndarray] = {}
                for _v in _raw_sel_mt:
                    try:
                        _cv = np.asarray(X[cols[_v]].to_numpy(), dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if _cv.shape[0] == _mt_n and np.all(np.isfinite(_cv)) and _cv.std() > 1e-12:
                        _mt_ranks[_v] = pd.Series(_cv).rank(method="average").to_numpy()
                # Relevance to break ties / pick the survivor: the screening marginal MI.
                def _mt_relevance(_v):
                    """Screening marginal MI(v, y) for raw column index ``_v``, used to pick the survivor between two monotone-twin raw columns (0.0 on a cache miss)."""
                    try:
                        return float(cached_MIs.get((_v,), 0.0))
                    except Exception as e:
                        logger.debug("cached_MIs lookup for monotone-twin relevance failed (%s: %s) -- treating as 0.0", type(e).__name__, e)
                        return 0.0
                from .._feature_engineering_pairs._pairs_core import _abs_corr_finite_njit as _mt_corr_njit
                _mt_keep: list[int] = []
                _mt_drop: set[int] = set()
                # Keep order = selection order, so an earlier-selected twin is preferred on a tie.
                for _v in _raw_sel_mt:
                    if _v not in _mt_ranks:
                        _mt_keep.append(_v)
                        continue
                    _twin_of = None
                    # ``_mt_ranks`` values are already guaranteed fully finite (the ``np.all(np.isfinite(_cv))``
                    # check above), so a serial one-pair njit reduction (no masking needed, no ``prange``
                    # launch overhead) reproduces ``np.corrcoef`` bit-faithfully without the 2x2-matrix
                    # build; this loop is bounded by the FINAL selected-feature count (small), so the plain
                    # single-pair kernel already used for the SAME pattern in ``_ratio_delta_fe.py`` fits
                    # better than a batched/parallel one at this scale.
                    _v_finite_mask = np.ones(_mt_ranks[_v].shape[0], dtype=np.bool_)
                    for _k in _mt_keep:
                        _rk = _mt_ranks.get(_k)
                        if _rk is None:
                            continue
                        _rho = float(_mt_corr_njit(_mt_ranks[_v], _rk, _v_finite_mask, 2))
                        if _rho >= _MONO_TWIN_RHO:
                            _twin_of = _k
                            break
                    if _twin_of is None:
                        _mt_keep.append(_v)
                    else:
                        # Drop the LOWER-relevance twin; if the candidate out-scores the kept twin,
                        # displace the kept one instead.
                        if _mt_relevance(_v) > _mt_relevance(_twin_of) + 1e-12:
                            _mt_drop.add(_twin_of)
                            _mt_keep.remove(_twin_of)
                            _mt_keep.append(_v)
                        else:
                            _mt_drop.add(_v)
                if _mt_drop:
                    selected_vars = [v for v in selected_vars if v not in _mt_drop]
                    if verbose:
                        logger.info(
                            "MRMR raw monotone-twin drop: removed %d raw decoy(s) that are pure "
                            "monotone re-encodings of a higher-relevance selected raw (|Spearman rho|"
                            ">=%.2f, rank-redundant): %s",
                            len(_mt_drop), _MONO_TWIN_RHO, [cols[v] for v in _mt_drop],
                        )
        except Exception as _exc_mt:
            logger.warning(
                "MRMR raw monotone-twin drop failed: %s; keeping the un-pruned support.",
                _exc_mt,
            )

    return selected_vars
