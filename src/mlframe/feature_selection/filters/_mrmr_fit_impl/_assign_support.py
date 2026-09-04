"""Split off ``mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core`` for the sub-split
that brings ``_fit_impl_core.py`` below the project's 1k-LOC module-size gate.

Holds ``_assign_support``: the "Assign support" section of ``MRMR._fit_impl`` -- everything from the
never-empty raw-representative re-attach through the final p>=n false-positive-control re-cap, i.e.
every post-screen support_ mutation pass (cluster-aggregate final exclusion, search-space restriction,
emit-both operand re-attach, C2 additive-fusion strip, usability-aware pure-form/raw retention,
post-retention raw-redundancy drop, raw-signal-retention augmentation, and the empty-support rescue).
Threads ``self`` plus every fit-body local this section reads as explicit keyword arguments (mirrors
the ``_finalise_fs_results`` carve-out's own pattern) rather than importing anything fresh, because
``_fit_impl`` itself resolves several helpers via LAZY imports inside its own body specifically to
avoid import cycles a fresh top-level import here would reintroduce.

``selected_vars`` is passed in by VALUE (its pre-block state) and never returned: every subsequent
consumer in ``_fit_impl`` (the "Report FS results" tail carved into ``_finalise_fs_results``) reads
the outcome via ``self.support_`` / ``self._engineered_recipes_`` / ``self.cached_MIs`` etc., which
this function sets directly -- ``selected_vars`` itself is not read again after this section returns.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ._helpers import _build_stability_replay_state, _engineered_recipe_name, _pgn_raw_budget
from ._assign_support_tail import _assign_support_tail

logger = logging.getLogger(__name__)


def _assign_support(
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
    fe_max_steps,
    cached_MIs,
    engineered_recipes,
    predictors,
    _eng_continuous_snapshot,
    selected_vars,
):
    """Assign MRMR's final raw ``support_`` (and its many post-screen adjustment passes) onto ``self``.

    See the module docstring for the full section this carves out and why ``selected_vars`` is
    accepted by value and never returned.
    """
    # ---------------------------------------------------------------------------------------------------------------

    # ``selected_vars`` holds integer column indices. Force int64 dtype so an
    # EMPTY selection (all signal folded into engineered recipes under the
    # full-mode default -> zero raw survivors) stays an integer index array.
    # ``np.array([])`` defaults to float64, and the ndarray transform path
    # (``X[:, support_]`` in _mrmr_validate_transform) then raises
    # ``IndexError: arrays used as indices must be of integer (or boolean)
    # type`` because a float array can't index. Integer dtype makes the empty
    # slice a valid no-op on both the DataFrame and the ndarray paths.
    #
    # NEVER-EMPTY RAW REPRESENTATIVE: when the ONLY confirmed feature(s) are engineered
    # recipes (their raw operands all judged redundant, so ``selected_vars`` is empty while
    # ``_engineered_recipes_`` is non-empty), the raw integer ``support_`` would be empty even though a
    # genuine signal-bearing feature WAS selected. That breaks any linear downstream that consumes the raw
    # ``support_`` (it sees zero columns) and the never-empty selection contract. Re-attach the single
    # highest-marginal-MI raw OPERAND of a surviving engineered feature as the cluster's raw stand-in -
    # mirrors ``reattach_raw_representative_after_aggregate_swap`` for the DCD aggregate case. One raw
    # column is added (the operand most relevant to y), never an unvalidated one, and the engineered
    # recipe still rides along via ``get_feature_names_out`` / ``transform``. Best-effort: any failure
    # leaves the empty support_ unchanged (no crash on a degenerate fit).
    # The conditional-redundancy sweep marks an INTENTIONALLY engineered-only support via
    # ``_redundancy_emptied_raw_``: every raw operand was FULLY subsumed by a surviving
    # engineered child (e.g. ``a`` + ``b`` both captured by ``div(neg(a),sqrt(b))`` for
    # ``y=a**2/b``), so re-attaching the "best" raw stand-in would resurrect exactly the
    # operand the n-invariant CMI verdict just dropped (observed at n=2000/5000: the sweep
    # dropped a+b, this block re-added a as the highest-marginal stand-in). When the empty
    # raw support is the redundancy sweep's deliberate outcome, the engineered recipes ARE
    # the complete feature set - skip the never-empty re-attach and let ``support_`` stay
    # empty (transform still emits the engineered columns). The re-attach remains active for
    # the genuine degenerate case (engineered-only with no redundancy verdict).
    if (not selected_vars) and getattr(self, "_engineered_recipes_", None) and not getattr(self, "_redundancy_emptied_raw_", False):
        try:
            from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _NE_TOK_SPLIT
            from ..info_theory import mi as _ne_mi
            _raw_names_ne = set(self.feature_names_in_)
            # Recipe -> column NAME. ``self._engineered_recipes_`` holds EngineeredRecipe
            # OBJECTS whose ``str()``/``repr()`` is the full dataclass repr, NOT the column
            # name - so ``str(r)`` neither matches ``cols`` nor is a clean token source.
            # Resolve the name from ``.name`` (the column the recipe materialises), falling
            # back to ``str(r)`` only for a legacy bare-string entry.
            _ne_recipe_name = _engineered_recipe_name
            # name -> index map built once (O(F)); reused below for ``_eng_survivor_cols`` too.
            # ``cols`` is not mutated for the remainder of ``_fit_impl`` past this point.
            _ne_cols_idx = {nm: i for i, nm in enumerate(cols)}
            _operand_idxs: set = set()
            for _r_obj in self._engineered_recipes_:
                for _tok in _NE_TOK_SPLIT.split(_ne_recipe_name(_r_obj)):
                    if not _tok:
                        continue
                    _base = _tok if _tok in _raw_names_ne else (_tok.split("__", 1)[0] if "__" in _tok else None)
                    if _base in _raw_names_ne:
                        _ne_ci = _ne_cols_idx.get(_base)
                        if _ne_ci is not None:
                            _operand_idxs.add(_ne_ci)
            # CONDITIONAL-REDUNDANCY GUARD on the re-attach (BUG1). The
            # operand picked below is the highest-MARGINAL-MI one, but a high marginal
            # does NOT mean it carries signal the engineered child lacks: a dominant
            # operand (``a`` in ``a**2/b``) has the largest marginal yet is FULLY
            # subsumed by the ``a**2/b`` ratio inside the surviving composite. Re-
            # attaching it re-introduces exactly the redundancy the campaign set out to
            # remove (observed at n=100k on the user fixture: the single full-target
            # composite ``add(mul(log(c),sin(d)),abs(div(sqr(a),abs(b))))`` rode as a
            # recipe -> ``selected_vars`` empty -> this block re-attached raw ``a``,
            # which the composite already captures). Restrict the candidate pool to
            # operands that carry a SIGNIFICANT INDEPENDENT RESIDUAL given the engineered
            # survivor(s), using the SAME n-invariant conditional-redundancy verdict as
            # the main drop. Only operands NOT judged subsumed are eligible; if every
            # operand is subsumed (the composite fully reconstructs y), leave the support
            # engineered-only - the recipe IS the complete feature set.
            _subsumed_operand_names: set = set()
            try:
                # emit_both keeps engineered operands; skip the subsumption restriction so the never-empty re-attach is not narrowed.
                if getattr(self, "redundancy_policy", "emit_both") != "drop":
                    raise RuntimeError("redundancy_policy=emit_both: skip subsumption restriction")
                from .._fe_raw_redundancy_drop import drop_redundant_raw_operands as _ne_drop
                _recipe_names = [_ne_recipe_name(r) for r in self._engineered_recipes_]
                _eng_survivor_cols = [_ne_cols_idx[_nm] for _nm in _recipe_names if _nm in _ne_cols_idx and _nm not in _raw_names_ne]
                if _eng_survivor_cols and _operand_idxs:
                    _trial_sel = sorted(set(_operand_idxs) | set(_eng_survivor_cols))
                    # name -> EngineeredRecipe so the verdict can isolate clean nested
                    # sub-expressions here too (BUG1 nested-operand consumer detection).
                    _ne_recipes = {_ne_recipe_name(_r): _r for _r in self._engineered_recipes_ if _ne_recipe_name(_r) is not None}
                    _, _ne_dropped = _ne_drop(
                        data=data, cols=cols, selected_cols_idx=_trial_sel,
                        raw_name_set=_raw_names_ne, y_binned=classes_y,
                        y_continuous=(y.values if hasattr(y, "values") else np.asarray(y)),
                        engineered_continuous=_eng_continuous_snapshot,
                        replayable_eng_names=set(_recipe_names),
                        recipes=_ne_recipes,
                        raw_X=X,
                        linear_usability_keep=bool(getattr(self, "fe_keep_linearly_usable_raw_operands", True)),
                        seed=int(getattr(self, "random_seed", 0) or 0), verbose=0,
                    )
                    _subsumed_operand_names = set(_ne_dropped or ())
            except Exception as exc:
                logger.debug("mrmr: subsumed-operand computation failed; falling back to MI-only pick (best-effort): %r", exc, exc_info=True)
                _subsumed_operand_names = set()  # best-effort: fall back to MI-only pick
            # C2 ADDITIVE-FUSION EXCLUSION: never re-attach a raw operand the
            # FE additive-fusion proposer already judged subsumed by the fused ``add(...)``
            # compound (``_raw_redundancy_dropped_``). The fused compound carries its additive
            # term, so resurrecting it as the never-empty stand-in re-injects the redundant
            # single-group fragment the fusion removed (the FUSION-blocked goal's leftover raw).
            _fused_dropped_ne = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
            _eligible_idxs = [_oi for _oi in _operand_idxs if cols[_oi] not in _subsumed_operand_names and cols[_oi] not in _fused_dropped_ne]
            if _eligible_idxs:
                _tgt_ne = np.asarray(target_indices, dtype=np.int64)
                _fn_ne = np.asarray(nbins, dtype=np.int64)
                _best_idx_ne, _best_rel_ne = -1, float("-inf")
                for _oi in sorted(_eligible_idxs):
                    try:
                        _rel_ne = float(_ne_mi(data, np.array([int(_oi)], dtype=np.int64), _tgt_ne, _fn_ne))
                    except Exception as exc:
                        logger.debug("mrmr: relevance computation failed for this non-engineered candidate; treating as zero (conservative): %r", exc, exc_info=True)
                        _rel_ne = 0.0
                    if _rel_ne > _best_rel_ne:
                        _best_rel_ne, _best_idx_ne = _rel_ne, int(_oi)
                if _best_idx_ne >= 0:
                    # ``_best_idx_ne`` is a COLS-space index (the augmented, categorize_dataset-reordered matrix that carries the injected target +
                    # engineered columns). ``support_`` must index ``feature_names_in_`` (raw user columns only), so remap the chosen operand by NAME -
                    # the same translation the main selection does at the ``selected_vars_names`` split. Assigning the raw cols-space index directly let an
                    # out-of-range index (>= n_features_in_) reach ``support_`` and crashed ``transform`` with IndexError when feature_names_in_ was narrower.
                    _operand_name_ne = cols[_best_idx_ne]
                    selected_vars = [list(self.feature_names_in_).index(_operand_name_ne)]
                    if verbose:
                        logger.info(
                            "MRMR never-empty raw representative: support_ would be empty (only engineered "
                            "feature(s) selected); re-attached raw operand %r (marginal MI %.4f) as the raw "
                            "stand-in (carries residual signal beyond the engineered child).",
                            _operand_name_ne, _best_rel_ne,
                        )
            elif _operand_idxs and _subsumed_operand_names:
                # EVERY engineered operand is conditionally subsumed by a surviving
                # engineered child - the engineered recipe(s) ARE the complete feature
                # set. Record the verdict so the DOWNSTREAM empty-raw rescue (the
                # ``else`` branch that tops up the support to ``min_features_fallback``
                # by marginal MI) does NOT resurrect a dropped operand. Without this the
                # rescue re-adds the highest-marginal operand (``a`` in the user's
                # ``a**2/b + log(c)sin(d)`` fixture, whose ``a**2/b`` is captured by the
                # composite), the BUG1 spurious-raw-kept regression - because the raw
                # operands were dropped by the EARLIER raw-retention pass, not the main
                # ``drop_redundant_raw_operands`` sweep, so neither
                # ``_raw_redundancy_dropped_`` nor ``_redundancy_emptied_raw_`` was set.
                # The ``elif`` at the rescue site keys on ``_redundancy_emptied_raw_`` and
                # the rescue / RFECV / augmentation pools all exclude
                # ``_raw_redundancy_dropped_``; populate both here so the engineered-only
                # support stands.
                self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | set(_subsumed_operand_names)
                self._redundancy_emptied_raw_ = True
                if verbose:
                    logger.info(
                        "MRMR never-empty raw representative: ALL %d engineered operand(s) are "
                        "conditionally subsumed by the surviving engineered child; leaving support "
                        "engineered-only (no spurious raw stand-in re-attached): %s",
                        len(_operand_idxs), sorted(_subsumed_operand_names),
                    )
        except Exception as _ne_exc:
            logger.warning("MRMR never-empty raw representative re-attach failed (%r); leaving support_ empty.", _ne_exc)

    # CLUSTER-AGGREGATE 'replace' FINAL EXCLUSION. Members folded into a denoised
    # MULTI-parent aggregate (``cluster_aggregate_mode='replace'`` -> ``_cluster_aggregate_removals_``,
    # or a DCD PC1/mean_z swap -> ``cluster_members_``) were dropped from ``selected_vars`` at the
    # replace step, but the many intervening raw-retention / masked-raw rescue / hinge / orth / pcr /
    # never-empty-representative / additional-RFECV passes can resurrect a removed member when it is an
    # OPERAND of a SURVIVING engineered child (e.g. ``add(refl0,sin(indep))`` keeps a private residual
    # given the aggregate). Several of those passes pre-date the cluster-aggregate feature and do not
    # consult ``_cluster_aggregate_removals_``, so rather than patch each call site we re-apply the
    # exclusion ONCE here - the single chokepoint right before ``support_`` is frozen, in
    # feature_names_in_ index space - guaranteeing a replaced member can never reach support_ /
    # get_feature_names_out regardless of which re-add path touched it. The denoised aggregate itself
    # (an engineered name in ``_engineered_recipes_``) is untouched.
    _ca_final_excl = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
    _cm_final = getattr(self, "cluster_members_", None)
    _raw_names_cmfinal = set(self.feature_names_in_)
    # Raw cluster representatives a DCD-aggregate-anchor swap would otherwise strip: force-kept / force-ADDED
    # below so every collapsed cluster retains >=1 raw column. Initialised at function-body level (NOT inside
    # the ``isinstance(_cm_final, dict)`` block) because it is referenced unconditionally further down - a
    # fit whose ``cluster_members_`` is not a dict (e.g. dcd_enable=False) must not hit an UnboundLocalError.
    _ca_keep_raw: set = set()
    if isinstance(_cm_final, dict):
        # ``cluster_members_`` is populated by mechanisms with DIFFERENT final-exclusion semantics:
        #   * an ENGINEERED-anchor cluster (DCD PC1/mean_z swap whose anchor is the denoised aggregate, a
        #     name NOT in feature_names_in_): the aggregate survives, so its raw members are stripped from
        #     any raw support a downstream pass resurrected.
        #   * a pure RAW redundancy cluster (exact-duplicate / collinear / DCD decoy pair, ALL names raw):
        #     exactly ONE representative must survive. The cluster dict's anchor/member DIRECTION is NOT
        #     reliable for which to keep (e.g. ``{'collinear_b': ['good_b']}`` labels the genuine ``good_b``
        #     as a member), so keep the highest cached-MI(.,y) column of the cluster and strip the rest.
        #     This de-duplicates (RC2 exact-duplicate / realistic-mixed-degenerate -> keep ``good_a``,
        #     ``good_b``) AND prunes genuine decoys (layer6 DCD second-decoy -> keep the strong driver).
        # Mixed clusters (raw anchor + pseudo-remix/engineered member) fall to the pseudo-remix-protected
        # member strip below.
        _nm2col_cm = {c: i for i, c in enumerate(cols)}
        # Use the in-scope LOCAL cached_MIs (populated by the screen, same dict used at the other read
        # sites) - self.cached_MIs is only assigned near the end of _fit_impl, so on a FRESH fit
        # hasattr(self,...) is False and this degraded to {} -> every rep tiebreak collapsed to 0.0.
        _cached_cm = cached_MIs if ("cached_MIs" in dir() and isinstance(cached_MIs, dict)) else {}
        _name2inidx_cm = {c: i for i, c in enumerate(self.feature_names_in_)}
        # Names ALREADY in selected_vars (raw, in feature_names_in_ index space). The greedy
        # screen / retention passes have already chosen these as the cluster's surviving
        # representative(s); the pure-raw-cluster strip below must KEEP one of them rather than
        # silently swapping in an unselected member.
        _sel_names_cm = {self.feature_names_in_[int(v)] for v in selected_vars if int(v) < len(self.feature_names_in_)}
        def _cm_mi(_nm):
            """Screening marginal MI(name, y) for a cluster-member column name, used to pick the strongest representative when collapsing a pure-raw redundancy cluster (0.0 when the column has no cached score)."""
            _ci = _nm2col_cm.get(_nm)
            return float(_cached_cm.get((_ci,), 0.0)) if _ci is not None else 0.0

        for _anchor, _members in _cm_final.items():
            _a = str(_anchor)
            _mlist = [str(_m) for _m in (_members or [])] if isinstance(_members, (list, tuple, set)) else []
            _group = [_a, *_mlist]
            if all(_nm in _raw_names_cmfinal for _nm in _group):
                # pure raw cluster - keep the single strongest representative, strip the rest.
                # KEEP-ONE-SELECTED-RAW: the cached-MI lookup the rep tiebreak relies
                # on is often a miss for these members (``cached_MIs`` is keyed on the screening
                # cols-space and a cluster member may never have been scored there), collapsing every
                # member's relevance to 0.0 -> the rep degenerates to the LOWEST feature-index member.
                # When that lowest-index member is NOT the one the greedy screen actually selected,
                # the cluster's genuine selected representative (which IS in ``selected_vars``) gets
                # stripped and the whole latent block vanishes from support_ (embedding cross-terms
                # layer20: 12-member e1 cluster, only the high-MI anchor ``e1_17`` was selected, yet
                # the rep collapsed to ``e1_1`` and e1 dropped entirely). PRINCIPLE: a member already
                # chosen by the screen is the de-facto representative - prefer it. Restrict the rep
                # candidate pool to the cluster members present in ``selected_vars`` when any are;
                # only fall back to the MI/index tiebreak over the whole group when none was selected.
                if len(_group) >= 2:
                    _rep_pool = [_nm for _nm in _group if _nm in _sel_names_cm]
                    if not _rep_pool:
                        _rep_pool = _group
                    _rep = min(_rep_pool, key=lambda _nm: (-_cm_mi(_nm), _name2inidx_cm.get(_nm, 1 << 30)))
                    _ca_final_excl.update(_nm for _nm in _group if _nm != _rep)
                    # KEEP-ONE-RAW for pure-raw PRUNED clusters (no denoised aggregate): when a
                    # within-pack SU cluster is merely pool-pruned (size below the swap threshold, so
                    # no aggregate column is ever built) AND its screen-selected representative was
                    # later dropped (e.g. a second screen pass re-prunes the pack and the anchor falls
                    # out of selected_vars), NONE of the group survives - the latent vanishes from
                    # support_ entirely and the RFECV rescue pool excludes every cluster member, so it
                    # is unrecoverable (scenario-A sensor mesh: L1 pack pruned, AUC -0.08). Force-keep
                    # the chosen representative exactly like the engineered-anchor branch below, so every
                    # collapsed cluster retains >=1 raw column. No support growth: this re-adds the SINGLE
                    # representative of a cluster that would otherwise contribute zero columns.
                    if _rep not in _sel_names_cm:
                        _ca_keep_raw.add(_rep)
            elif _a not in _raw_names_cmfinal:
                # engineered/aggregate anchor (DCD PC1/mean_z swap) - strip its (raw) members; the
                # aggregate itself survives.
                #
                # KEEP-ONE-RAW-REPRESENTATIVE: a DCD denoised-aggregate swap collapses an
                # entire raw cluster into a single engineered column and prunes every raw member. When
                # the aggregate is the cluster's ONLY survivor, the latent block has no RAW column in
                # ``support_`` at all - any downstream consumer that reads the raw support names (a
                # linear model fed the raw matrix, a feature-importance report, the layer20 embedding
                # cross-terms contract) sees the whole block as dropped even though it was merely
                # denoised. PRINCIPLE: the engineered aggregate is a SUPPLEMENT, not a replacement for
                # the cluster's presence - always leave at least one genuine raw representative of the
                # cluster alive. Keep the strongest raw member (highest cached MI, lowest-index
                # tiebreak) and strip the rest; the kept member is force-added to ``selected_vars``
                # below so it survives even if no raw member reached the support chokepoint.
                _raw_mem = [_m for _m in _mlist if _m in _raw_names_cmfinal]
                if _raw_mem:
                    _agg_rep = min(_raw_mem, key=lambda _nm: (-_cm_mi(_nm), _name2inidx_cm.get(_nm, 1 << 30)))
                    _ca_keep_raw.add(_agg_rep)
                    _ca_final_excl.update(_m for _m in _mlist if _m != _agg_rep)
                else:
                    _ca_final_excl.update(_mlist)
            else:
                # raw anchor + engineered/pseudo member(s) - strip only the non-raw members (pseudo-remix
                # protection below keeps a raw operand the cluster pairs with a pseudo-remix built from it).
                _ca_final_excl.update(_m for _m in _mlist if _m not in _raw_names_cmfinal)
    # PSEUDO-REMIX SELF-SOURCE PROTECTION. A conditional-gate / binned-numeric-agg /
    # row-argmax anchor (``gate_mask__a__b`` / ``binagg_mean(d|qbin(a))`` / ``argmax__a__b``) is a
    # LOSSY threshold/binning RE-MIX of its raw source(s): it cannot carry a raw operand's private
    # LINEAR term (a binary gate of ``a`` does not span ``10*a``). When the clustering folds a RAW
    # column into a cluster ANCHORED by such a pseudo-remix BUILT FROM that raw and strips the raw as
    # a "member", a genuine private term is lost (test_private_raw_a_kept: raw ``a`` with a dominant
    # ``10*a`` term clustered under ``gate_mask__a__b`` and dropped). Mirror the redundancy gate's
    # ``_is_pseudo_remix_child`` exclusion here: never strip a RAW column that the cluster pairs with
    # a pseudo-remix BUILT FROM that raw, in EITHER direction -
    #   (A) pseudo-remix ANCHOR + raw-source MEMBER  (``gate_mask__a__b`` anchors raw ``a``); or
    #   (B) raw ANCHOR + pseudo-remix MEMBER of it    (``x2`` anchors ``gate_mask__x2__x1``).
    # The lossy gate/binagg/argmax cannot carry the raw's continuous value a LINEAR downstream needs
    # (measured: a 5-class LogReg macro-F1 0.62 when x2 was stripped as such a cluster anchor vs >0.70
    # protected; and the test_private_raw_a_kept ``10*a`` case for direction A). Engineered members +
    # genuine (non-pseudo) aggregate members are untouched -> byte-identical when no such pairing exists.
    if _ca_final_excl and isinstance(_cm_final, dict):
        from .._fe_raw_redundancy_drop import _is_pseudo_remix_child, _PSEUDO_SRC_SPLIT
        _raw_names_ca = set(self.feature_names_in_)
        _protect_ca = set()
        for _anchor, _members in _cm_final.items():
            _a = str(_anchor)
            _mlist = [str(_m) for _m in (_members or [])]
            # (A) pseudo-remix anchor -> protect any raw member that is one of its sources.
            if _is_pseudo_remix_child(_a):
                _anchor_raw_srcs = {t for t in _PSEUDO_SRC_SPLIT.split(_a) if t in _raw_names_ca}
                for _m in _mlist:
                    if _m in _raw_names_ca and _m in _anchor_raw_srcs:
                        _protect_ca.add(_m)
            # (B) raw anchor -> protect it when a member is a pseudo-remix built from that raw.
            if _a in _raw_names_ca:
                for _m in _mlist:
                    if _is_pseudo_remix_child(_m) and _a in set(_PSEUDO_SRC_SPLIT.split(_m)):
                        _protect_ca.add(_a)
                        break
        if _protect_ca:
            _ca_final_excl -= _protect_ca
    # KEEP-ONE-RAW-REPRESENTATIVE force-keep: the designated raw representative of each
    # DCD-aggregate-collapsed cluster must never be stripped, even if another cluster's strip set or
    # a redundancy pass nominated it. Remove it from the exclusion set first.
    if _ca_keep_raw:
        _ca_final_excl -= _ca_keep_raw
    if _ca_final_excl and selected_vars:
        _fni = self.feature_names_in_
        _pre_n = len(selected_vars)
        selected_vars = [v for v in selected_vars if _fni[v] not in _ca_final_excl]
        if verbose and len(selected_vars) != _pre_n:
            logger.info(
                "MRMR cluster-aggregate 'replace': re-stripped %d cluster member(s) a downstream "
                "retention/rescue pass had resurrected; only the denoised aggregate survives.",
                _pre_n - len(selected_vars),
            )
    # Force-ADD the kept raw representative of each aggregate-collapsed cluster when no raw member of
    # that cluster reached the support chokepoint (the swap pruned them all). Guarantees every
    # denoised cluster keeps >=1 genuine raw column in ``support_`` alongside its engineered aggregate.
    if _ca_keep_raw:
        _name2inidx_add = {c: i for i, c in enumerate(self.feature_names_in_)}
        _sel_set = set(int(v) for v in selected_vars)
        for _kr in _ca_keep_raw:
            _ki = _name2inidx_add.get(_kr)
            if _ki is not None and _ki not in _sel_set:
                selected_vars.append(_ki)
                _sel_set.add(_ki)

    # SEARCH-SPACE RESTRICTION FINAL ENFORCEMENT. When the caller pins the candidate
    # pool via ``factors_names_to_use`` / ``factors_to_use``, the SCREEN honours it, but the many
    # post-screen raw-retention / masked-raw rescue / hinge / orth / pcr / never-empty / count-floor
    # re-add passes do NOT all consult the restriction, so a forbidden raw column (e.g. ``good2`` when
    # the pool is pinned to ``["good1"]``) leaks into ``support_`` - and because the in-object fit-skip
    # / _FIT_CACHE replay a stale selection unless every param change invalidates it, the bug also shows
    # as a stale-replay regression. Enforce the restriction ONCE at the support chokepoint (raw indices
    # into feature_names_in_): a raw column the user excluded can never reach support_ regardless of which
    # re-add path admitted it. Engineered survivors (in ``_engineered_recipes_``) are untouched - they are
    # built only from allowed raws by the screen, which already respects the restriction.
    _allowed_raw_idx = None
    _fn_restrict = getattr(self, "factors_names_to_use", None)
    _fi_restrict = getattr(self, "factors_to_use", None)
    if _fn_restrict:
        _allowed_names = set(_fn_restrict)
        _allowed_raw_idx = {_j for _j, _nm in enumerate(self.feature_names_in_) if _nm in _allowed_names}
    elif _fi_restrict is not None:
        _allowed_raw_idx = set(int(_j) for _j in _fi_restrict)
    if _allowed_raw_idx is not None and selected_vars:
        _pre_r = len(selected_vars)
        selected_vars = [v for v in selected_vars if int(v) in _allowed_raw_idx]
        if verbose and len(selected_vars) != _pre_r:
            logger.info(
                "MRMR: dropped %d raw feature(s) outside the pinned factors_names_to_use / "
                "factors_to_use search space that a downstream re-add pass had admitted.",
                _pre_r - len(selected_vars),
            )

    # P>=N FP-CONTROL TOTAL CAP. In the p>>n regime some pure-noise column WILL correlate with y by chance, and the post-screen
    # retention / rescue passes can admit a few of them (measured: 51 raws at p=150, 103 at p=300 - 1-3 over the multiple-comparison
    # ceiling). When p >= n, cap the total selected raw set at ``max(20, p//3)`` features chosen by descending relevance MI(X_j, y),
    # mirroring the RFECV ``p_ge_n_fp_control_cap``. Confined to p >= n so the well-powered p<n path is byte-unchanged. Engineered
    # survivors are counted toward the cap (they reach the output too) but never the dropped tail - only raw ``selected_vars`` is trimmed.
    _pgn_n = int(data.shape[0]) if "data" in dir() else 0
    _pgn_p = int(getattr(self, "n_features_in_", 0) or 0)
    if _pgn_p > 0 and _pgn_n > 0 and _pgn_p >= _pgn_n and selected_vars:
        _pgn_ceiling = max(20, _pgn_p // 3)
        # ``n_engineered_out`` is not yet bound at this point (assigned further below near ``n_features_``); read the
        # engineered count straight off ``self._engineered_recipes_`` (populated by the main sweep above) so engineered
        # survivors are actually charged against the p>=n ceiling instead of silently degrading the count to 0.
        _pgn_eng = len(getattr(self, "_engineered_recipes_", None) or [])
        _pgn_budget = _pgn_raw_budget(_pgn_ceiling, _pgn_eng)
        if len(selected_vars) > _pgn_budget:
            # LOCAL cached_MIs (see the cluster-rep note above): self.cached_MIs is unset until the end of
            # _fit_impl, so on a fresh fit this read degraded to {} and the p>=n cap sort collapsed to index order.
            _pgn_cached = cached_MIs if ("cached_MIs" in dir() and isinstance(cached_MIs, dict)) else {}
            _pgn_n2ci = {c: i for i, c in enumerate(cols)} if "cols" in dir() else {}
            _fni_pgn = self.feature_names_in_

            def _pgn_rel(_v):
                """Screening marginal MI(v, y) for raw index ``_v``, used to rank raw features for the p>=n false-positive-control cap (0.0 when unresolvable)."""
                _nm = _fni_pgn[_v] if _v < len(_fni_pgn) else None
                _ci = _pgn_n2ci.get(_nm)
                return float(_pgn_cached.get((_ci,), 0.0)) if _ci is not None else 0.0
            # Descending relevance, stable secondary key on the raw index so ties are column-order invariant.
            selected_vars = [v for v in sorted(selected_vars, key=lambda v: (-_pgn_rel(v), int(v)))][:_pgn_budget]
            if verbose:
                logger.info(
                    "MRMR p>=n FP-control: capped raw support to top-%d by relevance (p=%d >= n=%d, ceiling=%d, engineered=%d).",
                    _pgn_budget, _pgn_p, _pgn_n, _pgn_ceiling, _pgn_eng,
                )

    # EMIT-BOTH OPERAND RE-ATTACH. A feature selector must not destroy linearly-usable raw signal: for every SELECTED engineered feature, surface its raw operand
    # columns (parsed from the recipe ``src_names`` or name tokens). Re-attach only operands that themselves carry MARGINAL signal toward y (a within-data
    # permutation-significance test, p<alpha): a SIGNAL operand of a selected engineered feature is kept (the linear-usability win), but a NOISE operand fused into a
    # composite (e.g. ``noise_3`` inside ``sub(...,prewarp(noise_3))``) does NOT clear its null and is NOT re-attached -> FS still rejects noise. Bounded to operands
    # of SELECTED engineered features, in feature_names_in_, not already selected, inside the pinned search space (``_allowed_raw_idx``).
    if getattr(self, "redundancy_policy", "emit_both") != "drop" and selected_vars:
        try:
            from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _EB_TOK_SPLIT
            from ..permutation import mi_direct as _eb_mi_direct
            _eb_raw_names = set(self.feature_names_in_)
            _eb_sel_set = set(int(v) for v in selected_vars)
            _eb_name_to_in = {nm: i for i, nm in enumerate(self.feature_names_in_)}
            _eb_cols_idx = {nm: i for i, nm in enumerate(cols)}
            _eb_recipes = {getattr(_r, "name", None): _r for _r in (getattr(self, "_engineered_recipes_", None) or [])}
            _eb_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
            _eb_qdtype = getattr(self, "quantization_dtype", np.int32)
            _eb_operands: list[str] = []
            for _enm, _erec in _eb_recipes.items():
                if _enm is None or _enm in _eb_raw_names or _enm not in _eb_cols_idx:
                    continue
                if _eb_cols_idx[_enm] not in _eb_sel_set:
                    continue  # only SELECTED engineered features
                _src = getattr(_erec, "src_names", None)
                _eb_toks = list(_src) if _src else [t for t in _EB_TOK_SPLIT.split(str(_enm)) if t]
                for _t in _eb_toks:
                    _base = _t if _t in _eb_raw_names else (_t.split("__", 1)[0] if "__" in _t else None)
                    if _base is not None and _base in _eb_raw_names and _base not in _eb_operands:
                        _eb_operands.append(_base)

            def _eb_operand_is_signal(_cols_i):
                """Permutation-significance test (32 permutations) for a raw operand of a selected engineered feature; True when it clears its own null (p<alpha) or the MI estimator errors, gating the emit-both re-attach so a noise operand fused into a composite is not resurrected."""
                try:
                    _r = _eb_mi_direct(data, x=np.array([int(_cols_i)], dtype=np.int64), y=target_indices,  # type: ignore[arg-type]
                                       factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                                       return_null_mean=True, parallelism="none", dtype=_eb_qdtype, prefer_gpu=False)
                    return float(_r[3]) < _eb_alpha  # p-value below alpha -> genuine marginal signal
                except Exception as e:
                    logger.debug("Marginal-MI significance probe failed (%s: %s) -- not silently dropping a possibly-genuine operand", type(e).__name__, e)
                    return True  # estimator error -> do not silently drop a possibly-genuine operand
            _eb_added = []
            for _op in _eb_operands:
                _idx = _eb_name_to_in.get(_op)
                if _idx is None or int(_idx) in _eb_sel_set:
                    continue
                if _allowed_raw_idx is not None and int(_idx) not in _allowed_raw_idx:
                    continue
                _eb_ci = _eb_cols_idx.get(_op)
                if _eb_ci is None or not _eb_operand_is_signal(_eb_ci):
                    continue  # noise operand of a composite -> FS keeps rejecting it
                selected_vars.append(int(_idx))
                _eb_sel_set.add(int(_idx))
                _eb_added.append(_op)
            if _eb_added and verbose:
                logger.info("MRMR emit_both operand re-attach: added %d signal raw operand(s) of selected engineered features: %s", len(_eb_added), _eb_added)
        except Exception as _eb_exc:
            logger.debug("MRMR emit_both operand re-attach skipped (%s: %s).", type(_eb_exc).__name__, _eb_exc)

    # C2 ADDITIVE-FUSION FINAL RAW STRIP. Raw operands the FE additive-fusion
    # proposer verified the fused ``add(...)`` compound fully captures (``_fused_subsumed_raws_``,
    # set via the production keep-probe against the WHOLE compound) must not survive in the raw
    # support, no matter which downstream retention / rescue / re-attach pass re-added them: those
    # passes condition a raw on the CLEAN nested sub-expression, which on a corrupted a/b half does
    # NOT capture the raw and so KEEPS it, whereas the fused compound DOES - this strip applies the
    # stronger whole-compound verdict. Only strips when the fused compound itself survives as a
    # recipe (so the additive term it carries is actually present); byte-identical (empty set) when
    # no fusion fired.
    _fused_subsumed = set(getattr(self, "_fused_subsumed_raws_", None) or set())
    if _fused_subsumed:
        # NOTE: ``self._engineered_recipes_`` is not populated until later in this function (the
        # UAED-trim / group-drop reassignments below), so reading it here (as the block previously
        # did) silently sees its initial ``[]`` default and the whole strip below no-ops, letting a
        # provably-subsumed raw (``_fused_subsumed``) ride into ``support_`` beside the compound that
        # captures it. ``selected_vars`` is not a reliable substitute either -- the fused compound can
        # legitimately not have reached ``selected_vars`` yet at this exact point in a multi-step fit
        # (it is registered in the recipe dict the moment the fusion is admitted, but folded into
        # ``selected_vars`` on the SAME step's re-screen, which this final-assembly code can run ahead
        # of on some step orderings). ``engineered_recipes`` (the local name -> recipe dict this
        # function has threaded throughout) is updated the instant a fusion is admitted and is the
        # authoritative "does this compound exist at all" source regardless of screen timing; a
        # ``_fused_subsumed`` entry only exists when its compound's OWN admission already passed the
        # production keep-probe, so trusting the recipe dict here (not gating on selection) does not
        # widen the strip's blast radius beyond what ``_fused_subsumed`` already vetted.
        _surv_eng = set(engineered_recipes.keys()) if isinstance(engineered_recipes, dict) else set()
        # Only strip a raw when a SURVIVING engineered compound actually references it (carries its
        # additive term) - otherwise leave it (the fusion that subsumed it did not survive).
        import re as _re_fsr
        _fsr_tok = _re_fsr.compile(r"[^A-Za-z0-9_]+")
        _covered: set = set()
        for _en in _surv_eng:
            for _t in _fsr_tok.split(str(_en) or ""):
                if not _t:
                    continue
                _base = _t if _t in set(self.feature_names_in_) else (
                    _t.split("__", 1)[0] if "__" in _t and _t.split("__", 1)[0] in set(self.feature_names_in_) else None)
                if _base is not None:
                    _covered.add(_base)
        _strip = _fused_subsumed & _covered
        if _strip:
            selected_vars = [v for v in selected_vars if not (0 <= int(v) < len(self.feature_names_in_) and self.feature_names_in_[int(v)] in _strip)]
            if verbose:
                logger.info(
                    "MRMR C2 additive-fusion: stripped %d raw operand(s) the fused compound fully "
                    "captures from the final raw support: %s", len(_strip), sorted(_strip),
                )

    self.support_ = np.array(selected_vars, dtype=np.int64)

    # USABILITY-AWARE MULTI-LIST POST-PASS. ``support_`` above is the pure-MI selection
    # (the nonlinear / tree list, byte-identical to today). When ``usability_aware_lists`` is on AND
    # a continuous target is available, ALSO produce a linear-downstream list (``support_linear_``)
    # and a blended universal list (``support_universal_``) - each a replayable candidate list -
    # WITHOUT touching ``support_``. Fully guarded: a degenerate pool / non-numeric target / row
    # mismatch leaves the extra lists ``None`` and never breaks the fit. ``support_nonlinear_`` is
    # always set as the alias of ``support_`` so downstream routing has a stable name to read.
    try:
        if getattr(self, "usability_aware_lists", False):
            from .._usability_lists import build_usability_lists
            build_usability_lists(self, X=X, y_cont=getattr(self, "_fe_prewarp_y_continuous_", None))
        else:
            self.support_nonlinear_ = self.support_
            self.support_linear_ = None
            self.support_universal_ = None
    except Exception as _usability_exc:  # never let the optional second list break a fit
        self.support_nonlinear_ = getattr(self, "support_", None)
        self.support_linear_ = None
        self.support_universal_ = None
        logger.debug("Usability-aware multi-list post-pass skipped (%s: %s).", type(_usability_exc).__name__, _usability_exc)

    # SELECTION-STABILITY REPLAY STATE (backlog W3). Store a compact slice of the
    # already-discretised screening matrix ``data`` + the target codes + the per-column selection
    # outcome so ``MRMR.selection_stability_report(n_boot=K)`` can recompute per-feature selection-
    # frequency by REPLAY (K cheap marginal-MI sweeps over the frozen bins) without refitting MRMR -
    # the #15 "replay not refit" trick applied to a user-facing confidence readout. Subsample rows to
    # cap the stored footprint (8GB-shared box); the bins are frozen so resampling rows is leak-free.
    try:
        _build_stability_replay_state(
            self, data=data, cols=cols, nbins=nbins,
            target_indices=target_indices, selected_vars=selected_vars,
            engineered_recipes=engineered_recipes,
        )
    except Exception as _stab_exc:  # never let the diagnostic accessor break a fit
        self._stability_replay_state_ = None
        logger.debug("Stability replay-state capture skipped (%s: %s).", type(_stab_exc).__name__, _stab_exc)

    # ROSTER RECONCILIATION: the per-stage engineered rosters (``hybrid_orth_features_``, ``_adaptive_fourier_features_``, the Layer-33/34/37/38/87+ family lists) are
    # populated as each FE stage APPENDS its columns, but the MRMR screen / accuracy gate / dedup then drop a subset before support is finalised. ``self._engineered_features_`` is the
    # authoritative set of engineered columns that actually survived into the output (reachable via ``get_feature_names_out``). Intersect every roster with it so a column the screen
    # dropped (and the adaptive-protection block did NOT re-add) no longer leaks into the user-facing roster. Runs AFTER the additional_rfecv rescue (which reads the FULL rosters to
    # exclude engineered columns from its raw-only rescue pool); the rescue never adds engineered columns, so ``_engineered_features_`` is final here. Order-preserving per roster.
    # Snapshot the PRE-intersection roster first: once the loop below keeps only survivors there is no way left
    # to distinguish "this FE stage never fired" from "it fired and its column lost the greedy to an equivalent
    # column from a sibling family" -- a distinction that otherwise costs a manual bisect to recover.
    self.hybrid_orth_candidates_ = list(
        dict.fromkeys(list(getattr(self, "hybrid_orth_candidates_", None) or []) + list(getattr(self, "hybrid_orth_features_", None) or []))
    )
    _surviving_eng = set(self._engineered_features_ or [])
    for _roster_attr in (
        "hybrid_orth_features_", "_adaptive_fourier_features_", "mi_greedy_features_",
        "kfold_te_features_", "count_encoding_features_", "frequency_encoding_features_",
        "cat_num_interaction_features_", "missingness_indicator_features_",
        "missingness_count_features_", "missingness_pattern_features_",
        "pairwise_ratio_features_", "pairwise_log_ratio_features_",
        "grouped_delta_features_", "lagged_diff_features_", "grouped_agg_features_",
        "composite_group_agg_features_", "grouped_quantile_features_",
        "cat_pair_features_", "cat_triple_features_", "numeric_decompose_features_",
        "modular_features_", "group_distance_features_", "rare_category_features_",
        "conditional_residual_features_", "conditional_dispersion_features_",
        "wavelet_features_",
        "rankgauss_features_", "temporal_agg_features_",
    ):
        _roster = getattr(self, _roster_attr, None)
        if _roster:
            setattr(self, _roster_attr, [c for c in _roster if c in _surviving_eng])

    # Always store ``cached_MIs`` - the empty-support fallback at the bottom
    # of this function reads ``self.cached_MIs`` to rank by raw MI(X_j, y), so
    # the attribute should exist regardless of ``retain_artifacts``. Cheap (a
    # dict of tuple->float; bounded by the screen's candidate pool).
    self.cached_MIs = cached_MIs

    # iter66: artifact retention for cross-selector reuse (off by default).
    # Captured at the cols-space stage so ``data`` / ``cols`` / ``nbins`` are
    # the active matrices the screen actually consumed; the export dict is
    # axis-aligned to the original ``feature_names_in_`` for the downstream
    # consumer's convenience.
    if getattr(self, "retain_artifacts", False):
        try:
            from .._mrmr_artifacts import compute_mrmr_artifacts
            self._artifacts_ = compute_mrmr_artifacts(
                data=data,
                cols=list(cols),
                nbins=nbins,
                target_indices=target_indices,
                cached_MIs=cached_MIs,
                feature_names_in=list(self.feature_names_in_),
                support_original=self.support_,
                retain_bins=bool(getattr(self, "retain_bins", True)),
                dtype=self.quantization_dtype,
            )
        except Exception as _exc:
            logger.warning(
                "MRMR.retain_artifacts: capture failed (%s); export_artifacts() will raise. " "Cause: %s",
                type(_exc).__name__,
                _exc,
            )
            self._artifacts_ = None
    # Populate ``mrmr_gains_``
    # so the documented ``uaed_auto_size=True`` post-fit elbow trim at
    # line 1020+ actually fires. Pre-fix the comment claimed
    # "Wave-7 audit landed this trace" but no code ever assigned the
    # attribute - ``getattr(self, "mrmr_gains_", [])`` defaulted to
    # empty, ``gains.size >= 3`` was False, the UAED block was
    # guaranteed dead code. ``MRMR(uaed_auto_size=True)`` silently
    # returned the full screen output regardless. Restore the
    # advertised behaviour: store per-selection-round gains in
    # screening order, aligned with the predictor log.
    try:
        self.mrmr_gains_ = np.asarray(
            [float(p.get("gain", 0.0)) for p in (predictors or [])],
            dtype=np.float64,
        )
    except Exception as exc:
        logger.debug("mrmr: mrmr_gains_ computation failed; using an empty array: %r", exc, exc_info=True)
        self.mrmr_gains_ = np.array([], dtype=np.float64)
    # Layer 54: stash the greedy predictor log on ``self`` so the FE
    # provenance helper can map engineered feature names back to their
    # support_rank / mrmr_gain entries. ``predictors`` is a list of
    # ``{"name", "indices", "gain", ...}`` dicts in selection order.
    # Light copy (per-entry shallow) to dodge accidental downstream
    # mutation of the screen's working list; ``indices`` is captured as
    # a tuple to keep the entry pickle-safe across processes.
    try:
        self._predictors_log_ = tuple(
            {
                "name": p.get("name"),
                "gain": float(p.get("gain", 0.0)),
                "indices": tuple(p.get("indices", ()) or ()),
            }
            for p in (predictors or [])
        )
    except Exception as exc:
        logger.debug("mrmr: predictors-log capture failed; using an empty tuple: %r", exc, exc_info=True)
        self._predictors_log_ = ()
    self.fallback_used_ = False
    self.fallback_metadata_ = None

    # USABILITY-AWARE PURE-FORM RETENTION. On an ADDITIVE target whose terms share
    # operands, the MI greedy keeps a high-MI CROSS-MIX feature and drops the pure single-pair forms
    # (``a**2/b`` / ``log(c)*sin(d)``) as conditionally redundant - the right call for a TREE model but
    # not for the LINEAR/additive downstream, which needs the clean pure form the lossy cross-mix cannot
    # replace. Re-attach a pure single-pair engineered form whenever a CROSS-VALIDATED linear wrapper
    # confirms it lowers the linear CV-MAE on top of the current selection AND the pair is not already
    # represented by a pure (<=2-operand) selected feature. Purely ADDITIVE (nothing MI-selected is
    # removed; support_ untouched) and no-op when the pure form adds no linear value -> byte-identical
    # there. Only when FE is enabled (fe_max_steps>0); skipped on the fe-disabled raw-only path.
    # Names of engineered survivors RE-ATTACHED by the retention passes below (AFTER the main raw-vs-
    # engineered redundancy sweep). The post-retention drop only re-litigates raws against THESE - the
    # main sweep already vetted raws against survivors it could see (so a genuine pair-interaction operand
    # the main sweep kept is not re-dropped by the stricter post-retention margin).
    _retention_added_eng_names: set = set()
    _assign_support_tail(
        self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        fe_max_steps=fe_max_steps,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _allowed_raw_idx=_allowed_raw_idx,
        _retention_added_eng_names=_retention_added_eng_names,
    )
