"""Sibling of ``_assign_support.py`` (the sub-split of ``_fit_impl_core.py``'s "Assign support" section,
itself further split for the 1k-LOC module-size gate).

Holds ``_assign_support_tail``: the second half of the post-screen support_ mutation passes --
the fe_max_steps>0-gated usability-aware pure-form/raw retention, post-retention raw-redundancy
drop, raw-signal-retention augmentation, and the empty-support rescue / final p>=n re-cap.
Called from ``_assign_support`` as a plain trailing statement (no return value consumed, mirrors
the parent's own contract with ITS caller) -- all state is set directly on ``self`` (``self.support_``,
``self.n_features_``, ``self._engineered_recipes_``, etc.).

``_allowed_raw_idx`` and ``_retention_added_eng_names`` are threaded in explicitly: both are locals
the parent computes/initialises BEFORE this section (search-space restriction, retention roster)
that this section reads -- confirmed via ``pyutilz.dev.freevar_analysis`` on the exact line range
this module carves out, not by inspection alone.
"""

from __future__ import annotations

import logging

import numpy as np

from ._helpers import _pgn_raw_budget

logger = logging.getLogger(__name__)


def _assign_support_tail(
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
    _eng_continuous_snapshot,
    selected_vars,
    _allowed_raw_idx,
    _retention_added_eng_names,
):
    """Run the second half of the post-screen support_ mutation passes.

    See the module docstring for the full section this carves out.
    """
    if fe_max_steps > 0:
        # SHARED RETENTION PREP: both retain_usable_pure_forms and retain_usable_raw_columns below
        # independently rebuild the same numeric-dtype base_names filter, std-trim-to-max_base_features,
        # and (same seed) row subsample from this SAME (X, y_cont) - computed once here and passed to
        # both so the identical-seed X.iloc[_idx] draw is materialized once, not twice, per fit.
        _retention_prep_cache = None
        try:
            import pandas as _ret_pd
            from .._fe_pure_form_retention import _retention_prep as _ret_prep_fn

            _ret_y_prep = getattr(self, "_fe_prewarp_y_continuous_", None)
            if isinstance(X, _ret_pd.DataFrame) and _ret_y_prep is not None:
                _retention_prep_cache = _ret_prep_fn(
                    self, X, _ret_y_prep, seed=int(getattr(self, "random_seed", 0) or 0),
                )
        except Exception as exc:
            logger.debug("mrmr: retention-prep cache build failed; pure-form retention will recompute per-call: %r", exc, exc_info=True)
            _retention_prep_cache = None
        try:
            from .._fe_pure_form_retention import retain_usable_pure_forms

            _retain_extra = retain_usable_pure_forms(
                self, X, getattr(self, "_fe_prewarp_y_continuous_", None),
                seed=int(getattr(self, "random_seed", 0) or 0), verbose=verbose,
                _prep=_retention_prep_cache,
            )
            # ENGINEERED-SUBSUMPTION GUARD. The pure-form retention runs AFTER the post-FE
            # engineered-vs-engineered CMI redundancy gate, so a re-attached pure form is never tested
            # against the engineered survivors admitted BEFORE retention. When an incumbent survivor is a
            # FUSED compound that already carries BOTH additive halves of the target (the canonical
            # ``add(neg(mul(sqr(a),reciproc(b))),neg(mul(log(c),sin(d))))`` for y=a**2/b+log(c)*sin(d)),
            # a re-attached pure half (``mul(log(c),sin(d))`` / ``div(sqr(a),sin(b))``) is FULLY redundant
            # given it - the fragmentation regression (one compound PLUS several sub-fragments). Re-run
            # the SAME n-invariant debiased-excess CMI subsumption check the S5 gate validated, conditioning
            # each retention candidate on the INCUMBENT (pre-retention) engineered survivors, and skip any
            # whose information collapses given them. A genuinely COMPLEMENTARY pure form (one the incumbents
            # do not span - the case this retention pass exists to rescue) keeps a large conditional excess
            # and is admitted; only sub-fragments of an incumbent compound are dropped. No-op (byte-identical)
            # when there is no incumbent engineered survivor to condition on.
            if _retain_extra:
                try:
                    from .._fe_retention_subsumption import retention_form_is_subsumed
                    from ..engineered_recipes._recipe_dispatch import apply_recipe as _ret_apply
                    _inc_names = [str(_n) for _n in (self._engineered_features_ or [])]
                    _inc_cont = []
                    for _in in _inc_names:
                        _iv = _eng_continuous_snapshot.get(_in)
                        if _iv is not None and np.asarray(_iv).shape[0] == int(data.shape[0]):
                            _inc_cont.append(np.asarray(_iv, dtype=np.float64).ravel())
                    if _inc_cont:
                        _ret_y = np.ascontiguousarray(np.asarray(classes_y)).ravel()
                        _ret_y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
                        if _ret_y_cont is None:
                            try:
                                _yv = y.values if hasattr(y, "values") else np.asarray(y)
                                _yv = np.asarray(_yv).reshape(-1)
                                if _yv.shape[0] == int(data.shape[0]) and np.issubdtype(np.asarray(_yv).dtype, np.number):
                                    _ret_y_cont = _yv
                            except Exception as exc:
                                logger.debug("mrmr: continuous-y coercion failed for pure-form retention; retention skipped for this column: %r", exc, exc_info=True)
                                _ret_y_cont = None
                        _ret_seed = int(getattr(self, "random_seed", 0) or 0)
                        _kept_extra = []
                        for _r_recipe, _r_name in _retain_extra:
                            try:
                                _cv = np.asarray(_ret_apply(_r_recipe, X), dtype=np.float64).ravel()
                                _cv = np.nan_to_num(_cv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            except Exception as exc:
                                logger.debug("mrmr: recipe replay failed while checking form subsumption; conservatively retaining (cannot prove subsumed): %r", exc, exc_info=True)
                                _kept_extra.append((_r_recipe, _r_name))  # cannot replay -> retain (conservative)
                                continue
                            if _cv.shape[0] == int(data.shape[0]) and retention_form_is_subsumed(
                                cand_continuous=_cv, incumbent_continuous=_inc_cont,
                                y_binned=_ret_y, y_continuous=_ret_y_cont, seed=_ret_seed,
                            ):
                                if verbose:
                                    logger.info(
                                        "MRMR usability-aware retention: SKIP re-attaching pure form %r -- "
                                        "fully subsumed by an incumbent engineered compound (conditional CMI "
                                        "collapses given the pre-retention survivors).", _r_name,
                                    )
                                continue
                            _kept_extra.append((_r_recipe, _r_name))
                        _retain_extra = _kept_extra
                except Exception as _subsume_exc:
                    logger.debug("MRMR retention engineered-subsumption guard skipped (%s: %s).", type(_subsume_exc).__name__, _subsume_exc)
            for _r_recipe, _r_name in _retain_extra:
                self._engineered_recipes_.append(_r_recipe)
                self._engineered_features_.append(_r_name)
                _retention_added_eng_names.add(str(_r_name))
            if _retain_extra and verbose:
                logger.info(
                    "MRMR usability-aware retention: re-attached %d linearly-usable pure pair form(s) " "the MI greedy dropped for a higher-MI cross-mix: %s",
                    len(_retain_extra),
                    [n for _, n in _retain_extra],
                )
        except Exception as _retain_exc:  # never let the optional retention break a fit
            logger.debug("MRMR usability-aware pure-form retention skipped (%s: %s).", type(_retain_exc).__name__, _retain_exc)

        # USABILITY-AWARE RAW RETENTION. The companion to the pure-form retention above for the
        # case where the genuinely useful structure is a RAW the MI greedy under-ranked, not a pair form.
        # MRMR ranks raws by binned MI, which under-values a linearly-usable raw whose marginal-MI estimate is
        # small - e.g. operands g/k of a WEAK additive ratio term ``+ g/k`` in y = w*a**2/b + g/k +
        # log(c)*sin(d): binned MI ~0.01-0.02 (below the relevance floor) yet linear corr ~0.15-0.24 and a tree
        # recovers the ratio. Both are dropped from support_, the pure-form retention cannot rescue the pair
        # (the clean g/k engineered form is a pool-generation lottery), and the marginal-MI re-attach skips
        # them (MI below floor, not a recipe operand) -> the FE space loses the g/k signal and a downstream
        # model scores BELOW raw-only (BUG3 "FE harmful"; the I5 ratio_plus_trig case). The CV-MAE linear
        # wrapper (the same one the pure-form retention trusts) run over the RAW passthroughs surfaces these
        # under-ranked raws and - crucially - rejects pure-noise raws (they do not lower the average CV-MAE).
        # Re-attaches only raws NOT already in support_; purely additive (no engineered recipe touched).
        try:
            from .._fe_pure_form_retention import retain_usable_raw_columns

            _raw_extra = retain_usable_raw_columns(
                self, X, getattr(self, "_fe_prewarp_y_continuous_", None),
                seed=int(getattr(self, "random_seed", 0) or 0), verbose=verbose,
                _prep=_retention_prep_cache,
            )
            # CLUSTER-COLLAPSE EXCLUSION. ``retain_usable_raw_columns`` ranks raws by
            # linear usability and is OBLIVIOUS to the cluster-aggregate / DCD redundancy collapse that
            # the support chokepoint above already applied. A perfectly-collinear duplicate (``z=2a+3``)
            # is maximally linearly-usable, so this pass happily re-attaches the very cluster member the
            # chokepoint stripped - re-injecting the redundancy and selecting BOTH members of a
            # collinear pair (test_duplicate_collinear_handled_and_recorded). Mirror the same exclusion
            # the raw-signal augmentation below applies: never re-attach a raw the cluster collapse
            # already folded into another representative / a denoised aggregate.
            if _raw_extra:
                # Exclude the NON-REPRESENTATIVE members of every redundancy cluster: per cluster
                # exactly ONE representative (the strongest member) stays eligible for re-attachment,
                # mirroring the chokepoint's keep-one-strip-rest. ``_cluster_aggregate_removals_`` (the
                # explicit 'replace'-mode removals) are excluded outright - they are folded into a
                # denoised aggregate that already represents the cluster.
                # Exclude at most all-but-one member of every cluster, so the pass can never select
                # BOTH members of a redundant pair, while still re-attaching ONE representative when the
                # whole cluster was dropped. For each cluster, of the members ``retain_usable_raw_columns``
                # surfaced, keep the first (it is the strongest by the pass's own usability ranking) and
                # exclude the rest; if a cluster member is ALREADY in ``selected_vars`` that member is the
                # representative, so exclude every cluster member from ``_raw_extra`` (no second copy).
                _rr_excl_names = set(str(_n) for _n in (getattr(self, "_cluster_aggregate_removals_", None) or []))
                _cm_rr = getattr(self, "cluster_members_", None)
                if isinstance(_cm_rr, dict):
                    _rr_raw = set(self.feature_names_in_)
                    _rr_sel_names = {self.feature_names_in_[int(v)] for v in selected_vars if int(v) < len(self.feature_names_in_)}
                    _rr_order = {str(_nm): _i for _i, _nm in enumerate(_raw_extra)}
                    for _rr_anchor, _rr_members in _cm_rr.items():
                        _a = str(_rr_anchor)
                        _ms = [str(_m) for _m in _rr_members] if isinstance(_rr_members, (list, tuple, set)) else []
                        if _a not in _rr_raw:
                            # aggregate/engineered anchor: every raw member is folded into the aggregate.
                            _rr_excl_names.update(_n for _n in _ms if _n in _rr_raw)
                            continue
                        _grp = [_n for _n in [_a, *_ms] if _n in _rr_raw]
                        if len(_grp) < 2:
                            continue
                        if any(_n in _rr_sel_names for _n in _grp):
                            # a representative already survived -> drop every cluster member from re-attach.
                            _rr_excl_names.update(_grp)
                        else:
                            # whole cluster dropped -> keep the single member the retention ranked highest.
                            _cands = [_n for _n in _grp if _n in _rr_order]
                            if _cands:
                                _keep = min(_cands, key=lambda _n: _rr_order[_n])
                                _rr_excl_names.update(_n for _n in _grp if _n != _keep)
                            else:
                                _rr_excl_names.update(_grp)
                # SUBSUMED-OPERAND EXCLUSION (signal-aware, variant-3). A raw that is an operand of a
                # SURVIVING engineered feature MAY be fully represented by that feature (re-attaching it then
                # re-injects the raw-redundancy I4b forbids) - but it may instead carry a large PRIVATE signal
                # the engineered child only partially tracks (e.g. a dominant linear term ``y += 2*a`` that a
                # nonlinear nesting ``sub(log(a),...)`` cannot capture). A blanket name-token exclusion drops
                # BOTH cases and silently destroys genuine raw signal (fs_robustness: a linear ``y`` whose raws
                # are all folded into nonlinear engineered survivors loses every raw -> empty support). Decide
                # PER RAW with the same conditional-redundancy discriminator the rescue/drop passes use: exclude
                # ONLY raws truly subsumed by the engineered survivors consuming them (no private signal given
                # those children); KEEP raws that retain >= RAW_SELF_RETAIN_FRAC of their marginal excess.
                from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _RR_TOK_SPLIT2
                _rr_raw_set = set(self.feature_names_in_)
                # raw name -> surviving engineered recipe names that consume it as an operand.
                # ``selected_vars`` is narrowed to RAW-ONLY indices just above (the ``selected_vars =
                # original_indices`` rebind, ~line 8527) -- engineered survivors live ONLY in
                # ``self._engineered_recipes_`` / ``self._engineered_features_`` from that point on, so a
                # ``cols[int(v)] not in _rr_raw_set`` scan of ``selected_vars`` here always finds nothing
                # (every ``v`` now indexes a raw column) and this exclusion silently never fires -- the SAME
                # staleness trap the sibling C2-fusion / usability-retention fixes closed for
                # ``self._engineered_recipes_`` reads made BEFORE that rebind; this read happens AFTER it, so
                # the freshly (re)populated attribute (set at ~line 8498-8514, above this block) is the
                # authoritative source here, not the now RAW-only ``selected_vars``.
                _rr_consumers: dict = {}
                _rr_sel_eng_names = {str(getattr(_r, "name", "")) for _r in (self._engineered_recipes_ or []) if getattr(_r, "name", None)}
                _rr_sel_eng_names |= {str(_n) for _n in (self._engineered_features_ or [])}
                for _en_name in _rr_sel_eng_names:
                    for _tok in _RR_TOK_SPLIT2.split(str(_en_name)):
                        if not _tok:
                            continue
                        _base = _tok if _tok in _rr_raw_set else (_tok.split("__", 1)[0] if "__" in _tok else None)
                        if _base in _rr_raw_set:
                            _rr_consumers.setdefault(_base, set()).add(str(_en_name))
                # Only the raws actually up for re-attachment need a verdict.
                _rr_cand_subsumed = {str(_n) for _n in _raw_extra if str(_n) in _rr_consumers}
                if _rr_cand_subsumed and not bool(getattr(self, "use_simple_mode", False)):
                    # FULL FE mode: the caller opted into replacing subsumed raws with engineered
                    # survivors, so exclude EVERY engineered operand from re-attachment unconditionally
                    # (the I4b subsumed-raw contract). The signal-aware verdict below is only for SIMPLE
                    # mode, where a linearly-usable raw must survive even when an engineered child encodes
                    # it nonlinearly.
                    _rr_excl_names.update(_rr_cand_subsumed)
                elif _rr_cand_subsumed:
                    try:
                        from .._fe_raw_redundancy_drop import raw_retains_signal_given_genuine_children as _rr_keep
                        from .._mi_greedy_cmi_fe import _quantile_bin as _rr_qbin
                        _rr_cols_idx = {nm: i for i, nm in enumerate(cols)}
                        _rr_y = np.ascontiguousarray(np.asarray(classes_y)).ravel().astype(np.int64)
                        _rr_eng_cont = _eng_continuous_snapshot or {}
                        _rr_seed = int(getattr(self, "random_seed", 0) or 0)
                        for _base in _rr_cand_subsumed:
                            _rr_ci = _rr_cols_idx.get(_base)
                            if _rr_ci is None:
                                _rr_excl_names.add(_base)  # cannot test -> keep the conservative exclusion
                                continue
                            _raw_b = np.asarray(data[:, _rr_ci]).astype(np.int64).ravel()
                            _child_bins = []
                            for _en_name in _rr_consumers.get(_base, ()):  # genuine engineered survivors
                                _cci = _rr_cols_idx.get(_en_name)
                                if _cci is not None:
                                    _child_bins.append(np.asarray(data[:, _cci]).astype(np.int64).ravel())
                                elif _en_name in _rr_eng_cont:
                                    _child_bins.append(
                                        np.asarray(_rr_qbin(np.asarray(_rr_eng_cont[_en_name], dtype=np.float64), nbins=10)).astype(np.int64).ravel()
                                    )
                            if not _child_bins:
                                continue  # no usable child to condition on -> not provably subsumed -> KEEP
                            try:
                                _retains = _rr_keep(raw_bin=_raw_b, y_bin=_rr_y,
                                                    genuine_child_bins=_child_bins,
                                                    allow_linear_usability=bool(getattr(self, "use_simple_mode", False)),
                                                    seed=_rr_seed)
                            except Exception as exc:
                                logger.debug("mrmr: discriminator estimator failed; conservatively retaining (never drop genuine signal): %r", exc, exc_info=True)
                                _retains = True  # estimator error -> never drop genuine signal
                            if not _retains:
                                _rr_excl_names.add(_base)  # truly subsumed -> exclude from re-attach
                    except Exception as exc:
                        # RETAIN on error, matching the inner per-candidate handler a few lines above. This used
                        # to do the opposite -- one exception anywhere in the enclosing block (an import fault, a
                        # shape error building the child bins, a dtype problem on the data slice) blanket-excluded
                        # EVERY candidate raw from the re-attach set, so features that would have been retained
                        # were silently absent from support_, on a debug line, with the two handlers disagreeing
                        # about polarity and no way to tell from the logs which had fired.
                        logger.warning(
                            "mrmr: the subsumption discriminator failed (%s: %s); RETAINING all %d candidate raw(s) for re-attach rather than "
                            "blanket-excluding them. A dropped feature set would otherwise be indistinguishable from a genuine subsumption verdict.",
                            type(exc).__name__,
                            exc,
                            len(_rr_cand_subsumed),
                        )
                if _rr_excl_names:
                    _raw_extra = [_nm for _nm in _raw_extra if str(_nm) not in _rr_excl_names]
            if _raw_extra:
                # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
                _name_to_in_idx = {nm: i for i, nm in enumerate(getattr(self, "feature_names_in_", []))}
                # Append to the local ``selected_vars`` (the canonical raw-support list every downstream
                # step - n_features_, the marginal-MI augmentation, the elbow trim, and the final
                # ``self.support_ = np.array(selected_vars)`` - reads), NOT directly to ``self.support_``:
                # a later block re-derives ``support_`` from ``selected_vars`` and would clobber a direct
                # ``support_`` edit. Keeps every consumer consistent.
                _cur_set = set(int(v) for v in selected_vars)
                _added_idx = []
                for _nm in _raw_extra:
                    _idx = _name_to_in_idx.get(_nm)
                    if _idx is not None and int(_idx) not in _cur_set:
                        # Honour the caller's pinned search space: the usability-aware retention runs AFTER the
                        # factors_names_to_use / factors_to_use chokepoint (above), so a re-attached raw outside
                        # the pinned pool would re-leak a forbidden column into support_.
                        if _allowed_raw_idx is not None and int(_idx) not in _allowed_raw_idx:
                            continue
                        selected_vars.append(int(_idx))
                        _cur_set.add(int(_idx))
                        _added_idx.append(int(_idx))
                if _added_idx:
                    self.support_ = np.array(selected_vars, dtype=np.int64)
                    if verbose:
                        logger.info(
                            "MRMR usability-aware raw retention: re-attached %d linearly-usable raw(s) the "
                            "MI greedy under-ranked: %s", len(_added_idx), _raw_extra,
                        )
        except Exception as _raw_retain_exc:  # never let the optional retention break a fit
            logger.debug("MRMR usability-aware raw retention skipped (%s: %s).", type(_raw_retain_exc).__name__, _raw_retain_exc)

    # POST-RETENTION RAW-REDUNDANCY DROP (BUG1). The main raw-vs-engineered
    # redundancy sweep (above, ~line 7915) runs on the screen-stage ``selected_vars`` BEFORE
    # the usability-aware pure-form retention re-attaches an engineered survivor. When that
    # retention adds a MULTI-OPERAND composite (e.g. ``div(qubed(a),sin(b))``) AFTER the
    # sweep, the raw operands it subsumes (``a``, ``b``) are still in ``selected_vars`` and no
    # later pass conditions them on the freshly-attached child - so a fully-subsumed raw rides
    # into ``support_`` beside the composite that captures it (the I4b end-to-end violation).
    # Re-run the SAME n-invariant conditional-redundancy verdict on the FINAL selection, with
    # the now-complete engineered survivor set (incl. the retained pure forms) as the anchor.
    # Only DROPS raws fully subsumed by a surviving MULTI-SOURCE child; a genuine private raw
    # (large independent residual) and a raw consumed by no surviving engineered feature are
    # KEPT (the DPI-trap filter + self-retention leg inside the helper enforce this). Off when
    # the drop sweep is disabled (shares ``fe_drop_redundant_raw_operands``).
    if (getattr(self, "fe_drop_redundant_raw_operands", True)
            and getattr(self, "redundancy_policy", "emit_both") == "drop"
            and selected_vars and getattr(self, "_engineered_recipes_", None)):
        try:
            from .._fe_raw_redundancy_drop import drop_redundant_raw_operands as _post_drop
            from ..engineered_recipes._recipe_dispatch import apply_recipe as _post_apply
            from .._mi_greedy_cmi_fe import _quantile_bin as _post_qbin

            _post_raw_set = set(self.feature_names_in_)
            # Final engineered survivor recipes (name -> EngineeredRecipe); these are the
            # columns that actually reach transform() output.
            # Anchor ONLY on engineered survivors RE-ATTACHED by the retention passes (after the main
            # sweep). The main raw-vs-engineered sweep already vetted every raw against the survivors it
            # could see, so re-litigating those raws here - with the stricter post-retention margin -
            # wrongly drops a genuine pair-interaction operand the main sweep KEPT (TestPairInteraction:
            # x_a/x_b in y=x_a+x_b+2*x_a*x_b, main-sweep cmi 1.21x floor -> KEEP, post 1.5x -> DROP). The
            # post-retention sweep exists ONLY for composites retention attached after the sweep ran.
            _post_recipes: dict = {}
            for _r in self._engineered_recipes_ or []:
                _nm = getattr(_r, "name", None)
                if _nm is not None and _nm not in _post_raw_set and str(_nm) in _retention_added_eng_names:
                    _post_recipes[str(_nm)] = _r
            # Selected raw operand cols-indices (selected_vars is in feature_names_in_ space here;
            # map each surviving raw back to its cols-space index by name).
            _post_sel_raw_names = [self.feature_names_in_[int(v)] for v in selected_vars if 0 <= int(v) < len(self.feature_names_in_)]
            if _post_recipes and _post_sel_raw_names:
                _post_cols = list(cols)
                _post_data = data
                _post_eng_cont = dict(_eng_continuous_snapshot or {})
                _post_extra_cols: list = []
                # Ensure each engineered survivor has a cols-space column + continuous snapshot;
                # replay any retained pure form that the FE-step matrix does not already carry.
                _n_rows_post = int(data.shape[0])
                for _enm, _erec in _post_recipes.items():
                    if _enm not in _post_eng_cont:
                        try:
                            _vals = np.asarray(_post_apply(_erec, X), dtype=np.float64).ravel()
                            _vals = np.nan_to_num(_vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            if _vals.shape[0] == _n_rows_post:
                                _post_eng_cont[_enm] = _vals
                        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                            logger.debug("mrmr: post-selection engineered-continuous coercion failed for %r: %r", _enm, e, exc_info=True)
                            continue
                    if _enm not in _post_cols and _enm in _post_eng_cont:
                        _post_extra_cols.append(_post_qbin(np.asarray(_post_eng_cont[_enm], dtype=np.float64), nbins=10))
                        _post_cols.append(_enm)
                if _post_extra_cols:
                    _post_data = np.column_stack([data] + [np.asarray(c).reshape(-1, 1) for c in _post_extra_cols])
                _post_name_to_idx = {nm: i for i, nm in enumerate(_post_cols)}
                _post_sel_idx = []
                for _rn in _post_sel_raw_names:
                    _ci1 = _post_name_to_idx.get(_rn)
                    if _ci1 is not None:
                        _post_sel_idx.append(_ci1)
                for _enm in _post_recipes:
                    _ci2 = _post_name_to_idx.get(_enm)
                    if _ci2 is not None and _ci2 not in _post_sel_idx:
                        _post_sel_idx.append(_ci2)
                _has_eng_post = any(_post_cols[i] not in _post_raw_set for i in _post_sel_idx)
                _has_raw_post = any(_post_cols[i] in _post_raw_set for i in _post_sel_idx)
                if _has_eng_post and _has_raw_post:
                    _y_cont_post = None
                    try:
                        _yv = y.values if hasattr(y, "values") else np.asarray(y)
                        _yv = np.asarray(_yv).reshape(-1)
                        if _yv.shape[0] == _n_rows_post and np.issubdtype(np.asarray(_yv).dtype, np.number):
                            _y_cont_post = _yv
                    except Exception as exc:
                        logger.debug("mrmr: continuous-y coercion failed for the post-selection drop pass; column skipped: %r", exc, exc_info=True)
                        _y_cont_post = None
                    _, _post_dropped = _post_drop(
                        data=_post_data, cols=_post_cols, selected_cols_idx=_post_sel_idx,
                        raw_name_set=_post_raw_set, y_binned=classes_y, y_continuous=_y_cont_post,
                        engineered_continuous=_post_eng_cont,
                        replayable_eng_names=set(_post_recipes.keys()), recipes=_post_recipes,
                        raw_X=X, floor_margin_mult=1.5,
                        linear_usability_keep=bool(getattr(self, "fe_keep_linearly_usable_raw_operands", True)),
                        seed=int(getattr(self, "random_seed", 0) or 0), verbose=verbose,
                    )
                    if _post_dropped:
                        _post_drop_set = set(_post_dropped)
                        self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | _post_drop_set
                        selected_vars = [
                            v for v in selected_vars if not (0 <= int(v) < len(self.feature_names_in_) and self.feature_names_in_[int(v)] in _post_drop_set)
                        ]
                        # NEVER-EMPTY RAW FLOOR (mirrors the main sweep): the post-retention drop may not
                        # empty the raw support. If no raw survives, re-add the strongest dropped raw (by
                        # marginal MI) as the representative; the engineered survivor still rides along.
                        if not selected_vars:
                            _bf_idx, _bf_rel = None, float("-inf")
                            _tgt_pf = np.asarray(target_indices, dtype=np.int64)
                            _fn_pf = np.asarray(nbins, dtype=np.int64)
                            # ``_post_name_to_idx`` (built above) is ``_post_cols`` = ``list(cols)`` plus
                            # any appended engineered columns, so it agrees with ``cols.index`` for every
                            # raw name here (``_post_drop_set`` only ever holds raw ``feature_names_in_``
                            # names, always present in the original ``cols`` prefix) - reuse it instead
                            # of a fresh ``.index()`` rescan of ``cols`` per ``_dn``.
                            for _dn in _post_drop_set:
                                _bf_ci = _post_name_to_idx.get(_dn)
                                if _bf_ci is None:
                                    continue
                                try:
                                    from ..info_theory import mi as _pf_mi
                                    _rel = float(_pf_mi(data, np.array([int(_bf_ci)], dtype=np.int64), _tgt_pf, _fn_pf))
                                except Exception as exc:
                                    logger.debug("mrmr: relevance computation failed for this candidate in the post-drop pass; treating as zero (conservative): %r", exc, exc_info=True)
                                    _rel = 0.0
                                if _rel > _bf_rel:
                                    _bf_rel, _bf_idx = _rel, _dn
                            if _bf_idx is not None and _bf_idx in self.feature_names_in_:
                                selected_vars = [list(self.feature_names_in_).index(_bf_idx)]
                                self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) - {_bf_idx}
                        self.support_ = np.array(selected_vars, dtype=np.int64)
                        if verbose:
                            logger.info(
                                "MRMR post-retention raw-redundancy drop: removed %d raw operand(s) "
                                "subsumed by an engineered survivor re-attached AFTER the main sweep: %s",
                                len(_post_dropped), sorted(_post_drop_set),
                            )
        except Exception as _post_exc:
            logger.warning(
                "MRMR post-retention raw-redundancy drop failed: %s; keeping the support.",
                _post_exc,
            )

    # n_features_ reports the column count produced by transform() = raw selected + engineered (replayable via _engineered_recipes_). Higher-order
    # engineered features without a replayable recipe were already warned about above and are NOT counted (they don't appear in transform output).
    n_engineered_out = len(self._engineered_recipes_)
    if selected_vars:
        self.n_features_ = len(selected_vars) + n_engineered_out
        # RAW-SIGNAL-RETENTION augmentation (Fix B). On a wide composite-FE pool the screen often confirms an ENGINEERED derivative of a strong raw signal (e.g.
        # ``x1__resid_by__cat_a`` whose cat-residual MI exceeds raw ``x1``), which then conditionally redundifies the raw column so raw ``x1`` is dropped from
        # ``support_`` even though it is genuine, generalising signal. The empirical-null debiasing makes the per-feature ``cached_MIs`` an honest relevance ranking
        # (cardinality / heavy-tail / monotone in-sample inflation removed), so a raw feature that clears the relevance floor AND is the SOURCE of a confirmed
        # engineered child is genuine signal that the greedy step merely shadowed behind its derivative - we re-attach it. The augmentation is deliberately narrow:
        # it rescues ONLY columns whose name appears as a source token in some engineered recipe name, so it can never re-inflate a redundant block of near-duplicate
        # raw columns (those have no engineered child) and never overrides DCD / cluster-aggregate redundancy collapse. ``min_features_fallback==0`` opts out.
        _min_fb_aug = int(getattr(self, "min_features_fallback", 0) or 0)
        if _min_fb_aug >= 1 and self.n_features_in_ > 0 and hasattr(self, "cached_MIs") and n_engineered_out > 0:
            try:
                # Source tokens referenced by any confirmed engineered recipe (split on the engineered-name separators ``__``, ``(``, ``|``, ``)``, ``,``).
                import re as _re_aug
                _eng_names = []
                for _r in self._engineered_recipes_ or []:
                    # EngineeredRecipe's real field is `name` (`output_name` is not an attribute of any
                    # recipe class here); the dict fallback covers plain-dict recipe representations.
                    _nm = getattr(_r, "name", None)
                    if _nm is None and isinstance(_r, dict):
                        _nm = _r.get("name")
                    if _nm:
                        _eng_names.append(str(_nm))
                _eng_tokens = set()
                for _nm in _eng_names:
                    for _tok in _re_aug.split(r"[^0-9A-Za-z_]+", _nm.replace("__", " ")):
                        if _tok:
                            _eng_tokens.add(_tok)
                # Members folded into a denoised aggregate - cluster_aggregate 'replace' mode (``_cluster_aggregate_removals_``) or a DCD PC1/mean_z swap (``cluster_members_``) - are
                # ALREADY represented by that aggregate and were deliberately removed from the support. The token scan above matches them anyway because the member NAME survives as a
                # token inside OTHER engineered recipe names (e.g. ``add(refl0,sin(indep))``), so without this exclusion the augmentation resurrects the very members 'replace' mode and
                # DCD just collapsed, re-injecting the redundancy. Mirror the same exclusion the raw-retention block and the additional-RFECV rescue pool apply.
                _aug_excluded_names = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
                _cm_for_aug = getattr(self, "cluster_members_", None)
                if isinstance(_cm_for_aug, dict):
                    for _anchor, _members in _cm_for_aug.items():
                        _aug_excluded_names.add(_anchor)
                        if isinstance(_members, (list, tuple, set)):
                            _aug_excluded_names.update(_members)
                _name_to_cols_idx_aug = {c: i for i, c in enumerate(cols)}
                _abs_floor_aug = float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
                _rel_frac_aug = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
                _raw_mi_aug = []
                for _i in range(self.n_features_in_):
                    _name = self.feature_names_in_[_i] if _i < len(self.feature_names_in_) else None
                    _aug_ci = _name_to_cols_idx_aug.get(_name)
                    _mi = self.cached_MIs.get((_aug_ci,), 0.0) if _aug_ci is not None else 0.0
                    _raw_mi_aug.append((_i, _name, float(_mi)))
                _max_mi_aug = max((m for _, _, m in _raw_mi_aug), default=0.0)
                _floor_aug = max(_abs_floor_aug, _max_mi_aug * _rel_frac_aug)
                _selected_set = set(int(v) for v in selected_vars)
                # LARGE-N SCOPE: this augmentation re-attaches a raw
                # column whose NAME is a source token of a confirmed engineered recipe and whose
                # MARGINAL MI clears the relevance floor. Marginal MI cannot tell a FULLY-ABSORBED
                # operand (``a`` in ``div(sqr(a),abs(b))`` for ``y=a**2/b`` - high marginal MI, ZERO
                # conditional signal beyond the ratio) from a genuine independent term, so on the
                # canonical composite fixtures it resurrected exactly the redundant raw operands the
                # post-FE re-selection had correctly dropped (support_rank -1, no gain). At large n the
                # re-selection's conditional-MI redundancy verdict is reliable, so we DEFER to it: skip
                # the token-based re-attach for any raw column that is an operand of a SURVIVING
                # engineered feature. The small-n regime (where the augmentation was validated) keeps
                # the marginal-MI re-attach. Threshold shared with the raw-retention pass above.
                # ``selected_vars`` here holds RAW indices into ``feature_names_in_`` (the surviving
                # engineered columns live in ``self._engineered_recipes_`` / ``_engineered_features_``,
                # not in ``selected_vars``). Derive the surviving engineered OPERANDS from the recipe
                # source tokens (``_eng_tokens`` already = every source token of every confirmed recipe,
                # restricted here to raw names), since every confirmed engineered child contributes its
                # operands to that set. A raw column that is such an operand was dropped by the
                # re-selection IN FAVOUR of its engineered child -> at large n, do not resurrect it.
                _aug_max_n = int(getattr(self, "fe_raw_retention_max_n", 20000) or 0)
                _aug_large_n = int(data.shape[0]) > _aug_max_n
                _raw_names_for_aug = set(self.feature_names_in_)
                _surviving_eng_operands = {t for t in _eng_tokens if t in _raw_names_for_aug} if _aug_large_n else set()
                # Operands the n-invariant conditional-redundancy sweep dropped are authoritative
                # at EVERY n - never re-attach them here (the marginal-MI token match cannot tell
                # a fully-subsumed operand from a genuine independent term; the excess-CMI sweep can).
                _redund_dropped_names = set(getattr(self, "_raw_redundancy_dropped_", None) or ())
                _to_add = [i for i, _name, m in sorted(_raw_mi_aug, key=lambda kv: (-kv[2], kv[0]))
                           if m > _floor_aug and i not in _selected_set and _name in _eng_tokens
                           and _name not in _aug_excluded_names
                           and _name not in _redund_dropped_names
                           and not (_aug_large_n and _name in _surviving_eng_operands)]
                if _to_add:
                    selected_vars.extend(_to_add)
                    self.support_ = np.array(selected_vars, dtype=np.int64)
                    self.n_features_ = len(selected_vars) + n_engineered_out
            except Exception as _exc_aug:
                logger.warning("MRMR raw-signal-retention augmentation failed: %s; keeping greedy support.", _exc_aug)
    elif getattr(self, "_redundancy_emptied_raw_", False):
        # The raw support is empty because the n-invariant conditional-redundancy sweep
        # deliberately dropped every raw operand (each fully subsumed by a surviving
        # engineered child) - an INTENDED, complete engineered-only support, NOT a
        # "screen returned 0 raw" emergency. SKIP the empty-raw rescue entirely; firing
        # it would resurrect the dropped operands or pull in the next pure-noise column
        # ranked by marginal MI (measured ws1: ``e`` rescued at n=1000, ``a`` re-added at
        # n=25000). n_features_ is the engineered-only count.
        self.n_features_ = n_engineered_out
    else:
        # Empty-RAW-support fallback rescue carved into _finalise.py (Tier E partial split).
        # Threads the instance + fit-body locals explicitly; mutates self.support_ / n_features_ /
        # fallback_used_ / fallback_metadata_ in place. Behaviour byte-for-byte identical to the
        # former inlined branch.
        from ._finalise import _finalise_empty_support_fallback
        _finalise_empty_support_fallback(self, n_engineered_out, cols, data, nbins, target_indices)

    # The p>=n FP-control cap above is enforced exactly once,
    # but the post-selection reconciliation passes below it (emit-both operand re-attach, usability-aware
    # raw retention, raw-signal-retention augmentation) can each append more raw columns afterward with no
    # re-check against the cap - letting the final raw (and n_features_) count silently exceed the
    # documented max(20, p//3) ceiling on a p>>n fit with real leftover linear-usable raw signal. Re-apply
    # the same cap here, at the true end of raw-selection mutation for this fit (nothing below this point
    # adds more raw columns - only the UAED elbow trim further down, which only shrinks).
    _pgn_n_final = int(data.shape[0]) if "data" in dir() else 0
    _pgn_p_final = int(getattr(self, "n_features_in_", 0) or 0)
    if _pgn_p_final > 0 and _pgn_n_final > 0 and _pgn_p_final >= _pgn_n_final and selected_vars:
        _pgn_ceiling_final = max(20, _pgn_p_final // 3)
        _pgn_eng_final = len(getattr(self, "_engineered_recipes_", None) or [])
        _pgn_budget_final = _pgn_raw_budget(_pgn_ceiling_final, _pgn_eng_final)
        if len(selected_vars) > _pgn_budget_final:
            _pgn_cached_final = self.cached_MIs if isinstance(getattr(self, "cached_MIs", None), dict) else {}
            _pgn_n2ci_final = {c: i for i, c in enumerate(cols)} if "cols" in dir() else {}
            _fni_pgn_final = self.feature_names_in_

            def _pgn_rel_final(_v):
                """Screening marginal MI(v, y) for raw index _v, used by the FINAL p>=n cap re-application (after post-selection retention passes)."""
                _nm = _fni_pgn_final[_v] if _v < len(_fni_pgn_final) else None
                _ci = _pgn_n2ci_final.get(_nm)
                return float(_pgn_cached_final.get((_ci,), 0.0)) if _ci is not None else 0.0

            _pgn_overflow_final = len(selected_vars) - _pgn_budget_final
            selected_vars = [v for v in sorted(selected_vars, key=lambda v: (-_pgn_rel_final(v), int(v)))][:_pgn_budget_final]
            self.support_ = np.array(selected_vars, dtype=np.int64)
            self.n_features_ = len(selected_vars) + n_engineered_out
            if verbose:
                logger.info(
                    "MRMR p>=n FP-control: re-capped raw support to top-%d by relevance after post-selection "
                    "retention passes added %d raw feature(s) beyond the ceiling (p=%d >= n=%d, ceiling=%d, engineered=%d).",
                    _pgn_budget_final, _pgn_overflow_final, _pgn_p_final, _pgn_n_final, _pgn_ceiling_final, _pgn_eng_final,
                )
