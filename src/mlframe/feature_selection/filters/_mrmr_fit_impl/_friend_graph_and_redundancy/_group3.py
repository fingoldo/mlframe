"""Sibling of ``_friend_graph_and_redundancy/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
friend-graph-and-redundancy post-screen block, itself further split for the 1k-LOC module-size gate).

Holds passes: usability-aware-raw-readd, post-DCD-cluster-pruning. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``(selected_vars, cols, data, nbins)`` threading contract
(all four are BOTH incoming parameters AND part of the return value -- mirrors the parent's own).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _friend_graph_and_redundancy_passes_group3(
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
    """Run the usability-aware-raw-readd, post-DCD-cluster-pruning pass(es) and return ``(selected_vars, cols, data, nbins)``.
    See the package docstring for the full section this carves out."""
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
            from ..._dynamic_cluster_discovery import (
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
            from ..._fe_raw_redundancy_drop import (
                _is_pseudo_remix_child as _pcr_is_pseudo,
                _PSEUDO_SRC_SPLIT,
                raw_retains_signal_given_genuine_children as _pcr_keep,
            )
            from ..._mi_greedy_cmi_fe import _quantile_bin as _pcr_qbin
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
                from ..._fe_raw_redundancy_drop import _TOKEN_SPLIT
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
                    from ...permutation import mi_direct as _pcr_mi_direct
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

    return selected_vars, cols, data, nbins
