"""Sibling of ``_friend_graph_and_redundancy/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
friend-graph-and-redundancy post-screen block, itself further split for the 1k-LOC module-size gate).

Holds passes: pseudo-remix-redundancy-drop, monotone-twin-drop. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``(selected_vars, cols, data, nbins)`` threading contract
(all four are BOTH incoming parameters AND part of the return value -- mirrors the parent's own).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _friend_graph_and_redundancy_passes_group4(
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
    """Run the pseudo-remix-redundancy-drop, monotone-twin-drop pass(es) and return ``(selected_vars, cols, data, nbins)``.
    See the package docstring for the full section this carves out."""
    if getattr(self, "fe_drop_redundant_raw_operands", True) and getattr(self, "redundancy_policy", "emit_both") == "drop" and len(selected_vars) >= 2:
        try:
            from ..._fe_raw_redundancy_drop import drop_redundant_raw_operands
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
                    linear_usability_keep=bool(getattr(self, "fe_keep_linearly_usable_raw_operands", True)),
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
                        from ..._fe_raw_redundancy_drop import raw_retains_linear_signal_given_children as _floor_lin
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
                                from ...info_theory import mi as _floor_mi
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
                from ..._feature_engineering_pairs._pairs_core import _abs_corr_finite_njit as _mt_corr_njit
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

    return selected_vars, cols, data, nbins
