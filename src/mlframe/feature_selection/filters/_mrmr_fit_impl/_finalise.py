"""``MRMR._fit_impl`` finalisation tail: the empty-RAW-support fallback rescue.

Carved verbatim out of the giant ``_fit_impl`` orchestration body in
``_fit_impl_core.py`` (Tier E partial split) to shrink the parent below the
monolith budget. ``_finalise_empty_support_fallback`` is the ``else`` branch
of the post-selection raw-support reconciliation (no raw feature survived the
greedy screen): it ranks raw inputs by debiased cached MI, gates them on a
permutation-significance null + a redundancy dedup conditioned on surviving
engineered features, applies the ``min_features_fallback`` count floor and the
never-empty guarantee, then sets ``support_`` / ``n_features_`` /
``fallback_used_`` / ``fallback_metadata_`` on the instance and emits the
fallback warning.

The block reads/writes the ``MRMR`` instance heavily, so ``self`` is threaded
explicitly along with the pure fit-body locals it consumes
(``n_engineered_out`` / ``cols`` / ``data`` / ``nbins`` / ``target_indices``).
It returns nothing -- every output is an attribute set on ``self`` -- so the
call site in ``_fit_impl`` is a single call. Behaviour is byte-for-byte
identical to the inlined branch. The lazy in-body ``from ..X import ...``
imports stay inside the function to preserve the original import timing and the
``mrmr -> _mrmr_fit_impl -> mrmr`` cycle break.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _finalise_empty_support_fallback(self, n_engineered_out, cols, data, nbins, target_indices):
    """Empty-RAW-support fallback rescue carved out of ``_fit_impl``.

    Threads the ``MRMR`` instance + fit-body locals explicitly and mutates
    ``self`` in place (``support_`` / ``n_features_`` / ``fallback_used_`` /
    ``fallback_metadata_``). Returns ``None``.
    """
    # No RAW feature survived selection. Engineered-only support (or empty support) lacks a raw signal anchor:
    # on a WIDE engineered candidate pool the top raw signal is frequently out-ranked by overfit-in-sample-MI
    # engineered / high-card columns that do not generalise, leaving 0 raw features despite recoverable signal.
    # Rescue the top-K raw feature(s) clearing the relevance floor (below); a pure-interaction fixture whose raw
    # marginals are all ~0 stays engineered-only. Only triggers when min_features_fallback >= 1.
    self.n_features_ = n_engineered_out
    _min_fb = int(getattr(self, "min_features_fallback", 0) or 0)
    # 2026-05-30 Wave 9.1 fix (loop iter 39): hoist the
    # ``warnings.warn`` OUT of the try block. Pre-fix the warning
    # was inside ``try:`` and the surrounding ``except Exception``
    # caught it under ``simplefilter('error', UserWarning)`` -
    # making the user-facing warning indistinguishable from a real
    # fallback failure (and silently dropping it). Now the
    # try/except scopes only the MI computation; the warning fires
    # afterwards on the successful path.
    _fallback_msg = None
    if _min_fb >= 1 and self.n_features_in_ > 0:
        try:
            # Rank by cached confident MI with the target; take top-K. cached_MIs may not be populated;
            # re-compute from the original frame as a last resort.
            #
            # CRITICAL index-space translation: ``cached_MIs`` is keyed by
            # the candidate index in COLS-SPACE (the screen's working matrix,
            # which ``categorize_dataset`` reorders whenever categoricals
            # exist and which carries the injected target + engineered
            # columns), NOT by the original ``feature_names_in_`` position
            # ``_i``. Reading ``cached_MIs[(_i,)]`` directly mis-aligns every
            # column once the screen reorders (observed: input feature 15
            # ``num_pos_a`` resolving to ``group_id``'s MI 0.075, so the
            # fallback rescued a pure-noise column over the genuine signal).
            # Map original name -> cols-space index exactly as
            # ``compute_mrmr_artifacts`` does (``name_to_data_col``), then
            # look the cached MI up in cols-space while keeping ``support_``
            # in original ``feature_names_in_`` space.
            _name_to_cols_idx = {c: i for i, c in enumerate(cols)}
            _cached = self.cached_MIs if hasattr(self, "cached_MIs") else {}
            # Operands the n-invariant conditional-redundancy sweep deliberately dropped
            # (fully subsumed by a surviving engineered child) must NOT be resurrected by
            # this empty-raw rescue -- the rescue exists for "the screen left 0 raw despite
            # recoverable signal", not to undo an intentional redundancy drop. Excluding them
            # leaves an engineered-only support, which is legitimate and non-empty (the
            # never-empty guarantee only forces a column when n_engineered_out == 0).
            _rescue_redund_dropped = set(getattr(self, "_raw_redundancy_dropped_", None) or ())
            # Cluster members folded into a denoised aggregate (cluster_aggregate 'replace' mode ->
            # ``_cluster_aggregate_removals_``, or a DCD PC1/mean_z swap -> ``cluster_members_``) are ALREADY
            # represented by that aggregate and were deliberately dropped from the support. The empty-raw
            # rescue ranks every raw input by MI(X_j, y) and would otherwise resurrect the highest-MI member
            # (e.g. ``refl0`` of a denoised reflection cluster) as the never-empty / count-floor stand-in,
            # re-injecting the very redundancy the aggregation collapsed. Mirror the same exclusion the
            # raw-retention block, the additional-RFECV rescue pool, and the augmentation already apply.
            _rescue_redund_dropped |= set(getattr(self, "_cluster_aggregate_removals_", None) or ())
            # Operands of SURVIVING engineered features: in the empty-screen case the conditional-redundancy
            # sweep never ran (0 raws selected) so it could not mark them in ``_raw_redundancy_dropped_`` --
            # compute them directly from the surviving recipes so the rescue does not resurrect a raw a
            # surviving engineered child already captures (the underselection redundancy-dedup invariant).
            from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _RESC_TOK_SPLIT
            _resc_raw_set = set(self.feature_names_in_)
            for _en in getattr(self, "_engineered_recipes_", {}) or {}:
                for _tok in _RESC_TOK_SPLIT.split(str(_en)):
                    if not _tok:
                        continue
                    _b = _tok if _tok in _resc_raw_set else (_tok.split("__", 1)[0] if "__" in _tok else None)
                    if _b in _resc_raw_set:
                        _rescue_redund_dropped.add(_b)
            _cm_rescue = getattr(self, "cluster_members_", None)
            if isinstance(_cm_rescue, dict):
                for _anchor, _members in _cm_rescue.items():
                    _rescue_redund_dropped.add(_anchor)
                    if isinstance(_members, (list, tuple, set)):
                        _rescue_redund_dropped.update(_members)
            # The empty-screen rescue must honour the user's search-space restriction (``factors_names_to_use`` /
            # ``factors_to_use``): without this gate it ranks EVERY raw input by MI(X_j, y) and resurrects the
            # global top-MI column even when the caller pinned a disjoint subset, silently leaking a forbidden feature
            # into ``support_``. Build the allowed input-space index set once; ``None`` means "no restriction".
            _rescue_allowed_idx = None
            _fnames_restrict = getattr(self, "factors_names_to_use", None)
            _fidx_restrict = getattr(self, "factors_to_use", None)
            if _fnames_restrict is not None:
                _allowed_names = set(_fnames_restrict)
                _rescue_allowed_idx = {_j for _j, _nm in enumerate(self.feature_names_in_) if _nm in _allowed_names}
            elif _fidx_restrict is not None:
                _rescue_allowed_idx = set(int(_j) for _j in _fidx_restrict)
            _raw_mi = []
            for _i in range(self.n_features_in_):
                if _rescue_allowed_idx is not None and _i not in _rescue_allowed_idx:
                    continue
                _name = self.feature_names_in_[_i] if _i < len(self.feature_names_in_) else None
                if _name in _rescue_redund_dropped:
                    continue
                _cols_idx = _name_to_cols_idx.get(_name)
                _mi = _cached.get((_cols_idx,), 0.0) if _cols_idx is not None else 0.0
                # Keep the cols-space index alongside the input-space index so the rescue can re-run the permutation-significance / redundancy tests on the screen's own matrices.
                _raw_mi.append((_i, float(_mi), _cols_idx))
            # Sort by MI desc; pick top-K.
            # Wave 57 (2026-05-20): secondary key on feature index so
            # tied MI doesn't make the empty-support fallback drift.
            _raw_mi.sort(key=lambda kv: (-kv[1], kv[0]))
            _abs_floor = float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
            _rel_frac = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
            _max_mi = max((m for _, m, _c in _raw_mi), default=0.0)
            # Rescue EVERY raw feature clearing the relevance floor (the stricter of the absolute floor and the relative-to-strongest floor), not just the top
            # ``min_features_fallback`` -- with the empirical-null-debiased ``cached_MIs`` the ranking is honest, so genuine multi-signal pools (e.g. x1/x2/x3 each shadowed by an
            # engineered child) recover fully. But two failure modes the debiased-MI floor alone does NOT catch and that the rescue MUST guard against:
            #   (1) PURE NOISE small-n: the coarse-binning plug-in MI is upward-biased, so a pure-noise leg can leave a tiny residual debiased MI ABOVE the (very small) absolute
            #       floor and be wrongly rescued. The relevance floor is a magnitude test, not a significance test; gate the rescue on a permutation p-value (re-run on the screen's
            #       OWN binning so it matches what produced ``cached_MIs``) so a candidate that sits WITHIN its null is dropped. The never-empty guarantee below still returns one
            #       column when nothing is significant, keeping ``support_`` non-empty.
            #   (2) ALGEBRAIC REDUNDANCY: a block of re-expressions of one signal (financial margin/profit/cost family; 50 copies of a latent) all clear the floor AND are all
            #       individually significant, so significance alone would rescue the whole block and BLOAT the support the conditional-MI screen / DCD deliberately collapsed.
            #       Greedily accept a candidate only when its MI with the already-accepted set is low relative to its own relevance, so a near-duplicate of an accepted column is
            #       dropped. Independent signals (x1/x2/x3, distinct latents) survive this dedup; algebraic twins collapse to one representative.
            _floor = max(_abs_floor, _max_mi * _rel_frac)
            # Cap the floor-based rescue so a pathological pool of many near-identical above-floor raw columns (e.g. 50 copies of one signal, which the empty screen
            # could not collapse) does not balloon the fallback support. ``min_features_fallback`` sets the requested count; we rescue at most a modest multiple of it
            # (the genuine multi-signal fixtures that need the floor-based rescue carry only a handful of distinct above-floor signals, well within this bound).
            _rescue_cap = max(int(_min_fb), 8)
            _above_floor = [(i, _mi, _c) for i, _mi, _c in _raw_mi if _mi > _floor]

            # (1) Permutation-significance gate + (2) redundancy dedup, computed on the screen's own ``data`` / ``nbins`` so the binning matches ``cached_MIs``. Both reuse the
            # CPU permutation / MI njit kernels the screen already uses. Best-effort: if a kernel call fails (degenerate joint, missing cols-space index) the candidate falls
            # through to the magnitude-only path so the never-empty guarantee still holds.
            from ..permutation import mi_direct as _mi_direct_fb
            from ..info_theory import mi as _mi_pair_fb
            _signif_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
            _redundancy_frac = float(os.environ.get("MLFRAME_MRMR_FALLBACK_REDUNDANCY_FRAC", "0.5"))
            _q_dtype = getattr(self, "quantization_dtype", np.int32)
            _accepted: list = []  # input-space indices accepted into the rescue
            _accepted_cols = []  # their cols-space indices (for redundancy MI)
            # ENGINEERED-SURVIVOR CONDITIONING (2026-06-08): seed the redundancy-dedup
            # conditioning set with the cols-space indices of every SURVIVING engineered
            # feature. The empty-RAW-screen rescue fires precisely when 0 raw columns
            # survived the greedy screen but engineered children DID (``n_engineered_out > 0``);
            # on a composite target (``y = a**2/b + log(c)*sin(d)``) the engineered children
            # ``div(sqr(a),abs(b))`` / ``mul(log(c),sin(d))`` fully carry their raw operands'
            # y-information, yet each raw operand a,b,c,d individually clears the relevance
            # floor AND its own permutation null (it IS a genuine operand), and -- being
            # mutually independent uniforms -- none is redundant with ANOTHER RAW operand.
            # So the raw-only dedup admitted all four, re-injecting exactly the operands the
            # engineered children already subsume (the F2/two-pairs regression: a,b,c,d all
            # rescued alongside the correct engineered pairs). Conditioning the dedup on the
            # engineered survivors makes a raw operand whose y-information flows entirely into
            # its engineered child fail the redundancy test (high MI with the child, a large
            # fraction of its own relevance) and drop, while a raw column carrying signal NO
            # engineered survivor captures still passes and is rescued. Structure-independent:
            # correct at every n, no tuning constant beyond the existing ``_redundancy_frac``.
            _name_to_cols_idx_eng = {c: i for i, c in enumerate(cols)}
            # SEED ONLY ON SURVIVING ENGINEERED FEATURES (2026-06-16, s319 under-selection).
            # Condition the dedup on the engineered features that ACTUALLY REACH THE OUTPUT
            # -- i.e. the replayable ``self._engineered_recipes_`` counted in ``n_engineered_out``
            # -- NOT ``self._engineered_features_``, which still carries composites that were
            # SELECTED by the greedy step but then DROPPED downstream (recipeless nested parents,
            # or features that failed the ``fe_min_engineered_mi_prevalence`` gate). A composite
            # about to be dropped must not suppress its raw operands here: doing so loses BOTH the
            # composite (dropped from transform) AND every operand it captures (flagged redundant
            # with it), collapsing the rescue. Measured s319 (y = 1.5*a*b + 0.5*g/k, uniform,
            # n=25000): ``mul(a,b)`` was formed but prevalence-gated out, yet still suppressed raw
            # ``b`` -> the rescue fell to a single raw ``a`` (fe R^2 0.245 vs raw-only 0.556,
            # delta -0.311). Seeding on the (empty here) survivor set lets b,g,k pass the raw-vs-raw
            # dedup -> support {a,b,g,k}, delta +0.0005. When engineered survivors DO reach output
            # (the F2 ``a**2/b + log(c)*sin(d)`` composite case) they remain in ``_engineered_recipes_``
            # and still correctly drop their subsumed operands -- behaviour unchanged there.
            def _surv_eng_name(_r):
                _nm = getattr(_r, "name", None)
                return str(_nm) if _nm is not None else str(_r)
            for _eng_name in (_surv_eng_name(_r) for _r in (self._engineered_recipes_ or [])):
                _eng_ci = _name_to_cols_idx_eng.get(_eng_name)
                if _eng_ci is not None:
                    _accepted_cols.append(_eng_ci)
            # Bound the number of permutation-significance probes: ``_above_floor`` is sorted by debiased MI desc, so the genuine signal sits at the top; on a pathological
            # all-noise wide pool where every candidate fails significance, examining the whole list would run one 32-perm test PER column. Scan at most a modest multiple of the
            # rescue cap (the genuine multi-signal fixtures carry only a handful of distinct above-floor signals, well inside this window).
            _scan_limit = max(int(_rescue_cap) * 4, 16)
            for _i, _mi, _cols_idx in _above_floor[:_scan_limit]:
                if len(_accepted) >= _rescue_cap:
                    break
                if _cols_idx is None:
                    continue
                # Significance gate (#1): keep only candidates that sit ABOVE their permutation null. Pure-noise legs sit within it (p >= alpha) and are dropped.
                try:
                    _sig = _mi_direct_fb(
                        data, x=np.array([_cols_idx], dtype=np.int64), y=target_indices,  # type: ignore[arg-type]  # mi_direct (permutation.py, sibling-owned) accepts this call shape at runtime; its x/y annotation (tuple) is stricter than actual usage
                        factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                        return_null_mean=True, parallelism="none", dtype=_q_dtype, prefer_gpu=False,
                    )
                    _p_value = float(_sig[3])
                except Exception:
                    _p_value = 0.0  # significance unavailable -> fall back to the magnitude-only decision (keep)
                if _p_value >= _signif_alpha:
                    continue
                # Redundancy dedup (#2): drop a candidate whose MI with an already-accepted column is a large fraction of its own relevance (an algebraic / near-duplicate twin).
                _is_redundant = False
                for _acc_cols in _accepted_cols:
                    try:
                        _pair_mi = float(_mi_pair_fb(
                            factors_data=data, x=np.array([_cols_idx], dtype=np.int64),
                            y=np.array([_acc_cols], dtype=np.int64), factors_nbins=nbins, dtype=_q_dtype,
                        ))
                    except Exception:
                        _pair_mi = 0.0
                    if _pair_mi >= _redundancy_frac * max(_mi, 1e-12):
                        _is_redundant = True
                        break
                if _is_redundant:
                    continue
                _accepted.append(_i)
                _accepted_cols.append(_cols_idx)
            _topk = list(_accepted)
            # ``min_features_fallback`` count floor: if the significance/redundancy gates left fewer than the requested K, top up from the remaining above-absolute-floor
            # candidates (magnitude order) so legacy callers asking for >=K always get at least K. The never-empty guarantee then keeps one column even on a fully-null pool.
            # SURVIVING ENGINEERED FEATURES COUNT TOWARD THE FLOOR (2026-06-17): the floor is "support is never empty / has >= K features", and ``get_feature_names_out`` returns
            # raw (``support_``) + engineered. When an engineered feature already survived (``n_engineered_out >= 1``), the floor is met WITHOUT a raw, so do NOT magnitude-top-up a
            # raw that FAILED the permutation-significance gate -- that force-added a pure-noise raw (``e`` in ``y=log(a)*c+0.4*f``: MI 0.0004, p=0.34, only candidate left after the
            # engineered operands a/c were excluded) purely to satisfy a floor the engineered feature already satisfies. Mirrors the ``_redundancy_emptied_raw_`` branch's engineered-
            # only support. The top-up also stays gated on the absolute relevance floor so it never adds a sub-floor column.
            if len(_topk) + n_engineered_out < _min_fb:
                for i, _mi, _c in _raw_mi:
                    if i not in _topk and _mi > _abs_floor:
                        _topk.append(i)
                    if len(_topk) + n_engineered_out >= _min_fb:
                        break
            if not _topk and n_engineered_out == 0 and _raw_mi:
                _topk = [_raw_mi[0][0]]
            elif not _topk and n_engineered_out == 0 and not _raw_mi:
                # The redundancy/cluster exclusion (``_rescue_redund_dropped``) emptied the rescue pool:
                # EVERY raw candidate was marked redundant -- but with a mutually-redundant cluster
                # (e.g. two ~0.997-collinear columns each recorded as the other's cluster member) that
                # leaves the support EMPTY even though one representative should survive. The never-empty
                # guarantee must keep the single strongest column REGARDLESS of the exclusion, so a
                # symmetric redundancy verdict de-duplicates the pair rather than dropping both.
                _raw_mi_all = []
                for _i in range(self.n_features_in_):
                    if _rescue_allowed_idx is not None and _i not in _rescue_allowed_idx:
                        continue
                    _name = self.feature_names_in_[_i] if _i < len(self.feature_names_in_) else None
                    _cols_idx = _name_to_cols_idx.get(_name)
                    _mi = _cached.get((_cols_idx,), 0.0) if _cols_idx is not None else 0.0
                    _raw_mi_all.append((_i, float(_mi)))
                if _raw_mi_all:
                    _raw_mi_all.sort(key=lambda kv: (-kv[1], kv[0]))
                    _topk = [_raw_mi_all[0][0]]
            if _topk:
                # int64 to match every other support_ assignment in the fit body; a bare np.array(list[int]) is
                # int32 on Windows, an inconsistency that can bite dtype-sensitive downstream concatenation.
                self.support_ = np.array(_topk, dtype=np.int64)
                self.n_features_ = len(_topk) + n_engineered_out
                self.fallback_used_ = True
                _top_mi = float(_raw_mi[0][1]) if _raw_mi else 0.0
                _uninformative = _top_mi <= 0.0
                _fallback_msg = (
                    f"MRMR: screening returned 0 features; falling "
                    f"back to the {self.n_features_} raw feature(s) "
                    f"clearing the relevance floor by debiased "
                    f"MI(X_j, y). Set min_features_fallback=0 to "
                    f"disable. fallback_used_=True is set on the "
                    f"estimator."
                )
                if _uninformative:
                    _fallback_msg = (
                        f"{_fallback_msg} All candidates have MI <= 0 "
                        f"(e.g. constant X columns or empty "
                        f"cached_MIs); the returned support_ carries "
                        f"NO signal."
                    )
                # Structured metadata so a downstream report can flag (without log-grepping) that the
                # support_ came from the count floor rather than the relevance gates. n_features==1 with
                # uninformative=True is the dangerous case: a single near-noise column handed to the model.
                self.fallback_metadata_ = {
                    "fallback_used": True,
                    "n_features": int(self.n_features_),
                    "top_mi": _top_mi,
                    "uninformative": bool(_uninformative),
                    "min_features_fallback": int(_min_fb),
                }
        except Exception as _exc:
            logger.warning(
                "MRMR fallback to top-K MI failed: %s. Returning empty support_.",
                _exc,
            )
    if _fallback_msg is not None:
        # logger.warning for log-grepping back-compat AND
        # warnings.warn so simplefilter('error', UserWarning) / test
        # suites can intercept programmatically.
        logger.warning(_fallback_msg)
        import warnings as _w_iter39
        _w_iter39.warn(_fallback_msg, UserWarning, stacklevel=2)
