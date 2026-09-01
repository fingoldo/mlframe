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
It returns nothing - every output is an attribute set on ``self`` - so the
call site in ``_fit_impl`` is a single call. Behaviour is byte-for-byte
identical to the inlined branch. The lazy in-body ``from ..X import ...``
imports stay inside the function to preserve the original import timing and the
``mrmr -> _mrmr_fit_impl -> mrmr`` cycle break.
"""
from __future__ import annotations

import logging
import os
import textwrap
from timeit import default_timer as timer

import numpy as np

from ._helpers import _engineered_recipe_name

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
    # Hoist the
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
            # this empty-raw rescue - the rescue exists for "the screen left 0 raw despite
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
            # sweep never ran (0 raws selected) so it could not mark them in ``_raw_redundancy_dropped_`` -
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
            # Secondary key on feature index so
            # tied MI doesn't make the empty-support fallback drift.
            _raw_mi.sort(key=lambda kv: (-kv[1], kv[0]))
            _abs_floor = float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
            _rel_frac = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
            _max_mi = max((m for _, m, _c in _raw_mi), default=0.0)
            # Rescue EVERY raw feature clearing the relevance floor (the stricter of the absolute floor and the relative-to-strongest floor), not just the top
            # ``min_features_fallback`` - with the empirical-null-debiased ``cached_MIs`` the ranking is honest, so genuine multi-signal pools (e.g. x1/x2/x3 each shadowed by an
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
            # ENGINEERED-SURVIVOR CONDITIONING: seed the redundancy-dedup
            # conditioning set with the cols-space indices of every SURVIVING engineered
            # feature. The empty-RAW-screen rescue fires precisely when 0 raw columns
            # survived the greedy screen but engineered children DID (``n_engineered_out > 0``);
            # on a composite target (``y = a**2/b + log(c)*sin(d)``) the engineered children
            # ``div(sqr(a),abs(b))`` / ``mul(log(c),sin(d))`` fully carry their raw operands'
            # y-information, yet each raw operand a,b,c,d individually clears the relevance
            # floor AND its own permutation null (it IS a genuine operand), and - being
            # mutually independent uniforms - none is redundant with ANOTHER RAW operand.
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
            # - i.e. the replayable ``self._engineered_recipes_`` counted in ``n_engineered_out``
            # - NOT ``self._engineered_features_``, which still carries composites that were
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
            # and still correctly drop their subsumed operands - behaviour unchanged there.
            _surv_eng_name = _engineered_recipe_name
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
                except Exception as e:
                    # FAIL CLOSED. Substituting p = 0.0 is "maximally significant", so a broken probe made the
                    # gate one line below pass for every candidate it scanned -- and this gate exists precisely
                    # because coarse-binned plug-in MI is upward-biased, so the magnitude-only decision it fell
                    # back to re-injects the noise the gate was added to remove. "Significance unavailable" is
                    # not evidence of significance; drop on uncertainty, and say so audibly.
                    logger.warning(
                        "MRMR rescue: permutation-significance probe raised %s for column index %s (%s); "
                        "dropping the candidate rather than admitting it on magnitude alone.",
                        type(e).__name__, _cols_idx, e,
                    )
                    _p_value = 1.0
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
                    except Exception as e:
                        # FAIL CLOSED, same reasoning as the significance gate above: 0.0 is exactly the value
                        # that makes the redundancy test below fail, so a failed pair-MI silently admitted an
                        # algebraic near-duplicate into the support, visible afterwards only as a redundancy
                        # regression with no trace.
                        logger.warning(
                            "MRMR rescue: pair-MI probe raised %s for columns %s vs %s (%s); treating the pair as redundant.",
                            type(e).__name__, _cols_idx, _acc_cols, e,
                        )
                        _pair_mi = float("inf")
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
            # SURVIVING ENGINEERED FEATURES COUNT TOWARD THE FLOOR: the floor is "support is never empty / has >= K features", and ``get_feature_names_out`` returns
            # raw (``support_``) + engineered. When an engineered feature already survived (``n_engineered_out >= 1``), the floor is met WITHOUT a raw, so do NOT magnitude-top-up a
            # raw that FAILED the permutation-significance gate - that force-added a pure-noise raw (``e`` in ``y=log(a)*c+0.4*f``: MI 0.0004, p=0.34, only candidate left after the
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
                # EVERY raw candidate was marked redundant - but with a mutually-redundant cluster
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
                    # Name WHICH of the two causes actually fired. The previous wording listed both as
                    # possibilities, which cost a debugging cycle on a fixture whose features were strongly
                    # informative: the MI table was populated, every value was just <= 0, so the constant-column
                    # explanation was a red herring and the real question was why the estimator returned zeros.
                    _n_scored = len(_raw_mi)
                    _cause = (
                        "the MI table is EMPTY (no candidate was scored at all -- cached_MIs never populated)"
                        if _n_scored == 0
                        else f"all {_n_scored} scored candidate(s) came back with MI <= 0 (top={_top_mi:.6g}); "
                        f"the table was populated, so this is a scoring result, not a missing-data problem -- "
                        f"suspect the discretisation (constant columns, or a binning mode collapsing the column "
                        f"to one bin) rather than the candidate list"
                    )
                    _fallback_msg = f"{_fallback_msg} {_cause}. The returned support_ carries NO signal."
                # Structured metadata so a downstream report can flag (without log-grepping) that the
                # support_ came from the count floor rather than the relevance gates. n_features==1 with
                # uninformative=True is the dangerous case: a single near-noise column handed to the model.
                self.fallback_metadata_ = {
                    "fallback_used": True,
                    "n_features": int(self.n_features_),
                    "top_mi": _top_mi,
                    "uninformative": bool(_uninformative),
                    "n_scored_candidates": len(_raw_mi),
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


def _finalise_fs_results(
    self,
    *,
    MRMR,
    X,
    classes_y,
    cols,
    data,
    nbins,
    predictors,
    start_time,
    verbose,
    cache_key,
    signature,
    ran_out_of_time,
    hashable_params_signature,
    mrmr_cache_bytes_total,
    align_mrmr_gains,
    fit_cache_lock,
):
    """Post-selection finalisation tail: report/logging, ``signature``/``ran_out_of_time_`` bookkeeping,
    the UAED post-fit auto-size elbow trim, ``mrmr_gains_`` length re-alignment, the
    ``support_nonlinear_`` alias re-sync, the group-aware FE demotion final choke point, and the
    process-wide ``MRMR._FIT_CACHE`` store. Carved verbatim out of ``_fit_impl``'s tail (Tier E partial
    split, same convention as ``_finalise_empty_support_fallback`` above) to shrink the parent below the
    monolith budget -- byte-for-byte identical to the inlined block. ``MRMR`` (the class), the fit-cache
    lock/helpers, ``signature``/``ran_out_of_time`` (the fit-body locals this tail finalises onto
    ``self``), and ``_hashable_params_signature``/``_align_mrmr_gains`` are threaded explicitly because
    ``_fit_impl`` resolves ``MRMR`` via a LAZY ``from ..mrmr import (MRMR, ...)`` inside its own body (to
    break the ``mrmr -> _mrmr_fit_impl -> mrmr`` import cycle -- see that function's own docstring); a
    fresh top-level import here would reintroduce the same cycle, and ``signature``/``ran_out_of_time``
    are plain fit-body locals with no ``self`` attribute to read them back from before this tail runs.
    Returns ``self`` (the direct return value of ``_fit_impl`` itself)."""
    _align_mrmr_gains = align_mrmr_gains
    _hashable_params_signature = hashable_params_signature
    _mrmr_cache_bytes_total = mrmr_cache_bytes_total
    _cache_key = cache_key
    _MRMR_FIT_CACHE_LOCK = fit_cache_lock
    if verbose:
        predictors_str = ", ".join([f"{el['name']}: {el['gain']:.4f}" for el in predictors[:50]])
        predictors_str = textwrap.shorten(predictors_str, width=300)
        logger.info("MRMR+ selected %d out of %d features: %s", self.n_features_, self.n_features_in_, predictors_str)

    # Refresh the params slot with POST-fit values before storing: should fit ever resolve/normalise a
    # param in place (RFECV does this with ``scoring``), the entry-time params fingerprint would never
    # match the NEXT fit's ``get_params`` and identical refits would never skip. The data slots
    # (shapes/hashes/columns) stay as computed at fit entry.
    try:
        signature = (*signature[:-1], _hashable_params_signature(self.get_params(deep=True)))
    except Exception as exc:
        logger.debug("mrmr: final signature hash failed; using a unique sentinel (forces a cache miss / no replay for this fit): %r", exc, exc_info=True)
        signature = (*signature[:-1], object())  # unique token => next identical fit refits (conservative)
    self.signature = signature
    # ran_out_of_time was set only by the outer FE-loop deadline (line ~6714). screen_predictors honours
    # self.max_runtime_mins on its OWN and can return a truncated selection without the FE loop ever tripping, so a
    # screen-level timeout was reported as ran_out_of_time_=False - misleading a caller inspecting why selection was
    # thin. OR-in a total-elapsed-vs-budget check so any stage that pushed the fit past its budget is reflected.
    if self.max_runtime_mins is not None and (timer() - start_time) / 60.0 >= self.max_runtime_mins:
        ran_out_of_time = True
    self.ran_out_of_time_ = ran_out_of_time

    # Post-fit UAED auto-size. When enabled, replaces the
    # configured ``min_features_fallback`` floor with an automatic elbow on
    # the per-feature MI gain curve. Relevance trace is taken from the
    # ``mrmr_gains_`` attribute (Wave-7 audit landed this trace in the
    # standard fit output); if missing, this step no-ops.
    if getattr(self, "uaed_auto_size", False):
        try:
            from .._cmi_perm_stop import uaed_elbow
            gains = np.asarray(getattr(self, "mrmr_gains_", []), dtype=np.float64)
            # UAED runs BEFORE the mrmr_gains_ length-alignment below, so at this point ``gains`` is the
            # raw GREEDY log (one entry per confirmed greedy round) - often SHORTER than n_features_ when
            # FE/retention appended features the greedy never scored. The public ``mrmr_gains_`` the caller
            # sees is the n_features_-aligned (zero-padded) trace, so the elbow must be computed on that SAME
            # trace; otherwise a frame whose greedy log has <3 rounds but >=3 final features silently skips
            # the elbow (uaed_elbow_ never set, support never trimmed). Zero-extend to n_features_ to match.
            _nf_uaed = int(getattr(self, "n_features_", gains.size) or gains.size)
            if 0 < gains.size < _nf_uaed:
                gains = np.concatenate([gains, np.zeros(_nf_uaed - gains.size, dtype=np.float64)])
            if gains.size >= 3:
                elbow = int(uaed_elbow(gains))
                if 0 < elbow < gains.size and hasattr(self, "support_"):
                    # ``gains`` is the COMBINED trace (raw greedy gains + zero-padded engineered tail), matching the
                    # transform-time feature order [support_ ..., engineered recipes ...]. The elbow index therefore
                    # lives in COMBINED space, but ``support_`` holds RAW indices only. Slicing raw support by a
                    # combined elbow (and setting n_features_ = support_.size) dropped the engineered count while the
                    # recipes still fired in transform - transform emitted MORE columns than n_features_/mrmr_gains_
                    # claimed (a hard support/output desync). Trim raw support AND engineered recipes in LOCKSTEP so
                    # the retained feature count is exactly elbow+1 in both the state and the transform output.
                    _sup = np.asarray(self.support_)
                    _uaed_recipes = list(getattr(self, "_engineered_recipes_", []) or [])
                    _uaed_keep = elbow + 1  # combined features to retain
                    _uaed_raw_keep = min(_uaed_keep, _sup.size)
                    _uaed_eng_keep = max(0, _uaed_keep - _sup.size)  # <= len(_uaed_recipes): gains was zero-extended to n_features_
                    self.support_ = _sup[:_uaed_raw_keep]
                    if _uaed_recipes and _uaed_eng_keep < len(_uaed_recipes):
                        self._engineered_recipes_ = _uaed_recipes[:_uaed_eng_keep]
                    self.n_features_ = int(self.support_.size) + min(_uaed_eng_keep, len(_uaed_recipes))
                    self.uaed_elbow_ = int(elbow)
        except Exception as e:  # nosec B110 - non-trivial body
            # UAED is best-effort post-fit; don't break fit() on internal hiccup.
            logger.debug("UAED post-fit adjustment failed (%s: %s) -- keeping the pre-UAED support", type(e).__name__, e)
    # Transient FE-escalation fitting target: full-n array, fit-time only.
    self._fe_escalation_y_rank_ = None
    # Transient prewarp ALS reconstruction target: full-n continuous y, fit-time only.
    self._fe_prewarp_y_continuous_ = None

    # MRMR_GAINS LENGTH ALIGNMENT (final form). ``mrmr_gains_`` is the GREEDY selection log;
    # the FINAL feature count diverges from it - SHORTER on a degenerate-frame collapse / redundancy /
    # cluster-aggregate exclusion / p>=n cap / UAED elbow trim, LONGER when FE / retention / pseudo-
    # remix re-add appended features the greedy log never scored. The public contract + downstream
    # expect ``len(mrmr_gains_) == n_features_`` (TestSupportGainsAlignment). Reconcile HERE, after every
    # support/n_features_ mutation above is final: keep the top screening gains (descending - what the
    # UAED elbow already consumed) and pad any FE tail with 0.0. Byte-identical when already aligned.
    # NB: re-run once more after the group-aware demotion below (the last n_features_ mutation).
    _align_mrmr_gains(self)

    # SUPPORT_NONLINEAR_ ALIAS RE-SYNC. ``support_nonlinear_`` is set right after the FIRST support_
    # assignment as an alias of the pure-MI support_, but several later passes (usability-aware RAW
    # retention, count-floor rescue, UAED elbow trim) REASSIGN self.support_ to a NEW array, leaving the
    # alias pointing at the stale pre-mutation array. By contract support_nonlinear_ IS the final pure-MI
    # support_, so re-point it here after every support_ mutation (the separate linear/universal lists,
    # when present, are untouched).
    if hasattr(self, "support_nonlinear_"):
        self.support_nonlinear_ = self.support_

    # GROUP-AWARE FE DEMOTION (final choke point, AFTER every reintroduction pass). Under
    # ``group_aware_mi=True`` the raw-feature relevance screen (``evaluate_candidate``) already demotes
    # a between-group-level "leak" raw feature via group-blocked I(X;Y|G) instead of the naive global MI
    # - but EVERY engineered-feature producer (the unary/binary pair search, the hybrid-orth Hermite
    # basis, polynom/orthogonal families, ...) scores its candidates with the PLAIN global plug-in MI;
    # none of them consult ``get_group_mi()``. A pair/basis interaction BUILT FROM a demoted leak raw
    # feature (e.g. ``mul(leak_raw, unrelated)`` or a Hermite product ``leak_raw__He2 * other__He1``)
    # still carries the leak's between-group signal and clears every naive-MI acceptance gate. An
    # EARLIER attempt at this check (right after the initial ``selected_vars``/engineered-recipe build)
    # was found to be undone by later passes - usability-aware retention (``_retain_extra`` above)
    # re-attaches recipes it judges linearly-useful by its OWN criterion, independent of any earlier
    # group-aware verdict - so this MUST run last, after UAED / retention / every other
    # ``self._engineered_recipes_``/``self._engineered_features_`` mutation, right before returning.
    # Recipes still materialised in ``data``/``cols`` (the common case) are re-scored directly; a
    # retention-only recipe (usability-aware retention re-attaches recipes from its own recompute,
    # independent of ``cols``/``data``) is replayed via ``apply_recipe`` + discretised the SAME way the
    # retention pass itself does, so it gets the SAME group-aware check either way. Demotes any whose
    # within-group MI comes back EXACTLY zero (no genuine within-group signal at all - a column with any real, however small, within-group
    # signal survives). A nan group-MI (misaligned segments) is inconclusive and left alone, mirroring the
    # raw-feature gate's own fallback. No-op - and therefore byte-identical - when group_aware_mi is off
    # / no groups were supplied this fit (``get_group_mi()`` returns ``None``).
    try:
        from ..info_theory._state_and_dispatch import get_group_mi as _get_group_mi_final
        _gmi_final_payload = _get_group_mi_final()
    except Exception as e:
        logger.debug("get_group_mi_final() failed: %s", e)
        _gmi_final_payload = None
    _eng_recipes_final = getattr(self, "_engineered_recipes_", None) or []
    if _gmi_final_payload is not None and _eng_recipes_final:
        try:
            from ..info_theory._group_mi import group_blocked_mi as _group_blocked_mi_final
            from ..info_theory._group_mi import group_relevance_mi as _group_relevance_mi_final

            _cols_idx_f = {nm: i for i, nm in enumerate(cols)}
            _gsi_f, _goff_f, _gmr_f, _gsw_f = _gmi_final_payload
            _classes_y_arr_f = np.asarray(classes_y)
            _g_n_bins_y_f = int(_classes_y_arr_f.max()) + 1
            _group_dropped_final: set = set()
            for _recipe in _eng_recipes_final:
                _rname = str(getattr(_recipe, "name", ""))
                _cidx = _cols_idx_f.get(_rname)
                if _cidx is not None:
                    # Fast path: the engineered column is still materialised in ``data``/``cols``.
                    _grp_mi_f = _group_relevance_mi_final(
                        data, (int(_cidx),), _classes_y_arr_f, np.asarray(nbins), _g_n_bins_y_f,
                        _gsi_f, _goff_f, min_rows=_gmr_f, size_weighted=_gsw_f, dtype=self.quantization_dtype,
                    )
                else:
                    # Retention-only recipe (usability-aware retention re-attaches it from its OWN
                    # recompute, bypassing ``cols``/``data`` entirely - see the comment above). Replay
                    # the SAME recompute + discretise the retention pass itself uses, then group-block
                    # directly (a single already-discretised column needs no ``merge_vars`` combination).
                    try:
                        from ..engineered_recipes._recipe_dispatch import apply_recipe as _apply_recipe_final
                        from ..discretization import discretize_array as _discretize_array_final

                        _cv_f = np.asarray(_apply_recipe_final(_recipe, X), dtype=np.float64).ravel()
                        _cv_f = np.nan_to_num(_cv_f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        if _cv_f.shape[0] != _classes_y_arr_f.shape[0]:
                            continue  # can't align - leave this recipe alone
                        _codes_f = _discretize_array_final(_cv_f, n_bins=int(self.quantization_nbins), method=self.quantization_method, dtype=self.quantization_dtype)
                        _grp_mi_f = _group_blocked_mi_final(
                            _codes_f, _classes_y_arr_f, _gsi_f, _goff_f,
                            n_bins_x=int(self.quantization_nbins), n_bins_y=_g_n_bins_y_f,
                            min_rows=_gmr_f, size_weighted=_gsw_f, use_mm=True,
                        )
                    except Exception as e:
                        logger.debug("_fit: within-group MI replay/discretise failed for recipe %r, leaving it alone: %s", _rname, e)
                        continue  # can't replay/discretise - leave this recipe alone (conservative)
                if _grp_mi_f == _grp_mi_f and _grp_mi_f <= 0.0:  # not nan and exactly zero within-group signal
                    _group_dropped_final.add(_rname)
            if _group_dropped_final:
                self._engineered_recipes_ = [_r for _r in _eng_recipes_final if str(getattr(_r, "name", "")) not in _group_dropped_final]
                _eng_features_final = getattr(self, "_engineered_features_", None) or []
                self._engineered_features_ = [_fn for _fn in _eng_features_final if str(_fn) not in _group_dropped_final]
                self.n_features_ = (
                    len(self.support_) + len(self._engineered_recipes_)
                    if hasattr(self, "support_")
                    else int(getattr(self, "n_features_", 0)) - len(_group_dropped_final)
                )
                if verbose:
                    logger.info(
                        "MRMR: demoted %d engineered feature(s) with zero within-group relevance MI "
                        "under group_aware_mi (between-group-only leak signature reaching the final selection): %s",
                        len(_group_dropped_final), sorted(_group_dropped_final),
                    )
        except Exception as _group_final_exc:
            logger.debug(
                "MRMR group-aware FE demotion (final choke point) failed (%s: %s); keeping the naive-MI selection.",
                type(_group_final_exc).__name__, _group_final_exc,
            )

    # Final re-alignment: the group-aware demotion just above is the LAST n_features_ mutation, and it does
    # not touch mrmr_gains_. Re-run the trim/pad here so the len(mrmr_gains_) == n_features_ contract holds
    # even when the demotion dropped >=1 engineered feature (idempotent no-op otherwise).
    _align_mrmr_gains(self)

    # Store self in process-wide cache so cloned MRMR instances fit on the same (X, y) arrays can replay
    # this fitted state instead of re-running cat-FE + permutation. Bound the LRU by ``fit_cache_max``;
    # the default (4) covers a typical model suite without thrashing and long-lived workers no longer leak.
    # ``_skip_fit_cache`` (private, non-BaseEstimator attr set by the stability-selection outer loop on
    # its throwaway bootstrap-replicate sub-fits): each replicate
    # fits a DIFFERENT row-subsample every call, so its cache key never repeats and is a guaranteed
    # future miss - storing it only serves to evict a legitimately-reusable entry from an unrelated
    # concurrent caller sharing the same process-wide 4-entry LRU. Unlike ``fit_cache_max=0`` (which
    # clears the WHOLE shared cache as an operator-level opt-out), this skips only THIS instance's own
    # store, leaving every other entry untouched.
    #
    # Placed at the very end of fit (after every post-fit self-mutation: UAED elbow, mrmr_gains_
    # alignment, support_nonlinear_ re-sync, group-aware FE demotion) rather than right after the main
    # selection loop -- a concurrent fit's own byte-cap eviction walks EVERY cached instance's
    # ``vars(instance)`` via ``_mrmr_cache_bytes_total`` (see below); publishing ``self`` into
    # ``_FIT_CACHE`` before those later blocks finished ADDING/REASSIGNING instance attributes let another
    # thread observe ``self.__dict__`` mid-mutation and raise ``RuntimeError: dictionary changed size
    # during iteration`` (reproduced live via ``test_concurrent_real_fits_no_exception_and_bounded_cache``
    # -- 6 threads x 12 real fits reliably triggered it within a few runs). The cache lock only ever
    # serialised the ``_FIT_CACHE`` container itself, not a stored VALUE's own further mutation by its
    # owning thread; the real fix is publishing ``self`` only once it is fully finalised, not widening the
    # lock (self's own post-store mutations are single-threaded from this thread's perspective and were
    # never a race against another thread's SAME instance -- the race was always a torn READ of one
    # thread's instance by ANOTHER thread's eviction walk).
    if _cache_key is not None and not getattr(self, "_skip_fit_cache", False):
        # Whole store + LRU/byte-cap eviction held under the cache lock so a concurrent fit cannot interleave its
        # own ``__setitem__``/``popitem``/``move_to_end`` (KeyError, wrong-entry eviction) or iterate ``.values()``
        # via ``_mrmr_cache_bytes_total`` while another thread mutates the dict.
        with _MRMR_FIT_CACHE_LOCK:
            # concurrency audit (TOCTOU): between this thread's earlier locked miss-check and this
            # locked store, another thread with the IDENTICAL cache key may have run its own full fit and
            # already stored its result here. Both instances are independently correct (same X/y/params), but
            # picking a FIRST-WRITER-WINS policy (``setdefault`` instead of unconditional overwrite) makes the
            # canonical cached entry deterministic by arrival order at this lock rather than by whichever
            # thread happened to reach this exact line last - and avoids uselessly replacing an already-valid
            # entry with an equivalent one. ``self`` remains fully usable to ITS OWN caller either way; only
            # which instance becomes the shared replay source for FUTURE cache hits is affected.
            MRMR._FIT_CACHE.setdefault(_cache_key, self)
            MRMR._FIT_CACHE.move_to_end(_cache_key)
            # ``fit_cache_max=0`` is the operator-explicit "disable LRU" sentinel
            # (e.g. for memory-constrained suites where the 4-entry cache pins
            # too much state). The previous ``or 4`` form silently restored the
            # default cap, so cache-off was a no-op. ``None`` (unset attr) still
            # folds to 4.
            _cap_raw = getattr(self, "fit_cache_max", 4)
            _cap = int(4 if _cap_raw is None else _cap_raw)
            if _cap <= 0:
                MRMR._FIT_CACHE.clear()
            else:
                while len(MRMR._FIT_CACHE) > _cap:
                    MRMR._FIT_CACHE.popitem(last=False)
            # Byte-size cap on top of entry count: a 1k-feature suite carrying 4 cached MRMR instances each
            # holding _selectors_ / _engineered_features_ state can exceed 1 GB of process RSS.
            # ``fit_cache_max_mb`` (default 1024 MB; env override ``MLFRAME_MRMR_FIT_CACHE_MAX_MB``) bounds the
            # aggregate cache footprint.
            _mb_cap_raw = getattr(self, "fit_cache_max_mb", None)
            if _mb_cap_raw is None:
                _env_mb = os.environ.get("MLFRAME_MRMR_FIT_CACHE_MAX_MB", "1024")
                try:
                    _mb_cap = float(_env_mb)
                except ValueError:
                    _mb_cap = 1024.0
            else:
                try:
                    _mb_cap = float(_mb_cap_raw)
                except (TypeError, ValueError):
                    _mb_cap = 1024.0
            if _mb_cap > 0 and _cap > 0 and len(MRMR._FIT_CACHE) > 0:
                _byte_cap = _mb_cap * (1024**2)
                while len(MRMR._FIT_CACHE) > 1 and _mrmr_cache_bytes_total() > _byte_cap:
                    MRMR._FIT_CACHE.popitem(last=False)

    return self
