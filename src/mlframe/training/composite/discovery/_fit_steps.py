"""Helpers carved out of ``_fit`` to keep that module under its size budget."""

from __future__ import annotations

import logging
import os
import threading
from timeit import default_timer as timer
from typing import Any

import numpy as np

from .screening import (
    _aggregate_mi_per_feature,
    _aggregate_mi_per_feature_excluding,
    _extract_column_array,
    _mi_to_target,
    _mi_to_target_prebinned,
)
from ..transforms import UnknownTransformError, get_transform
from ._skew_gate import left_skewed_right_tail_skips
from ._fit_ram import _phase_ram_report, _process_mem_mb  # noqa: F401 -- _process_mem_mb re-exported for back-compat
from ._eval import build_unary_base_context, eval_one_transform, release_context_matrices
from ._fit_multibase import apply_multi_base_forward_stepwise
from ._per_base_x import release_fold_caches
from ._eval_stats import near_collinear_keep_mask
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# Sentinel base key for the dedicated UNARY (``requires_base=False``)
# evaluation context. Unary transforms ignore the base column entirely, so they
# are scored ONCE against the FULL feature matrix (no base dropped) rather than
# bound to an arbitrary first base. The empty string is also the
# ``CompositeTargetEstimator`` default ``base_column`` for base-less specs, and
# ``compose_target_name(..., base="")`` renders the base-free 2-segment name.
_UNARY_BASE_SENTINEL = ""


logger = logging.getLogger(__name__)


_UNARY_BASE_SENTINEL = ""


def _apply_honest_holdout_stages(self, df, target_col, kept_specs, usable_features, train_idx, y_full, _honest_holdout_idx, _ram_profiler_on, _ram_state, _phase_ram_report):
    """Run the holdout RMSE gate on the selection rows, then stamp the honest gain from the report rows.

    The gate drops specs, so it reads only the selection half; the stamped number comes from rows no gate saw, which
    is what keeps it free of the winner's curse. A holdout too small to halve gives both stages the whole of it.
    """
    # Honest-holdout OOS predictive-error gate. MI (and the MI-based honest re-score below) is
    # monotone-invariant, so a spec can raise MI while WORSENING the y-scale OOS RMSE (canonical case:
    # a ratio dividing by a small noisy base amplifies noise). Replicate the real prediction objective
    # on the never-touched holdout with a tiny model and DROP specs whose y-scale holdout RMSE loses to
    # raw y. This is the only OOS predictive gate on the ``screening="mi"`` path. Runs before the MI
    # re-score so the heavier per-spec MI pass only touches survivors (``honest_rmse_gate_enabled``).
    if kept_specs and getattr(self.config, "honest_rmse_gate_enabled", True):
        from ._honest_rmse_gate import apply_honest_rmse_gate

        # The SELECTION half: this gate drops specs, so it must not read the rows the reported honest number comes from.
        _select_idx = getattr(self, "honest_holdout_select_idx_", _honest_holdout_idx)
        # Both passes share their fit rows and read the two halves of one holdout: gather each once, fit each model once.
        self._honest_gate_memo = {"key": (id(df), tuple(usable_features)), "mats": {}, "fits": {},
                                  "holdout": None if _honest_holdout_idx is None else np.sort(np.asarray(_honest_holdout_idx)), "holdout_x": None}
        try:
            kept_specs = apply_honest_rmse_gate(self, df, target_col, kept_specs, usable_features, train_idx, _select_idx, y_full)
            # The exported RMSE gain comes from the report half: the one above is conditioned on having passed the gate.
            _report_idx = getattr(self, "honest_holdout_report_idx_", None)
            if kept_specs and _report_idx is not None and _select_idx is not None and not np.array_equal(_report_idx, _select_idx):
                apply_honest_rmse_gate(self, df, target_col, kept_specs, usable_features, train_idx, _report_idx, y_full, record_only=True)
        finally:
            self._honest_gate_memo = None
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "honest_rmse_gate_done")

    # Honest holdout re-score (SA27). The winner set is now FINAL; re-score ONLY these
    # survivors on the holdout the discovery never touched (see ``apply_honest_holdout``).
    if kept_specs and _honest_holdout_idx is not None and _honest_holdout_idx.size:
        from ._honest_holdout import apply_honest_holdout

        # The REPORT half: no gate or ranking reads these rows, so the stamped gain is free of the winner's curse the
        # carve exists to remove (with a holdout too small to halve, both roles share it and this is the old behaviour).
        apply_honest_holdout(
            self, df, target_col, kept_specs, usable_features,
            train_idx, getattr(self, "honest_holdout_report_idx_", _honest_holdout_idx), y_full,
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "honest_holdout_rescore_done")
    return kept_specs


def _evaluate_work_items(self, base_candidates, _base_contexts, _skip_right_tail, _unary_evaluated, _unary_context_available, y_train, y_screen, target_col) -> list:
    """Score every (base, transform) pair against its base context, in (base, transform) order.

    Unary transforms go to the full-X sentinel context once each; base transforms run per base. The dispatch is a
    threading pool, so the large per-base matrices are shared by reference rather than pickled.
    """
    _work_items: list[tuple[str, str, Any]] = []
    for base in base_candidates:
        if base not in _base_contexts:
            continue
        for transform_name in self.config.transforms:
            if transform_name in _skip_right_tail:
                continue
            try:
                transform = get_transform(transform_name)
            except UnknownTransformError as exc:
                log_throttle(
                    logger, "composite_discovery_fit_unknown_transform", logging.WARNING,
                    "[CompositeTargetDiscovery] %s; skipping.",
                    exc,
                )
                continue
            if not transform.requires_base:
                if transform_name in _unary_evaluated:
                    continue
                _unary_evaluated.add(transform_name)
                # Score the unary against the FULL-X sentinel context, not the
                # current loop's ``base``. Fall back to the real base only if
                # the sentinel context could not be built (degenerate empty
                # feature matrix) so the unary still gets evaluated.
                _unary_base = _UNARY_BASE_SENTINEL if _unary_context_available else base
                _work_items.append((_unary_base, transform_name, transform))
                continue
            _work_items.append((base, transform_name, transform))

    # A base's gathered matrices live from its first scored transform to its last: the work list is in base order, so
    # only the bases in flight hold copies.
    _pending: dict = {}
    for _b, _tn, _t in _work_items:
        _pending[_b] = _pending.get(_b, 0) + 1
    _pending_lock = threading.Lock()

    def _eval_and_release(_b, _tn, _t):
        """Evaluate one transform, then free the base's matrices once its last pending transform is done."""
        try:
            return eval_one_transform(self, _b, _tn, _t, base_contexts=_base_contexts, y_train=y_train, y_screen=y_screen, target_col=target_col)
        finally:
            with _pending_lock:
                _pending[_b] -= 1
                _last = _pending[_b] == 0
            if _last:
                release_context_matrices(_base_contexts[_b])

    # Single parallel dispatch over the flat
    # ``_work_items`` list. Joblib preserves input order so
    # ``candidates`` ends up in (base, transform) iteration order
    # identical to the legacy nested-loop serial path. joblib
    # threading backend keeps closure capture cheap (no pickling),
    # which is critical for the large ``x_remaining_matrix`` /
    # ``_x_prebinned`` arrays the body reads. Most of the compute
    # (transform.fit / transform.forward / _mi_to_target_prebinned
    # / bootstrap MI loop) is numpy / numba which releases the GIL,
    # so threading scales close to linearly up to cpu_count.
    # 0 = auto: cap at the number of work items and cpu_count. 1 = serial.
    _n_jobs_raw = getattr(self.config, "discovery_n_jobs", 0)
    _n_jobs_raw = 1 if _n_jobs_raw is None else int(_n_jobs_raw)
    if _n_jobs_raw == 0:
        _n_jobs_disc = max(1, min(len(_work_items), os.cpu_count() or 1))
    else:
        _n_jobs_disc = max(1, _n_jobs_raw)
    if _n_jobs_disc > 1 and len(_work_items) > 1:
        from joblib import Parallel as _Parallel, delayed as _delayed

        _results = _Parallel(
            n_jobs=_n_jobs_disc,
            backend="threading",
            prefer="threads",
        )(_delayed(_eval_and_release)(_b, _tn, _t) for _b, _tn, _t in _work_items)
    else:
        _results = [_eval_and_release(_b, _tn, _t) for _b, _tn, _t in _work_items]
    return [c for _r in _results if _r for c in _r]


def _mi_y_baseline(self, *, x_prebinned, per_feat_y_full, surviving_orig_idx, base, col_index, drop_idx, mi_aggregation,
                   per_feat_y_knn_full, x_remaining_matrix, y_screen, mi_kwargs) -> float:
    """MI(y, X_remaining) for one base, over exactly the columns MI(T, X_remaining) will see."""
    if x_prebinned is not None:
        if per_feat_y_full is not None and surviving_orig_idx is not None:
            # Aggregate over exactly the columns MI(T, X) sees: the base dropped AND the near-duplicates the dedup pruned.
            # Averaging MI(y, X) over every column let a high-MI duplicate block inflate mi_y and bias mi_gain low for
            # the bases the dedup exists for (the knn branch below was already pruned in lockstep).
            return _aggregate_mi_per_feature(per_feat_y_full[surviving_orig_idx], mi_aggregation)
        elif per_feat_y_full is not None and base in col_index:
            # Decompose: aggregate the precomputed per-feature MI over all
            # features except the base column (bit-identical to re-MI'ing
            # x_remaining vs y, since per-feature MI is base-invariant). The
            # exclude-aware aggregate masks out the base entry in place, so
            # no per-base (n, F-1) np.delete copy is materialised for the
            # baseline -- only the held-alive transform-consumer matrices remain.
            return _aggregate_mi_per_feature_excluding(
                per_feat_y_full,
                mi_aggregation,
                drop_idx,
            )
        elif per_feat_y_full is not None:
            return _aggregate_mi_per_feature(
                per_feat_y_full,
                mi_aggregation,
            )
        else:
            return _mi_to_target_prebinned(
                x_prebinned,
                y_screen,
                **mi_kwargs,
            )
    elif per_feat_y_knn_full is not None and surviving_orig_idx is not None:
        # knn baseline from the precomputed base-invariant per-feature vector: aggregate over the surviving
        # original-column indices. Bit-identical to _mi_to_target(x_remaining_matrix, y_screen, knn) -- the same
        # set of single-column MI(y, x_j) values (each on its own per-pair-finite rows), aggregated in the same
        # mean/sum reduction -- without re-running ~50 Kraskov estimators per base.
        return _aggregate_mi_per_feature(
            per_feat_y_knn_full[surviving_orig_idx],
            mi_aggregation,
        )
    else:
        return _mi_to_target(
            x_remaining_matrix,
            y_screen,
            n_neighbors=self.config.mi_n_neighbors,
            random_state=self.config.random_state,
            estimator=self.config.mi_estimator,
            **mi_kwargs,
        )


def _point_mass_skips_logged(y_train) -> set:
    """Curved y-compressors to skip because the target is a point mass, logging the skip when there is one.

    Their convex inverse cannot reconstruct spread from a point mass -- a production run measured pred_std at 1.3-1.8%
    of target_std for log/cbrt on a zero-inflated amount, after paying for the fits.
    """
    from ._point_mass_gate import point_mass_curved_inverse_skips, point_mass_fraction

    _skip_curved = point_mass_curved_inverse_skips(y_train)
    if _skip_curved:
        logger.info(
            "[CompositeTargetDiscovery] %.0f%% of the target sits on a single value; skipping curved y-compressors %s "
            "(their convex inverse cannot reconstruct spread from a point mass -- a production run measured "
            "pred_std at 1.3-1.8%% of target_std for log/cbrt on a zero-inflated amount, after paying for the fits). "
            "Clipping-style y-transforms keep a piecewise-linear inverse and stay.",
            100.0 * point_mass_fraction(y_train), sorted(_skip_curved),
        )
    return set(_skip_curved)


def _reason_from_ledger(self, spec_name: Any) -> str:
    """The rejecting stage and reason the ledger recorded last for ``spec_name``, or the list of gates when it recorded none.

    Every gate downstream of the MI gate appends its verdict to the rejection ledger, so the spec's own stage is known;
    the report used to print one fixed list of gate names for all of them, which named neither the y-scale holdout gate,
    the honest RMSE gate, the honest-OOF floor nor the structural-fragility gate that drop the most specs today.
    """
    rows = [r for r in (getattr(self, "rejection_ledger_", None) or []) if r.get("spec_name") == str(spec_name)]
    if rows:
        last = rows[-1]
        reason = str(last.get("reason") or "").strip()
        return f"rejected at the {last.get('stage')} stage" + (f": {reason}" if reason else "")
    return "dropped after the MI gate by a filter that records no per-spec verdict " "(top_k_after_mi trim / multi-base dedup)"


def _fit_step1_strict_no_op(self, st, df, train_idx):
    """Step 1 of fit: lines starting at ``for base in st.base_candidates:``."""
    for base in st.base_candidates:
        base_train = _extract_column_array(df, base)[train_idx]
        self._auto_base_pool[base] = base_train
        base_screen = base_train[st.sample_idx]
        # A synthetic interaction base is not a feature; its parents carry it, so they leave x_remaining as a base does.
        from mlframe.training.composite._synthetic_bases import dropped_columns

        _dropped = dropped_columns(base, st._col_index)
        # A base the user named explicitly past the corr filter is not among the usable features, so nothing leaves x_remaining for it.
        _readmitted_by_name = bool(base) and not _dropped and base in (getattr(self, "_corr_filtered_bases_", None) or {})
        if _dropped or _readmitted_by_name:
            _drop_idx: int | list[int] = st._col_index[base] if base in st._col_index else [st._col_index[c] for c in _dropped]
            st._x_prebinned = np.delete(st._full_x_prebinned, _drop_idx, axis=1) if st._full_x_prebinned is not None else None
            if st._use_lazy_prebin:
                # No float plane on the lazy path -- the base-dropped float matrix
                # is never read by the bin-estimator eval (it consumes only the
                # prebinned codes). Carry a zero-row float32 proxy of the right
                # WIDTH so the ``x_remaining_matrix.shape[1]`` index/empty checks
                # and the eval body's shape reads stay correct without allocating
                # the (n, F-1) plane. Dedup reads the streamed Gram, not these
                # values.
                _rem_cols = st._x_prebinned.shape[1] if st._x_prebinned is not None else 0
                st.x_remaining_matrix = np.empty((0, _rem_cols), dtype=np.float32)
            else:
                assert st._full_x_matrix is not None  # built above whenever not _use_lazy_prebin
                st.x_remaining_matrix = np.delete(st._full_x_matrix, _drop_idx, axis=1)
            # Original-column indices that survive base-drop (used to derive the knn mi_y baseline from the
            # precomputed base-invariant per-feature vector); dedup prunes this in lockstep with x_remaining_matrix.
            _surviving_orig_idx = np.delete(np.arange(len(st._usable_features_list)), _drop_idx)
            _ctx_keep = None
            if st._dedup_x_remaining and st.x_remaining_matrix.shape[1] > 1:
                _keep = st._streamed_dedup.keep_mask(_drop_idx) if st._use_lazy_prebin else near_collinear_keep_mask(
                    st.x_remaining_matrix,
                    corr_threshold=st._dedup_corr_thr,
                )
                if not _keep.all():
                    st.x_remaining_matrix = st.x_remaining_matrix[:, _keep]
                    if st._x_prebinned is not None:
                        st._x_prebinned = st._x_prebinned[:, _keep]
                    if _surviving_orig_idx is not None:
                        _surviving_orig_idx = _surviving_orig_idx[_keep]
                    _ctx_keep = _keep
        else:
            # Invariant: every base in usable_features is in _col_index, so this arm is unreachable; keeping the base in its own x_remaining would leak it into the MI baseline, so skip rather than mis-score.
            log_throttle(
                logger, "composite_discovery_fit_base_not_in_usable_features", logging.ERROR,
                "[CompositeTargetDiscovery] invariant violated: base %r not in usable_features index; skipping (would leak base into its own x_remaining).",
                base,
            )
            continue
        if st.x_remaining_matrix.shape[1] == 0:
            continue
        _mi_kwargs: dict[str, Any] = dict(
            nbins=int(self.config.mi_nbins),
            aggregation=getattr(self.config, "mi_aggregation", "mean"),
        )
        mi_y_for_base = _mi_y_baseline(
            self, x_prebinned=st._x_prebinned, per_feat_y_full=st._per_feat_y_full, surviving_orig_idx=_surviving_orig_idx, base=base,
            col_index=st._col_index, drop_idx=_drop_idx, mi_aggregation=st._mi_aggregation, per_feat_y_knn_full=st._per_feat_y_knn_full,
            x_remaining_matrix=st.x_remaining_matrix, y_screen=st.y_screen, mi_kwargs=_mi_kwargs,
        )
        # The context keeps the surviving column indices, not the (rows x features) copies: every base's copies held at
        # once were the discovery peak. The evaluator gathers them when the base's first transform runs and drops them
        # after its last (``discovery._eval.context_matrices``); a gather equals the delete-then-keep build byte for byte.
        st._base_contexts[base] = dict(
            base_train=base_train,
            base_screen=base_screen,
            x_remaining_matrix=None,
            _x_prebinned=None,
            _cols=(_drop_idx, _ctx_keep),
            mi_y_for_base=mi_y_for_base,
            _mi_kwargs=_mi_kwargs,
            # Shrunk-domain ``mi_y_compare`` memo shared by all transforms on this base (they share the ``valid_screen`` mask); lock guards the eval threads.
            _mi_y_compare_memo={},
            _mi_y_compare_memo_lock=threading.Lock(),
        )

    # Dedicated UNARY context (full feature matrix, sentinel base) so unary
    # (``requires_base=False``) transforms are scored ONCE against full X and
    # their mi_gain is invariant to auto-base ranking order. Built in the
    # ``_eval`` sibling to keep this module under the LOC threshold; see
    # ``build_unary_base_context`` for the full rationale.
    # On the lazy path the float plane is None; the unary context (bin
    # estimator) reads ``full_x_matrix`` only for its ``.shape[1]`` width guard
    # and the dead ``x_remaining_matrix`` store, so hand it a zero-row proxy of
    # the full column count -- never read for values on this gate.
    st._unary_full_x = st._full_x_matrix


def _fit_step2_full_column_count(self, st, train_idx, target_col, df):
    """Step 2 of fit: lines starting at ``if st._full_x_prebinned is not None: # the bin estimator reads only co``."""
    if st._full_x_prebinned is not None:  # the bin estimator reads only codes: the float plane is dead from here on
        _full_width = st._full_x_prebinned.shape[1] if st._full_x_prebinned is not None else 0
        st._unary_full_x = np.empty((0, _full_width), dtype=np.float32)
    assert st._unary_full_x is not None  # built above whenever not _use_lazy_prebin
    st._unary_ctx = build_unary_base_context(
        full_x_matrix=st._unary_full_x,
        full_x_prebinned=st._full_x_prebinned,
        per_feat_y_full=st._per_feat_y_full,
        y_screen=st.y_screen,
        n_train=train_idx.size,
        sample_idx=st.sample_idx,
        mi_aggregation=st._mi_aggregation,
        mi_nbins=int(self.config.mi_nbins),
        mi_n_neighbors=self.config.mi_n_neighbors,
        random_state=self.config.random_state,
        mi_estimator=self.config.mi_estimator,
    )
    if st._unary_ctx is not None:
        st._base_contexts[_UNARY_BASE_SENTINEL] = st._unary_ctx
    st._full_views = {"x": None if st._full_x_prebinned is not None else st._full_x_matrix, "pb": st._full_x_prebinned}
    st.x_remaining_matrix = st._x_prebinned = st._unary_full_x = None
    if st._full_x_prebinned is not None:
        st._full_x_matrix = None
    for _ctx in st._base_contexts.values():
        if "_cols" in _ctx:
            _ctx["_full_views"] = st._full_views
            _ctx["_matrices_lock"] = threading.Lock()

    # Build flat (base, transform_name, transform) work list. Base-dependent
    # transforms iterate per base normally. Unary (``requires_base=False``)
    # transforms route to the dedicated ``_UNARY_BASE_SENTINEL`` context exactly
    # ONCE (full-X scoring, base-free name) instead of being bound to whichever
    # real base they happened to pair with first. ``_unary_evaluated`` still
    # dedups so each unary appears once. Keeping the build serial outside the
    # parallel dispatch preserves deterministic (base, transform) ordering.
    st._unary_context_available = _UNARY_BASE_SENTINEL in st._base_contexts
    st._skip_right_tail = left_skewed_right_tail_skips(st.y_train)
    if st._skip_right_tail:
        logger.info(
            "[CompositeTargetDiscovery] target is left-skewed; skipping right-tail compressors %s (they would deepen the "
            "skew). yeo_johnson_y stays: it fits lambda > 1 for a left tail.", sorted(st._skip_right_tail),
        )
    st.candidates.extend(_evaluate_work_items(
        self, st.base_candidates, st._base_contexts, st._skip_right_tail | _point_mass_skips_logged(st.y_train), st._unary_evaluated, st._unary_context_available, st.y_train, st.y_screen, target_col,
    ))
    # The per-base float and prebinned copies and the full matrices are read by nothing past this point; holding them
    # kept (bases + 1) x rows x features x 6 bytes resident through the rerank, the holdout gates and the re-score.
    st._base_contexts = st._full_x_matrix = st._full_x_prebinned = st._unary_full_x = st._unary_ctx = st.x_remaining_matrix = st._x_prebinned = self._screen_matrix_stash = None
    if st._ram_profiler_on:
        _phase_ram_report(st._ram_state, "transforms_evaluated")

    # FDR control + eps_mi_gain gate + top-k sort + alpha-drift gate + linear_residual/diff
    # collapse + structural-fragility gate (lifted to ``_filter_and_gate`` to keep this file
    # under the monolith threshold).
    from mlframe.training.composite.discovery._filter_and_gate import filter_sort_and_gate_candidates

    st.kept_specs = filter_sort_and_gate_candidates(
        self, st.candidates, df=df, train_idx=train_idx, y_full=st.y_full, y_train=st.y_train,
        extract_column_array=_extract_column_array,
    )

    # Phase B: tiny-model rerank. Re-rank the MI-survivors by
    # CV-RMSE on the y-scale (the actual prediction objective).
    # Skip when ``screening == "mi"`` -- callers who want only
    # MI ranking pay zero rerank cost.
    if st.kept_specs and self.config.screening in ("tiny_model", "hybrid") and self.config.tiny_screening_models in ("single_lgbm", "per_family"):
        st.kept_specs = self._tiny_model_rerank(
            kept_specs=st.kept_specs,
            df=df,
            target_col=target_col,
            usable_features=st.usable_features,
            train_idx=train_idx,
            y_full=st.y_full,
        )
        release_fold_caches()
        if st._ram_profiler_on:
            _phase_ram_report(st._ram_state, "tiny_model_rerank_done")

    if not st.kept_specs:
        mode = self.config.fail_on_no_gain
        msg = f"[CompositeTargetDiscovery] no candidate cleared mi_gain > " f"{self.config.eps_mi_gain} on target='{target_col}'."
        if mode == "raise":
            raise RuntimeError(msg)
        logger.warning("%s (fail_on_no_gain=%r)", msg, mode)

    # Multi-base forward-stepwise auto-promotion of linear_residual specs; carved to
    # ``_fit_multibase`` to keep this module under the 1k-LOC monolith threshold.
    st._kept_specs_before_multibase = st.kept_specs
    st.kept_specs = apply_multi_base_forward_stepwise(self, st.kept_specs, df, target_col, train_idx, st.y_train)
    if st.kept_specs is not st._kept_specs_before_multibase and st._ram_profiler_on:
        _phase_ram_report(st._ram_state, "forward_stepwise_done")


def _fit_step3_opt_discovery_steps(self, st, df, target_col, train_idx, val_df, val_y):
    """Step 3 of fit: lines starting at ``if st.kept_specs and (``."""
    if st.kept_specs and (
        getattr(self.config, "region_adaptive_enabled", False)
        or getattr(self.config, "interaction_base_discovery_enabled", True)
        or getattr(self.config, "auto_chain_discovery_enabled", True)
    ):
        from mlframe.training.composite.discovery._opt_in_steps import run_optional_discovery_steps

        _extra = run_optional_discovery_steps(self, df, target_col, st.usable_features, train_idx, st.kept_specs, self.config)
        st.kept_specs = list(st.kept_specs) + list(_extra) if _extra else st.kept_specs
        if st._ram_profiler_on:
            _phase_ram_report(st._ram_state, "opt_in_steps_done")
        # Re-run the structural-fragility gate on the auto-chain-DISCOVERED specs: the early pass (before the rerank)
        # could not see them (auto_chain surfaces chains only here), so a base-additive chain like
        # ``chain_monotonic_residual_yj`` on a per-group-level base slipped through to the val-split gate, which
        # (val != test) let it survive and collapse to R^2<0 on test. This pass is idempotent for the already-passed
        # specs and cheap (30k sample); it runs only when auto-chain actually appended new specs.
        if _extra and st.kept_specs and getattr(self.config, "structural_fragility_gate_enabled", True):
            from mlframe.training.composite.discovery._yscale_holdout_gate import apply_structural_fragility_gate

            st.kept_specs = apply_structural_fragility_gate(self, df, st.kept_specs, train_idx, st.y_full)

    # y-scale group-aware holdout gate. Drop specs whose predict-T -> invert-to-y pipeline collapses
    # on a group-disjoint holdout (the prod failure the forward-only MI / i.i.d. honest-holdout never
    # sees). No-op without group ids. Runs BEFORE the honest re-score so the (heavier) MI re-score only
    # touches survivors. (The structural-fragility gate now runs earlier, before the tiny-rerank.)
    if st.kept_specs and getattr(self.config, "yscale_holdout_gate_enabled", True):
        from mlframe.training.composite.discovery._yscale_holdout_gate import apply_yscale_holdout_gate

        st.kept_specs = apply_yscale_holdout_gate(
            self, df, target_col, st.kept_specs, st.usable_features, train_idx, st.y_full,
            val_df=val_df, val_y=val_y,
        )
        if st._ram_profiler_on:
            _phase_ram_report(st._ram_state, "yscale_holdout_gate_done")

    st.kept_specs = _apply_honest_holdout_stages(
        self, df, target_col, st.kept_specs, st.usable_features, train_idx, st.y_full,
        st._honest_holdout_idx, st._ram_profiler_on, st._ram_state, _phase_ram_report,
    )

    st.elapsed = timer() - st.t0
    logger.info(
        "[CompositeTargetDiscovery] target='%s' discovered %d spec(s) " "from %d candidate(s) in %.2fs",
        target_col,
        len(st.kept_specs),
        len(st.candidates),
        st.elapsed,
    )
    if st._ram_profiler_on:
        _phase_ram_report(st._ram_state, "fit_exit")

    # Alpha-drift WARNINGs only for the SURVIVING specs. Inline emits during scoring are at DEBUG; the user sees a single, actionable warning at the end of discovery rather than a wall of warnings for specs that the raw-y baseline gate / Wilcoxon filter dropped anyway.
    st._drift_flags = getattr(self, "_alpha_drift_flags", {})
    if st._drift_flags and st.kept_specs:
        _drift_threshold = float(
            getattr(
                self.config,
                "alpha_drift_z_threshold",
                3.0,
            )
        )
        _surviving_drift = [
            (s.name, st._drift_flags[s.name]) for s in st.kept_specs if s.name in st._drift_flags and st._drift_flags[s.name].get("z_score", 0.0) > _drift_threshold
        ]
        if _surviving_drift:
            for _spec_name, _info in _surviving_drift:
                log_throttle(
                    logger, "composite_discovery_fit_alpha_drift_detected", logging.WARNING,
                    "[CompositeTargetDiscovery] alpha drift "
                    "detected for KEPT spec=%s (alpha first-half="
                    "%.4f, second-half=%.4f, z=%.2f > %.2f). "
                    "Concept drift -- linear_residual may "
                    "underperform on test. Set "
                    "reject_on_alpha_drift=True in "
                    "CompositeTargetDiscoveryConfig to drop "
                    "automatically.",
                    _spec_name,
                    _info["alpha_first_half"],
                    _info["alpha_second_half"],
                    _info["z_score"],
                    _drift_threshold,
                )

    # Reconcile the report ``kept`` flag against the FINAL surviving specs.
    # ``entry['kept']`` is stamped True at the eps_mi_gain gate, but the specs
    # then pass through top_k_after_mi trim, the alpha-drift gate, the
    # linear_residual->diff collapse, the tiny-model rerank, and multi-base
    # name-swaps -- none of which write back to the candidate entries. Without
    # this pass ``report()`` claims kept=True for specs that were actually
    # dropped (or renamed) downstream, contradicting its "all evaluated
    # candidates with their final disposition" contract. Reconcile by spec name
    # against ``kept_specs``; a multi-base upgrade swaps a seed
    # ``linear_residual`` for a ``linear_residual_multi`` of a NEW name, so its
    # seed entry is recorded as upgraded (not silently dropped).
    st._final_kept_names = {getattr(s, "name", None) for s in st.kept_specs}


def _fit_step4_seed_entry_recorded(self, st):
    """Step 4 of fit: lines starting at ``for _entry in st.candidates:``."""
    for _entry in st.candidates:
        _espec = _entry.get("spec")
        if _espec is None:
            continue  # already a reject row; reason already set.
        _ename = getattr(_espec, "name", None)
        if _ename in st._final_kept_names:
            _entry["kept"] = True
            continue
        # Spec did NOT survive to the final set. Flip kept and record why,
        # unless the eps gate already rejected it (kept was never set True).
        if _entry.get("kept"):
            if _espec.transform_name == "linear_residual" and _espec.base_column in st._multi_seed_primaries:
                _entry["reason"] = "upgraded into a linear_residual_multi spec " "(multi-base forward-stepwise)"
            else:
                _entry["reason"] = _reason_from_ledger(self, _ename)
            _entry["kept"] = False
