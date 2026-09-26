"""Per-round stages of the screen / FE loop in ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from timeit import default_timer as timer
# --- end imports ---


def _attach_dcd_results(self, _dcd_state, X, data, cols, nbins):
    """Publish DCD results on the estimator (dcd_, cluster_members_, cluster_hierarchy_) and adopt the matrix DCD extended with PC1 aggregates, so later FE sees it."""
    try:
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import dcd_summary as _dcd_summary

        self.dcd_ = _dcd_summary(_dcd_state)
    except Exception as exc:
        logger.debug("mrmr: DCD result attachment failed; dcd_ unavailable: %r", exc, exc_info=True)
        self.dcd_ = None
    # Layer 41: self-describing cluster membership accessor.
    # Mirror ``dcd_["cluster_anchors_names"]`` onto the estimator as a
    # first-class fitted attribute so downstream code can read the
    # discovered clusters without indexing through ``self.dcd_`` (the
    # raw summary dict). ``cluster_members_`` is None when DCD was
    # disabled, matching ``dcd_`` semantics. Pure additive metadata -
    # no effect on ``support_`` or ``transform`` output.
    if isinstance(self.dcd_, dict):
        self.cluster_members_ = dict(self.dcd_.get("cluster_anchors_names", {}))
    else:
        self.cluster_members_ = None
    # Layer 48: hierarchical post-hoc cluster map. Pure
    # additive analyser over ``dcd_["cluster_anchors_names"]`` -
    # surfaces super-cluster ties DCD's greedy single-anchor rule
    # cannot. Empty dict when DCD found <2 anchors / no super-tau
    # crossings. None mirrors ``cluster_members_`` semantics for the
    # DCD-disabled case.
    if isinstance(self.dcd_, dict):
        try:
            from mlframe.feature_selection.filters._cluster_hierarchy import build_cluster_hierarchy

            self.cluster_hierarchy_ = build_cluster_hierarchy(
                self.dcd_,
                X,
                super_tau=float(getattr(self, "dcd_super_tau", 0.5)),
                max_levels=int(getattr(self, "dcd_hierarchy_max_levels", 3)),
                distance=str(getattr(self, "dcd_distance", "su")),
            )
        except Exception as exc:
            logger.debug("mrmr: cluster-hierarchy accessor failed; using an empty mapping: %r", exc, exc_info=True)
            self.cluster_hierarchy_ = {}
    else:
        self.cluster_hierarchy_ = None
    # 2026-05-30 Wave 9.1 fix (loop iter 1, agent-found bug):
    # When DCD's ``commit_swap`` extended ``factors_data`` inside screen
    # with PC1 aggregate columns, the swap targets land in ``selected_vars``
    # at indices >= len(nbins) here - the outer-scope ``data/cols/nbins``
    # still point at the pre-swap matrix, so downstream ``_run_fe_step``
    # crashes in ``merge_vars`` with "negative dimensions" once an
    # aggregate index is looked up. Adopt the extended matrices back
    # from DCDState so downstream FE / final remap sees them.
    if _dcd_state is not None:
        try:
            _new_p = int(_dcd_state.factors_data.shape[1])
            _cur_p = int(data.shape[1])
            if _new_p > _cur_p:
                data = _dcd_state.factors_data
                cols = list(_dcd_state.cols)
                nbins = np.asarray(_dcd_state.factors_nbins, dtype=np.int64)
        except Exception as e:  # nosec B110 - non-trivial body
            # Best-effort - if DCDState is malformed, fall through.
            logger.debug("DCDState looked malformed (%s: %s) -- falling through without it", type(e).__name__, e)
    return cols, data, nbins


def _runtime_budget_exhausted(self, start_time, num_fs_steps, verbose):
    """True once the fit has used its ``max_runtime_mins`` budget; the FE loop then stops before another step."""
    if self.max_runtime_mins is None:
        return False
    elapsed_min = (timer() - start_time) / 60.0
    if elapsed_min < self.max_runtime_mins:
        return False
    if verbose:
        logger.info("MRMR.fit: runtime budget %.1f min exceeded at FE step %d; stopping.", self.max_runtime_mins, num_fs_steps)
    return True


def _sufficient_summary_reached(self, data, nbins, cols, selected_vars, target_indices, X, y, num_fs_steps, verbose):
    """Sufficient-summary early stop: True when the residual of y given the current selection is noise w.r.t. every raw
    feature and small relative to y, so by the Data-Processing Inequality no engineered candidate can add information.
    Never changes the final selection; it only skips FE steps that could find nothing. Sets ``self.sufficient_summary_``."""
    if not (bool(getattr(self, "fe_sufficient_summary_early_stop", True)) and len(selected_vars) > 0):
        return False
    from mlframe.feature_selection.filters._fe_sufficient_summary import check_sufficient_summary_for_mrmr

    _ss_verdict = check_sufficient_summary_for_mrmr(
        self,
        data=data,
        nbins=nbins,
        cols=cols,
        selected_vars=selected_vars,
        target_indices=target_indices,
        X=X,
        y=y,
        verbose=verbose,
    )
    self.sufficient_summary_ = _ss_verdict
    if _ss_verdict.reached:
        if verbose:
            logger.info(
                "MRMR.fit: sufficient-summary early-stop at FE step %d -- %s. Skipping the remaining FE search (selection unchanged).",
                num_fs_steps,
                _ss_verdict.reason,
            )
        return True
    return False


def _fe_step_params(fe):
    """The ``fe_*`` keyword arguments of ``MRMR._run_fe_step``, from the resolved FE parameters of this fit."""
    names = (
        "max_steps",
        "npermutations",
        "max_pair_features",
        "print_best_mis_only",
        "min_nonzero_confidence",
        "min_engineered_mi_prevalence",
        "good_to_best_feature_mi_threshold",
        "max_external_validation_factors",
        "min_pair_mi",
        "min_pair_mi_prevalence",
        "smart_polynom_iters",
        "smart_polynom_optimization_steps",
        "min_polynom_degree",
        "max_polynom_degree",
        "min_polynom_coeff",
        "max_polynom_coeff",
        "unary_preset",
        "binary_preset",
    )
    return {"fe_" + name: getattr(fe, name) for name in names}


def _adaptive_relax_retry(self, fe, step_kw, data, cols, nbins, X, selected_vars, verbose):
    """Retry the first FE step once with relaxed prevalence thresholds; the result of ``_run_fe_step`` or None.

    When the first-pass FE produces 0 engineered features, the most likely culprit on heavily-correlated feature sets is
    the strict ``fe_min_engineered_mi_prevalence`` gate - pair-level MI is near the individual-MI sum and the engineered
    candidate cannot beat 98% of pair MI. The retry re-evaluates every pair (fresh ``checked_pairs``) and skips the
    already-completed expensive Hermite Optuna phase (``fe_smart_polynom_iters=0``). Off with fe_adaptive_threshold_relax=False.
    """
    if not bool(getattr(self, "fe_adaptive_threshold_relax", True)):
        return None
    relax_factor = float(getattr(self, "fe_adaptive_relax_factor", 0.9))
    # fe_min_pair_mi_prevalence may be the sentinel string "auto" (debiased-ratio mode - see _step_core.py's own isinstance
    # check) rather than a float; resolve it to that same established numeric convention (1.05) just for this relaxation
    # arithmetic, as _step_core.py/_step_pairs_rank.py do at their own use sites.
    prevalence = fe.min_pair_mi_prevalence
    pair_prevalence = 1.05 if isinstance(prevalence, str) and prevalence.strip().lower() == "auto" else float(prevalence)
    relaxed_engineered = fe.min_engineered_mi_prevalence * relax_factor
    relaxed_pair = max(1.001, pair_prevalence * relax_factor)
    if verbose:
        logger.info(
            "MRMR FE: first pass found 0 engineered features; retrying with relaxed thresholds "
            "(engineered_mi_prevalence: %.3f -> %.3f, pair_mi_prevalence: %.3f -> %.3f). "
            "Skipping Hermite Optuna re-run (already cached in _hermite_features_).",
            fe.min_engineered_mi_prevalence,
            relaxed_engineered,
            pair_prevalence,
            relaxed_pair,
        )
    kw = dict(step_kw, fe_min_engineered_mi_prevalence=relaxed_engineered, fe_min_pair_mi_prevalence=relaxed_pair, fe_smart_polynom_iters=0)
    result = self._run_fe_step(data=data, cols=cols, nbins=nbins, X=X, selected_vars=selected_vars, checked_pairs=set(), **kw)
    if result is not None and verbose:
        logger.info("MRMR FE adaptive retry produced %d engineered features.", result[-1])
    return result


def _take_engineered_continuous_store(self):
    """Detach the fit-time continuous engineered-value store from the estimator (keeps the pickle lean) and return a local
    snapshot of it for the raw-vs-engineered redundancy drop, which needs the continuous values, not the screening bins."""
    _eng_continuous_snapshot = dict(getattr(self, "_engineered_continuous_", None) or {})
    if hasattr(self, "_engineered_continuous_"):
        try:
            del self._engineered_continuous_
        except Exception as exc:
            logger.debug("mrmr: engineered-continuous store failed; using an empty mapping: %r", exc, exc_info=True)
            self._engineered_continuous_ = {}
    return _eng_continuous_snapshot
