"""Post-while-loop finalisation for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s post-loop tail. Three responsibilities:

1. SFFS swap pass (``self.swap_top_k``) and best-score refresh.
2. Write the public ``self.n_features_in_`` / ``self.feature_names_in_``
   / ``self.estimators_`` / ``self.feature_importances_`` /
   ``self.selected_features_`` / ``self.cv_results_`` slots; call
   ``self.select_optimal_nfeatures_`` to pick the final ``support_``.
3. Glue must_include into ``self.support_``; expand
   ``self.feature_groups`` for all-or-nothing group decisions; stamp
   ``self.signature``; cache the resolved column list for ``transform``.

Re-imported at the parent's module bottom so historical
``from ._fit import _finalize_fit_results`` keeps resolving
transparently.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")


def _finalize_fit_results(
    self,
    *,
    X,
    y,
    X_estimator=None,
    col_pos=None,
    estimator,
    cv,
    scoring,
    best_nfeatures,
    best_score,
    selected_features_per_nfeatures,
    feature_importances,
    original_features,
    evaluated_scores_mean,
    evaluated_scores_std,
    verbose,
    ndigits,
    fitted_estimators,
    must_include_resolved,
    feature_cost,
    smooth_perf,
    use_all_fi_runs,
    use_last_fi_run_only,
    use_one_freshest_fi_run,
    use_fi_ranking,
    votes_aggregation_method,
    show_plot,
    signature,
):
    """Run the SFFS swap pass + write the public RFECV result slots."""
    # SFFS swap pass: greedy 1-in / 1-out tweak around the current best.
    # ``_sffs_swap_pass`` does NOT honour fit_params / val_cv / early stopping.
    # E2 (Wave 1, 2026-05-28): when val_cv is set AND the estimator actually
    # uses early stopping (CB / LGB / XGB), the bare cross_val_score used in
    # _sffs_swap_pass trains to convergence with no ES, so any accepted swap
    # may be an artifact of letting the swap-in feature overfit relative to
    # the ES-bounded best subset. Skip swap on those configurations.
    # Estimators like LR don't use val_cv even when early_stopping_val_nsplits
    # is set -- the gate must check BOTH the knob AND the estimator type.
    # Opt-out via self.swap_top_k_allow_no_es=True for benchmarking.
    _val_cv_knob_set = bool(getattr(self, "early_stopping_val_nsplits", None))
    try:
        from mlframe.core.helpers import has_early_stopping_support as _has_es
        from sklearn.pipeline import Pipeline as _Pipeline
        _est_type_name = (
            type(estimator.steps[-1][1]).__name__
            if isinstance(estimator, _Pipeline) else type(estimator).__name__
        )
        _es_active = bool(_has_es(_est_type_name))
    except Exception:
        _es_active = False
    _has_val_cv = _val_cv_knob_set and _es_active
    _allow_swap_without_es = bool(getattr(self, "swap_top_k_allow_no_es", False))
    if self.swap_top_k and self.swap_top_k > 0 and best_nfeatures > 0 \
            and (not _has_val_cv or _allow_swap_without_es):
        try:
            self._sffs_swap_pass(
                X=X, y=y, X_estimator=X_estimator, col_pos=col_pos, estimator=estimator, cv=cv, scoring=scoring,
                best_nfeatures=best_nfeatures, best_score_ref=best_score,
                selected_features_per_nfeatures=selected_features_per_nfeatures,
                feature_importances=feature_importances,
                original_features=original_features,
                evaluated_scores_mean=evaluated_scores_mean,
                evaluated_scores_std=evaluated_scores_std,
                verbose=verbose, ndigits=ndigits,
            )
            # After the swap pass best_nfeatures may have changed.
            if evaluated_scores_mean:
                new_best_nf = max(evaluated_scores_mean, key=evaluated_scores_mean.get)
                if evaluated_scores_mean[new_best_nf] > best_score:
                    best_nfeatures = new_best_nf
                    best_score = evaluated_scores_mean[new_best_nf]
        except Exception as _swap_exc:
            if verbose:
                logger.warning("RFECV: SFFS swap pass failed (%s); continuing.", _swap_exc)
    elif self.swap_top_k and self.swap_top_k > 0 and _has_val_cv and not _allow_swap_without_es:
        if verbose:
            logger.info(
                "RFECV: swap_top_k=%d skipped because val_cv-driven early "
                "stopping is active and the swap pass uses bare "
                "cross_val_score (would compare ES vs non-ES scores). "
                "Set swap_top_k_allow_no_es=True to override.",
                self.swap_top_k,
            )

    # Save best result found so far as final.
    self.n_features_in_ = X.shape[1]
    # Row count at fit, consumed by select_optimal_nfeatures_'s p>=n FP-control gate (the collapsed-search below-dummy reject).
    self._n_samples_fit_ = int(X.shape[0])
    self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(map(str, np.arange(self.n_features_in_)))

    self.estimators_ = fitted_estimators  # dict keyed by nfeatures_nfold
    self.feature_importances_ = feature_importances  # dict keyed by nfeatures_nfold
    self.selected_features_ = selected_features_per_nfeatures  # dict keyed by nfeatures

    checked_nfeatures = sorted(evaluated_scores_mean.keys())
    cv_std_perf = [evaluated_scores_std[n] for n in checked_nfeatures]
    cv_mean_perf = [evaluated_scores_mean[n] for n in checked_nfeatures]
    self.cv_results_ = {"nfeatures": checked_nfeatures, "cv_mean_perf": cv_mean_perf, "cv_std_perf": cv_std_perf}

    # 2026-05-28 sklearn-parity: per-split keys ``split{k}_test_score`` exposing per-fold scores per N.
    # ``per_fold_scores`` is dict[N -> list of fold scores]; pivot into len(checked_nfeatures)-aligned arrays.
    _pfs = getattr(self, "_per_fold_scores", None)
    if _pfs is None:
        _pfs = {}
    # Determine the max number of folds across all N (defensive: some N may have fewer folds due to early skips).
    _max_folds = 0
    for _scores in _pfs.values():
        if _scores is not None:
            _max_folds = max(_max_folds, len(_scores))
    if _max_folds > 0:
        for _k in range(_max_folds):
            _key = f"split{_k}_test_score"
            _col: list = []
            for _n in checked_nfeatures:
                _arr = _pfs.get(_n, [])
                _col.append(float(_arr[_k]) if (_arr is not None and _k < len(_arr)) else float("nan"))
            self.cv_results_[_key] = _col

    self.select_optimal_nfeatures_(
        checked_nfeatures=checked_nfeatures,
        cv_mean_perf=cv_mean_perf,
        cv_std_perf=cv_std_perf,
        feature_cost=feature_cost,
        smooth_perf=smooth_perf,
        use_all_fi_runs=use_all_fi_runs,
        use_last_fi_run_only=use_last_fi_run_only,
        use_one_freshest_fi_run=use_one_freshest_fi_run,
        use_fi_ranking=use_fi_ranking,
        votes_aggregation_method=votes_aggregation_method,
        verbose=verbose,
        show_plot=show_plot,
    )

    # Glue must_include into the final support_. The optimiser produced support_ over the search-universe complement only;
    # must_include features are always in the final selection regardless of what the optimiser picked.
    if must_include_resolved and hasattr(self, "support_") and len(self.support_) > 0:
        if isinstance(self.support_[0], (bool, np.bool_)):
            # support_ is bool-mask aligned with feature_names_in_; set the must_include positions to True.
            support_mask = np.asarray(self.support_, dtype=bool)
            for col in must_include_resolved:
                if col in self.feature_names_in_:
                    support_mask[self.feature_names_in_.index(col)] = True
            self.support_ = support_mask
        else:
            # support_ is integer indices; prepend must_include positions.
            idx_must = [self.feature_names_in_.index(c) for c in must_include_resolved if c in self.feature_names_in_]
            merged = list(idx_must) + [i for i in self.support_ if i not in idx_must]
            self.support_ = np.asarray(merged)
        self.n_features_ = int(np.sum(self.support_)) if isinstance(self.support_[0], (bool, np.bool_)) else len(self.support_)

    # feature_groups: all-or-nothing decision per group. If ANY member of group G is in support_, ALL members are added; if NONE, all
    # stay out. Resolves the "5 collinear copies" caveat at config level when the operator knows the group structure.
    if self.feature_groups and hasattr(self, "support_") and len(self.support_) > 0:
        # Convert support_ to bool-mask form for uniform handling.
        if isinstance(self.support_[0], (bool, np.bool_)):
            support_mask = np.asarray(self.support_, dtype=bool)
        else:
            support_mask = np.zeros(len(self.feature_names_in_), dtype=bool)
            for i in self.support_:
                support_mask[i] = True
        _name_to_idx = {n: i for i, n in enumerate(self.feature_names_in_)}
        _expanded = 0
        for _group_name, _members in self.feature_groups.items():
            _idx = [_name_to_idx[m] for m in _members if m in _name_to_idx]
            if not _idx:
                continue
            _any_selected = any(support_mask[i] for i in _idx)
            if _any_selected:
                for i in _idx:
                    if not support_mask[i]:
                        _expanded += 1
                        support_mask[i] = True
        if _expanded > 0:
            if verbose:
                logger.info(
                    "RFECV: feature_groups expanded support_ by %d column(s) "
                    "for all-or-nothing group decisions.", _expanded,
                )
            self.support_ = support_mask
            self.n_features_ = int(support_mask.sum())

    # Refresh the params slot with POST-fit values before storing: fit resolves some params in place
    # (``scoring=None -> make_scorer(...)``, ``force_parallel`` thread pinning on the wrapped estimator),
    # so the entry-time params fingerprint would never match the NEXT fit's ``get_params`` and identical
    # refits would never skip. The data slots (shapes/hashes/columns) stay as computed at fit entry.
    from ._fit_init import _current_params_signature

    self.signature = signature[:-1] + (_current_params_signature(self),)

    # Cache resolved column list so transform() avoids per-call reconstruction.
    self._selected_cols_cache = None
    support = getattr(self, "support_", None)
    if support is not None and len(support) > 0:
        if isinstance(support[0], (bool, np.bool_)):
            self._selected_cols_cache = [col for col, selected in zip(self.feature_names_in_, support) if selected]
        else:
            self._selected_cols_cache = [self.feature_names_in_[i] for i in support]

    _persist_fitted_estimators(self, estimator=estimator, fitted_estimators=fitted_estimators, verbose=verbose)


def _persist_fitted_estimators(self, *, estimator, fitted_estimators, verbose):
    """Persist the fitted estimators + a required-features/metrics summary to ``self.estimators_save_path`` (documented ctor knob).

    Layout (per the ctor docstring): each fitted estimator -> ``join(save_path, estimator_type_name, "{key}.dump")``; the kept feature
    list and achieved CV metrics -> ``join(save_path, "required_features.dump")``. No-op when ``estimators_save_path`` is unset (default).
    """
    save_path = getattr(self, "estimators_save_path", None)
    if not save_path:
        return

    import os
    import joblib

    from sklearn.pipeline import Pipeline

    try:
        os.makedirs(save_path, exist_ok=True)

        required_features = list(self._selected_cols_cache) if self._selected_cols_cache is not None else []
        summary = {
            "required_features": required_features,
            "n_features": int(getattr(self, "n_features_", len(required_features))),
            "cv_results": getattr(self, "cv_results_", None),
        }
        joblib.dump(summary, os.path.join(save_path, "required_features.dump"))

        if getattr(self, "keep_estimators", False) and fitted_estimators:
            estimator_type = type(estimator.steps[-1][1]).__name__ if isinstance(estimator, Pipeline) else type(estimator).__name__
            est_dir = os.path.join(save_path, estimator_type)
            os.makedirs(est_dir, exist_ok=True)
            for key, est in fitted_estimators.items():
                joblib.dump(est, os.path.join(est_dir, f"{key}.dump"))
    except Exception as exc:
        if verbose:
            logger.warning("RFECV: estimators_save_path persistence failed (%s); continuing.", exc)
