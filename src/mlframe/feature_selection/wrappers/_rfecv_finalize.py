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
``from ._rfecv_fit import _finalize_fit_results`` keeps resolving
transparently.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def _finalize_fit_results(
    self,
    *,
    X,
    y,
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
    if self.swap_top_k and self.swap_top_k > 0 and best_nfeatures > 0:
        try:
            self._sffs_swap_pass(
                X=X, y=y, estimator=estimator, cv=cv, scoring=scoring,
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

    # Save best result found so far as final.
    self.n_features_in_ = X.shape[1]
    self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else list(map(str, np.arange(self.n_features_in_)))

    self.estimators_ = fitted_estimators  # dict keyed by nfeatures_nfold
    self.feature_importances_ = feature_importances  # dict keyed by nfeatures_nfold
    self.selected_features_ = selected_features_per_nfeatures  # dict keyed by nfeatures

    checked_nfeatures = sorted(evaluated_scores_mean.keys())
    cv_std_perf = [evaluated_scores_std[n] for n in checked_nfeatures]
    cv_mean_perf = [evaluated_scores_mean[n] for n in checked_nfeatures]
    self.cv_results_ = {"nfeatures": checked_nfeatures, "cv_mean_perf": cv_mean_perf, "cv_std_perf": cv_std_perf}

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

    self.signature = signature

    # Cache resolved column list so transform() avoids per-call reconstruction.
    self._selected_cols_cache = None
    support = getattr(self, "support_", None)
    if support is not None and len(support) > 0:
        if isinstance(support[0], (bool, np.bool_)):
            self._selected_cols_cache = [col for col, selected in zip(self.feature_names_in_, support) if selected]
        else:
            self._selected_cols_cache = [self.feature_names_in_[i] for i in support]
