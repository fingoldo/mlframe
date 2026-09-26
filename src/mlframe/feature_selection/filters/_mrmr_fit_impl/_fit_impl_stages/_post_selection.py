"""Stages of ``_fit_impl`` that run after the screen: support, RFECV, index mapping, friend graph, temp-target cleanup."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from sklearn.metrics import make_scorer

# --- end imports ---


def _assign_support(
    self, X, classes_y, cols, data, nbins, target_indices, y, verbose, fe, cached_MIs, engineered_recipes, predictors, _eng_continuous_snapshot, selected_vars
):
    """Assign ``support_`` and the related fitted attributes from the final selection."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._assign_support import _assign_support

    _assign_support(
        self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        fe_max_steps=fe.max_steps,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        predictors=predictors,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
    )


def _run_additional_rfecv(self, X, selected_vars, verbose, y, categorical_vars_names, _fni_idx):
    """Optional RFECV pass over the MRMR-selected columns (``additional_rfecv``)."""
    from mlframe.feature_selection.filters.mrmr import (
        RFECV,
        CatBoostClassifier,
        compute_probabilistic_multiclass_error,
    )

    if self.run_additional_rfecv_minutes:
        """On the factors discarded by MRMR, let's run RFECV to see if any of them participate in interactions"""
        n_unexplored = X.shape[1] - len(selected_vars)
        if n_unexplored > 0:
            if verbose:
                logger.info(
                    "Running RFECV for %s minute(s) over %s feature(s) discarded by MRMR to extract interactions...",
                    self.run_additional_rfecv_minutes,
                    f"{n_unexplored:_}",
                )

            from mlframe.training import get_training_configs

            configs = get_training_configs(has_time=True)

            params = configs.COMMON_RFECV_PARAMS.copy()
            params["max_runtime_mins"] = self.run_additional_rfecv_minutes
            # Wire MRMR.cv / cv_shuffle into the additional RFECV pass; pre-fix they were dead constructor params. ``params`` may already carry ``cv`` from
            # configs.COMMON_RFECV_PARAMS; MRMR's explicit setting wins.
            params.update(self._rfecv_cv_kwargs())
            # Parsimony for the rescue: RFECV's recall-oriented default ('one_se_max') keeps the LARGEST subset within 1 SE, which on a noise-robust booster
            # re-admits ~the whole discarded pool and undoes MRMR's selection. Pin the smallest-within-1-SE rule so the rescue re-adds only discarded features
            # that genuinely lift CV. setdefault lets COMMON_RFECV_PARAMS / additional_rfecv_kwargs win.
            params.setdefault("n_features_selection_rule", getattr(self, "additional_rfecv_selection_rule", "one_se_min"))
            _extra_rfecv = getattr(self, "additional_rfecv_kwargs", None)
            if _extra_rfecv:
                params.update(_extra_rfecv)

            # Classifier-vs-regressor detection. Preference order:
            #   1) Explicit ``target_type`` attribute on self (set by the caller / harness).
            #   2) Honest dtype + cardinality heuristic: float dtype is regression by
            #      construction (zero-inflated targets like ``[0]*900 + [1.7, 2.4, ...]``
            #      satisfy the legacy ratio>100 but are NOT classification). Integer
            #      dtype with ratio>100 AND small absolute cardinality (<=64 unique
            #      values) is classification. Everything else is regression.
            # Pre-fix, the regression else-branch silently skipped the additional-RFECV pass entirely, so regression callers got no benefit
            # from run_additional_rfecv_minutes. The dtype guard prevents misclassifying
            # zero-inflated float targets. fix audit row FS-L-2.
            _explicit_tt = getattr(self, "target_type", None)
            if _explicit_tt is not None:
                _tt_str = str(_explicit_tt).lower()
                _is_classification = "classif" in _tt_str or _tt_str in ("binary", "multiclass", "multilabel")
            else:
                _y_arr = np.asarray(y)
                _n_unique = len(np.unique(_y_arr))
                _ratio = len(_y_arr) / max(1, _n_unique)
                _is_float = _y_arr.dtype.kind == "f"
                _is_classification = (not _is_float) and _ratio > 100 and _n_unique <= 64
                if _ratio > 100 and _is_float:
                    logger.warning(
                        "MRMR.run_additional_rfecv: target is float dtype with %d unique values; "
                        "treating as regression despite samples/unique ratio %.1f>100. Pass "
                        "target_type='classification' explicitly to override.",
                        _n_unique,
                        _ratio,
                    )
            # order-preserving set difference. The prior ``list(set(X.columns) - set(...))`` produced a HASH-SEED-DEPENDENT column order because Python's
            # randomized string hashing reorders ``set`` iteration across processes. That order flowed into RFECV's CatBoost feature importances, whose
            # tie-breaks then gave different ``self.support_`` across runs that differed only in ``PYTHONHASHSEED``. Concrete demo: 5/5 distinct orderings
            # observed across seeds 0-4. Breaks the "same random_seed -> identical support_" contract for any user with ``run_additional_rfecv_minutes`` > 0.
            # ``selected_vars`` indexes ``feature_names_in_`` (full, includes passthrough); ``X`` here is the passthrough-narrowed working frame, so map names
            # via ``feature_names_in_`` rather than ``X.columns[...]`` (positional mismatch when passthrough is active). Passthrough columns are never in the
            # narrowed X and never enter the RFECV rescue pool below regardless.
            _sel_names = {self.feature_names_in_[i] for i in selected_vars}
            # Cluster members already folded into a denoised aggregate (post-hoc cluster_aggregate 'replace' mode,
            # _cluster_aggregate_removals_) or into a DCD PC1/mean_z swap (cluster_members_) are REPRESENTED by that
            # aggregate. Excluding them from the rescue pool stops RFECV re-admitting the raw members and re-injecting
            # the very redundancy the aggregation removed - only features dropped for low marginal/joint relevance get reconsidered.
            _excluded_from_rescue = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
            _cm = getattr(self, "cluster_members_", None)
            if isinstance(_cm, dict):
                for _anchor, _members in _cm.items():
                    _excluded_from_rescue.add(_anchor)
                    if isinstance(_members, (list, tuple, set)):
                        _excluded_from_rescue.update(_members)
            # Engineered FE columns (univariate basis a__T2, hybrid/pair/triplet crosses,
            # MI-greedy) survive in X.columns but were deliberately excluded from
            # feature_names_in_ (raw columns only, line above). They cannot be indexed
            # into support_ via feature_names_in_.index() -> ValueError. Exclude them from
            # the rescue pool so RFECV only reconsiders RAW discarded columns.
            _excluded_from_rescue.update(getattr(self, "hybrid_orth_features_", None) or [])
            _excluded_from_rescue.update(getattr(self, "mi_greedy_features_", None) or [])
            # Raw operands the conditional-redundancy sweep judged FULLY SUBSUMED by a
            # surviving engineered child (``_raw_redundancy_dropped_``) must NOT re-enter
            # via the RFECV rescue pool. The n-invariant CMI verdict is authoritative: a
            # raw whose entire y-information is captured by an admitted engineered feature
            # (e.g. ``a`` / ``b`` in ``a**2/b`` once ``div(neg(a),sqrt(b))`` is selected)
            # carries no independent signal, but CatBoost RFECV - which scores raw
            # MARGINAL usefulness, blind to the engineered child's coverage - would re-admit
            # it, resurrecting the exact redundancy the sweep removed (observed at n=2000/5000
            # on ``y=0.30 a**2/b``: the sweep dropped a+b, RFECV re-added a). Excluding the
            # dropped set keeps the redundancy decision consistent across both the FE-step
            # finalisation AND the downstream RFECV rescue.
            _excluded_from_rescue.update(getattr(self, "_raw_redundancy_dropped_", None) or set())
            # The rescue maps pool columns back through feature_names_in_, so only raw input columns are eligible. Filtering by membership,
            # not by per-family rosters, keeps every FE family's engineered columns out (an unary/binary pair column reached this pool and
            # would have raised KeyError if RFECV had selected it).
            _raw_rescue_names = set(self.feature_names_in_)
            temp_columns = [c for c in X.columns if c in _raw_rescue_names and c not in _sel_names and c not in _excluded_from_rescue]

            if not temp_columns:
                # Every raw column is already selected or excluded (cluster members, subsumed operands, engineered columns); the
                # count above mixes the working frame (with engineered columns) and feature_names_in_, so it can be positive here.
                if verbose:
                    logger.info("RFECV rescue skipped: no discarded raw column is eligible after exclusions.")
            else:
                if _is_classification:
                    cb_num_rfecv = RFECV(
                        estimator=CatBoostClassifier(**configs.CB_CLASSIF),
                        fit_params=dict(plot=False),
                        cat_features=categorical_vars_names,
                        scoring=make_scorer(score_func=compute_probabilistic_multiclass_error, response_method="predict_proba", greater_is_better=False),
                        **params,
                    )
                else:
                    # Regression branch: CatBoostRegressor with the same shared params; default scoring lets
                    # RFECV pick from the estimator (negative-MSE-like). Keeping the import local avoids
                    # paying the CatBoostRegressor import cost when only classification is exercised.
                    from catboost import CatBoostRegressor

                    cb_num_rfecv = RFECV(
                        estimator=CatBoostRegressor(**configs.CB_REGR),
                        fit_params=dict(plot=False),
                        cat_features=categorical_vars_names,
                        **params,
                    )
                cb_num_rfecv.fit(X[temp_columns], y)

                if cb_num_rfecv.n_features_ > 0:
                    new_features = np.array(temp_columns)[cb_num_rfecv.support_]
                    if verbose:
                        logger.info("RFECV selected %d additional feature(s): %s", cb_num_rfecv.n_features_, new_features)
                    # Reuse the name -> index map built above (``feature_names_in_`` is fit-invariant).
                    for feature in new_features:
                        selected_vars.append(_fni_idx[feature])
                else:
                    if verbose:
                        logger.info("RFECV selected no additional features.")


def _map_selected_to_original_indices(self, cols, selected_vars, verbose, engineered_recipes):
    """Map ``selected_vars`` from working-column indices to original-frame indices (categorize_dataset may reorder cat columns)."""
    selected_vars_names = np.array(cols)[np.array(selected_vars, dtype=np.intp)]

    # BUG2: the cross-fold stability vote in ``_run_fe_step`` pops a
    # fold-unstable engineered recipe AND de-selects its column for that step, but the
    # materialised bin-code column stays in ``cols``/``data``, so the downstream greedy
    # screen (step>1 re-screen / final selection) re-admits it on marginal MI - it then
    # arrives here with NO recipe and was silently DROPPED from transform output (a
    # select-then-drop contract violation: a feature in support_/discovered MUST survive
    # transform). The vote is authoritative, so strip every vote-rejected engineered name
    # from the selection BEFORE finalising support_/discovered: the column never re-enters
    # support_, get_feature_names_out, or _engineered_features_. ``selected_vars`` is filtered
    # in lockstep (by cols-index) so the raw integer support stays consistent.
    _vote_dropped_names = getattr(self, "_fe_stability_vote_dropped_", None)
    if _vote_dropped_names:
        _keep_mask = np.array([nm not in _vote_dropped_names for nm in selected_vars_names], dtype=bool)
        if not _keep_mask.all():
            _kept_idx_positions = np.nonzero(_keep_mask)[0]
            selected_vars = [selected_vars[i] for i in _kept_idx_positions]
            selected_vars_names = selected_vars_names[_keep_mask]
            if verbose:
                logger.info(
                    "MRMR.fit: stripped %d cross-fold-vote-rejected engineered feature(s) from the "
                    "final selection so they cannot re-enter support_ without a replayable recipe.",
                    int((~_keep_mask).sum()),
                )
    # Tolerate FE-engineered names: screening output may include synthetic feature names not in
    # feature_names_in_; record them in self._engineered_features_ instead of raising on the .index() lookup.
    # Also surface matching EngineeredRecipe (built during _run_fe_step) so transform() can replay each
    # engineered column on test data. An engineered name without a recipe (e.g. higher-order interaction
    # whose parents are themselves engineered) is recorded by name only and dropped from transform output.
    self._engineered_features_ = []
    self._engineered_recipes_ = []
    original_indices = []
    engineered_without_recipe = []
    # feature_names_in_ is an ndarray (sklearn convention); name -> index map built once (O(F))
    # instead of an ``in`` test + ``.index()`` rescan per ``col`` (O(F) each) - turns the O(K*F)
    # loop below into O(K+F).
    _fni_idx = {nm: i for i, nm in enumerate(self.feature_names_in_)}
    for col in selected_vars_names:
        _fni_i = _fni_idx.get(col)
        if _fni_i is not None:
            original_indices.append(_fni_i)
        else:
            self._engineered_features_.append(col)
            recipe = engineered_recipes.get(col)
            if recipe is not None:
                self._engineered_recipes_.append(recipe)
            else:
                engineered_without_recipe.append(col)
    if engineered_without_recipe and verbose:
        # Happens with fe_max_steps>1 when a higher-order interaction's parents are themselves engineered features. The recipe replay path can only
        # reconstruct 1-deep engineering; deeper nests are recorded in self._engineered_features_ but DROPPED from transform output. Surface the cost.
        logger.warning(
            "MRMR.fit: %d engineered feature(s) selected without replayable recipe (nested-engineered parents at fe_max_steps=%d); they will be DROPPED from transform output: %s",
            len(engineered_without_recipe),
            self.fe_max_steps,
            engineered_without_recipe[:8],
        )
    # ``selected_vars`` is downstream re-bound to the integer indices of the RAW columns only; engineered features are appended in transform() via
    # ``_append_engineered`` using ``self._engineered_recipes_``. This split mirrors the on-disk contract: support_ indexes feature_names_in_; engineered output
    # columns come from the recipes list. n_features_ counts BOTH (see assignment below).
    selected_vars = original_indices

    # PSEUDO-REMIX OPERAND RE-ADD. A surviving conditional-gate / binned-numeric-agg /
    # row-argmax composite (``gate_mask__a__b`` / ``binagg_*(c|qbin(a))`` / ``argmax__a__b``) is a LOSSY
    # threshold/binning re-mix of its raw operands: it survived because it captures the INTERACTION, but
    # it destroys each operand's continuous value that a LINEAR downstream needs (measured: a 5-class
    # LogReg scored macro-F1 0.62 when x2 lived ONLY inside ``gate_mask__x1__x2`` vs >0.70 with raw x2
    # restored). The operands typically have WEAK MARGINAL MI (signal is in the joint), so the screen /
    # marginal retention never surface them. When a CO-operand is ALREADY in the raw support (e.g. x1
    # selected beside ``gate_mask__x1__x2``) the composite is a vouched genuine multi-source interaction,
    # so restore the other raw operand(s). Runs here (engineered roster + raw support both final). A
    # single-operand self-gate gets no vouch; a noise-paired gate has low joint MI and rarely survives.
    # PASSTHROUGH RE-ATTACH. Embedding/text columns excluded from the MI screen above are re-added to the selected set so transform() emits them unchanged. Their
    # indices are looked up in ``feature_names_in_`` (which includes them, in original order). Appended AFTER the screen so they never participate in MI/redundancy
    # but always survive to the estimator (the learnable-embedding network + boundary encoder consume them).
    if self._passthrough_features_:
        _existing = set(selected_vars)
        # Reuse the name -> index map built above (``feature_names_in_`` is fit-invariant).
        for _pname in self._passthrough_features_:
            _pidx = _fni_idx.get(_pname)
            if _pidx is not None and _pidx not in _existing:
                selected_vars.append(_pidx)
                _existing.add(_pidx)
    return _fni_idx, selected_vars


def _friend_graph_analysis(
    self,
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
    recipes,
    _persisted_dcd_state,
    _y_np,
    _fe_family_on,
):
    """Friend-graph post-analysis: diagnostic, with optional pruning of the selection."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy import _friend_graph_and_redundancy_passes
    from mlframe.feature_selection.filters._fe_frame_ops import fe_to_pandas

    selected_vars, cols, data, nbins = _friend_graph_and_redundancy_passes(
        self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _effective_min_relevance_gain=_effective_min_relevance_gain,
        _hinge_deferred_recipes=recipes.hinge_deferred,
        _hinge_deferred_values=recipes.hinge_deferred_values,
        _hybrid_orth_pre_recipes=recipes.hybrid_orth,
        _miss_ind_pre_recipes=recipes.miss_ind,
        _persisted_dcd_state=_persisted_dcd_state,
        _y_np=_y_np,
        fe_to_pandas=fe_to_pandas,
        _fe_family_on=_fe_family_on,
    )
    return cols, data, nbins, selected_vars


def _drop_temporary_targets(_is_polars_input, X, target_names, _dcd_state, selected_vars):
    """Drop the targets temporarily injected into X for the fit."""
    if _is_polars_input:
        X = X.drop(target_names)  # no-copy lazy op; caller's X untouched
    else:
        # option_context silences the conservative SettingWithCopy heuristic (fires when the caller passed a sliced
        # view); the in-place drop reverses this function's own targ_<id> injection on the same object, no copy.
        with pd.option_context("mode.chained_assignment", None):
            X.drop(columns=target_names, inplace=True)  # noqa: PD002 - must mutate the caller's frame OBJECT in place (restores its original schema by identity), not rebind a local; `X = X.drop(...)` would silently stop touching the caller's actual frame

    # DCD orphaned-cluster raw re-attach. A DCD AGGREGATE swap replaces the raw
    # anchor with the (engineered, non-support_) aggregate column; when that
    # anchor was the cluster's only selected raw column the latent disappears
    # from the raw ``support_`` (which indexes feature_names_in_ only) even
    # though the denoised aggregate survives in ``get_feature_names_out`` /
    # ``transform``. Run on the FINAL ``selected_vars`` (after the confirm-
    # rescreen loop has fully settled, so this can never perturb a subsequent
    # re-selection) to re-attach one raw cluster member per orphaned aggregate,
    # keeping each collapsed latent visible in BOTH the raw support and the
    # transform output. Best-effort; never breaks fit.
    if _dcd_state is not None and len(selected_vars):
        try:
            from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
                reattach_raw_representative_after_aggregate_swap as _dcd_reattach_raw,
            )

            _sv_list = list(selected_vars)
            _sv_set = {int(s) for s in _sv_list}
            _agg_indices = [
                int(e.get("new_col_idx"))
                for e in (getattr(_dcd_state, "swap_log", None) or [])
                if str(e.get("branch", "aggregate")) == "aggregate" and e.get("aggregate_name") and e.get("new_col_idx") is not None
            ]
            for _agg_idx in _agg_indices:
                if _agg_idx in _sv_set:
                    _dcd_reattach_raw(_dcd_state, _agg_idx, _sv_list)
            selected_vars = _sv_list
        except Exception as _reattach_exc:
            logger.warning(
                "DCD orphaned-cluster raw re-attach failed (%s); continuing.",
                _reattach_exc,
            )
    return X, selected_vars
