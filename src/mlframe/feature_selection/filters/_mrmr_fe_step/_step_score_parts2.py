"""Helpers carved out of ``_step_score_parts`` to keep that module under its size budget."""
from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection.cv_policy import get_cv_policy

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection


def _fe_tail_budget_spent(stage: str, verbose: int = 0) -> bool:
    """True when the FE wall-clock deadline has already passed, so this optional tail stage is skipped whole.

    The escalation / fusion / stability-vote tails are per-candidate enrichments that run after the pair search, and a fit handed a small
    ``max_runtime_mins`` used to run every one of them to completion past the budget. They are skipped in one piece rather than broken
    mid-loop: the escalation and fusion blocks pre-size their code arrays and then append ``nbins`` entries by length, so an early break
    inside those fills would leave ``data``/``cols``/``nbins`` desynchronised.
    """
    from .._fe_deadline import fe_deadline_passed

    if not fe_deadline_passed():
        return False
    if verbose:
        logger.info("MRMR FE: wall-clock budget spent; skipping the %s stage (the selection so far is kept).", stage)
    return True


def _device_codes_to_host(transformed_vals, nbins, dtype):
    """Quantile codes of a device-backed survivor matrix, binned ON the device (the float values never cross to the host); only the narrow int codes -
    the augmented frame / data matrix the host stages consume - are copied back. ``None`` when the matrix is a host array or the device path faults."""
    if not hasattr(transformed_vals, "dev"):
        return None
    try:
        import cupy as cp

        from mlframe.feature_selection.filters._gpu_resident_discretize import _gpu_resident_discretize_codes
        from mlframe.feature_selection.filters.discretization.shared import safe_code_dtype as _safe_code_dtype

        code_dtype = np.dtype(_safe_code_dtype(int(nbins), dtype))
        dev = transformed_vals.dev
        codes = _gpu_resident_discretize_codes(dev.reshape(-1, 1) if dev.ndim == 1 else dev, int(nbins), out_dtype=code_dtype)
        return np.asarray(cp.asnumpy(codes.astype(cp.dtype(code_dtype), copy=False)))
    except Exception as e:
        logger.debug("device binning of the survivor matrix failed, binning on the host: %s", e)
        return None


def _materialise_and_fina_step3_gate_composite_drop(self, _gate_composite_drop, prospective_additions, st, cols, num_fs_steps, verbose):
    """Step 3 of _materialise_and_fina_step2_pruned_here_byte: lines starting at ``if _gate_composite_drop:``."""
    if _gate_composite_drop:
        from mlframe.feature_selection.filters.mrmr import get_new_feature_name as _gnf_gate
        _filtered: dict = {}
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in prospective_additions.items():
            if not _tpf or _tvals is None or not _ncols:
                _filtered[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                continue
            st._keep_idx = [i for i, nm in enumerate(_ncols) if nm not in _gate_composite_drop]
            if not st._keep_idx:
                continue
            if len(st._keep_idx) == len(_ncols):
                _filtered[_rp] = (_tpf, _tvals, _ncols, _nnb, _msgs)
                continue
            st._n2c = {_gnf_gate(_cfg, cols): _cfg for _cfg, _ in _tpf}
            st._new_tpf = set()
            for _np, _oi in enumerate(st._keep_idx):
                _cfg = st._n2c.get(_ncols[_oi])
                if _cfg is not None:
                    st._new_tpf.add((_cfg, _np))
            # a length-mismatched _nnb used to pass through
            # UNFILTERED (not narrowed to _keep_idx), silently reintroducing the nbins-vs-cols length
            # mismatch this filtering exists to prevent. Log so a genuine producer-shape bug is never
            # indistinguishable from the expected None case.
            if _nnb is not None and (not hasattr(_nnb, "__len__") or len(_nnb) != len(_ncols)):
                log_throttle(
                    logger, "step_score_nnb_length_mismatch_filtered", logging.WARNING,
                    "mrmr: _nnb length mismatch (expected %d columns) while filtering to kept columns; "
                    "passing it through UNFILTERED -- a downstream nbins/cols length assertion may now fire.",
                    len(_ncols),
                )
            st._new_nnb = [_nnb[i] for i in st._keep_idx] if _nnb is not None and hasattr(_nnb, "__len__") and len(_nnb) == len(_ncols) else _nnb
            _filtered[_rp] = (st._new_tpf, _tvals[:, st._keep_idx], [_ncols[i] for i in st._keep_idx], st._new_nnb, _msgs)
        prospective_additions = _filtered
        for _dn in _gate_composite_drop:
            _record_fe_rejection(
                self, gate="gate_composite_overmaterialization",
                candidate=str(_dn), operands=None, operator="engineered",
                observed=float("nan"), threshold=float("nan"),
                reason="raw_coverage_subset_of_clean_survivors", step=int(num_fs_steps),
            )
        if verbose:
            logger.info(
                "MRMR FE: pruned %d gate-operand composite(s) whose raw coverage is already provided by "
                "clean non-gate engineered survivors (over-materialization): %s",
                len(_gate_composite_drop), sorted(_gate_composite_drop),
            )
    return prospective_additions


def _materialise_and_fina_step3_cols_space_index(self, prospective_additions, engineered_features, verbose, fe_max_steps, X, discretize_array, cols, st, nbins, _is_polars_input, engineered_recipes, get_new_feature_name, _poly_coefs, fe_unary_preset, fe_binary_preset, n_recommended_features, checked_pairs):
    """Step 3 of materialise_and_finalise_fe_candidates: lines starting at ``for raw_vars_pair, (this_pair_features, transformed_vals, new_cols, ne``."""
    import polars as pl

    for raw_vars_pair, (this_pair_features, transformed_vals, new_cols, new_nbins, messages) in prospective_additions.items():
        if this_pair_features:
            engineered_features.update(this_pair_features)
            if verbose:
                for mes in messages:
                    logger.info(mes)
            if fe_max_steps >= 1:
                _n_forms = len(this_pair_features)
                if self.quantization_method == "quantile":
                    # BATCHED: one ``discretize_2d_quantile_batch`` call over the whole
                    # (n, _n_forms) ``transformed_vals`` block instead of ``_n_forms`` separate
                    # ``discretize_array`` calls - bit-identical per its own docstring (same quantile
                    # grid / percentile-edge / searchsorted per column), the SAME pattern already the
                    # default one layer up in ``_pairs_score.py``/``_pairs_emit.py`` (gated on
                    # ``quantization_method == "quantile"`` there too, see ``_use_batch_disc``).
                    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
                    new_vals = _device_codes_to_host(transformed_vals, self.quantization_nbins, self.quantization_dtype)
                    if new_vals is None:
                        new_vals = discretize_2d_quantile_batch(
                            np.asarray(transformed_vals), n_bins=self.quantization_nbins, dtype=self.quantization_dtype,
                        )
                else:
                    # Pre-widen the buffer dtype the SAME way ``discretize_array`` widens its own return
                    # value internally (``_safe_code_dtype``): the pre-fix code preallocated ``new_vals``
                    # at the raw (possibly too-narrow) ``self.quantization_dtype`` and wrote each
                    # already-widened column into it, which silently DOWNCASTS back to the narrow dtype on
                    # assignment - wrapping codes negative for ``n_bins > 127`` under the (non-default)
                    # ``quantization_dtype=int8`` config, exactly the bug ``_safe_code_dtype`` exists to
                    # prevent everywhere else it's used (``discretize_array``/``discretize_2d_quantile_batch``/
                    # ``discretize_2d_array``).
                    from mlframe.feature_selection.filters.discretization.shared import safe_code_dtype as _safe_code_dtype
                    _safe_dtype = _safe_code_dtype(self.quantization_nbins, self.quantization_dtype, reserve_nan_slot=(self.quantization_method == "uniform"))
                    new_vals = np.empty(shape=(len(X), _n_forms), dtype=_safe_dtype)
                    for j in range(_n_forms):
                        new_vals[:, j] = discretize_array(
                            arr=transformed_vals[:, j],
                            n_bins=self.quantization_nbins,
                            method=self.quantization_method,
                            dtype=self.quantization_dtype,
                        )
                _n_cols_before = len(cols)
                st._data_chunks.append(new_vals)
                # ``nbins`` is a numpy.ndarray (returned by categorize_dataset), so plain ``+`` does
                # element-wise addition / broadcasting, not concatenation. Use np.concatenate so nbins
                # grows in lockstep with data.shape[1] (otherwise screen_predictors trips its
                # targets_data.shape[1] == len(targets_nbins) assertion when engineered cols feed back).
                nbins = np.concatenate([
                    np.asarray(nbins),
                    np.asarray(new_nbins, dtype=nbins.dtype),
                ])
                cols = cols + new_cols
                # cols-space indices of the freshly appended engineered columns.
                st._newly_engineered_indices.extend(range(_n_cols_before, len(cols)))
                # Use the DISCRETISED codes (``new_vals``) for the augmented
                # output frame, NOT the raw ``transformed_vals``. The fit-time
                # frame must match what ``transform()`` reproduces on test data
                # (the recipe replay emits quantised bin codes), otherwise a
                # consumer reading the fit-time augmented frame would see raw
                # floats while transform() emits codes - a silent fit/transform
                # skew. ``transformed_vals`` (raw) is still used below to pin the
                # recipe's quantile edges.
                if _is_polars_input:
                    # Polars is immutable: with_columns returns a new frame sharing buffers; caller's X untouched.
                    _series_to_add = [pl.Series(col, new_vals[:, j]) for j, col in enumerate(new_cols)]
                    X = X.with_columns(_series_to_add)
                else:
                    # Index by the per-column position, not the
                    # leaked loop variable ``j`` (which held len-1 after the
                    # discretize loop above, so EVERY appended pandas column
                    # silently received the LAST survivor's values).
                    for _jc, col in enumerate(new_cols):
                        X[col] = new_vals[:, _jc]

                # ENGINEERED-OPERAND FEED-FORWARD: stash the CONTINUOUS
                # engineered values (``transformed_vals``) keyed by column name. The
                # augmented frame ``X`` only carries the DISCRETISED bin codes (needed
                # for screening), but the NEXT FE step's pair search must combine the
                # CONTINUOUS values: ``add(bin_codes(eng1), bin_codes(eng2))`` is
                # severely lossy (measured: the additive composite of the two real
                # step-1 features keeps MI 0.88 from bin codes vs 1.81 - the full
                # signal - from continuous values, so the code form fails the
                # engineered-MI gate). ``check_prospective_fe_pairs`` reads this store
                # (threaded as ``engineered_operand_values``) so ``(eng_i, eng_j)``
                # composites are built on the continuous values and recover the signal.
                st._eng_cont_store = getattr(self, "_engineered_continuous_", None)
                if st._eng_cont_store is None:
                    st._eng_cont_store = {}
                    self._engineered_continuous_ = st._eng_cont_store
                for _jc, col in enumerate(new_cols):
                    if transformed_vals.shape[1] > _jc:
                        # A device-backed survivor matrix hands over a lazy device column: the next FE step's pair search and the device stages read it
                        # in place, and a host consumer copies it back (once) only when it actually reads it.
                        _cv = transformed_vals[:, _jc]
                        st._eng_cont_store[col] = _cv if hasattr(_cv, "dev") else np.asarray(_cv, dtype=np.float64)

                # Build EngineeredRecipe for each newly-appended column so transform() can replay it.
                # Runs whenever columns were added (fe_max_steps >= 1). NESTED-ENGINEERED PARENTS
                #: a parent that is itself an engineered column (a higher-order
                # composite, e.g. add(div(sqr(a),abs(b)), mul(log(c),sin(d)))) is now REPLAYABLE -
                # we pass the parent's own EngineeredRecipe (already in ``engineered_recipes`` from
                # the prior step) so replay recomputes it recursively. Only when a parent is
                # engineered AND has no replayable recipe do we skip (cannot reconstruct it).
                if engineered_recipes is not None:
                    from mlframe.feature_selection.filters.engineered_recipes import build_unary_binary_recipe
                    _raw_names = set(self.feature_names_in_)
                    for config, _j in this_pair_features:
                        # config = (transformations_pair, bin_func_name, i)
                        # transformations_pair = ((var_a_idx, unary_a_name),
                        #                        (var_b_idx, unary_b_name))
                        transformations_pair, bin_func_name, _ = config
                        var_a_idx, unary_a_name = transformations_pair[0]
                        var_b_idx, unary_b_name = transformations_pair[1]
                        # Map cols-index -> name. A RAW parent resolves to a ``feature_names_in_``
                        # name; an ENGINEERED parent resolves to its prior recipe (nested replay).
                        src_a_name_raw = cols[var_a_idx]
                        src_b_name_raw = cols[var_b_idx]
                        # NESTED-PARENT RESOLUTION also consults the SUBSUMED-FRAGMENT recipe store
                        # . When C2 additive-fusion subsumes a fragment at a prior step it
                        # POPS the fragment from ``engineered_recipes`` (so it is not re-selected
                        # bare), but a LATER step's pair / escalation search may legitimately re-derive
                        # a composite that nests that fragment (e.g. ``abs(div(sqr(a),neg(b)))``). The
                        # fragment's recipe must still be reachable so the re-derived composite stays
                        # REPLAYABLE - otherwise it is recorded recipe-less and DROPPED from transform
                        # output, collapsing the selection back to raw operands (the F2 scaled_1_5
                        # DOMINANT-CAPTURE leak). The preserved store keeps the fragment recipe object
                        # available for nested replay WITHOUT re-admitting the bare fragment to selection.
                        st._subsumed_store = getattr(self, "_fe_subsumed_recipes_", None) or {}
                        _nested_a = None if src_a_name_raw in _raw_names else (engineered_recipes.get(src_a_name_raw) or st._subsumed_store.get(src_a_name_raw))
                        _nested_b = None if src_b_name_raw in _raw_names else (engineered_recipes.get(src_b_name_raw) or st._subsumed_store.get(src_b_name_raw))
                        # Skip only when an operand is engineered but its parent recipe is missing
                        # (un-replayable) - e.g. a parent from a stage that did not register one.
                        _a_unreplayable = (src_a_name_raw not in _raw_names) and (_nested_a is None)
                        _b_unreplayable = (src_b_name_raw not in _raw_names) and (_nested_b is None)
                        if _a_unreplayable or _b_unreplayable:
                            if verbose:
                                logger.info(
                                    "Skipping recipe construction for nested engineered feature " "'%s' (parent %s has no replayable recipe).",
                                    get_new_feature_name(config, cols),
                                    src_a_name_raw if _a_unreplayable else src_b_name_raw,
                                )
                            continue
                        eng_name = get_new_feature_name(config, cols)
                        #
                        # pass the fit-time engineered values
                        # ``transformed_vals[:, _j]`` so the recipe
                        # persists the quantile edges. Pre-fix replay
                        # re-quantiled on test data, silently shifting
                        # bin codes between fit and transform under
                        # distribution drift.
                        _fit_vals = transformed_vals[:, _j] if transformed_vals.shape[1] > _j else None
                        # Per-operand pre-warp: when a side used the learned
                        # ``prewarp`` pseudo-unary, hand its fitted spec to the
                        # recipe so replay reproduces the closed-form warp.
                        # Pair-scoped spec first (the warp this pair's column was actually built with); var key is the legacy fallback.
                        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import _prewarp_pair_spec_key

                        _pw_a = st._prewarp_specs.get(_prewarp_pair_spec_key(raw_vars_pair, var_a_idx), st._prewarp_specs.get(var_a_idx)) if unary_a_name == "prewarp" else None
                        _pw_b = st._prewarp_specs.get(_prewarp_pair_spec_key(raw_vars_pair, var_b_idx), st._prewarp_specs.get(var_b_idx)) if unary_b_name == "prewarp" else None
                        # Per-operand median gate: when a side used the
                        # ``gate_med`` pseudo-unary, hand its fitted TRAIN
                        # median to the recipe so replay reproduces the
                        # closed-form ``(x > median)`` gate.
                        _gm_a = st._gate_med_specs.get(var_a_idx) if unary_a_name == "gate_med" else None
                        _gm_b = st._gate_med_specs.get(var_b_idx) if unary_b_name == "gate_med" else None
                        # Freeze the fit-time ``smart_log`` shift
                        # anchor per ``log`` side. ``smart_log`` shifts non-positive
                        # inputs by ``(1e-5 - nanmin(operand))``; that anchor is
                        # data-dependent, so a transform row-slice recomputes a
                        # different shift and the log output (then the bin code)
                        # drifts. Reconstruct the CONTINUOUS fit-time operand exactly
                        # as replay does (raw column from X, or the nested parent's
                        # continuous replay) and compute the frozen anchor so replay is
                        # byte-exact. Best-effort: None leaves the legacy refit path.
                        from mlframe.feature_selection.filters._mrmr_fe_step._step_log_anchor import smart_log_anchor

                        _ls_a = smart_log_anchor(src_a_name_raw, _nested_a, X, st._ls_anchor_memo) if unary_a_name == "log" else None
                        _ls_b = smart_log_anchor(src_b_name_raw, _nested_b, X, st._ls_anchor_memo) if unary_b_name == "log" else None
                        engineered_recipes[eng_name] = build_unary_binary_recipe(
                            name=eng_name,
                            src_a_name=src_a_name_raw,
                            src_b_name=src_b_name_raw,
                            unary_a_name=unary_a_name,
                            unary_b_name=unary_b_name,
                            binary_name=bin_func_name,
                            # Persist the hermite coef of a poly_<coef> unary so recipe replay can hermval it.
                            poly_a_coef=(_poly_coefs.get(unary_a_name) if _poly_coefs is not None and unary_a_name.startswith("poly_") else None),
                            poly_b_coef=(_poly_coefs.get(unary_b_name) if _poly_coefs is not None and unary_b_name.startswith("poly_") else None),
                            unary_preset=fe_unary_preset,
                            binary_preset=fe_binary_preset,
                            quantization_nbins=self.quantization_nbins,
                            quantization_method=self.quantization_method,
                            quantization_dtype=self.quantization_dtype,
                            fit_values_for_edges=_fit_vals,
                            prewarp_a=_pw_a,
                            prewarp_b=_pw_b,
                            gate_med_a=_gm_a,
                            gate_med_b=_gm_b,
                            # Nested-engineered parents: None for raw operands,
                            # else the parent's recipe so replay recomputes it recursively.
                            nested_parent_a=_nested_a,
                            nested_parent_b=_nested_b,
                            log_shift_a=_ls_a,
                            log_shift_b=_ls_b,
                        )

            n_recommended_features += len(this_pair_features)

        # factors_to_use / factors_names_to_use are
        # already threaded through the upstream FE loop (MRMR.fit -> FE-pair
        # iteration consults these via `self.factors_to_use` and the
        # caller-supplied filter); no extra plumbing needed at this
        # bookkeeping site. The pair-cache only tracks "raw pair already
        # processed", which is name-agnostic.
        checked_pairs.add(raw_vars_pair)
    return X, cols, n_recommended_features, nbins


def _materialise_and_fina_step1_try(self, num_fs_steps, prospective_additions, prospective_pairs, _prevalence_failed_synergy, X, cols, classes_y, st, _pair_maxt_floor, verbose, _is_polars_input, discretize_array, data, nbins, n_recommended_features, engineered_recipes):
    """Step 1 of _materialise_and_fina_step2_produced_admitted_column: lines starting at ``try:``."""

    try:
        # Per-fit escalation ledger: a pair escalated once is never re-escalated
        # in a later FE step of the SAME fit (a step-2 retry would re-propose the
        # identical candidate on identical data and emit a duplicate ``..._2``
        # column). Reset on the first FE step so re-fits start clean.
        _esc_done, _esc_failed = _materialise_and_fina_step1_column_reset_first(self, num_fs_steps, prospective_additions, prospective_pairs, _prevalence_failed_synergy, X, cols, classes_y)
        X, cols, data, n_recommended_features, nbins = _materialise_and_fina_step2_esc_failed(self, _esc_failed, classes_y, st, prospective_additions, X, cols, _pair_maxt_floor, verbose, _prevalence_failed_synergy, _esc_done, _is_polars_input, discretize_array, data, nbins, n_recommended_features, engineered_recipes)
    except Exception:
        logger.warning(
            "MRMR FE auto-escalation failed; continuing with the unary/binary survivors only.",
            exc_info=True,
        )
    return X, cols, data, n_recommended_features, nbins


def _materialise_and_fina_step1_column_reset_first(self, num_fs_steps, prospective_additions, prospective_pairs, _prevalence_failed_synergy, X, cols, classes_y):
    """Step 1 of _materialise_and_fina_step1_try: lines starting at ``if num_fs_steps == 0 or not hasattr(self, "_fe_escalation_done_pairs_"``."""
    if num_fs_steps == 0 or not hasattr(self, "_fe_escalation_done_pairs_"):
        self._fe_escalation_done_pairs_ = set()
        self.fe_escalation_history_ = []
    _esc_done = self._fe_escalation_done_pairs_
    _esc_pairs_with_additions = {_rp for _rp, _v in prospective_additions.items() if _v[0]}
    _esc_failed = [(_k[0], float(_k[1])) for _k in prospective_pairs if _k[0] not in _esc_pairs_with_additions and _k[0] not in _esc_done]
    # PREVALENCE-FAILED SYNERGY RESCUE (F2 a**2/b miss): synergy
    # pairs that cleared the order-2 maxT floor but missed the stricter raw-MI
    # synergy prevalence ratio (the raw-MI ratio under-estimates a smooth ratio
    # interaction - the genuine a**2/b scores ~1.11 < 1.5). Feed them to the
    # escalation as failed pairs: ``_propose_poly``'s leak-safe held-out
    # pair-vs-single |corr| margin re-decides, then the full gates. These never
    # entered ``prospective_pairs`` (the prevalence gate dropped them), so the
    # set-difference above cannot contain them; add directly (skip ones already
    # escalated this fit / already admitted by the unary search).
    _esc_failed.extend((_pp, _pmi) for _pp, _pmi in _prevalence_failed_synergy.items() if _pp not in _esc_pairs_with_additions and _pp not in _esc_done)
    # UNDERDELIVERY trigger: a pair that DID admit a column but
    # whose best capture leaves SIGNIFICANT conditional pair MI on the table
    # (leftover CMI(joint(a,b); y | best admitted) above its conditional-
    # permutation null) is escalated too - e.g. the ``y=sin(3.7a)*b``
    # envelope capture ``mul(sin(a),qubed(b))`` that the marginal-uplift
    # fallback admits while most of the detected signal stays unexpressed.
    # Stride-subsampled + 8-perm null keeps the every-pair-delivers common
    # case cheap; a false trigger only PROPOSES - the full gates (incl. the
    # S5 CMI gate vs the pair's own admitted column) still decide. See
    # ``find_underdelivering_pairs``.
    if bool(getattr(self, "fe_escalation_underdelivery_enable", True)):
        from mlframe.feature_selection.filters._fe_auto_escalation import find_underdelivering_pairs
        _esc_failed.extend(find_underdelivering_pairs(
            self,
            prospective_pairs=prospective_pairs,
            prospective_additions=prospective_additions,
            X=X, cols=cols, classes_y=classes_y, done=_esc_done,
        ))
    return _esc_done, _esc_failed


def _materialise_and_fina_step2_esc_failed(self, _esc_failed, classes_y, st, prospective_additions, X, cols, _pair_maxt_floor, verbose, _prevalence_failed_synergy, _esc_done, _is_polars_input, discretize_array, data, nbins, n_recommended_features, engineered_recipes):
    """Step 2 of _materialise_and_fina_step1_try: lines starting at ``if _esc_failed:``."""
    import polars as pl

    if _esc_failed:
        from mlframe.feature_selection.filters._fe_auto_escalation import run_fe_auto_escalation
        from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned as _esc_mi, _quantile_bin as _esc_qbin
        # Admitted-support context for the S5 gate: the engineered columns
        # the main path just materialised (continuous values + marginal MI).
        from mlframe.feature_selection.filters._mrmr_fe_step._step_class_codes import dense_class_codes

        _esc_y_dense = dense_class_codes(classes_y)
        _esc_admitted_pool: dict = {}
        _esc_batched_done = False
        if st._gate_resident:
            from mlframe.feature_selection.filters._mrmr_fe_step._step_batched_marginals import batched_device_marginals

            _e_names, _e_vals = [], []
            for _tpf, _tvals, _ncols, _nnb, _msgs in prospective_additions.values():
                if not _tpf or _tvals is None or not _ncols:
                    continue
                for _jc, _cname in enumerate(_ncols):
                    if _tvals.shape[1] > _jc:
                        _e_names.append(_cname)
                        _ev = _tvals[:, _jc]
                        _e_vals.append(_ev if hasattr(_ev, "dev") else np.asarray(_ev, dtype=np.float64))
            _e_mi = batched_device_marginals(_e_vals, _esc_y_dense, int(self.quantization_nbins))
            if _e_mi is not None:
                _esc_admitted_pool = {_nm: (_vv, _mm) for _nm, _vv, _mm in zip(_e_names, _e_vals, _e_mi)}
                _esc_batched_done = True
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in (() if _esc_batched_done else prospective_additions.items()):
            if not _tpf or _tvals is None or not _ncols:
                continue
            for _jc, _cname in enumerate(_ncols):
                if _tvals.shape[1] <= _jc:
                    continue
                _cv = np.asarray(_tvals[:, _jc], dtype=np.float64)
                # DEVICE-BORN admitted-pool marginal MI (same residency contract as the gate's _cmi_cands
                # build above): bin the admitted engineered column ONCE on device and score its marginal
                # MI from the RESIDENT codes so the candidate float + codes never re-cross H2D. Host
                # fallback per-column on any cupy fault. Selection-equivalent (same device partition).
                _cb = None
                if st._gate_resident and np.isfinite(_cv).all():
                    try:
                        from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin_gpu_resident as _qbr_esc
                        _cb = _qbr_esc(_cv, int(self.quantization_nbins))
                    except Exception as e:
                        logger.debug("_quantile_bin_gpu_resident (escape) failed, falling back to the host path: %s", e)
                        _cb = None
                if _cb is None:
                    _cb = _esc_qbin(_cv, nbins=int(self.quantization_nbins))
                _esc_admitted_pool[_cname] = (_cv, float(_esc_mi(_cb, _esc_y_dense, None, kx=int(self.quantization_nbins))))
        # Per-pair admitted-capture values: UNDERDELIVERY-triggered pairs
        # get their proposers fit on the RESIDUAL of the target given the
        # existing capture (see ``run_fe_auto_escalation``); zero-admission
        # pairs have no entry and fit the full target.
        _esc_capture_vals: dict = {}
        for _pp, _pmi in _esc_failed:
            _v = prospective_additions.get(_pp)
            if _v and _v[0] and _v[1] is not None and _v[2]:
                _esc_capture_vals[tuple(_pp)] = _v[1][:, : min(int(_v[1].shape[1]), len(_v[2]))]
        _esc_admitted = run_fe_auto_escalation(
            self,
            failed_pairs=_esc_failed,
            X=X, cols=cols,
            classes_y=classes_y,
            pair_maxt_floor=float(_pair_maxt_floor),
            admitted_pool=_esc_admitted_pool,
            verbose=verbose,
            capture_vals=_esc_capture_vals,
            rescue_pairs=set(_prevalence_failed_synergy.keys()),
        )
        # Mark the pairs escalation actually PROCESSED (budget-selected
        # eligible) as done for this fit, admitted or not - a retry on
        # identical data cannot change the verdict.
        _esc_done.update(getattr(self, "fe_escalation_info_", {}).get("eligible_idx", []) or [])
        if _esc_admitted:
            # Materialise exactly like the unary/binary survivors above:
            # discretised codes into data/X, name into cols, nbins in
            # lockstep, recipe registered, continuous values stashed for
            # the engineered-operand feed-forward, index promoted below.
            if not _is_polars_input and hasattr(X, "columns") and not st._x_is_owned:
                X = X.copy(deep=False)
                st._x_is_owned = True
            # Pre-widen to _safe_code_dtype, not the raw (possibly too-narrow) quantization_dtype:
            # discretize_array widens its OWN return value internally and assigning that widened
            # value into a narrow-dtype buffer silently downcasts it back, wrapping codes negative
            # for quantization_nbins > 127 under quantization_dtype=int8 (same bug already fixed
            # above at the unary/binary materialize block via _safe_code_dtype).
            from mlframe.feature_selection.filters.discretization.shared import safe_code_dtype as _safe_code_dtype
            _esc_safe_dtype = _safe_code_dtype(self.quantization_nbins, self.quantization_dtype, reserve_nan_slot=(self.quantization_method == "uniform"))
            _esc_new_codes: np.ndarray = np.empty(
                shape=(len(X), len(_esc_admitted)), dtype=_esc_safe_dtype,
            )
            for _je, _ec in enumerate(_esc_admitted):
                _esc_new_codes[:, _je] = discretize_array(
                    arr=np.asarray(_ec["values"], dtype=np.float64),
                    n_bins=self.quantization_nbins,
                    method=self.quantization_method,
                    dtype=self.quantization_dtype,
                )
            _n_cols_before_esc = len(cols)
            # Preserve the compact codes dtype (COMPACT CODES STORAGE in _fit_impl_core): nbins-bounded codes,
            # cast down to data.dtype rather than upcasting the whole matrix back to int32.
            data = np.append(data, _esc_new_codes.astype(data.dtype, copy=False), axis=1)
            nbins = np.concatenate([
                np.asarray(nbins),
                np.asarray([self.quantization_nbins] * len(_esc_admitted), dtype=np.asarray(nbins).dtype),
            ])
            cols = cols + [_ec["name"] for _ec in _esc_admitted]
            st._newly_engineered_indices.extend(range(_n_cols_before_esc, len(cols)))
            n_recommended_features += len(_esc_admitted)
            st._eng_cont_store = getattr(self, "_engineered_continuous_", None)
            if st._eng_cont_store is None:
                st._eng_cont_store = {}
                self._engineered_continuous_ = st._eng_cont_store
            if _is_polars_input:
                X = X.with_columns([pl.Series(_ec["name"], _esc_new_codes[:, _je]) for _je, _ec in enumerate(_esc_admitted)])
            for _je, _ec in enumerate(_esc_admitted):
                if not _is_polars_input:
                    X[_ec["name"]] = _esc_new_codes[:, _je]
                st._eng_cont_store[_ec["name"]] = np.asarray(_ec["values"], dtype=np.float64)
                if engineered_recipes is not None:
                    engineered_recipes[_ec["name"]] = _ec["recipe"]
    return X, cols, data, n_recommended_features, nbins


def _materialise_and_fina_step4_contract_stay_off(self, engineered_recipes, st, verbose, cols, classes_y, X, _is_polars_input, discretize_array, data, nbins, n_recommended_features, selected_vars):
    """Step 4 of materialise_and_finalise_fe_candidates: lines starting at ``if (``."""
    import polars as pl

    if (
        engineered_recipes is not None
        and bool(getattr(self, "fe_additive_fusion_enable", True))
        and int(getattr(self, "fe_max_engineered_operands", 8)) != 0
        and st._newly_engineered_indices and not _fe_tail_budget_spent("additive fusion", verbose)
    ):
        try:
            from mlframe.feature_selection.filters._fe_additive_fusion import propose_additive_fusions
            st._eng_cont_store = getattr(self, "_engineered_continuous_", None) or {}
            # Names of the engineered columns just materialised this step (cols-space indices
            # -> names) that have a registered (replayable) recipe.
            _newly_names = [cols[i] for i in st._newly_engineered_indices if 0 <= i < len(cols) and cols[i] in engineered_recipes]
            _raw_name_set = set(self.feature_names_in_)
            _fused, _subsumed, _subsumed_raws = propose_additive_fusions(
                self,
                engineered_recipes=engineered_recipes,
                engineered_continuous=st._eng_cont_store,
                newly_engineered_names=_newly_names,
                raw_name_set=_raw_name_set,
                cols=cols,
                classes_y=classes_y,
                X=X,
                nbins=int(self.quantization_nbins),
                seed=int(getattr(self, "random_seed", 0) or 0),
                verbose=int(verbose),
            )
            if _fused:
                # Materialise each fused compound exactly like an escalation survivor.
                if not _is_polars_input and hasattr(X, "columns") and not st._x_is_owned:
                    X = X.copy(deep=False)
                    st._x_is_owned = True
                # Same narrow-preallocation bug/fix as the escalation-materialize block above:
                # pre-widen to _safe_code_dtype so discretize_array's internally-widened return value
                # isn't silently downcast back on assignment.
                from mlframe.feature_selection.filters.discretization.shared import safe_code_dtype as _safe_code_dtype
                _fz_safe_dtype = _safe_code_dtype(self.quantization_nbins, self.quantization_dtype, reserve_nan_slot=(self.quantization_method == "uniform"))
                _fz_codes: np.ndarray = np.empty(shape=(len(X), len(_fused)), dtype=_fz_safe_dtype)
                for _jf, _fc in enumerate(_fused):
                    _dev_codes = _device_codes_to_host(_fc["values"], self.quantization_nbins, self.quantization_dtype) if self.quantization_method == "quantile" else None
                    if _dev_codes is not None:
                        _fz_codes[:, _jf] = _dev_codes[:, 0]
                        continue
                    _fz_codes[:, _jf] = discretize_array(
                        arr=np.asarray(_fc["values"], dtype=np.float64),
                        n_bins=self.quantization_nbins,
                        method=self.quantization_method,
                        dtype=self.quantization_dtype,
                    )
                _n_cols_before_fz = len(cols)
                data = np.append(data, _fz_codes.astype(data.dtype, copy=False), axis=1)
                nbins = np.concatenate([
                    np.asarray(nbins),
                    np.asarray([self.quantization_nbins] * len(_fused), dtype=np.asarray(nbins).dtype),
                ])
                cols = cols + [_fc["name"] for _fc in _fused]
                _fused_indices = list(range(_n_cols_before_fz, len(cols)))
                n_recommended_features += len(_fused)
                if _is_polars_input:
                    X = X.with_columns([pl.Series(_fc["name"], _fz_codes[:, _jf]) for _jf, _fc in enumerate(_fused)])
                for _jf, _fc in enumerate(_fused):
                    if not _is_polars_input:
                        X[_fc["name"]] = _fz_codes[:, _jf]
                    st._eng_cont_store[_fc["name"]] = _fc["values"] if hasattr(_fc["values"], "dev") else np.asarray(_fc["values"], dtype=np.float64)
                    engineered_recipes[_fc["name"]] = _fc["recipe"]
                self._engineered_continuous_ = st._eng_cont_store
                # Promote the fused compounds into selection and DROP the subsumed fragments
                # (by cols-space index) from selected_vars + the recipe dict so neither
                # support_ nor _engineered_recipes_ keeps a now-redundant fragment.
                _drop_names = set(_subsumed) | set(_subsumed_raws)
                _subsumed_idx = {i for i in selected_vars if 0 <= i < len(cols) and cols[i] in _drop_names}
                st._sv = [i for i in selected_vars if i not in _subsumed_idx]
                st._sv_set = set(st._sv)
                selected_vars = st._sv + [i for i in _fused_indices if i not in st._sv_set]
                # The fused compounds are freshly-engineered survivors too: register their
                # cols-indices so the stability vote (below) + the step>1 re-screen treat them
                # as engineered (the vote replays them via ``engineered_recipes`` regardless,
                # but this keeps the engineered-index bookkeeping consistent), and drop the
                # subsumed fragments' indices so they are not re-screened as engineered.
                st._newly_engineered_indices = [i for i in st._newly_engineered_indices if i not in _subsumed_idx] + [
                    i for i in _fused_indices if i not in set(st._newly_engineered_indices)
                ]
                # Preserve each subsumed fragment's recipe in a side-store BEFORE popping it from
                # the active recipe dict, so a later FE step that re-derives a composite nesting
                # the fragment can still resolve a replayable recipe for it (without re-admitting
                # the bare fragment). See the nested-parent resolution above.
                st._subsumed_store = getattr(self, "_fe_subsumed_recipes_", None)
                if st._subsumed_store is None:
                    st._subsumed_store = {}
                    self._fe_subsumed_recipes_ = st._subsumed_store
                for _sn in _subsumed:
                    _sr = engineered_recipes.get(_sn)
                    if _sr is not None:
                        st._subsumed_store[_sn] = _sr
                for _sn in _subsumed:
                    engineered_recipes.pop(_sn, None)
                # Record the subsumed-fragment names so the fit-end selection finaliser
                # strips them (they stay in cols/data and could otherwise be re-admitted by
                # a downstream marginal-MI screen with no recipe -> select-then-drop skew).
                _fz_dropped = getattr(self, "_fe_stability_vote_dropped_", None)
                if _fz_dropped is None:
                    _fz_dropped = set()
                    self._fe_stability_vote_dropped_ = _fz_dropped
                _fz_dropped.update(str(_sn) for _sn in _subsumed)
                # Register the fused-out RAW operands as redundancy-dropped so the
                # downstream raw-retention / rescue / augmentation passes
                # (``_prefe_screened_raw_`` re-add, never-empty rescue) do NOT resurrect a
                # raw the fused compound now fully subsumes - the SAME contract the
                # ``drop_redundant_raw_operands`` sweep relies on via ``_raw_redundancy_dropped_``.
                if _subsumed_raws:
                    _rrd = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
                    _rrd.update(str(_rn) for _rn in _subsumed_raws)
                    self._raw_redundancy_dropped_ = _rrd
                    # AUTHORITATIVE FUSED-SUBSUMED RAW SET: the raws the fused
                    # compound verifiably captures (the keep-probe said no independent
                    # residual). The fit-end support finaliser strips these unconditionally -
                    # the downstream retention / rescue / emit-both-reattach passes evaluate a
                    # raw against the CLEAN nested sub-expression (which, on a CORRUPTED a/b
                    # half like ``div(neg(b),a__p2sin1)``, does NOT capture the raw and so
                    # KEEPS it), but the FUSED compound DOES capture it - so this set carries
                    # the stronger (whole-compound) verdict the general sweep cannot reach once
                    # the raw is re-attached as an operand of the surviving compound.
                    _fsr = set(getattr(self, "_fused_subsumed_raws_", None) or set())
                    _fsr.update(str(_rn) for _rn in _subsumed_raws)
                    self._fused_subsumed_raws_ = _fsr
        except Exception:
            logger.warning(
                "MRMR FE additive-fusion failed; continuing with the un-fused engineered survivors.",
                exc_info=True,
            )
    return X, cols, data, n_recommended_features, nbins, selected_vars


def _materialise_and_fina_step5_byte_reproduces_pre(self, engineered_recipes, st, verbose, X, classes_y, num_fs_steps, selected_vars, cols):
    """Step 5 of materialise_and_finalise_fe_candidates: lines starting at ``if engineered_recipes and bool(getattr(self, "fe_stability_vote_enable``."""
    if engineered_recipes and bool(getattr(self, "fe_stability_vote_enable", True)) and st._newly_engineered_indices and not _fe_tail_budget_spent("stability vote", verbose):
        try:
            from mlframe.feature_selection.filters._fe_stability_vote import confirm_recipes_cross_fold, resolve_adaptive_vote_k

            _vote_diag: dict = {}
            _failed_eng = confirm_recipes_cross_fold(
                recipes=engineered_recipes,
                X=X,
                y_codes=classes_y,
                feature_names_in=list(self.feature_names_in_),
                nbins=int(self.quantization_nbins),
                # guarded-hybrid K: explicit int honoured verbatim (default 5 -> byte-identical);
                # only "auto" adapts, and only downward for tiny n (== 5 for n >= 500).
                k=resolve_adaptive_vote_k(getattr(self, "fe_stability_vote_k", 5), int(getattr(X, "shape", (0,))[0]) or len(X)),
                quorum=float(getattr(self, "fe_stability_vote_quorum", 0.6)),
                rng=np.random.default_rng(int(getattr(self, "random_seed", 0) or 0)),
                verbose=int(verbose),
                diagnostics_out=_vote_diag, cv_policy=get_cv_policy(self),
            )
            # REJECTION LEDGER (additive): record each recipe the cross-fold vote
            # dropped, with observed=folds-passed vs threshold=quorum bar (need_eff).
            for _fn in _failed_eng:
                _vd = _vote_diag.get(_fn, {})
                _record_fe_rejection(
                    self, gate="stability_vote",
                    candidate=str(_fn), operands=_vd.get("src_names"), operator="engineered",
                    observed=_vd.get("passes", float("nan")),
                    threshold=_vd.get("need_eff", float("nan")),
                    reason="below_quorum", step=int(num_fs_steps),
                )
            if _failed_eng:
                # Drop the failed engineered names from selected_vars (by cols-index)
                # and from the recipe dict so neither support_ nor _engineered_recipes_
                # admits a fold-specific winner.
                _failed_idx = {i for i in selected_vars if 0 <= i < len(cols) and cols[i] in _failed_eng}
                if _failed_idx:
                    selected_vars = [i for i in selected_vars if i not in _failed_idx]
                for _fn in _failed_eng:
                    engineered_recipes.pop(_fn, None)
                # The vote pops the recipe + de-selects the column,
                # but the materialised bin-code column STAYS in ``cols``/``data`` and is
                # therefore still visible to the downstream greedy screen (the step>1
                # re-screen / final selection). That screen re-admits it on its marginal
                # MI, so it lands in ``selected_vars_names`` at fit-end with NO recipe and
                # is silently DROPPED from transform output - a select-then-drop contract
                # violation. Record the vote-rejected engineered NAMES on ``self`` so the
                # fit-end selection finaliser strips them before they can re-enter support_.
                # The vote is authoritative: a fold-unstable recipe must not reappear.
                _vote_dropped = getattr(self, "_fe_stability_vote_dropped_", None)
                if _vote_dropped is None:
                    _vote_dropped = set()
                    self._fe_stability_vote_dropped_ = _vote_dropped
                _vote_dropped.update(str(_fn) for _fn in _failed_eng)
        except Exception as _vote_exc:
            logger.debug(
                "MRMR cross-fold stability vote failed (%s: %s); keeping the un-voted FE support.",
                type(_vote_exc).__name__, _vote_exc,
            )
    return selected_vars
