"""Per-candidate scoring / quantile-discretization materialise stage of ``MRMR._run_fe_step``.

Carved verbatim from ``_step_core.py`` (the irreducible single-function FE-step body) to bring that
module under the 1k-LOC ceiling. ``materialise_and_finalise_fe_candidates`` is the back half of the
``if True:`` FE-pair pipeline: the conditional-MI redundancy gate, the gate-composite / fast-search
over-materialisation prunes, the discretise+append+recipe materialise loop, the auto-escalation tail,
the ROOT-CAUSE-5 ``selected_vars`` promotion, and the cross-fold stability vote. It mutates the loop
locals threaded in as explicit keyword args (no closure capture) and returns the values the parent
re-binds; the dict/set/recipe containers (``engineered_features`` / ``checked_pairs`` /
``engineered_recipes``) are mutated in place. Selection is byte-for-byte identical to the inline block.

The two helper callables ``discretize_array`` / ``get_new_feature_name`` are passed in (they are lazily
imported in the parent from ``..mrmr`` to avoid the mrmr<->this-package import cycle); polars is imported
in-body only on the polars path. All other intra-filters dependencies are imported lazily in-body exactly
as they were in the inline block.
"""
from __future__ import annotations

import os

import numpy as np


from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection


from ._step_score_parts import (  # noqa: F401  -- carved helpers
    logger,
    _fe_tail_budget_spent,
    _materialise_and_fina_step1_st_simplenamespace_long,
    _materialise_and_fina_step3_single_column_reduces,
    _materialise_and_fina_step2_pruned_here_byte,
    _materialise_and_fina_step1_isinstance_cmi_cands,
    _materialise_and_fina_step2_final_support_gate,
    _materialise_and_fina_step3_gate_composite_drop,
    _materialise_and_fina_step3_cols_space_index,
    _materialise_and_fina_step1_no_op_byte,
    _materialise_and_fina_step2_produced_admitted_column,
    _materialise_and_fina_step1_try,
    _materialise_and_fina_step1_column_reset_first,
    _materialise_and_fina_step2_esc_failed,
    _materialise_and_fina_step3_cluster_aggregate_self,
    _materialise_and_fina_step4_contract_stay_off,
    _materialise_and_fina_step5_byte_reproduces_pre,
)


def materialise_and_finalise_fe_candidates(
    self,
    *,
    prospective_additions,
    prospective_pairs,
    _prevalence_failed_synergy,
    _pair_maxt_floor,
    _polynom_engineered_indices,
    data, cols, nbins, X,
    classes_y,
    selected_vars,
    engineered_features,
    engineered_recipes,
    checked_pairs,
    n_recommended_features,
    num_fs_steps,
    fe_max_steps,
    fe_unary_preset,
    fe_binary_preset,
    _is_polars_input,
    verbose,
    discretize_array,
    get_new_feature_name,
    _poly_coefs=None,
):
    """Run the redundancy gates, materialise admitted FE candidates, escalate, and stability-vote.

    Returns ``(prospective_additions, data, cols, nbins, X, selected_vars, n_recommended_features)``.
    ``engineered_features`` / ``checked_pairs`` / ``engineered_recipes`` are mutated in place.
    """
    st = _materialise_and_fina_step1_st_simplenamespace_long(self, _is_polars_input, X, cols)
    if st._fe_acceptance == "conditional_mi" and prospective_additions:
        from .._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate
        from ..mrmr import discretize_array  # already imported above; re-bind for clarity

        # Build the surviving-candidate pool: {engineered_col_name -> (continuous_vals,
        # marginal_mi)}. The continuous values are the pair search's ``transformed_vals``
        # (full-n float, NOT pre-binned). Marginal MI is computed cheaply from the binned
        # values via the same plug-in primitive (z=None) so the seed/relative-bar anchor
        # matches the production CMI estimator - no separate MI kernel.
        from .._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

        # y codes: reuse the discretised target the MI sweep scored against.
        from ._step_class_codes import dense_class_codes

        _y_dense = dense_class_codes(classes_y)

        # GATE SCORING SUBSAMPLE. The conditional-MI redundancy gate only DECIDES which engineered
        # candidates are redundant (drop) vs carry private y-information (keep) - an admit/drop decision with
        # wide margins (a redundant column collapses to ~0 CMI; a genuine private interaction keeps a large CMI).
        # That decision is selection-equivalent under a large strided subsample, while binning every candidate +
        # its marginal MI + the O(M^2) greedy CMI on full 1M rows dominates. Strided-subsample the candidate
        # continuous values AND y together (same stride -> aligned rows) above MLFRAME_FE_GATE_MAX_ROWS (default
        # 250k, 0=full-n). The stored ``_cmi_cands`` arrays and ``_y_dense_g`` feed ONLY the gate; the admitted
        # candidates' full-n values are materialised downstream unchanged, so this caps scoring cost only.
        _gate_max_rows = int(os.environ.get("MLFRAME_FE_GATE_MAX_ROWS", "250000"))
        _gate_n = int(_y_dense.shape[0])
        _gate_stride = int(_gate_n // _gate_max_rows) if _gate_max_rows > 0 and _gate_n > _gate_max_rows else 1
        _y_dense_g = _y_dense[::_gate_stride] if _gate_stride > 1 else _y_dense

        # DEVICE-BORN marginal MI (default ON under fe_gpu_strict_resident_enabled; opt-out
        # MLFRAME_FE_GATE_RESIDENT_CANDS=0). The seed/relative-bar anchor marginal MI is computed by binning
        # each candidate's continuous ``_vals`` ONCE on device and scoring MI from the RESIDENT int64 codes
        # (``_cmi_from_binned`` dispatches to the cupy resident-input branch), so the candidate float + codes
        # never re-cross H2D at this ``qbin_x`` / ``cmi_cand_x`` site. Falls back per-candidate to the host
        # ``_quantile_bin`` on any cupy fault. Selection-equivalent (same device partition as the gate's binning).
        # ``_gate_resident`` is computed once at function scope above (shared with the escalation admitted-pool build).
        st._cmi_cands = {}
        # One batched device binning + marginal-MI workload for every candidate on the resident path; the per-candidate loop below is the fallback.
        _batched_done = False
        if st._gate_resident:
            from ._step_batched_marginals import batched_device_marginals

            _b_names, _b_vals = [], []
            for _tpf, _tvals, _ncols, _nnb, _msgs in prospective_additions.values():
                if not _tpf or _tvals is None or not _ncols:
                    continue
                for _jc, _cname in enumerate(_ncols):
                    if _tvals.shape[1] <= _jc:
                        continue
                    _v = np.asarray(_tvals[:, _jc], dtype=np.float64)
                    _b_names.append(_cname)
                    _b_vals.append(_v[::_gate_stride] if _gate_stride > 1 else _v)
            _b_mi = batched_device_marginals(_b_vals, _y_dense_g, int(self.quantization_nbins))
            if _b_mi is not None:
                st._cmi_cands = {_nm: (_vv, _mm) for _nm, _vv, _mm in zip(_b_names, _b_vals, _b_mi)}
                _batched_done = True
        for _rp, (_tpf, _tvals, _ncols, _nnb, _msgs) in (() if _batched_done else prospective_additions.items()):
            if not _tpf or _tvals is None or not _ncols:
                continue
            for _jc, _cname in enumerate(_ncols):
                if _tvals.shape[1] <= _jc:
                    continue
                _vals = np.asarray(_tvals[:, _jc], dtype=np.float64)
                if _gate_stride > 1:
                    _vals = _vals[::_gate_stride]
                _vb = None
                if st._gate_resident and np.isfinite(_vals).all():
                    try:
                        from .._mi_greedy_cmi_fe import _quantile_bin_gpu_resident
                        _vb = _quantile_bin_gpu_resident(_vals, int(self.quantization_nbins))
                    except Exception as e:
                        logger.debug("_quantile_bin_gpu_resident failed, falling back to the host path: %s", e)
                        _vb = None
                if _vb is None:
                    _vb = _quantile_bin(_vals, nbins=int(self.quantization_nbins))
                _marg = float(_cmi_from_binned(_vb, _y_dense_g, None, kx=int(self.quantization_nbins)))
                st._cmi_cands[_cname] = (_vals, _marg)

        if len(st._cmi_cands) >= 2:
            _retain = float(getattr(self, "fe_engineered_cmi_retain_frac", 0.15))
            _escape = float(getattr(self, "fe_engineered_cmi_significance_escape_margin", 3.0))
            _cmi_max_cands = int(getattr(self, "fe_engineered_cmi_max_candidates", 64))
            _accepted, _diag = apply_cmi_redundancy_gate(
                st._cmi_cands, _y_dense_g,
                nbins=int(self.quantization_nbins),
                retain_frac=_retain,
                significance_escape_margin=_escape,
                max_candidates=_cmi_max_cands,
                seed=int(self._effective_random_seed() or 0),
                verbose=int(bool(verbose)),
            )
            st._cmi_dropped = set(st._cmi_cands) - _accepted
            if st._cmi_dropped and verbose:
                logger.info(
                    "CMI-redundancy gate: dropped %d/%d engineered survivors as redundant "
                    "given the admitted engineered support (TAU=%.3f): %s",
                    len(st._cmi_dropped), len(st._cmi_cands), _retain, sorted(st._cmi_dropped),
                )
            # REJECTION LEDGER (additive): the CMI gate already returns a per-name
            # ``_diag`` dict carrying observed CMI, the permutation floor, the relative
            # bar and the reason - harvest it for the dropped names (no recompute).
            for _dn in st._cmi_dropped:
                _d = _diag.get(_dn, {}) if isinstance(_diag, dict) else {}
                _reason = str(_d.get("reason", "redundant"))
                # Pick the bar this candidate actually missed: below_floor -> the perm
                # floor (observed = cmi); below_rel_bar -> the relative bar (observed =
                # debiased excess). Margin = observed - threshold (negative => missed).
                if _reason == "redundant_below_floor":
                    _obs = _d.get("cmi", float("nan"))
                    _thr = _d.get("floor", float("nan"))
                else:
                    _obs = _d.get("cmi_excess", float("nan"))
                    _thr = _d.get("rel_bar", float("nan"))
                _record_fe_rejection(
                    self, gate="cmi_redundancy",
                    candidate=str(_dn), operands=None, operator="engineered",
                    observed=_obs, threshold=_thr, reason=_reason,
                    step=int(num_fs_steps),
                )

    # Apply the CMI-redundancy drops to ``prospective_additions`` IN PLACE so the
    # materialise / recipe loop below never appends a redundant engineered column.
    # Each entry's parallel arrays (``this_pair_features`` set of (config, j),
    # ``transformed_vals`` columns, ``new_cols`` names, ``new_nbins``) are filtered to
    # the surviving columns by NAME. ``new_cols[i]`` is the name of the i-th
    # ``transformed_vals`` column; the matching ``(config, j)`` is the one whose
    # ``get_new_feature_name(config, cols)`` equals that name. Both downstream
    # consumers index ``transformed_vals`` by the per-column position
    # (materialise: ``for j in range(len(this_pair_features))``; recipe: the tuple's
    # stored ``j``), so the kept tuples are re-emitted as ``(config, new_position)``
    # with the new packed column position. Entries whose every column was dropped
    # are removed entirely. In the common one-best-per-pair case a pair holds a
    # single column, so this reduces to keep-entry / drop-entry.
    prospective_additions = _materialise_and_fina_step3_single_column_reduces(self, st, prospective_additions, cols, engineered_recipes, num_fs_steps, verbose, _polynom_engineered_indices)
    if not _is_polars_input and hasattr(X, "columns") and any(v[0] for v in prospective_additions.values()):
        X = X.copy(deep=False)
        st._x_is_owned = True
    # Accumulate the per-pair discretised code blocks and concatenate ONCE after the loop instead of
    # ``np.append``-ing the whole (n, K) code matrix per pair (each append reallocated + copied all of
    # ``data``, O(pairs * n * K)). ``cols`` / ``nbins`` still grow per iteration (cheap), so the
    # cols-space index bookkeeping is unchanged; ``data`` itself is not read inside the loop.
    st._data_chunks = []
    X, cols, n_recommended_features, nbins = _materialise_and_fina_step3_cols_space_index(self, prospective_additions, engineered_features, verbose, fe_max_steps, X, discretize_array, cols, st, nbins, _is_polars_input, engineered_recipes, get_new_feature_name, _poly_coefs, fe_unary_preset, fe_binary_preset, n_recommended_features, checked_pairs)

    # Flush the accumulated per-pair code blocks into ``data`` in a SINGLE concatenate (column order
    # matches the loop's append order, so it is byte-identical to the prior per-pair ``np.append``).
    # Must run before the escalation block below reads / appends ``data``.
    if st._data_chunks:
        data = np.concatenate([data, *st._data_chunks], axis=1)

    # AUTO-ESCALATION to the richer SHIPPED bases (backlog idea B,
    # default-ON). A pair that PASSED the pair-MI prescreen (ratio gate + order-2
    # maxT floor) but for which the unary/binary search above admitted NOTHING used
    # to end in the log_fe_summary WARNING below - detected signal, silently
    # abandoned. Escalate instead: PROPOSE candidates from the richer shipped basis
    # families (signal-adaptive orth-poly ALS warp across the 4 polynomial bases at
    # a higher degree + DEMODULATED adaptive-frequency Fourier/chirp warps - e.g.
    # the sin(3.7*a)*b inner frequency no library unary can express) and let the
    # EXISTING gates decide (maxT floor on MM-debiased MI + marginal-permutation
    # floor + the S5 conditional-MI redundancy gate vs the admitted engineered
    # support). Structurally a no-op (one set-difference) when every surviving pair
    # produced an admitted column - the common case. See ``_fe_auto_escalation``.
    X, cols, data, n_recommended_features, nbins = _materialise_and_fina_step2_produced_admitted_column(self, prospective_pairs, _prevalence_failed_synergy, verbose, num_fs_steps, prospective_additions, X, cols, classes_y, st, _pair_maxt_floor, _is_polars_input, discretize_array, data, nbins, n_recommended_features, engineered_recipes)

    # Promote the freshly-appended engineered
    # columns directly into ``selected_vars`` (cols-space). They already
    # cleared every FE gate (pair-MI prevalence, engineered-MI prevalence,
    # external validation) - the gates ARE the selection criterion for FE
    # survivors. Without this, the only path to ``support_`` was the
    # screening re-run at the top of the NEXT outer-loop iteration, which
    # never executes under the default ``fe_max_steps=1`` (the loop breaks
    # first), so ``_engineered_features_`` stayed empty. On multi-step the
    # re-screen still re-evaluates them and may prune weak ones. Mirrors the
    # cluster_aggregate self-selection pattern below.
    selected_vars = _materialise_and_fina_step3_cluster_aggregate_self(st, selected_vars)

    # C2 ADDITIVE-FUSION (default-ON). MUST RUN BEFORE the cross-fold stability vote
    # below (ordering fix for the F2 DOMINANT-CAPTURE / weak-half class): the weak c/d half
    # ``mul(log(c),sin(d))`` alone FAILS the cross-fold vote (its uplift is carried by too few
    # rows to clear the per-fold quorum), so if the vote ran first it would DROP that half before
    # C2 ever saw it - leaving no clean half to fuse. Running C2 first fuses the two surviving
    # engineered halves with DISJOINT raw-token sets + GENUINE additive separability (the fused
    # ``add`` MI exceeds both halves OR the 2-half OLS multiple-R beats the best single half) into
    # the single ``add(half_a, half_b)`` compound via the EXISTING unary_binary + nested-parent
    # recipe (no new recipe kind; byte-exact replay); the FUSED compound is registered in
    # ``engineered_recipes`` so it then FACES the stability vote (the fused compound passes the
    # vote - it reconstructs the whole target - while the bare weak half it subsumed does not
    # matter). The two now-subsumed fragments are popped from the recipe dict + dropped from
    # selection BEFORE the vote, so the vote never sees them. Self-gates to a no-op (byte-identical)
    # when fewer than two relevant disjoint engineered halves are present - the common case.
    # ``fe_max_engineered_operands == 0`` is the documented raw-only-pool contract (no composites
    # whose operands are themselves engineered features). Additive fusion combines two engineered
    # halves into ``add(half_a, half_b)``, i.e. a composite of engineered operands, so it must honor
    # that contract and stay off when the feed-forward cap is 0.
    X, cols, data, n_recommended_features, nbins, selected_vars = _materialise_and_fina_step4_contract_stay_off(self, engineered_recipes, st, verbose, cols, classes_y, X, _is_polars_input, discretize_array, data, nbins, n_recommended_features, selected_vars)

    # CROSS-FOLD RECIPE STABILITY VOTING. A near-free
    # consensus layer OVER the existing FE gates. The expensive search above ran
    # ONCE on the full data; here we add a cheap K-fold CONFIRMATION - each
    # surviving unary_binary recipe is REPLAYED (leak-safe: the recipe is frozen,
    # only the rows change) on K held-out folds, its uplift gate statistic
    # recomputed per fold, and the recipe ADMITTED only if it clears the gate in
    # >= ceil(q*K) folds. This complements the order-2/order-3 maxT floors: maxT
    # kills the chance-MAX candidate WITHIN a fold (best-of-pool selection bias);
    # this kills a recipe that won only on a fold-specific QUIRK of the full-data
    # split (its uplift carried by a few rows in the train split, collapses on the
    # held-out folds). NO REFIT - only K plug-in-MI replays per recipe, so the
    # cost is negligible. Failed recipes are dropped from BOTH ``engineered_recipes``
    # (so they never reach ``self._engineered_recipes_`` at fit-end) and from
    # ``selected_vars`` (so they never reach ``support_``). Default-ON; self-gates
    # to a no-op below 2 unary_binary survivors / k<2 / tiny n. ``fe_stability_vote_enable=False``
    # byte-reproduces the pre-vote support.
    selected_vars = _materialise_and_fina_step5_byte_reproduces_pre(self, engineered_recipes, st, verbose, X, classes_y, num_fs_steps, selected_vars, cols)

    # GROUP-AWARE FE DEMOTION: NOT done here. An early attempt at this check (dropping a zero-within-
    # group-MI engineered survivor right after the plain pair-search, before "usability-aware retention"
    # runs later in ``_fit_impl``) was found to be UNDONE: retention re-attaches recipes it judges
    # linearly-useful by its OWN criterion, independent of any earlier group-aware verdict, from its own
    # cached recompute - bypassing ``cols``/``data`` (and this early strip) entirely. The group-aware
    # demotion therefore runs ONCE, as the LAST thing before ``_fit_impl`` returns (after every
    # retention/UAED/escalation pass that could still mutate ``self._engineered_recipes_``), covering
    # both materialised columns and retention-only recipes uniformly - see ``_fit_impl_core.py``'s
    # "final choke point" block.
    return prospective_additions, data, cols, nbins, X, selected_vars, n_recommended_features
