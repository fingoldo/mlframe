"""Sibling of ``_fit_impl_core.py`` (part of the sub-split that brings the parent below
the project's 1k-LOC module-size gate).

Holds ``_fe_stage_cascade_early_b``: k-fold target encoding (Layer 33), count/frequency/cat-num-interaction encoding (Layer 34), missingness-aware FE (Layer 37), cross-feature ratio/grouped-delta/lagged-diff FE (Layer 38). Every FE family stage here reads the (possibly
already-augmented) ``X`` and appends its own winning engineered columns via
``fe_append_columns``/``fe_extract_columns`` -- mirrors the ``_hybrid_orth_family_variants``
siblings' own ``X``-in-``X``-out contract, confirmed the same way (grepping every
``X = fe_append_columns(X, ...)`` reassignment in range).

``_fit_entry_nan_mask`` is threaded in explicitly: a dict the parent builds BEFORE this
section (at fit-body setup) that the missingness-aware family (Layer 37) reads -- confirmed
via ``pyutilz.dev.freevar_analysis`` never reassigned here, only read, so no return needed.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .._fe_frame_ops import fe_to_pandas, fe_append_columns, fe_extract_columns, fe_polars_exceeds

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _fe_stage_cascade_early_b(
    self, *, X, y, verbose, fe_max_steps, _y_np, _fe_family_on, _fit_entry_nan_mask, _raw_input_cols_pre_fe,
    _kfold_te_pre_recipes, _binned_agg_pre_recipes, _count_enc_pre_recipes, _freq_enc_pre_recipes, _cat_num_pre_recipes,
    _miss_ind_pre_recipes, _miss_cnt_pre_recipes, _miss_pat_pre_recipes,
    _ratio_pre_recipes, _log_ratio_pre_recipes, _grouped_delta_pre_recipes, _lagged_diff_pre_recipes,
):
    """Run the k-fold target encoding (Layer 33), count/frequency/cat-num-interaction encoding (Layer 34), missingness-aware FE (Layer 37), cross-feature ratio/grouped-delta/lagged-diff FE (Layer 38) FE family stage(s) and return the (possibly column-augmented) ``X``.

    See the module docstring for the full section this carves out. All ``_*_pre_recipes`` dicts are
    caller-owned and mutated in place (never reassigned here -- confirmed via a systematic
    reassignment-vs-mutation check), so no return is needed for them.
    """
    # 2026-05-31 Layer 33 — K-fold target encoding for raw categorical
    # columns. Runs after hybrid + MI-greedy because TE is the standard
    # prod pattern for cardinality > 5 categoricals that the other two
    # stages do not touch. Recipes (kind ``kfold_target_encoded``) carry
    # only the full-data per-category lookup - no y at replay time.
    # Engineered columns route through ``hybrid_orth_features_`` so the
    # end-of-fit remap treats them as engineered features (same routing
    # as Layer 23 / 26 / 32).
    self.kfold_te_features_ = []
    if _fe_family_on("fe_kfold_te_enable", False):
        # K-fold target encoding is an OOF stat (no closed-form subsample-replay), so it needs the full frame: gate the
        # polars->pandas materialisation on size and skip a > ~2 GiB frame rather than whole-copy it (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: fe_kfold_te_enable=True but X is a large polars frame (> ~2 GiB); K-fold target encoding needs a "
                "full-frame OOF decision and is skipped to avoid a whole-frame to_pandas copy. Materialise a subset or "
                "pass pandas if you need it.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._target_encoding_fe import (
                    kfold_target_encode_with_recipes,
                )

                _te_cols_cfg = tuple(getattr(self, "fe_kfold_te_cols", ()) or ())
                # Explicit empty tuple -> auto-detect; explicit names -> use
                # exactly those (after intersecting with X.columns).
                _te_cols = list(_te_cols_cfg) if _te_cols_cfg else None
                if _te_cols is not None:
                    _hybrid_appended = set(self.hybrid_orth_features_ or [])
                    _mig_appended_set = set(self.mi_greedy_features_ or [])
                    _te_cols = [c for c in _te_cols if c in X.columns and c not in _hybrid_appended and c not in _mig_appended_set]
                _y_for_te = _y_np
                # TE works for both binary classification and regression as-
                # is (mean of {0,1} = P(y=1); mean of continuous = mean).
                # Cast bool / object to float to avoid type errors inside
                # the mean computation.
                _y_for_te = np.asarray(_y_for_te, dtype=np.float64).ravel()
                _X_before_te_cols = list(X.columns)
                # W6 follow-up: record this family's unified local-MI abs-MAD
                # floor kills into the FE rejection ledger (pure-record; the
                # kept set is unchanged so selection is byte-identical).
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
                _te_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _te_reject_sink(**_kw):
                    """Reject-sink callback for the k-fold target-encoding FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_te_step, **_kw)

                X_te, _te_appended, _te_recipes = kfold_target_encode_with_recipes(
                    fe_to_pandas(X), _y_for_te,
                    cat_cols=_te_cols,
                    n_folds=int(getattr(self, "fe_kfold_te_folds", 5)),
                    smoothing=float(getattr(self, "fe_kfold_te_smoothing", 10.0)),
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_te_reject_sink,
                    # Multi-stat target encoding: beyond the per-cell mean(y), also emit std / skew / kurt of y per
                    # category when requested. Helps when the category MODULATES a raw feature (heteroscedastic /
                    # varying-slope): +0.04..+0.09 OOS R^2 in those regimes (bench_multistat_cell_encoding). Default
                    # ("mean",) is byte-identical to the prior single-stat behaviour.
                    stats=tuple(getattr(self, "fe_kfold_te_stats", ("mean",)) or ("mean",)),
                )
                # Guard against silent overlap with prior stages: the
                # ``{col}__te`` suffix is dedicated to this stage so the
                # collision pre-condition would require a user-supplied
                # source column literally named ``{src}__te``. Drop any
                # accidental name collision rather than overwrite.
                _te_appended = [c for c in _te_appended if c not in _X_before_te_cols]
                if _te_appended:
                    X = fe_append_columns(X, fe_extract_columns(X_te, _te_appended))
                    self.kfold_te_features_ = list(_te_appended)
                    # Route through hybrid_orth_features_ so the end-of-fit
                    # remap routes by-name selected items into
                    # _engineered_recipes_ (Layer 23 routing path).
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_te_appended)
                    for _r in _te_recipes:
                        if _r.name in _te_appended:
                            _kfold_te_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit kfold_te: appended %d engineered " "column(s): %s",
                            len(_te_appended),
                            _te_appended[:8],
                        )
            except Exception as _te_exc:
                logger.warning(
                    "MRMR.fit kfold_te FE raised %s: %s; continuing " "without target-encoded columns.",
                    type(_te_exc).__name__,
                    _te_exc,
                )

    # GROUPED AGGREGATION OVER QUANTILE-BINNED NUMERIC CELLS. Appends leak-safe per-cell
    # mean/std/skew/kurt of numeric columns grouped by quantile-binned cells of other numerics. Runs in the
    # pre-FE region (before categorize_dataset) so the appended columns enter screening like any numeric, and
    # routes recipes through hybrid_orth_features_ so a selected binagg column lands in _engineered_recipes_.
    if _fe_family_on("fe_binned_numeric_agg_enable", False) and fe_polars_exceeds(X):
        warnings.warn(
            "MRMR: fe_binned_numeric_agg_enable=True but X is a large polars frame (> ~2 GiB); binned-agg is an OOF stat "
            "needing a full-frame decision and is skipped to avoid a whole-frame to_pandas copy.",
            UserWarning, stacklevel=3,
        )
    elif _fe_family_on("fe_binned_numeric_agg_enable", False):
        try:
            from .._binned_numeric_agg_fe import binned_numeric_agg_with_recipes
            _ba_y = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).ravel()
            _X_before_ba = list(X.columns)
            _bas_raw = getattr(self, "fe_binned_numeric_agg_stats", None)
            X_ba, _ba_appended, _ba_recipes = binned_numeric_agg_with_recipes(
                fe_to_pandas(X), _ba_y,
                stats=tuple(_bas_raw) if _bas_raw is not None else ("mean", "std", "skew", "kurt"),
                nbins_base=int(getattr(self, "fe_binned_numeric_agg_nbins", 10)),
                n_folds=int(getattr(self, "fe_kfold_te_folds", 5)),
                random_state=int(getattr(self, "random_seed", 0) or 0),
                max_pairs=int(getattr(self, "fe_binned_numeric_agg_max_pairs", 64)),
                redundancy_gate=bool(getattr(self, "fe_binned_numeric_agg_redundancy_gate", True)),
                min_cmi_gain=float(getattr(self, "fe_binned_numeric_agg_min_cmi_gain", 0.005)),
            )
            _ba_appended = [c for c in _ba_appended if c not in _X_before_ba]
            if _ba_appended:
                X = fe_append_columns(X, fe_extract_columns(X_ba, _ba_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ba_appended)
                for _r in _ba_recipes:
                    if _r.name in _ba_appended:
                        _binned_agg_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit binned_numeric_agg: appended %d engineered column(s): %s",
                        len(_ba_appended), _ba_appended[:8],
                    )
        except Exception as _ba_exc:
            logger.warning(
                "MRMR.fit binned_numeric_agg FE raised %s: %s; continuing without binned-agg columns.",
                type(_ba_exc).__name__, _ba_exc,
            )

    # 2026-05-31 Layer 34 — COUNT + FREQUENCY ENCODING + CAT x NUM
    # INTERACTION (target-mean residual). Three independent master switches;
    # each appends its own engineered columns AND emits one recipe per col.
    # Recipes route through ``hybrid_orth_features_`` so the end-of-fit
    # remap (Layer 23 pattern) routes them into ``_engineered_recipes_``.
    self.count_encoding_features_ = []
    self.frequency_encoding_features_ = []
    self.cat_num_interaction_features_ = []
    if (
        _fe_family_on("fe_count_encoding_enable", False)
        or _fe_family_on("fe_frequency_encoding_enable", False)
        or _fe_family_on("fe_cat_num_interaction_enable", False)
    ):
        # Count / frequency / cat-num-residual encodings are OOF / full-cardinality stats (no closed-form subsample-replay),
        # so they need the full frame: gate the materialisation on size and skip a > ~2 GiB polars frame (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: Layer 34 FE (count/frequency/cat_num) enabled but X is a large polars frame (> ~2 GiB); these OOF/"
                "cardinality encodings need a full-frame decision and are skipped to avoid a whole-frame to_pandas copy.",
                UserWarning, stacklevel=3,
            )
        else:
            from .._count_freq_interaction_fe import (
                count_encode_with_recipes,
                frequency_encode_with_recipes,
                cat_num_interaction_with_recipes,
            )
            from .._target_encoding_fe import auto_detect_te_cols
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

            # W6 follow-up: shared sink for the count/freq/cat-num family's
            # unified local-MI abs-MAD floor kills (pure-record; selection
            # byte-identical).
            _l34_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _l34_reject_sink(**_kw):
                """Shared reject-sink for the count/frequency/cat-num-interaction FE family; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_l34_step, **_kw)

            _hybrid_appended_l34 = set(self.hybrid_orth_features_ or [])
            _mig_appended_l34 = set(self.mi_greedy_features_ or [])
            _te_appended_l34 = set(self.kfold_te_features_ or [])
            _engineered_seen_l34 = _hybrid_appended_l34 | _mig_appended_l34 | _te_appended_l34

            # ----- Count encoding ----------------------------------------
            if _fe_family_on("fe_count_encoding_enable", False):
                try:
                    _cnt_cfg = tuple(getattr(self, "fe_count_encoding_cols", ()) or ())
                    if _cnt_cfg:
                        _cnt_cols = [c for c in _cnt_cfg if c in X.columns and c not in _engineered_seen_l34]
                    else:
                        _cnt_cols = auto_detect_te_cols(
                            X, min_card=5, max_card=500,
                        )
                    _X_before_cnt_cols = list(X.columns)
                    _y_for_cnt = _y_np
                    X_c, _cnt_appended, _cnt_recipes = count_encode_with_recipes(
                        fe_to_pandas(X), cat_cols=_cnt_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_cnt,
                        reject_sink=_l34_reject_sink,
                    )
                    _cnt_appended = [c for c in _cnt_appended if c not in _X_before_cnt_cols]
                    if _cnt_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_c, _cnt_appended))
                        self.count_encoding_features_ = list(_cnt_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cnt_appended)
                        for _r in _cnt_recipes:
                            if _r.name in _cnt_appended:
                                _count_enc_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit count_encoding: appended %d " "engineered column(s): %s",
                                len(_cnt_appended),
                                _cnt_appended[:8],
                            )
                except Exception as _cnt_exc:
                    logger.warning(
                        "MRMR.fit count_encoding FE raised %s: %s; " "continuing without count-encoded columns.",
                        type(_cnt_exc).__name__,
                        _cnt_exc,
                    )

            # ----- Frequency encoding ------------------------------------
            if _fe_family_on("fe_frequency_encoding_enable", False):
                try:
                    _freq_cfg = tuple(getattr(self, "fe_frequency_encoding_cols", ()) or ())
                    if _freq_cfg:
                        _freq_cols = [c for c in _freq_cfg if c in X.columns and c not in _engineered_seen_l34]
                    else:
                        _freq_cols = auto_detect_te_cols(
                            X, min_card=5, max_card=500,
                        )
                    _X_before_freq_cols = list(X.columns)
                    _y_for_freq = _y_np
                    X_f, _freq_appended, _freq_recipes = frequency_encode_with_recipes(
                        fe_to_pandas(X), cat_cols=_freq_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_freq,
                        reject_sink=_l34_reject_sink,
                    )
                    _freq_appended = [c for c in _freq_appended if c not in _X_before_freq_cols]
                    if _freq_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_f, _freq_appended))
                        self.frequency_encoding_features_ = list(_freq_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_freq_appended)
                        for _r in _freq_recipes:
                            if _r.name in _freq_appended:
                                _freq_enc_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit frequency_encoding: appended %d " "engineered column(s): %s",
                                len(_freq_appended),
                                _freq_appended[:8],
                            )
                except Exception as _freq_exc:
                    logger.warning(
                        "MRMR.fit frequency_encoding FE raised %s: %s; " "continuing without frequency-encoded columns.",
                        type(_freq_exc).__name__,
                        _freq_exc,
                    )

            # ----- Cat x Num interaction (OOF residual) ------------------
            if _fe_family_on("fe_cat_num_interaction_enable", False):
                try:
                    _cn_cats = tuple(getattr(self, "fe_cat_num_interaction_cat_cols", ()) or ())
                    _cn_nums = tuple(getattr(self, "fe_cat_num_interaction_num_cols", ()) or ())
                    _cn_cats = tuple(c for c in _cn_cats if c in X.columns)
                    _cn_nums = tuple(c for c in _cn_nums if c in X.columns)
                    if _cn_cats and _cn_nums:
                        _y_for_cn = _y_np
                        _y_for_cn = np.asarray(_y_for_cn, dtype=np.float64).ravel()
                        _X_before_cn_cols = list(X.columns)
                        X_cn, _cn_appended, _cn_recipes = cat_num_interaction_with_recipes(
                            fe_to_pandas(X),
                            _y_for_cn,
                            cat_cols=_cn_cats,
                            num_cols=_cn_nums,
                            n_folds=int(getattr(self, "fe_cat_num_interaction_folds", 5)),
                            smoothing=float(getattr(self, "fe_cat_num_interaction_smoothing", 10.0)),
                            random_state=int(getattr(self, "random_seed", 0) or 0),
                            mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                            mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                            reject_sink=_l34_reject_sink,
                        )
                        _cn_appended = [c for c in _cn_appended if c not in _X_before_cn_cols]
                        if _cn_appended:
                            X = fe_append_columns(X, fe_extract_columns(X_cn, _cn_appended))
                            self.cat_num_interaction_features_ = list(_cn_appended)
                            self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cn_appended)
                            for _r in _cn_recipes:
                                if _r.name in _cn_appended:
                                    _cat_num_pre_recipes[_r.name] = _r
                            if verbose:
                                logger.info(
                                    "MRMR.fit cat_num_interaction: appended %d " "engineered column(s): %s",
                                    len(_cn_appended),
                                    _cn_appended[:8],
                                )
                except Exception as _cn_exc:
                    logger.warning(
                        "MRMR.fit cat_num_interaction FE raised %s: %s; " "continuing without cat x num residual columns.",
                        type(_cn_exc).__name__,
                        _cn_exc,
                    )

    # 2026-05-31 Layer 37 — MISSINGNESS-AWARE FE. Three independent master
    # switches (indicator / count / pattern); each appends its own engineered
    # columns AND emits one recipe per column. Recipes route through
    # ``hybrid_orth_features_`` so the end-of-fit remap (Layer 23 pattern)
    # routes them into ``_engineered_recipes_``.
    self.missingness_indicator_features_ = []
    self.missingness_count_features_ = []
    self.missingness_pattern_features_ = []
    # Unlike every other fe_*_enable family, missingness indicator/count/pattern are not a FE SEARCH step
    # consuming the fe_max_steps budget - each is a deterministic, explicit-opt-in-only static derivation
    # from the input's own NaN structure (no candidate scan, no round-trip cost the budget was meant to
    # bound). Deliberately checked directly (not via _fe_family_on, which requires fe_max_steps>0) so
    # ``fe_max_steps=0`` + an explicit fe_missingness_*_enable=True still emits the requested column(s) -
    # exactly the "disable the FE search but keep this one explicit static feature" contract callers rely on.
    if (
        bool(getattr(self, "fe_missingness_indicator_enable", False))
        or bool(getattr(self, "fe_missingness_count_enable", False))
        or bool(getattr(self, "fe_missingness_pattern_enable", False))
    ):
        # Missingness indicator/count/pattern read whole-column NaN structure (no closed-form subsample-replay), so they
        # need the full frame: gate the materialisation on size and skip a > ~2 GiB polars frame (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: Layer 37 FE (missingness indicator/count/pattern) enabled but X is a large polars frame (> ~2 GiB); "
                "the missingness encodings need a full-frame decision and are skipped to avoid a whole-frame to_pandas copy.",
                UserWarning, stacklevel=3,
            )
        else:
            from .._missingness_fe import (
                auto_detect_missing_cols,
                missing_indicator_with_recipes,
                missingness_count_with_recipes,
                missingness_pattern_with_recipes,
            )
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

            # Restore the fit-entry NaN positions on the snapshot columns before deriving missingness encodings. An earlier include_numeric /
            # binned_numeric_agg cat-FE stage GPU-categorizes and imputes X in place (when CUDA_PATH is set), which erases the very NaNs the
            # missingness-FE family encodes - is_missing__ would be all-zeros and missingness_pattern would collapse to a single pattern. The raw
            # NaNs are the user's input; MRMR's nan_strategy='separate_bin' scorer handles them downstream, so reinstating them here is correct, not a hack.
            if _fit_entry_nan_mask and isinstance(X, pd.DataFrame):
                for _mc, _mask in _fit_entry_nan_mask.items():
                    if _mc in X.columns and len(_mask) == len(X):
                        _col_now = X[_mc]
                        if not _col_now.isna().to_numpy().any():
                            _restored = _col_now.to_numpy().astype(np.float64, copy=True)
                            _restored[_mask] = np.nan
                            X[_mc] = _restored

            # W6 follow-up: missingness-indicator family's unified local-MI
            # abs-MAD floor kills (pure-record; selection byte-identical).
            _l37_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _l37_reject_sink(**_kw):
                """Shared reject-sink for the missingness-indicator FE family; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_l37_step, **_kw)

            _engineered_seen_l37 = (
                set(self.hybrid_orth_features_ or [])
                | set(self.mi_greedy_features_ or [])
                | set(getattr(self, "kfold_te_features_", []) or [])
                | set(getattr(self, "count_encoding_features_", []) or [])
                | set(getattr(self, "frequency_encoding_features_", []) or [])
                | set(getattr(self, "cat_num_interaction_features_", []) or [])
            )

            def _resolve_missing_cols(cfg):
                """Resolve the missingness-indicator family's candidate columns: explicit ``cfg`` when given, else auto-detect NaN-rate-in-[1%,99%] columns; always excludes columns already engineered by an earlier FE stage."""
                _cfg = tuple(cfg or ())
                if _cfg:
                    return [c for c in _cfg if c in X.columns and c not in _engineered_seen_l37]
                # Auto-detect candidate cols with NaN rate in [1%, 99%].
                return [c for c in auto_detect_missing_cols(fe_to_pandas(X)) if c not in _engineered_seen_l37]

            # ----- Per-column indicator ------------------------------------
            if bool(getattr(self, "fe_missingness_indicator_enable", False)):
                try:
                    _ind_cols = _resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))
                    _X_before_ind_cols = list(X.columns)
                    _y_for_ind = _y_np
                    # Anchor the indicator's MI noise floor on the RAW input columns, not the engineered-polluted X: an earlier adaptive-Fourier stage appended high-(plug-in)-MI hijacker columns that would otherwise inflate the floor above a genuine MNAR indicator's MI and drop it (a >2%-missing source's signal lives in the NaN pattern the Fourier MI inflates).
                    _raw_floor_X = fe_to_pandas(X)[[c for c in _raw_input_cols_pre_fe if c in X.columns]] if _raw_input_cols_pre_fe else None
                    X_i, _ind_appended, _ind_recipes = missing_indicator_with_recipes(
                        fe_to_pandas(X), cols=_ind_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_ind,
                        raw_X=_raw_floor_X,
                        reject_sink=_l37_reject_sink,
                    )
                    _ind_appended = [c for c in _ind_appended if c not in _X_before_ind_cols]
                    if _ind_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_i, _ind_appended))
                        self.missingness_indicator_features_ = list(_ind_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ind_appended)
                        for _r in _ind_recipes:
                            if _r.name in _ind_appended:
                                _miss_ind_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_indicator: appended %d " "engineered column(s): %s",
                                len(_ind_appended),
                                _ind_appended[:8],
                            )
                except Exception as _ind_exc:
                    logger.warning(
                        "MRMR.fit missingness_indicator FE raised %s: %s; " "continuing without missingness indicator columns.",
                        type(_ind_exc).__name__,
                        _ind_exc,
                    )

            # ----- Per-row missingness count -------------------------------
            if bool(getattr(self, "fe_missingness_count_enable", False)):
                try:
                    _cnt_cols = _resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))
                    _X_before_mc_cols = list(X.columns)
                    X_c, _mc_appended, _mc_recipes = missingness_count_with_recipes(
                        fe_to_pandas(X), cols=_cnt_cols,
                    )
                    _mc_appended = [c for c in _mc_appended if c not in _X_before_mc_cols]
                    if _mc_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_c, _mc_appended))
                        self.missingness_count_features_ = list(_mc_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_mc_appended)
                        for _r in _mc_recipes:
                            if _r.name in _mc_appended:
                                _miss_cnt_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_count: appended %d " "engineered column(s): %s",
                                len(_mc_appended),
                                _mc_appended[:8],
                            )
                except Exception as _mc_exc:
                    logger.warning(
                        "MRMR.fit missingness_count FE raised %s: %s; " "continuing without missingness count column.",
                        type(_mc_exc).__name__,
                        _mc_exc,
                    )

            # ----- Per-row top-K pattern -----------------------------------
            if bool(getattr(self, "fe_missingness_pattern_enable", False)):
                try:
                    _pat_cols = _resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))
                    _top_k = int(getattr(self, "fe_missingness_pattern_top_k", 5))
                    _X_before_pat_cols = list(X.columns)
                    X_p, _pat_appended, _pat_recipes = missingness_pattern_with_recipes(
                        fe_to_pandas(X), cols=_pat_cols, top_k=_top_k,
                    )
                    _pat_appended = [c for c in _pat_appended if c not in _X_before_pat_cols]
                    if _pat_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_p, _pat_appended))
                        self.missingness_pattern_features_ = list(_pat_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_pat_appended)
                        for _r in _pat_recipes:
                            if _r.name in _pat_appended:
                                _miss_pat_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_pattern: appended %d " "engineered column(s): %s",
                                len(_pat_appended),
                                _pat_appended[:8],
                            )
                except Exception as _pat_exc:
                    logger.warning(
                        "MRMR.fit missingness_pattern FE raised %s: %s; " "continuing without missingness pattern column.",
                        type(_pat_exc).__name__,
                        _pat_exc,
                    )

    # 2026-05-31 Layer 38 — CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF.
    # Four independent master switches (ratio / log_ratio / grouped_delta /
    # lagged_diff); each appends its engineered columns AND emits one recipe
    # per column. Routing piggybacks on hybrid_orth_features_ (same Layer 23
    # remap pattern used by Layers 33/34/37).
    self.pairwise_ratio_features_ = []
    self.pairwise_log_ratio_features_ = []
    self.grouped_delta_features_ = []
    self.lagged_diff_features_ = []
    self.grouped_agg_features_ = []
    self.composite_group_agg_features_ = []
    self.grouped_quantile_features_ = []
    self.cat_pair_features_ = []
    self.cat_triple_features_ = []
    self.numeric_decompose_features_ = []
    self.temporal_agg_features_ = []
    self.modular_features_ = []
    self.pairwise_modular_features_ = []
    self.integer_lattice_features_ = []
    self.row_argmax_features_ = []
    self.conditional_gate_features_ = []
    # RAW SOURCE OPERANDS of the selected gate_mask / row_argmax features (their recipe src_names).
    # The FE pair step re-classifies these from synergy-bootstrap to REGULARLY-selected operands so
    # the elementary pair over a gate's raw sources competes on the LENIENT prevalence bar instead of
    # being demoted to the stricter synergy bar (a high-MI gate built FROM a raw col evicts that col
    # from selected_vars, so its clean elementary pair would otherwise be suppressed). 2026-06-13.
    self._gate_raw_operands_ = set()
    # Per-gate-column -> set of its RAW source variables (recipe ``src_names``). The FE step uses this to
    # resolve the raw-variable coverage of a gate-operand COMPOSITE (whose gate operand buries its raw
    # vars inside the column name) so it can drop a composite whose entire raw coverage is already provided
    # by clean non-gate engineered survivors (CASE1) while keeping one that adds genuinely new (c,d)
    # coverage no clean form expresses (CASE2). Empty when no gate fired. 2026-06-13.
    self._gate_col_src_vars_ = {}
    self.group_distance_features_ = []
    if (
        _fe_family_on("fe_pairwise_ratio_enable", False)
        or _fe_family_on("fe_pairwise_log_ratio_enable", False)
        or _fe_family_on("fe_grouped_delta_enable", False)
        or _fe_family_on("fe_lagged_diff_enable", False)
    ):
        # grouped_delta / lagged_diff are cross-row (group / time ordered) and ratio / log-ratio rank their mi_gate on the
        # full frame, none wired for closed-form subsample-replay - so this block needs the full frame: gate the materialisation
        # on size and skip a > ~2 GiB polars frame (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: Layer 38 FE (ratio/log-ratio/grouped-delta/lagged-diff) enabled but X is a large polars frame "
                "(> ~2 GiB); these families need a full-frame decision and are skipped to avoid a whole-frame to_pandas copy.",
                UserWarning, stacklevel=3,
            )
        else:
            from .._ratio_delta_fe import (
                pairwise_ratio_with_recipes,
                pairwise_log_ratio_with_recipes,
                grouped_delta_with_recipes,
                lagged_diff_with_recipes,
            )

            _l38_mi_gate = bool(getattr(self, "fe_local_mi_gate", False))
            _l38_mi_gate_top_k = int(getattr(self, "fe_local_mi_gate_top_k", 20))
            _y_for_l38 = _y_np
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

            # W6 follow-up: shared sink for the ratio/log-ratio/grouped-delta/
            # lagged-diff family's unified local-MI abs-MAD floor kills
            # (pure-record; selection byte-identical).
            _l38_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _l38_reject_sink(**_kw):
                """Shared reject-sink for the ratio/log-ratio/grouped-delta/lagged-diff FE family; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_l38_step, **_kw)

            # ----- Pairwise ratio --------------------------------------------
            if _fe_family_on("fe_pairwise_ratio_enable", False):
                try:
                    _ratio_cols = tuple(getattr(self, "fe_pairwise_ratio_cols", ()) or ())
                    _ratio_cols = tuple(c for c in _ratio_cols if c in X.columns)
                    _eps = float(getattr(self, "fe_pairwise_ratio_eps", 1e-9))
                    _X_before_r_cols = list(X.columns)
                    X_r, _r_appended, _r_recipes = pairwise_ratio_with_recipes(
                        fe_to_pandas(X), cols=_ratio_cols, eps=_eps,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _r_appended = [c for c in _r_appended if c not in _X_before_r_cols]
                    if _r_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_r, _r_appended))
                        self.pairwise_ratio_features_ = list(_r_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_r_appended)
                        for _r in _r_recipes:
                            if _r.name in _r_appended:
                                _ratio_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit pairwise_ratio: appended %d " "engineered column(s): %s",
                                len(_r_appended),
                                _r_appended[:8],
                            )
                except Exception as _r_exc:
                    logger.warning(
                        "MRMR.fit pairwise_ratio FE raised %s: %s; " "continuing without ratio columns.",
                        type(_r_exc).__name__,
                        _r_exc,
                    )

            # ----- Pairwise log-ratio ----------------------------------------
            if _fe_family_on("fe_pairwise_log_ratio_enable", False):
                try:
                    _lr_cols = tuple(getattr(self, "fe_pairwise_log_ratio_cols", ()) or ())
                    _lr_cols = tuple(c for c in _lr_cols if c in X.columns)
                    _eps_lr = float(getattr(self, "fe_pairwise_ratio_eps", 1e-9))
                    _X_before_lr_cols = list(X.columns)
                    X_lr, _lr_appended, _lr_recipes = pairwise_log_ratio_with_recipes(
                        fe_to_pandas(X), cols=_lr_cols, eps=_eps_lr,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _lr_appended = [c for c in _lr_appended if c not in _X_before_lr_cols]
                    if _lr_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_lr, _lr_appended))
                        self.pairwise_log_ratio_features_ = list(_lr_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_lr_appended)
                        for _r in _lr_recipes:
                            if _r.name in _lr_appended:
                                _log_ratio_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit pairwise_log_ratio: appended %d " "engineered column(s): %s",
                                len(_lr_appended),
                                _lr_appended[:8],
                            )
                except Exception as _lr_exc:
                    logger.warning(
                        "MRMR.fit pairwise_log_ratio FE raised %s: %s; " "continuing without log-ratio columns.",
                        type(_lr_exc).__name__,
                        _lr_exc,
                    )

            # ----- Grouped delta ---------------------------------------------
            if _fe_family_on("fe_grouped_delta_enable", False):
                try:
                    _gd_group = getattr(self, "fe_grouped_delta_group_col", None)
                    _gd_nums = tuple(getattr(self, "fe_grouped_delta_num_cols", ()) or ())
                    _gd_nums = tuple(c for c in _gd_nums if c in X.columns)
                    _X_before_gd_cols = list(X.columns)
                    X_gd, _gd_appended, _gd_recipes = grouped_delta_with_recipes(
                        fe_to_pandas(X), group_col=_gd_group, num_cols=_gd_nums,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _gd_appended = [c for c in _gd_appended if c not in _X_before_gd_cols]
                    if _gd_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_gd, _gd_appended))
                        self.grouped_delta_features_ = list(_gd_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_gd_appended)
                        for _r in _gd_recipes:
                            if _r.name in _gd_appended:
                                _grouped_delta_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit grouped_delta: appended %d " "engineered column(s): %s",
                                len(_gd_appended),
                                _gd_appended[:8],
                            )
                except Exception as _gd_exc:
                    logger.warning(
                        "MRMR.fit grouped_delta FE raised %s: %s; " "continuing without grouped-delta columns.",
                        type(_gd_exc).__name__,
                        _gd_exc,
                    )

            # ----- Lagged diff -----------------------------------------------
            if _fe_family_on("fe_lagged_diff_enable", False):
                try:
                    _ld_time = getattr(self, "fe_lagged_diff_time_col", None)
                    _ld_vals = tuple(getattr(self, "fe_lagged_diff_value_cols", ()) or ())
                    _ld_vals = tuple(c for c in _ld_vals if c in X.columns)
                    _ld_periods = tuple(getattr(self, "fe_lagged_diff_periods", (1, 2)) or (1, 2))
                    _X_before_ld_cols = list(X.columns)
                    X_ld, _ld_appended, _ld_recipes = lagged_diff_with_recipes(
                        fe_to_pandas(X), time_col=_ld_time, value_cols=_ld_vals,
                        periods=_ld_periods,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _ld_appended = [c for c in _ld_appended if c not in _X_before_ld_cols]
                    if _ld_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_ld, _ld_appended))
                        self.lagged_diff_features_ = list(_ld_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ld_appended)
                        for _r in _ld_recipes:
                            if _r.name in _ld_appended:
                                _lagged_diff_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit lagged_diff: appended %d " "engineered column(s): %s",
                                len(_ld_appended),
                                _ld_appended[:8],
                            )
                except Exception as _ld_exc:
                    logger.warning(
                        "MRMR.fit lagged_diff FE raised %s: %s; " "continuing without lagged-diff columns.",
                        type(_ld_exc).__name__,
                        _ld_exc,
                    )

    return X
