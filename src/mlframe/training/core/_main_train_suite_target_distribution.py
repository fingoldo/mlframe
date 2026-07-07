"""mini-HPT target + feature distribution analyzer, carved out of
``train_mlframe_models_suite`` (in ``_main_train_suite``).

Inspects the FIRST target of the most-prevalent target_type, logs any
detected pathologies, and merges gap-fill recommendations into
``hyperparams_config``. Both the target-side and feature-side reports
are stamped into ``metadata`` for downstream observability regardless
of whether anything was merged.

Re-imported at the parent's module bottom so historical
``from ._main_train_suite import _run_target_distribution_analyzer``
keeps resolving transparently.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.training.core._main_train_suite")


def _maybe_auto_drop_after_feature_analyzer(
    *,
    fd_report,
    train_df,
    val_df,
    test_df,
    behavior_config,
    metadata: dict,
    verbose: bool,
):
    """Apply the auto-drop knobs (auto_drop_distribution_analyzer_candidates,
    auto_drop_near_duplicate_threshold) from behavior_config to all three split
    frames in a single pass.

    Train-only derivation contract: the drop SET is built from the analyzer's
    report (drop_candidates and redundant_feature_pairs) which both come from
    train_df + train-y only. val/test simply lose the same columns to keep
    schemas aligned (no val/test stats touch the drop decision).

    Returns ``(train_df, val_df, test_df, dropped_cols)``. When nothing is
    dropped (flags disabled, no candidates, or columns already missing) the
    frames are returned unchanged and ``dropped_cols`` is empty.
    """
    if behavior_config is None or fd_report is None:
        return train_df, val_df, test_df, []
    _do_drop_candidates = bool(getattr(behavior_config, "auto_drop_distribution_analyzer_candidates", False))
    _dup_threshold = float(getattr(behavior_config, "auto_drop_near_duplicate_threshold", 2.0))
    drop_set: set = set()
    if _do_drop_candidates:
        drop_set.update(getattr(fd_report, "drop_candidates", []) or [])
    if _dup_threshold <= 1.0:
        # Walk the analyzer's diagnostics.redundant_feature_pairs (list of
        # (a, b, |corr|)) and drop the alphabetically-larger of each pair whose
        # corr clears the threshold. Reproducible pick keeps train/val/test
        # column order stable across runs.
        _pairs = (fd_report.diagnostics or {}).get("redundant_feature_pairs") or []
        # diagnostics["redundant_feature_pairs"] is capped to top-50 by the
        # analyzer; the FULL list is still available via the warning trail.
        # For drop purposes the top-50 captures the strongest dupes which is
        # exactly the worthwhile target -- the long tail is sub-1.0 corr.
        for entry in _pairs:
            try:
                # Each entry is dict {a, b, corr} from the analyzer return.
                _a = str(entry.get("a") if isinstance(entry, dict) else entry[0])
                _b = str(entry.get("b") if isinstance(entry, dict) else entry[1])
                _c = float(entry.get("corr") if isinstance(entry, dict) else entry[2])
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _main_train_suite_target_distribution.py:70: %s", e)
                continue
            if abs(_c) >= _dup_threshold:
                # Drop the alphabetically-larger; if either is already in
                # drop_set, skip so we don't drop both halves of the pair.
                drop_set.add(_b if _a not in drop_set and _b not in drop_set else (_a if _b in drop_set else _b))
    if not drop_set:
        return train_df, val_df, test_df, []
    # Filter the drop set to columns actually present in train_df.
    train_cols = set(getattr(train_df, "columns", []) or [])
    drop_list = sorted(c for c in drop_set if c in train_cols)
    if not drop_list:
        return train_df, val_df, test_df, []

    def _drop(df, cols):
        if df is None:
            return df
        present = [c for c in cols if c in getattr(df, "columns", [])]
        if not present:
            return df
        # polars + pandas both expose .drop(cols, ...); polars wants list, pandas wants list (axis=1).
        try:
            return df.drop(present)
        except TypeError:
            return df.drop(columns=present)
    train_df = _drop(train_df, drop_list)
    val_df = _drop(val_df, drop_list)
    test_df = _drop(test_df, drop_list)
    metadata.setdefault("feature_distribution_report", {})["auto_dropped_columns"] = list(drop_list)
    if verbose:
        _preview = ", ".join(drop_list[:8])
        if len(drop_list) > 8:
            _preview += f", ... (+{len(drop_list) - 8} more)"
        logger.info(
            "[mini-HPT] auto-drop applied: %d column(s) removed from train/val/test (analyzer candidates + |corr|>=%.4f duplicates): %s",
            len(drop_list), _dup_threshold, _preview,
        )
    return train_df, val_df, test_df, drop_list


def _run_target_distribution_analyzer(
    *,
    enable_target_distribution_analyzer: bool,
    target_by_type,
    train_idx,
    group_ids,
    timestamps,
    train_df,
    verbose: bool,
    metadata: dict,
    hyperparams_config: Any,
    ctx,
    val_df=None,
    test_df=None,
    behavior_config=None,
):
    """Run the target-side and feature-side analyzers; merge recommendations.

    Returns the (possibly updated) ``hyperparams_config``. ``metadata`` is
    mutated in place. ``ctx.hyperparams_config`` is also reflected when
    the target-side merge fires so downstream phases see the merged config.

    The whole block is wrapped in a broad except: the analyzer is purely
    observational/recommendation-only, so any failure MUST NOT block
    training. WARN once so the operator can debug.
    """
    if not (enable_target_distribution_analyzer and target_by_type):
        # Must mirror the 4-tuple contract of the main return below
        # (hyperparams_config, train_df, val_df, test_df); the frames pass
        # through unchanged when the analyzer is disabled / has no targets.
        # Returning the bare ``hyperparams_config`` here (stale from before the
        # function grew to also thread the frames) made the caller's
        # ``a, b, c, d = _run_target_distribution_analyzer(...)`` raise
        # "ValueError: too many values to unpack (expected 4)" on EVERY combo
        # with the analyzer off or an empty target_by_type.
        return hyperparams_config, train_df, val_df, test_df

    try:
        from ..targets import analyze_target_distribution
        from ..configs import TargetTypes as _TT

        # Pick a representative target: prefer regression (the typical
        # scenario class), fall back to the first available type. ``target_by_type``
        # maps TargetTypes -> dict[target_name -> array-like]. Skip empty buckets.
        _picked_target = None
        _picked_type_name = None
        _picked_target_name = None
        for _tt_key in (_TT.REGRESSION, _TT.BINARY_CLASSIFICATION, _TT.MULTICLASS_CLASSIFICATION):
            _bag = target_by_type.get(_tt_key)
            if isinstance(_bag, dict) and _bag:
                _picked_target_name, _picked_target = next(iter(_bag.items()))
                _picked_type_name = "regression" if _tt_key == _TT.REGRESSION else "classification"
                break
        if _picked_target is None:
            # Fallback: walk all keys, take the first non-empty target.
            for _tt_key, _bag in target_by_type.items():
                if isinstance(_bag, dict) and _bag:
                    _picked_target_name, _picked_target = next(iter(_bag.items()))
                    _picked_type_name = "regression" if str(_tt_key).endswith("REGRESSION") else "classification"
                    break

        if _picked_target is not None and train_idx is not None:
            _y_arr = np.asarray(_picked_target)
            _y_train = _y_arr[train_idx] if _y_arr.size >= np.max(train_idx) + 1 else _y_arr
            _g_train = None
            if group_ids is not None:
                _g_arr = np.asarray(group_ids).reshape(-1)
                if _g_arr.size >= np.max(train_idx) + 1:
                    _g_train = _g_arr[train_idx]
            # has_time_axis: rows are pre-split in the order the caller built
            # the frame. We can only trust the AR detector when timestamps are
            # supplied (the suite carries them as ``timestamps`` after
            # _phase_load_and_preprocess). Also AUTO-DETECT common time-axis
            # column names when the caller didn't explicitly pass timestamps.
            # Some domains use a depth/position column as the sequence axis;
            # log / event data uses timestamp/date/time. Without auto-detection
            # the AR detector skips entirely on such data unless the caller
            # remembers to set timestamps_column -- easy to forget in practice
            # (the per-group AR fallback now catches it via group_ids,
            # but auto-detected has_time_axis fires the GLOBAL detector first
            # and provides additional diagnostics).
            _TIME_AXIS_HINT_NAMES = ("timestamp", "date", "time", "datetime", "md", "depth")
            _has_time = timestamps is not None
            if not _has_time and train_df is not None:
                try:
                    _cols_lower_map = {str(c).lower(): str(c) for c in getattr(train_df, "columns", [])}
                    _hit_lower = set(_cols_lower_map) & set(_TIME_AXIS_HINT_NAMES)
                    # Monotonicity gate: a column NAMED ``date`` does NOT
                    # automatically imply rows are sorted by it. Auto-flipping
                    # has_time_axis without checking would let the AR detector
                    # fire on randomly-shuffled data and produce spurious low-AR
                    # readings (which is worse than silently skipping the
                    # detector). Sample-check the column: if at least one
                    # matching column is monotonic (non-strictly increasing or
                    # decreasing) on a 1024-row stride, accept the time-axis hint.
                    _verified_hits: list[str] = []
                    for _hint_lower in sorted(_hit_lower):
                        _orig_col = _cols_lower_map[_hint_lower]
                        try:
                            _stride = max(1, len(train_df) // 1024)
                            if hasattr(train_df, "iloc"):
                                _sample = train_df.iloc[::_stride][_orig_col].to_numpy()
                            else:
                                # polars: gather only the strided indices instead of materialising the full column then slicing -- avoids
                                # paying for a multi-GB column to throw away >99% of it on a 1024-row monotonicity probe.
                                _n = len(train_df)
                                _idx = list(range(0, _n, _stride))
                                _sample = train_df.get_column(_orig_col).gather(_idx).to_numpy()
                            _sample = np.asarray(_sample)
                            if _sample.dtype.kind in "Mm":
                                _sample = _sample.astype("int64")
                            elif _sample.dtype.kind in "OU":
                                # Object/string columns -- can't trivially compare; skip
                                # monotonicity check and trust the name.
                                _verified_hits.append(_orig_col)
                                continue
                            if _sample.size > 1:
                                _diffs = np.diff(_sample.astype(np.float64))
                                if np.all(_diffs >= 0) or np.all(_diffs <= 0):
                                    _verified_hits.append(_orig_col)
                        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                            logger.debug("suppressed in _main_train_suite_target_distribution.py:232: %s", e)
                            pass
                    if _verified_hits:
                        _has_time = True
                        if verbose:
                            logger.info(
                                "[mini-HPT] auto-detected monotonic time-axis column(s) %s; " "AR detector will run on global lag-1.",
                                _verified_hits,
                            )
                    elif _hit_lower and verbose:
                        logger.info(
                            "[mini-HPT] candidate time-axis column(s) %s present but NOT "
                            "monotonic -- rows aren't sorted by them; skipping AR detector "
                            "(per-group AR via group_ids still fires if applicable).",
                            sorted(_hit_lower),
                        )
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _main_train_suite_target_distribution.py:249: %s", e)
                    pass
            _td_report = analyze_target_distribution(
                _y_train,
                group_ids=_g_train,
                target_type=_picked_type_name,
                has_time_axis=_has_time,
            )
            if verbose:
                logger.info(
                    "[mini-HPT] target_distribution_analyzer on %s target %r "
                    "(n=%d, time_axis=%s): pathologies=%s, knob_overrides=%s, diagnostics=%s",
                    _picked_type_name, _picked_target_name, _td_report.n_samples,
                    _has_time, _td_report.pathologies or "(none)",
                    _td_report.knob_overrides or "(none)",
                    _td_report.diagnostics,
                )
            metadata["target_distribution_report"] = {
                "target_type": _td_report.target_type,
                "picked_target_name": _picked_target_name,
                "n_samples": _td_report.n_samples,
                "pathologies": list(_td_report.pathologies),
                "knob_overrides": dict(_td_report.knob_overrides),
                "knob_overrides_provenance": dict(getattr(_td_report, "knob_overrides_provenance", {}) or {}),
                "diagnostics": dict(_td_report.diagnostics),
            }
            try:
                from mlframe.training.provenance import record_provenance as _record_provenance
                _record_provenance(
                    metadata,
                    "target_distribution_analyzer",
                    source="train_only",
                    n_rows=int(_td_report.n_samples),
                    extra={"target_type": _td_report.target_type, "n_pathologies": len(_td_report.pathologies)},
                )
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _main_train_suite_target_distribution.py:284: %s", e)
                pass
            # Maintain a per-knob "hyperparams_used" provenance dict so downstream consumers (model factories, audit
            # reports) can distinguish analyzer-injected knobs from caller-supplied defaults. User overrides take
            # precedence: if a slot+knob already lives in caller's hyperparams_config, keep that source as "user".
            try:
                _hpd = metadata.setdefault("hyperparams_used", {})
                _user_hpd = {}
                if hyperparams_config is not None:
                    if hasattr(hyperparams_config, "model_dump"):
                        _user_hpd = hyperparams_config.model_dump()
                    elif isinstance(hyperparams_config, dict):
                        _user_hpd = dict(hyperparams_config)
                for _slot, _knobs in (getattr(_td_report, "knob_overrides_provenance", {}) or {}).items():
                    _slot_store = _hpd.setdefault(_slot, {})
                    _user_slot = _user_hpd.get(_slot) if isinstance(_user_hpd, dict) else None
                    for _knob_name, _stamp in _knobs.items():
                        _has_user = isinstance(_user_slot, dict) and _knob_name.split(".")[0] in _user_slot
                        if _has_user:
                            _slot_store[_knob_name] = {"value": _user_slot[_knob_name.split(".")[0]], "source": "user"}
                        else:
                            _slot_store[_knob_name] = dict(_stamp)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _main_train_suite_target_distribution.py:306: %s", e)
                pass
            # Gap-fill merge into hyperparams_config. The config can be a
            # pydantic ModelHyperparamsConfig (dump+rebuild) or a dict (merge
            # in place via the report helper). For the Pydantic path the merge
            # only touches the per-model kwargs slots referenced by the
            # recommendations (mlp_kwargs / lgb_kwargs / xgb_kwargs /
            # cb_kwargs); other fields are left intact.
            if _td_report.knob_overrides:
                if isinstance(hyperparams_config, dict):
                    hyperparams_config = _td_report.merge_into_config(
                        hyperparams_config, override_existing=False,
                    )
                elif hyperparams_config is not None:
                    _hp_dict = hyperparams_config.model_dump() if hasattr(hyperparams_config, "model_dump") else dict(hyperparams_config.__dict__)
                    _hp_merged = _td_report.merge_into_config(_hp_dict, override_existing=False)
                    # Reapply by updating per-knob slots only -- avoids
                    # accidentally clobbering Pydantic-validated nested fields.
                    for _slot in ("mlp_kwargs", "lgb_kwargs", "xgb_kwargs", "cb_kwargs", "split_config"):
                        if _slot in _td_report.knob_overrides and hasattr(hyperparams_config, _slot):
                            try:
                                setattr(hyperparams_config, _slot, _hp_merged.get(_slot))
                            except Exception:
                                # Pydantic v2 frozen models reject direct setattr;
                                # fall back to model_copy(update={...}).
                                try:
                                    hyperparams_config = hyperparams_config.model_copy(update={_slot: _hp_merged.get(_slot)})
                                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                                    logger.debug("suppressed in _main_train_suite_target_distribution.py:333: %s", e)
                                    pass
                # Reflect any mutation back onto ctx so downstream phases see
                # the merged config instead of the caller's original.
                ctx.hyperparams_config = hyperparams_config

            # FEATURE-SIDE analyzer (mini-HPT v2). The target-side detector recommends model
            # objectives / layernorm flags / class weights; the feature-side detector surfaces
            # low-variance / NaN-heavy / high-cardinality / redundant / suspected-leakage
            # features. Both reports are stamped into metadata so operators see the full
            # diagnostic table on every default run.
            try:
                from ..targets import analyze_feature_distribution
                # Use train_df_polars_pre / train_df / whatever is available at this point;
                # _phase_train_val_test_split has already returned train_df at this scope.
                if train_df is not None:
                    _fd_report = analyze_feature_distribution(
                        train_df,
                        y=_y_train,
                        target_type=_picked_type_name,
                    )
                    if verbose:
                        logger.info(
                            "[mini-HPT] feature_distribution_analyzer (n_samples=%d, n_features=%d): "
                            "pathologies=%s, drop_candidates=%s, leakage_candidates=%s",
                            _fd_report.n_samples, _fd_report.n_features,
                            _fd_report.pathologies or "(none)",
                            _fd_report.drop_candidates or "(none)",
                            _fd_report.leakage_candidates or "(none)",
                        )
                    metadata["feature_distribution_report"] = {
                        "n_samples": _fd_report.n_samples,
                        "n_features": _fd_report.n_features,
                        "pathologies": list(_fd_report.pathologies),
                        "drop_candidates": list(_fd_report.drop_candidates),
                        "leakage_candidates": list(_fd_report.leakage_candidates),
                        "feature_warnings": dict(_fd_report.feature_warnings),
                        "diagnostics": dict(_fd_report.diagnostics),
                        "knob_overrides": dict(_fd_report.knob_overrides),
                    }
                    # Auto-drop: when behavior_config opts in (defaults ON for
                    # candidate-list, threshold 0.999 for near-duplicate), strip the
                    # nan-heavy / low-variance / exact-duplicate columns from all
                    # three splits before Phase 3. The drop SET is train-derived so
                    # val/test never enter the decision; they just lose the same
                    # columns to keep schemas aligned. The legacy diagnostic-only
                    # behaviour is preserved by setting both knobs off.
                    try:
                        train_df, val_df, test_df, _dropped = _maybe_auto_drop_after_feature_analyzer(
                            fd_report=_fd_report,
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            behavior_config=behavior_config,
                            metadata=metadata,
                            verbose=verbose,
                        )
                        # Mirror drops onto ctx so any later phase reading from ctx
                        # (the in-progress ctx-form migration in main_train_suite) sees
                        # the trimmed frames too.
                        if _dropped and ctx is not None:
                            for _slot in ("train_df", "val_df", "test_df"):
                                if hasattr(ctx, _slot):
                                    try:
                                        setattr(ctx, _slot, locals()[_slot])
                                    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                                        logger.debug("suppressed in _main_train_suite_target_distribution.py:398: %s", e)
                                        pass
                    except Exception as _drop_err:
                        logger.warning(
                            "[mini-HPT] auto-drop after feature_distribution_analyzer failed (%s); " "training continues with full column set.",
                            _drop_err,
                        )
            except Exception as _fd_err:
                logger.warning(
                    "[mini-HPT] feature_distribution_analyzer crashed (%s); proceeding without feature warnings.",
                    _fd_err, exc_info=False,
                )
    except Exception as _td_err:
        # The analyzer is observational + recommendation-only; any failure
        # MUST NOT block training. WARN once so the operator can debug.
        logger.warning(
            "[mini-HPT] target_distribution_analyzer crashed (%s); proceeding without recommendations.",
            _td_err, exc_info=False,
        )

    return hyperparams_config, train_df, val_df, test_df
