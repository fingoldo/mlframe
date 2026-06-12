"""Per-target helpers carved out of ``mlframe.training.core._phase_train_one_target``.

Holds ``_build_feature_selection_report`` and ``_maybe_run_feature_handling_apply`` --
both module-level helpers used exclusively by ``_train_one_target``. Lifting
them here keeps the parent below the 1k-line monolith threshold.

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _build_feature_selection_report`` /
``_maybe_run_feature_handling_apply`` resolves transparently.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _build_feature_selection_report(
    pre_pipeline,
    pre_pipeline_name: str | None,
    fitted_columns_in: list[str] | None,
    kept_columns: list[str] | None,
) -> dict:
    """Build the per-model FS report stamped onto ``metadata["model_schemas"][file_name]``.

    Layout (per task spec):
        selector_name : "MRMR" | "RFECV" | "BorutaShap" | None
        selector_params_hash : 8-byte blake2b hex of selector.get_params(), or None
        kept_features : list[str] -- post-FS surviving feature names
        dropped_features : list[str] -- pre-FS names NOT in kept_features
        scores : {feature: float} -- per-feature score from the selector's natural attribute
                                     (None when the selector exposes no per-feature score)
        reason_per_feature : {feature: str} -- per-feature decision label
                                                (None when the selector lacks a reason)

    Reads selector-specific attributes:
      * MRMR: ``support_`` (integer index list into ``feature_names_in_``); no per-feature score is
        exposed, so ``scores=None``. Reason: "kept" / "dropped".
      * RFECV: ``feature_importances_`` is a ``{"<nfeatures>_<fold>": {feature: score}}`` dict; a feature's
        score is the mean of its per-fold importance across every run it appears in. ``ranking_`` (when
        present) gives the eliminated-order; we surface it as a per-feature reason
        ("kept@rank=N" / "dropped@rank=N").
      * BorutaShap: ``history_x`` is a DataFrame of per-iteration shap importances (one row per
        iteration, one column per feature); mean across rows is the canonical "average importance"
        score. Reason: "accepted" / "rejected" / "tentative" via ``self.accepted`` / ``self.rejected``
        / ``self.tentative`` (set when ``calculate_rejected_accepted_tentative`` ran).

    Falls back to a minimal report (selector_name + kept/dropped only) if any attribute access fails;
    a failed FS report must never abort the training run.
    """
    # Lazy import: ``.._phase_train_one_target`` re-imports this module at its
    # bottom for re-export -> any top-level ``from ._phase_train_one_target
    # import ...`` here would create a hard import cycle.
    from ._phase_train_one_target import (
        _unwrap_selector,
        _selector_kind,
        _selector_params_hash,
    )
    selector = _unwrap_selector(pre_pipeline)
    _kind = _selector_kind(selector)
    _report: dict = {
        "selector_name": _kind,
        "selector_params_hash": _selector_params_hash(selector),
        "kept_features": list(kept_columns) if kept_columns is not None else None,
        "dropped_features": None,
        "scores": None,
        "reason_per_feature": None,
    }
    # Compute dropped = feature_names_in_ \ kept; selectors stamp ``feature_names_in_`` post-fit.
    _all_in = None
    if selector is not None:
        try:
            _all_in = getattr(selector, "feature_names_in_", None)
        except Exception:
            _all_in = None
    if _all_in is None:
        _all_in = fitted_columns_in
    if _all_in is not None and kept_columns is not None:
        try:
            _kept_set = set(kept_columns)
            _report["dropped_features"] = [c for c in _all_in if c not in _kept_set]
        except Exception:
            _report["dropped_features"] = None

    if _kind == "MRMR":
        # MRMR exposes ``support_`` as integer indices into ``feature_names_in_``; no per-feature score.
        _report["scores"] = None
        if _all_in is not None and kept_columns is not None:
            try:
                _kept_set = set(kept_columns)
                _report["reason_per_feature"] = {
                    str(c): ("kept" if c in _kept_set else "dropped") for c in _all_in
                }
            except Exception:
                pass
    elif _kind == "RFECV":
        # ``feature_importances_`` is a dict keyed by "<nfeatures>_<fold>". Each value is a per-fold
        # ``{feature_name: importance}`` map carrying the importance of every feature present in that
        # fold's candidate subset (the real fitted-RFECV surface). A feature's score is the mean of its
        # importance across every fold run it appears in -- early-eliminated features only appear in the
        # wider-subset runs. Legacy callers/stubs supply ndarray-valued rows aligned to
        # ``feature_names_in_``; both shapes are aggregated here. ``ranking_`` (when present) gives the
        # per-feature elimination order surfaced as the reason below.
        try:
            _fi_dict = getattr(selector, "feature_importances_", None)
            if isinstance(_fi_dict, dict) and _fi_dict:
                _acc: dict = {}
                for _row in _fi_dict.values():
                    if isinstance(_row, dict):
                        _items = _row.items()
                    elif _all_in is not None:
                        _vals = np.asarray(_row, dtype=np.float64).ravel()
                        if _vals.shape[0] != len(_all_in):
                            continue
                        _items = zip(_all_in, _vals)
                    else:
                        continue
                    for _feat, _val in _items:
                        try:
                            _fval = float(_val)
                        except (TypeError, ValueError):
                            continue
                        if np.isnan(_fval):
                            continue
                        _acc.setdefault(str(_feat), []).append(_fval)
                _scores = {_f: float(np.mean(_vals)) for _f, _vals in _acc.items() if _vals}
                if _scores:
                    _report["scores"] = _scores
        except Exception:
            _report["scores"] = None
        # Ranking-based reason
        try:
            _ranking = getattr(selector, "ranking_", None)
            if _ranking is not None and _all_in is not None and kept_columns is not None:
                _kept_set = set(kept_columns)
                _rank_arr = list(_ranking)
                if len(_rank_arr) == len(_all_in):
                    _report["reason_per_feature"] = {
                        str(c): (
                            f"kept@rank={_rank_arr[i]}" if c in _kept_set
                            else f"dropped@rank={_rank_arr[i]}"
                        )
                        for i, c in enumerate(_all_in)
                    }
        except Exception:
            pass
    elif _kind == "BorutaShap":
        # ``history_x`` columns map 1:1 to ``all_columns``; the per-feature score is the mean
        # historical SHAP importance across all Boruta iterations. ``accepted`` / ``rejected`` /
        # ``tentative`` carry the final per-feature verdicts.
        try:
            _history = getattr(selector, "history_x", None)
            if _history is not None and hasattr(_history, "mean"):
                _means = _history.mean(axis=0)
                if hasattr(_means, "to_dict"):
                    _report["scores"] = {str(k): float(v) for k, v in _means.to_dict().items()}
        except Exception:
            _report["scores"] = None
        try:
            _accepted = set(getattr(selector, "accepted", None) or [])
            _rejected = set(getattr(selector, "rejected", None) or [])
            _tentative = set(getattr(selector, "tentative", None) or [])
            _all_cols_attr = getattr(selector, "all_columns", None)
            _iter = list(_all_cols_attr) if _all_cols_attr is not None else (_all_in or [])
            if _iter:
                _reasons = {}
                for c in _iter:
                    _cs = str(c)
                    if c in _accepted:
                        _reasons[_cs] = "accepted"
                    elif c in _rejected:
                        _reasons[_cs] = "rejected"
                    elif c in _tentative:
                        _reasons[_cs] = "tentative"
                    else:
                        _reasons[_cs] = "unknown"
                _report["reason_per_feature"] = _reasons
        except Exception:
            pass

    # Friend-graph post-analysis summary (MRMR only; absent on other selectors). Compact,
    # JSON-serializable: per-class counts, suspected-sink / pruned feature names, and per-node
    # entropy / relevance / redundancy stats. A failed read must never abort the run.
    try:
        _fg = getattr(selector, "friend_graph_", None)
        if _fg is not None and hasattr(_fg, "to_meta"):
            _report["friend_graph"] = _fg.to_meta()
    except Exception:
        pass

    # Clustered-feature aggregation summary (MRMR only). Lists each denoised aggregate built from a
    # correlated-reflection cluster: its name, chosen combiner method, member features, and the
    # MI(aggregate;y) vs best-member-MI gain. Already JSON-serializable. Read must never abort the run.
    try:
        _ca = getattr(selector, "cluster_aggregate_", None)
        if _ca:
            _report["cluster_aggregate"] = _ca
    except Exception:
        pass

    return _report


def _maybe_run_feature_handling_apply(
    ctx,
    *,
    cur_target_name: str,
    train_df,
    val_df,
    test_df,
    current_train_target,
    sample_weight=None,
):
    """Run feature_handling_apply once per target when ctx carries a FeatureHandlingConfig; else no-op.

    Returns the FeatureHandlingResult on success or None when disabled / failed. Fitted state is also
    stashed under ctx.artifacts["feature_handling_fitted"][cur_target_name] so a future predict-side
    wave can replay handlers without re-fitting. ctx.artifacts is the only ctx slot we may write to
    here -- TrainingContext uses slots=True so adding a new attribute would AttributeError, and the
    SCOPE constraint forbids touching _training_context.py in this wave.

    sample_weight is accepted for forward compatibility: feature_handling_apply does not yet take it
    (validated against the current apply.py). The keyword is plumbed through so a later apply.py
    extension picks it up without a second wire-in change here. NOTE: the underlying handlers do
    consume sample_weight via LeakageSafeEncoder -- once apply.py grows the kwarg, drop the silent
    discard below.

    model_kind comes from ctx.sorted_mlframe_models[0] -- the first concrete kind drives FHC
    validation; the resulting fitted state is model-agnostic for the handlers wired in v1 (TF-IDF,
    target-encoder, custom), so one call seeds the FeatureCache for every model that follows.
    """
    fhc = getattr(ctx, "feature_handling_config", None)
    if fhc is None:
        # `_phase_config_setup.setup_configuration` stores the validated config under `ctx.artifacts`
        # because TrainingContext uses slots=True and exposes no dedicated slot. Honour that storage path.
        fhc = ctx.artifacts.get("feature_handling_config") if isinstance(getattr(ctx, "artifacts", None), dict) else None
    if fhc is None:
        return None
    # sample_weight is accepted for forward compatibility but feature_handling_apply does NOT yet
    # consume it. Under target-encoder handlers (LeakageSafeEncoder), the OOF means are computed
    # UNWEIGHTED even when the suite is running recency-weighted training. Loud-warn so the operator
    # sees this BEFORE diagnosing "production AUC degraded vs uniform-trained baseline" -- the
    # silent discard was the cause. Once apply.py grows the sample_weight kwarg + threads it into
    # the handler chain, drop the WARN and pass the value through.
    if sample_weight is not None:
        # Rate-limit log: emit at most once per process per (target, handler-count) to avoid
        # spamming the log on multi-target suites where the warning is the same root cause.
        _key = ("_fhc_sw_discard_warned", cur_target_name)
        if not getattr(ctx, "artifacts", {}).get(_key):
            logger.warning(
                "_maybe_run_feature_handling_apply: sample_weight provided for target %r but "
                "feature_handling_apply does not yet consume it; target-encoder OOF means will "
                "compute UNWEIGHTED. Production AUC may degrade vs uniform-trained baseline. "
                "Track removal of this warning when apply.py threads sample_weight through.",
                cur_target_name,
            )
            if isinstance(getattr(ctx, "artifacts", None), dict):
                ctx.artifacts[_key] = True
    try:
        from mlframe.training.feature_handling import feature_handling_apply  # local: avoid suite-import cost when FHC is off
    except ImportError:  # pragma: no cover
        return None

    sorted_models = getattr(ctx, "sorted_mlframe_models", None) or getattr(ctx, "mlframe_models", None) or []
    if not sorted_models:
        return None
    model_kind = sorted_models[0]

    # Propagate ctx.cat_features as the explicit candidate_cat_columns list. Without this
    # the suite-internal call fell into feature_handling_apply's candidate_cat_columns=None
    # branch which pre-2026-05-20 silently dropped EVERY target_mean / WoE handler the
    # FHC was configured for (the by-dtype auto-detect now kicks in as a fallback, but
    # the suite already knows the cat list via the convention at _phase_helpers.py:920-931
    # and should pass it explicitly so the FHC handler chain operates on exactly the same
    # cat universe the rest of the suite uses). External direct callers of
    # feature_handling_apply still benefit from the by-dtype auto-detect when they don't
    # set their own candidate list.
    _cat_for_fhc = list(ctx.cat_features) if getattr(ctx, "cat_features", None) else None
    try:
        result = feature_handling_apply(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_target=current_train_target,
            fhc=fhc,
            model_kind=model_kind,
            candidate_cat_columns=_cat_for_fhc,
        )
    except ValueError as fhc_err:
        # Surface configuration errors with the kwarg name so users grep the right place; chain so the
        # original validation traceback is preserved.
        raise ValueError(
            f"feature_handling_config rejected for model_kind={model_kind!r} on target "
            f"{cur_target_name!r}: {fhc_err}"
        ) from fhc_err
    except Exception as fhc_err:
        logger.warning(
            "feature_handling_apply failed for target %r (model_kind=%s): %s; continuing without FHC enrichment for this target.",
            cur_target_name, model_kind, fhc_err,
        )
        return None

    # ctx.artifacts is a plain dict on the dataclass, so we can nest a sub-dict here without slots issues.
    fitted_store = ctx.artifacts.setdefault("feature_handling_fitted", {})
    fitted_store[cur_target_name] = result

    # Wave 63 (2026-05-20): replaced "wave-N" placeholder TODO with concrete tracking
    # comment. Phase F (CB embedding_features) and Phase G (TabularInputEncoder)
    # downstream routing are tracked separately; the current call exists to seed
    # the per-suite FeatureCache and exercise the validate_against_models guard so
    # misconfig is caught at fit time. No action required in this code path; the
    # `result.train/val/test` matrices are intentionally fit-and-stash-only here.
    return result
