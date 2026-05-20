"""Recurrent model training (LSTM, GRU, RNN, Transformer). Neural max_time defaults to P95 of prior non-neural model train times.

Recurrent models are trained AFTER the per-target booster loop; they are integrated into the ensemble via
``_rerun_ensemble_with_recurrent`` immediately after their fit, before the suite returns. Prior to the integration
helper added in this module, recurrent models silently never joined the blend: ``score_ensemble`` had already
run for the target by the time ``train_recurrent_models`` appended them to ``ctx.models[type][target]``.

TODO: ``core/predict.py`` (currently locked) does not re-run the recurrent-augmented ensemble at predict time; the
predict-side replay needs a follow-up wave. Today the recurrent member predictions are baked into the persisted
ensemble result via ``ctx.ensembles[type][target]`` at train time, so static metric reads on the persisted suite
already see the blend, but a live predict path that rebuilds the ensemble in-memory will still miss the recurrent
members until predict.py is updated.
"""
from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pyutilz.system import tqdmu_lazy_start
from sklearn.base import clone

from ..configs import TargetTypes
from ..utils import log_phase, log_ram_usage
from ..trainer import _configure_recurrent_params
from ._misc_helpers import _compute_neural_max_time

logger = logging.getLogger(__name__)


def _coerce_to_numpy(arr):
    """Tolerate Series / polars Series / numpy / list / None at array boundaries."""
    if arr is None:
        return None
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    if hasattr(arr, "values"):
        return arr.values
    return np.asarray(arr)


def _coerce_features_to_float32(
    features,
    *,
    cache: Optional[Dict[Any, np.ndarray]] = None,
    cache_key: Any = None,
) -> Optional[np.ndarray]:
    """Convert a DataFrame / Series / ndarray to a contiguous float32 ndarray.

    POLARS-PANDAS-CHURN fix: the recurrent path predicts on the same train/val/test frames three
    times per member (train_preds / val_preds / test_preds), each time paying a polars->numpy +
    np.asarray(float32) round-trip. When ``cache`` is supplied, the result is memoised by
    ``cache_key`` (typically a (split_label, id(frame)) tuple) so the second and third hits return
    the cached buffer. Already-float32 ndarrays are returned unchanged without the asarray copy.
    """
    if features is None:
        return None
    if cache is not None and cache_key is not None:
        _hit = cache.get(cache_key)
        if _hit is not None:
            return _hit
    if isinstance(features, np.ndarray):
        if features.dtype == np.float32 and features.flags["C_CONTIGUOUS"]:
            arr = features
        else:
            arr = np.ascontiguousarray(features, dtype=np.float32)
    else:
        try:
            raw = features.to_numpy() if hasattr(features, "to_numpy") else features
            if isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.flags["C_CONTIGUOUS"]:
                arr = raw
            else:
                arr = np.ascontiguousarray(raw, dtype=np.float32)
        except Exception as exc:
            logger.warning("Recurrent predict: failed to coerce features to ndarray (%s); dropping member.", exc)
            return None
    if cache is not None and cache_key is not None:
        cache[cache_key] = arr
    return arr


def _safe_predict_recurrent(
    *,
    model,
    sequences,
    features,
    is_classification: bool,
    ctx=None,
    split: str = "?",
) -> np.ndarray | None:
    """Predict on val/test for a fitted recurrent model. Returns None on any failure so the caller can degrade.

    Features arrive as pandas/polars DataFrames from the suite splits; the wrapper's predict path calls
    ``_compute_cache_key(features, ...)`` which reads ``features.dtype`` BEFORE ``_create_dataset`` would
    normalise to ndarray, so a DataFrame at the predict boundary blows up with ``'DataFrame' has no attribute
    'dtype'``. Cast to a float32 ndarray here (the same shape contract the wrapper documents).

    POLARS-PANDAS-CHURN: when ``ctx`` is supplied, cache the coerced float32 array on
    ``ctx._recurrent_numpy_cache`` keyed by ``(split, id(features))``. The cache is invalidated by
    ``_release_ctx_polars_frames`` (id-recycle safe via the strong-ref window).
    """
    if sequences is None and features is None:
        return None
    cache = getattr(ctx, "_recurrent_numpy_cache", None) if ctx is not None else None
    cache_key = (split, id(features)) if features is not None else None
    features = _coerce_features_to_float32(features, cache=cache, cache_key=cache_key)
    if features is None and sequences is None:
        return None
    try:
        if is_classification and hasattr(model, "predict_proba"):
            preds = model.predict_proba(features=features, sequences=sequences)
        else:
            preds = model.predict(features=features, sequences=sequences)
        preds = np.asarray(preds)
        if preds.size == 0 or not np.all(np.isfinite(preds)):
            return None
        return preds
    except Exception as exc:
        logger.warning("Recurrent predict failed (%s); recurrent member dropped for this split.", exc)
        return None


def _build_recurrent_member_entry(
    *,
    recurrent_model_name: str,
    model,
    train_preds,
    val_preds,
    test_preds,
    is_classification: bool,
) -> SimpleNamespace:
    """Wrap a fitted recurrent model into the same member-entry shape that ``score_ensemble`` reads (val_preds,
    test_preds, train_preds, val_probs, test_probs, train_probs, model, model_name, columns)."""
    if is_classification:
        # For classifiers the recurrent emits class probs; map them onto both *_probs (canonical) and *_preds
        # so the ensemble's split-detection (which scans *_preds first, then *_probs) sees the recurrent member.
        val_probs, test_probs, train_probs = val_preds, test_preds, train_preds
        val_pred_array = None if val_preds is None else (val_preds[:, 1] if val_preds.ndim == 2 and val_preds.shape[1] == 2 else val_preds)
        test_pred_array = None if test_preds is None else (test_preds[:, 1] if test_preds.ndim == 2 and test_preds.shape[1] == 2 else test_preds)
        train_pred_array = None if train_preds is None else (train_preds[:, 1] if train_preds.ndim == 2 and train_preds.shape[1] == 2 else train_preds)
    else:
        val_probs = test_probs = train_probs = None
        val_pred_array, test_pred_array, train_pred_array = val_preds, test_preds, train_preds
    return SimpleNamespace(
        val_preds=val_pred_array,
        test_preds=test_pred_array,
        train_preds=train_pred_array,
        val_probs=val_probs,
        test_probs=test_probs,
        train_probs=train_probs,
        model=model,
        model_name=recurrent_model_name,
        columns=None,
    )


def _validate_member_shape_uniformity(members: list, *, target_name: str) -> bool:
    """Defensive check: every member's val_preds (where present) must share shape[0]; same for test_preds.
    The booster-ensemble path already enforces this implicitly via the splits; this guard surfaces a shape
    drift coming from the recurrent side BEFORE handing the augmented list to ``score_ensemble``."""
    for attr in ("val_preds", "test_preds", "train_preds"):
        sizes = []
        for m in members:
            arr = getattr(m, attr, None)
            if arr is None:
                continue
            sizes.append(int(np.asarray(arr).shape[0]))
        if len(set(sizes)) > 1:
            logger.warning(
                "Recurrent ensemble integration: split %s has non-uniform member row counts for target %s (sizes=%s); "
                "skipping rerun to avoid a broken blend.",
                attr, target_name, sizes,
            )
            return False
    return True


def _apply_recurrent_to_ensemble(
    *,
    ctx,
    ensemble_dict: dict,
    target_type,
    target_name: str,
    target_values,
) -> dict:
    """Idempotent helper that re-runs the score_ensemble step with recurrent members included.

    SKEW-RECURRENT: both train (``_rerun_ensemble_with_recurrent``) and predict (``core/predict.py``)
    need to replay the same recurrent-augmented blend. Extracted here so the predict side can call
    it without duplicating the slicing / kwarg-filtering logic. The function is idempotent: calling
    it twice on the same ``ensemble_dict`` returns the same result modulo numerical noise from the
    second score_ensemble run (which sees the same member list and the same targets).

    Args:
        ctx: TrainingContext (used for train_idx / val_idx / test_idx and verbose).
        ensemble_dict: prior ensemble payload (``{method_name: ens_result}``). Returned unchanged
            if rebuild fails or recurrent members are absent.
        target_type: per-target target_type enum.
        target_name: per-target name string.
        target_values: aligned target array for index slicing.

    Returns:
        Either the rebuilt ``{method_name: ens_result}`` dict or ``ensemble_dict`` unchanged on
        failure / no-op.
    """
    members = ctx.models.get(target_type, {}).get(target_name) or []
    if len(members) < 2:
        return ensemble_dict

    if not _validate_member_shape_uniformity(members, target_name=target_name):
        return ensemble_dict

    from mlframe.models.ensembling import score_ensemble

    try:
        # pandas Series ``[positional_idx]`` is LABEL-indexed, so for a non-default Series index this returns
        # the wrong rows. Route Series through .iloc for positional semantics; ndarray / pl.Series keep
        # __getitem__ semantics which ARE positional. Mirrors the guard in _phase_train_one_target:895-899.
        def _slice_positional(values, idx):
            if values is None or idx is None:
                return None
            if isinstance(values, pd.Series):
                return values.iloc[idx]
            if hasattr(values, "__getitem__"):
                return values[idx]
            return None
        train_target = _coerce_to_numpy(_slice_positional(target_values, ctx.train_idx))
        val_target = _coerce_to_numpy(_slice_positional(target_values, ctx.val_idx))
        test_target = _coerce_to_numpy(_slice_positional(target_values, ctx.test_idx))
    except Exception as exc:
        logger.warning("apply_recurrent_to_ensemble: failed to slice targets for %s (%s); returning prior ensemble.", target_name, exc)
        return ensemble_dict

    # C-P1-10: recurrent members may be trained on sequence-aligned slices whose row count differs from
    # the dense (train|val|test)_idx the booster ensemble uses. If any member's *_preds row count fails
    # to match the sliced target row count, the score_ensemble blend mixes predictions for different rows
    # than the booster -- silent contamination. Refuse to rerun in that case rather than emitting a bad
    # blend; the original ensemble_dict is returned and the suite reports the booster-only result.
    for _split_name, _sliced, _attr in (
        ("train", train_target, "train_preds"),
        ("val", val_target, "val_preds"),
        ("test", test_target, "test_preds"),
    ):
        if _sliced is None:
            continue
        _expected = int(np.asarray(_sliced).shape[0])
        for _m in members:
            _arr = getattr(_m, _attr, None)
            if _arr is None:
                continue
            _got = int(np.asarray(_arr).shape[0])
            if _got != _expected:
                logger.warning(
                    "apply_recurrent_to_ensemble: target=%s split=%s row-count drift: member %r has "
                    "%d rows but sliced target has %d. Refusing to blend (recurrent member rows likely "
                    "belong to a different row set than the booster). Returning prior ensemble.",
                    target_name, _split_name, getattr(_m, "model_name", _m), _got, _expected,
                )
                return ensemble_dict

    sig = inspect.signature(score_ensemble)
    accepted = set(sig.parameters.keys())
    # Per-target sample_weight pulled from ctx.sample_weights (suite-supplied via FTE.get_sample_weights).
    # Without this, the recurrent-rerun ensemble's gate / NNLS / RRF stages compute UNWEIGHTED
    # even when the suite was running recency- or fairness-weighted training -> the rebuilt
    # ensemble (which OVERWRITES the prior ensemble at L331) is strictly weaker than the
    # pre-recurrent build on weighted suites.
    _ctx_sw_dict = getattr(ctx, "sample_weights", None) or {}
    _sw_for_target = (
        _ctx_sw_dict.get(target_name)
        if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict
        else None
    )
    kwargs = {
        "models_and_predictions": list(members),
        "ensemble_name": f"{ctx.model_name or 'mdl'}__{target_name}__recurrent_rerun",
        "train_target": train_target,
        "val_target": val_target,
        "test_target": test_target,
        "train_idx": ctx.train_idx,
        "val_idx": ctx.val_idx,
        "test_idx": ctx.test_idx,
        "verbose": bool(ctx.verbose),
        # ctx-derived kwargs the score_ensemble signature accepts when available.
        "group_ids": getattr(ctx, "group_ids", None),
        "sample_weight": _sw_for_target,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in accepted or k == "models_and_predictions"}

    try:
        rebuilt = score_ensemble(**kwargs)
    except Exception as exc:
        logger.warning(
            "apply_recurrent_to_ensemble: rerun failed for target %s (%s); returning prior ensemble.",
            target_name, exc,
        )
        return ensemble_dict
    # ``rebuilt is None`` => score_ensemble decided not to return an ensemble
    # (caller may want the prior); ``rebuilt == {}`` => rebuild succeeded but
    # the gate pruned every member, which is operationally distinct from "no
    # rebuild". The previous ``rebuilt or ensemble_dict`` form conflated the
    # two and silently swapped {} for the pre-recurrent ensemble, masking the
    # signal that recurrent member dragged the gate too tight.
    if rebuilt is None:
        return ensemble_dict
    if not rebuilt:
        logger.warning(
            "apply_recurrent_to_ensemble: rebuilt ensemble is empty for target %s "
            "(all members gated out); returning empty dict, not prior ensemble.",
            target_name,
        )
    return rebuilt


def _rerun_ensemble_with_recurrent(
    *,
    ctx,
    target_type,
    target_name: str,
    target_values,
) -> bool:
    """Re-run ``score_ensemble`` on the augmented member list for one (target_type, target_name), so the recurrent
    entries that ``train_recurrent_models`` just appended to ``ctx.models[type][target]`` actually participate.

    Returns True when the ensemble was successfully rebuilt; False when there was nothing to rebuild (no prior
    ensemble, or the per-method rerun failed and the original was kept).
    """
    # Single-member "ensemble" is degenerate; the booster path would not have built one either.
    members = ctx.models.get(target_type, {}).get(target_name) or []
    if len(members) < 2:
        return False

    # Note: we proceed even when ``existing_ensembles`` is empty. A common case is exactly one booster + one
    # recurrent member - there was no prior ensemble (score_ensemble needs >=2 members) but post-recurrent we
    # now do, and the rebuild here is the FIRST time the target gets a real blend.
    existing_ensembles = ctx.ensembles.get(target_type, {}).get(target_name) if ctx.ensembles else None

    # Delegate to the shared helper so train-side and (future) predict-side replays agree byte-for-byte.
    rebuilt = _apply_recurrent_to_ensemble(
        ctx=ctx,
        ensemble_dict=existing_ensembles or {},
        target_type=target_type,
        target_name=target_name,
        target_values=target_values,
    )

    if not rebuilt or rebuilt is existing_ensembles:
        return False

    if existing_ensembles:
        logger.info("Recurrent ensemble rerun: rebuilt %d-member ensemble for target %s (previously %d methods).",
                    len(members), target_name, len(existing_ensembles) if isinstance(existing_ensembles, dict) else 0)
    else:
        logger.info("Recurrent ensemble rerun: built FIRST ensemble for target %s with %d members (recurrent + booster).",
                    target_name, len(members))
    ctx.ensembles.setdefault(target_type, {})[target_name] = rebuilt

    # Bookkeeping so downstream consumers (and the integration test) can confirm the rerun fired and which
    # recurrent members joined. Stored under ctx.metadata since ctx is slots=True (no new top-level attrs).
    rec_meta = ctx.metadata.setdefault("recurrent_ensemble_integration", {})
    rec_meta_target = rec_meta.setdefault(str(target_type), {})
    recurrent_member_names = [
        getattr(m, "model_name", None) for m in members
        if getattr(m, "model_name", None) and str(getattr(m, "model_name", "")).lower() in {"lstm", "gru", "rnn", "transformer"}
    ]
    rec_meta_target[target_name] = {
        "recurrent_members": recurrent_member_names,
        "total_members": len(members),
        "rebuilt_methods": sorted(rebuilt.keys()) if isinstance(rebuilt, dict) else [],
    }
    return True


def train_recurrent_models(
    *,
    models: dict,
    recurrent_models: list[str] | None,
    recurrent_config,
    train_sequences,
    val_sequences,
    test_sequences,
    train_df,
    train_df_pd,
    val_df_pd,
    target_by_type: dict,
    train_idx,
    val_idx,
    test_idx,
    _non_neural_train_times: list[float],
    model_name: str,
    verbose: bool,
    ctx=None,
    test_df_pd=None,
) -> dict:
    """Train recurrent models across all target types and targets.

    When ``ctx`` is supplied (current wiring), each fitted recurrent model gets val/test/train preds
    computed and is wrapped into a member-entry compatible with ``score_ensemble``; then
    ``_rerun_ensemble_with_recurrent`` rebuilds the per-target ensemble so the recurrent members actually
    participate. Without ctx the legacy raw-model-append behaviour is preserved for back-compat with any
    direct caller outside the suite orchestrator.
    """
    if not recurrent_models:
        return models
    if train_sequences is None and train_df is None:
        return models

    if verbose:
        log_phase("PHASE 5: Recurrent Model Training")

    use_regression = TargetTypes.REGRESSION in target_by_type

    recurrent_params = _configure_recurrent_params(
        recurrent_models=recurrent_models,
        recurrent_config=recurrent_config,
        sequences_train=train_sequences,
        features_train=train_df_pd if train_df_pd is not None else train_df,
        use_regression=use_regression,
    )

    # Track which (target_type, target_name) pairs gained at least one recurrent member, so the helper only
    # reruns the ensemble once per target (not once per recurrent model). Avoids paying score_ensemble's cost
    # N times when N recurrent variants train successfully.
    targets_with_recurrent: dict = {}

    for recurrent_model_name in tqdmu_lazy_start(recurrent_models, desc="recurrent model"):
        model_name_lower = recurrent_model_name.lower()
        if model_name_lower not in recurrent_params:
            logger.warning("Recurrent model %s not configured, skipping...", recurrent_model_name)
            continue

        recurrent_model = recurrent_params[model_name_lower]["model"]

        for target_type, targets in target_by_type.items():
            # ``TargetTypes.is_classification`` is a @property (bool), not a method - read it as an attribute.
            _is_clf_attr = getattr(target_type, "is_classification", None)
            is_classification = bool(_is_clf_attr) if _is_clf_attr is not None else (str(target_type).lower() != "regression")
            for cur_target_name, target_values in targets.items():
                if verbose:
                    logger.info("Training %s for target %s...", recurrent_model_name, cur_target_name)

                # ``hasattr(pd.Series, "__getitem__")`` is True but Series[positional] is LABEL-indexed --
                # mirror the isinstance(...) guard used in _phase_train_one_target:895-899 so non-default
                # indices don't silently produce label-indexed (wrong-row) slices.
                def _pos_slice(values, idx):
                    if values is None or idx is None:
                        return None
                    if isinstance(values, pd.Series):
                        return values.iloc[idx]
                    if hasattr(values, "__getitem__"):
                        return values[idx]
                    return None
                train_target = _coerce_to_numpy(_pos_slice(target_values, train_idx))
                val_target = _coerce_to_numpy(_pos_slice(target_values, val_idx))
                test_target = _coerce_to_numpy(_pos_slice(target_values, test_idx))

                model_clone = clone(recurrent_model)

                _timeout = _compute_neural_max_time(_non_neural_train_times)
                if _timeout is not None:
                    _max_time_dict, _p95_r, _n = _timeout
                    _r_inner = getattr(model_clone, "regressor", model_clone)
                    if hasattr(_r_inner, "trainer_params"):
                        _r_inner.trainer_params["max_time"] = _max_time_dict
                        if verbose:
                            logger.info(
                                "  [NeuralTimeout] %s max_time=%dh%02dm%02ds "
                                "(P95 of %d prior non-neural train times: %.0fs)",
                                recurrent_model_name,
                                _max_time_dict["hours"], _max_time_dict["minutes"], _max_time_dict["seconds"],
                                _n, _p95_r,
                            )

                # Build the eval_set tuple the wrapper actually accepts. Pre-fix this passed val_sequences=,
                # val_features=, val_labels= as kwargs - none of which exist on RecurrentRegressorWrapper.fit
                # / RecurrentClassifierWrapper.fit, so every recurrent fit raised TypeError and the existing
                # smoke test only survived via except-skip. The wrapper's _create_eval_dataset accepts a
                # 2-tuple (features, labels) or 3-tuple (sequences, features, labels).
                eval_set = None
                if val_target is not None:
                    if val_sequences is not None:
                        eval_set = (val_sequences, val_df_pd if val_df_pd is not None else None, val_target)
                    elif val_df_pd is not None:
                        eval_set = (val_df_pd, val_target)

                # Thread per-target sample_weight from ctx so the recurrent model
                # trains on the same weighted loss surface every other suite member
                # uses. Pre-fix the recurrent wrapper trained UNWEIGHTED even on
                # recency- / fairness-weighted suites, so the ensemble blend was
                # silently biased toward the unweighted recurrent member. Defensive
                # try/except per fit-call signature inspection: older wrapper
                # versions may not accept sample_weight; fall back to unweighted
                # fit + WARN so the caller sees the path was taken.
                _ctx_sw_dict = getattr(ctx, "sample_weights", None) or {}
                _sw_for_target = (
                    _ctx_sw_dict.get(cur_target_name)
                    if isinstance(_ctx_sw_dict, dict) and _ctx_sw_dict
                    else None
                )
                # Slice weights to train_idx so length matches train_target.
                if _sw_for_target is not None and hasattr(ctx, "train_idx") and ctx.train_idx is not None:
                    try:
                        _sw_for_target = np.asarray(_sw_for_target)[ctx.train_idx]
                    except (TypeError, IndexError):
                        _sw_for_target = None
                _fit_kwargs = dict(
                    sequences=train_sequences,
                    features=train_df_pd if train_df_pd is not None else None,
                    labels=train_target,
                    eval_set=eval_set,
                )
                if _sw_for_target is not None:
                    _fit_kwargs["sample_weight"] = _sw_for_target
                try:
                    try:
                        model_clone.fit(**_fit_kwargs)
                    except TypeError as _fit_te:
                        if "sample_weight" in str(_fit_te) and "sample_weight" in _fit_kwargs:
                            logger.warning(
                                "Recurrent wrapper %s did not accept sample_weight (%s); "
                                "falling back to unweighted fit. Ensemble blend may be biased on "
                                "weighted suites. Upgrade the wrapper or remove weighting.",
                                type(model_clone).__name__, _fit_te,
                            )
                            _fit_kwargs.pop("sample_weight", None)
                            model_clone.fit(**_fit_kwargs)
                        else:
                            raise
                except Exception as e:
                    logger.error("Failed to train %s for %s: %s", recurrent_model_name, cur_target_name, e)
                    continue

                if ctx is None:
                    # Legacy back-compat: caller did not thread ctx, so we cannot compute predictions against
                    # the same train/val/test splits the booster path used. Keep the original raw-model append.
                    models[target_type][cur_target_name].append(model_clone)
                    if verbose:
                        logger.info("Successfully trained %s for %s (legacy append, no ensemble rerun)", recurrent_model_name, cur_target_name)
                    continue

                # ctx-aware path: compute preds on each split, build a member entry, append.
                # test_df_pd is threaded through the kwarg (main.py passes it); fallback to None if missing.
                # POLARS-PANDAS-CHURN: ``ctx`` + ``split`` enable the per-frame numpy-coercion cache
                # so the (train/val/test) features arrays aren't re-coerced for each recurrent member.
                train_preds = _safe_predict_recurrent(
                    model=model_clone, sequences=train_sequences,
                    features=train_df_pd if train_df_pd is not None else None,
                    is_classification=is_classification,
                    ctx=ctx, split="train",
                )
                val_preds_arr = _safe_predict_recurrent(
                    model=model_clone, sequences=val_sequences,
                    features=val_df_pd if val_df_pd is not None else None,
                    is_classification=is_classification,
                    ctx=ctx, split="val",
                ) if (val_sequences is not None or val_df_pd is not None) else None
                test_preds_arr = _safe_predict_recurrent(
                    model=model_clone, sequences=test_sequences,
                    features=test_df_pd if test_df_pd is not None else None,
                    is_classification=is_classification,
                    ctx=ctx, split="test",
                ) if (test_sequences is not None or test_df_pd is not None) else None

                if val_preds_arr is None and test_preds_arr is None and train_preds is None:
                    # All splits failed prediction - graceful skip, keep moving (per continue_on_model_failure semantics).
                    logger.warning("Recurrent %s for %s: all splits failed prediction; skipping ensemble integration.", recurrent_model_name, cur_target_name)
                    models[target_type][cur_target_name].append(model_clone)
                    continue

                entry = _build_recurrent_member_entry(
                    recurrent_model_name=recurrent_model_name.lower(),
                    model=model_clone,
                    train_preds=train_preds,
                    val_preds=val_preds_arr,
                    test_preds=test_preds_arr,
                    is_classification=is_classification,
                )
                models[target_type][cur_target_name].append(entry)
                targets_with_recurrent.setdefault(target_type, {})[cur_target_name] = target_values

                if verbose:
                    logger.info("Successfully trained %s for %s; entry appended to ensemble member list.", recurrent_model_name, cur_target_name)

    # Per-target single rerun of score_ensemble with the augmented member list.
    if ctx is not None:
        for target_type, by_name in targets_with_recurrent.items():
            for cur_target_name, target_values in by_name.items():
                _rerun_ensemble_with_recurrent(
                    ctx=ctx,
                    target_type=target_type,
                    target_name=cur_target_name,
                    target_values=target_values,
                )

    return models
