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


def _safe_predict_recurrent(
    *,
    model,
    sequences,
    features,
    is_classification: bool,
) -> np.ndarray | None:
    """Predict on val/test for a fitted recurrent model. Returns None on any failure so the caller can degrade.

    Features arrive as pandas/polars DataFrames from the suite splits; the wrapper's predict path calls
    ``_compute_cache_key(features, ...)`` which reads ``features.dtype`` BEFORE ``_create_dataset`` would
    normalise to ndarray, so a DataFrame at the predict boundary blows up with ``'DataFrame' has no attribute
    'dtype'``. Cast to a float32 ndarray here (the same shape contract the wrapper documents).
    """
    if sequences is None and features is None:
        return None
    if features is not None and not isinstance(features, np.ndarray):
        try:
            features = np.asarray(features.to_numpy() if hasattr(features, "to_numpy") else features, dtype=np.float32)
        except Exception as exc:
            logger.warning("Recurrent predict: failed to coerce features to ndarray (%s); dropping member.", exc)
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
    members = ctx.models.get(target_type, {}).get(target_name) or []
    if len(members) < 2:
        # Single-member "ensemble" is degenerate; the booster path would not have built one either.
        return False

    # Note: we proceed even when ``existing_ensembles`` is empty. A common case is exactly one booster + one
    # recurrent member - there was no prior ensemble (score_ensemble needs >=2 members) but post-recurrent we
    # now do, and the rebuild here is the FIRST time the target gets a real blend.
    existing_ensembles = ctx.ensembles.get(target_type, {}).get(target_name) if ctx.ensembles else None

    if not _validate_member_shape_uniformity(members, target_name=target_name):
        return False

    # Lazy import keeps the module-import cost low for runs that disable recurrent entirely.
    from mlframe.models.ensembling import score_ensemble

    try:
        train_target = _coerce_to_numpy(target_values[ctx.train_idx] if ctx.train_idx is not None and hasattr(target_values, "__getitem__") else None)
        val_target = _coerce_to_numpy(target_values[ctx.val_idx] if ctx.val_idx is not None and hasattr(target_values, "__getitem__") else None)
        test_target = _coerce_to_numpy(target_values[ctx.test_idx] if ctx.test_idx is not None and hasattr(target_values, "__getitem__") else None)
    except Exception as exc:
        logger.warning("Recurrent ensemble rerun: failed to slice targets for %s (%s); skipping.", target_name, exc)
        return False

    # Pass-through only kwargs ``score_ensemble`` declares - keeps us forward-compatible if the suite-level path
    # adds new optional args without breaking this rerun.
    sig = inspect.signature(score_ensemble)
    accepted = set(sig.parameters.keys())
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
    }
    # The booster path also feeds ``ensembling_methods`` from suite config; we let the default kick in for the
    # rerun to keep this helper config-free. Drop any kwargs ``score_ensemble`` doesn't accept (defensive).
    kwargs = {k: v for k, v in kwargs.items() if k in accepted or k == "models_and_predictions"}

    try:
        rebuilt = score_ensemble(**kwargs)
    except Exception as exc:
        logger.warning(
            "Recurrent ensemble rerun failed for target %s (%s); keeping pre-recurrent ensemble.",
            target_name, exc,
        )
        return False

    if not rebuilt:
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

    When ``ctx`` is supplied (post-2026-05-16 wiring), each fitted recurrent model gets val/test/train preds
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

                train_target = _coerce_to_numpy(target_values[train_idx] if hasattr(target_values, "__getitem__") else target_values.iloc[train_idx])
                val_target = _coerce_to_numpy(target_values[val_idx]) if (val_idx is not None and hasattr(target_values, "__getitem__")) else None
                test_target = _coerce_to_numpy(target_values[test_idx] if hasattr(target_values, "__getitem__") else target_values.iloc[test_idx])

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

                try:
                    model_clone.fit(
                        sequences=train_sequences,
                        features=train_df_pd if train_df_pd is not None else None,
                        labels=train_target,
                        eval_set=eval_set,
                    )
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
                train_preds = _safe_predict_recurrent(
                    model=model_clone, sequences=train_sequences,
                    features=train_df_pd if train_df_pd is not None else None,
                    is_classification=is_classification,
                )
                val_preds_arr = _safe_predict_recurrent(
                    model=model_clone, sequences=val_sequences,
                    features=val_df_pd if val_df_pd is not None else None,
                    is_classification=is_classification,
                ) if (val_sequences is not None or val_df_pd is not None) else None
                test_preds_arr = _safe_predict_recurrent(
                    model=model_clone, sequences=test_sequences,
                    features=test_df_pd if test_df_pd is not None else None,
                    is_classification=is_classification,
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
