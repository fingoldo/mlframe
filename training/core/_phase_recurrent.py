"""
Phase 6: recurrent model training (LSTM, GRU, RNN, Transformer).

Handles sequence-based models with variable-length support. Neural max_time
defaults to P95 of prior non-neural model train times.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from pyutilz.system import tqdmu_lazy_start
from sklearn.base import clone

from ..configs import TargetTypes
from ..phases import log_phase, log_ram_usage
from ..trainer import _configure_recurrent_params

logger = logging.getLogger(__name__)


def train_recurrent_models(
    *,
    models: Dict,
    recurrent_models: Optional[List[str]],
    recurrent_config,
    train_sequences,
    val_sequences,
    test_sequences,
    train_df,
    train_df_pd,
    val_df_pd,
    target_by_type: Dict,
    train_idx,
    val_idx,
    test_idx,
    _non_neural_train_times: List[float],
    model_name: str,
    verbose: bool,
) -> Dict:
    """Train recurrent models across all target types and targets.

    Returns updated models dict.
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

    for recurrent_model_name in tqdmu_lazy_start(recurrent_models, desc="recurrent model"):
        model_name_lower = recurrent_model_name.lower()
        if model_name_lower not in recurrent_params:
            logger.warning("Recurrent model %s not configured, skipping...", recurrent_model_name)
            continue

        recurrent_model = recurrent_params[model_name_lower]["model"]

        for target_type, targets in target_by_type.items():
            for cur_target_name, target_values in targets.items():
                if verbose:
                    logger.info("Training %s for target %s...", recurrent_model_name, cur_target_name)

                # Extract train/val/test targets
                train_target = target_values[train_idx] if hasattr(target_values, '__getitem__') else target_values.iloc[train_idx]
                val_target = target_values[val_idx] if val_idx is not None and hasattr(target_values, '__getitem__') else None
                test_target = target_values[test_idx] if hasattr(target_values, '__getitem__') else target_values.iloc[test_idx]

                # Convert to numpy if needed
                if hasattr(train_target, 'to_numpy'):
                    train_target = train_target.to_numpy()
                elif hasattr(train_target, 'values'):
                    train_target = train_target.values

                if val_target is not None:
                    if hasattr(val_target, 'to_numpy'):
                        val_target = val_target.to_numpy()
                    elif hasattr(val_target, 'values'):
                        val_target = val_target.values

                if hasattr(test_target, 'to_numpy'):
                    test_target = test_target.to_numpy()
                elif hasattr(test_target, 'values'):
                    test_target = test_target.values

                # Clone model for this target
                model_clone = clone(recurrent_model)

                # Neural max_time from P95 of prior non-neural train times
                if _non_neural_train_times:
                    _p95_r = float(np.percentile(_non_neural_train_times, 95))
                    _max_s_r = max(int(round(_p95_r)), 300)
                    _dd_r = _max_s_r // 86400
                    _hh_r = (_max_s_r % 86400) // 3600
                    _mm_r = (_max_s_r % 3600) // 60
                    _ss_r = _max_s_r % 60
                    _r_inner = getattr(model_clone, "regressor", model_clone)
                    if hasattr(_r_inner, "trainer_params"):
                        _r_inner.trainer_params["max_time"] = {
                            "days": _dd_r, "hours": _hh_r,
                            "minutes": _mm_r, "seconds": _ss_r,
                        }
                        if verbose:
                            logger.info(
                                "  [NeuralTimeout] %s max_time=%dh%02dm%02ds "
                                "(P95 of %d prior non-neural train times: %.0fs)",
                                recurrent_model_name, _hh_r, _mm_r, _ss_r,
                                len(_non_neural_train_times), _p95_r,
                            )

                try:
                    model_clone.fit(
                        sequences=train_sequences,
                        features=train_df_pd if train_df_pd is not None else None,
                        labels=train_target,
                        val_sequences=val_sequences,
                        val_features=val_df_pd if val_df_pd is not None else None,
                        val_labels=val_target,
                    )

                    models[target_type][cur_target_name].append(model_clone)

                    if verbose:
                        logger.info("Successfully trained %s for %s", recurrent_model_name, cur_target_name)

                except Exception as e:
                    logger.error("Failed to train %s for %s: %s", recurrent_model_name, cur_target_name, e)
                    continue

    return models
