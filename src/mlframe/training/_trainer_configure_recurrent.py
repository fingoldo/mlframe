"""``_configure_recurrent_params``, carved out of ``trainer.py`` (X_EFFICIENCY_ARCHITECTURE-1 fix,
mrmr_audit_2026-07-22) to clear the repo's enforced hard 1000-LOC CI gate (that file was 1001 lines).
Behaviour preserved bit-for-bit; ``trainer.py`` re-exports this function so every existing import
keeps working unchanged.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _configure_recurrent_params(
    recurrent_models: list[str],
    recurrent_config: Any | None,
    sequences_train: list[np.ndarray] | None,
    features_train: pd.DataFrame | np.ndarray | None,
    use_regression: bool,
    metamodel_func: Callable | None = None,
) -> dict[str, dict]:
    """Configure recurrent model (LSTM, GRU, RNN, Transformer) parameters.

    Parameters
    ----------
    recurrent_models : list of str
        List of recurrent model types to configure (e.g., ["lstm", "gru"]).
    recurrent_config : RecurrentConfig or None
        Configuration for recurrent models. If None, uses defaults.
    sequences_train : list of np.ndarray or None
        Training sequences (variable length).
    features_train : DataFrame or np.ndarray or None
        Tabular features for HYBRID mode.
    use_regression : bool
        Whether to use regression (MSELoss) or classification (CrossEntropyLoss).
    metamodel_func : callable, optional
        Function to wrap the model (e.g., for calibration).

    Returns
    -------
    dict
        Dictionary mapping model names to their configurations.
    """
    from mlframe.training.neural import (
        RNNType,
        InputMode,
        RecurrentConfig,
        RecurrentClassifierWrapper,
        RecurrentRegressorWrapper,
    )

    if metamodel_func is None:

        def metamodel_func(x):
            """Identity fallback used when no metamodel post-processing function is supplied."""
            return x

    # Determine input mode based on available data
    has_sequences = sequences_train is not None and len(sequences_train) > 0
    has_features = features_train is not None
    if features_train is not None and hasattr(features_train, "shape"):
        has_features = has_features and features_train.shape[1] > 0

    if has_sequences and has_features:
        input_mode = InputMode.HYBRID
    elif has_sequences:
        input_mode = InputMode.SEQUENCE_ONLY
    else:
        input_mode = InputMode.FEATURES_ONLY

    # Use provided config or create default
    if recurrent_config is None:
        recurrent_config = RecurrentConfig()

    # seq_input_dim / features_dim are computed at fit-time by _RecurrentWrapperBase from input shapes (see neural/recurrent.py _aux_input_size / _seq_input_size).

    result = {}

    for model_name in recurrent_models:
        model_name_lower = model_name.lower()

        # Map model name to RNNType
        rnn_type_map = {
            "lstm": RNNType.LSTM,
            "gru": RNNType.GRU,
            "rnn": RNNType.RNN,
            "transformer": RNNType.TRANSFORMER,
        }
        if model_name_lower not in rnn_type_map:
            raise ValueError(f"Unknown recurrent model type: {model_name}. " f"Supported: {list(rnn_type_map.keys())}")

        rnn_type = rnn_type_map[model_name_lower]

        # RecurrentConfig field names are exact: n_heads (not num_heads) and mlp_hidden_sizes (not mlp_hidden_dims); seq/features dims are inferred at fit-time.
        config = RecurrentConfig(
            input_mode=input_mode,
            rnn_type=rnn_type,
            hidden_size=recurrent_config.hidden_size,
            num_layers=recurrent_config.num_layers,
            dropout=recurrent_config.dropout,
            bidirectional=recurrent_config.bidirectional,
            n_heads=recurrent_config.n_heads,
            use_attention=recurrent_config.use_attention,
            mlp_hidden_sizes=recurrent_config.mlp_hidden_sizes,
            num_classes=recurrent_config.num_classes,
            learning_rate=recurrent_config.learning_rate,
            weight_decay=recurrent_config.weight_decay,
            max_epochs=recurrent_config.max_epochs,
            batch_size=recurrent_config.batch_size,
            early_stopping_patience=recurrent_config.early_stopping_patience,
        )

        # Select wrapper class based on task type
        WrapperClass = RecurrentRegressorWrapper if use_regression else RecurrentClassifierWrapper
        wrapper = WrapperClass(config=config)

        result[model_name_lower] = dict(model=metamodel_func(wrapper))

    return result
