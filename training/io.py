"""
Model serialization and I/O utilities for mlframe.

Provides functions for saving, loading, and cleaning mlframe models using
zstandard compression and dill serialization.
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional, Dict, Any

import dill
import zstandard as zstd

logger = logging.getLogger(__name__)


def save_mlframe_model(
    model: object,
    file: str,
    zstd_kwargs: Optional[Dict[str, Any]] = None,
    verbose: int = 1,
) -> bool:
    """
    Save an mlframe model to a compressed file.

    Uses zstandard compression and dill serialization to handle complex
    Python objects including lambdas and closures.

    Args:
        model: The model object to save (typically a SimpleNamespace).
        file: Path to the output file.
        zstd_kwargs: Optional compression parameters for zstandard.
            Defaults to level=4, with checksum and content size.
        verbose: Verbosity level. If > 0, logs the file size.

    Returns:
        True if save was successful, False otherwise.
    """
    if zstd_kwargs is None:
        zstd_kwargs = dict(
            level=4,
            write_checksum=True,
            write_content_size=True,
            threads=-1,
        )
    try:
        with open(file, "wb") as f:
            compressor = zstd.ZstdCompressor(**zstd_kwargs)
            with compressor.stream_writer(f) as zf:
                dill.dump(model, zf)
        if verbose > 0:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            logger.info(f"Model saved successfully to {file}. Size: {size_mb:.2f} Mb")
        return True
    except Exception as e:
        logger.error(f"Could not save model to file {file}: {e}")
        return False


def load_mlframe_model(file: str) -> Optional[object]:
    """
    Load an mlframe model from a compressed file.

    Args:
        file: Path to the model file.

    Returns:
        The loaded model object, or None if loading failed.
    """
    try:
        with open(file, "rb") as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as zf:
                model = dill.load(zf)
        return model
    except Exception as e:
        logger.error(f"Could not load model from file {file}: {e}")
        return None


def clean_mlframe_model(model: SimpleNamespace) -> SimpleNamespace:
    """
    Remove extra fields from a model's namespace to reduce RAM usage.

    Removes prediction arrays, target arrays, and outlier detection indices
    that are typically not needed after training.

    Args:
        model: The model namespace to clean.

    Returns:
        The cleaned model namespace (modified in place).
    """
    fields_to_remove = [
        "test_preds",
        "test_probs",
        "test_target",
        "val_preds",
        "val_probs",
        "val_target",
        "train_preds",
        "train_probs",
        "train_target",
        "train_od_idx",
        "val_od_idx",
        "trainset_features_stats",
    ]
    for field in fields_to_remove:
        if hasattr(model, field):
            delattr(model, field)
    return model


__all__ = [
    "save_mlframe_model",
    "load_mlframe_model",
    "clean_mlframe_model",
]
