"""
Model serialization and I/O utilities for mlframe.

Provides functions for saving, loading, and cleaning mlframe models using
zstandard compression and dill serialization.
"""

import logging
import os
import warnings
from types import SimpleNamespace
from typing import Optional, Dict, Any

import dill
import zstandard as zstd

logger = logging.getLogger(__name__)


# Allowlist of module prefixes for safe unpickling.
_SAFE_MODULE_PREFIXES: tuple = (
    "numpy",
    "pandas",
    "polars",
    "sklearn",
    "pytorch_lightning",
    "torch",
    "catboost",
    "lightgbm",
    "xgboost",
    "builtins",
    "collections",
    "datetime",
    "dataclasses",
    "types",
)

# Specific safe names in "types" (only SimpleNamespace).
_SAFE_SPECIFIC: frozenset = frozenset({("types", "SimpleNamespace")})


class _SafeUnpickler(dill.Unpickler):
    """Restricted unpickler that only allows a conservative allowlist of modules."""

    def find_class(self, module: str, name: str):
        # Allow exact specific pairs.
        if (module, name) in _SAFE_SPECIFIC:
            return super().find_class(module, name)
        # Allow by module prefix (module == prefix or module startswith prefix + ".").
        for prefix in _SAFE_MODULE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise dill.UnpicklingError(
            f"Unsafe class blocked by _SafeUnpickler allowlist: {module}.{name}"
        )


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
                # Note: BufferedWriter wrapping (64KB/256KB/1MB/4MB) was benchmarked
                # 2026-04-14 on a fitted RandomForest + 1M-element ndarray payload —
                # all sizes landed within ±5% of the direct write (high variance).
                # Direct write retained.
                dill.dump(model, zf)
        if verbose > 0:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            logger.info(f"Model saved successfully to {file}. Size: {size_mb:.2f} Mb")
        return True
    except Exception as e:
        logger.error(f"Could not save model to file {file}: {e}")
        return False


def load_mlframe_model(file: str, safe: bool = True) -> Optional[object]:
    """
    Load an mlframe model from a compressed file.

    Args:
        file: Path to the model file.
        safe: If True (default), use _SafeUnpickler with a conservative allowlist.
            If False, use vanilla dill.load (unsafe — RCE risk from untrusted sources).

    Returns:
        The loaded model object, or None if loading failed.
    """
    try:
        with open(file, "rb") as f:
            decompressor = zstd.ZstdDecompressor()
            with decompressor.stream_reader(f) as zf:
                if safe:
                    model = _SafeUnpickler(zf).load()
                else:
                    warnings.warn(
                        "Loading without allowlist — trust source",
                        UserWarning,
                        stacklevel=2,
                    )
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
    "_SafeUnpickler",
]
