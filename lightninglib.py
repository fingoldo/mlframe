"""
DEPRECATED: Import from mlframe.training.neural instead.

This module is kept for backwards compatibility only.
All classes and functions have been moved to mlframe.training.neural.

Example migration:
    # Old (deprecated):
    from mlframe.lightninglib import TorchDataset, TorchDataModule

    # New (recommended):
    from mlframe.training.neural import TorchDataset, TorchDataModule
"""

import warnings

warnings.warn(
    "mlframe.lightninglib is deprecated. "
    "Import from mlframe.training.neural instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything for backwards compatibility
from mlframe.training.neural import *
