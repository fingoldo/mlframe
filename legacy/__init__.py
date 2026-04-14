"""Frozen legacy modules.

These modules are retained for historical reference and archival lookup only.
They are not maintained, not imported by the modern `mlframe.training` package,
and may depend on APIs that have since been removed. High-value symbols have
been salvaged into the modern package (see CHANGELOG audit #02 salvage entry).

Do not add new imports from here.
"""

import warnings as _warnings

_warnings.warn(
    "mlframe.legacy is a frozen archive of superseded modules (training_old, "
    "OldEnsembling). Use mlframe.training.* instead.",
    DeprecationWarning,
    stacklevel=2,
)
