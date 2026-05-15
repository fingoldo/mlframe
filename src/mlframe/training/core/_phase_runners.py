"""Namespace module aggregating the per-phase entry points used by ``main.py``.

CODE-P1-8: each per-phase wrapper used to be imported individually at the top of ``main.py``
(8 ``from ._phase_X import Y`` lines). Consolidating them here lets ``main.py`` say::

    from . import _phase_runners as pr

then call e.g. ``pr.apply_polars_categorical_fixes(...)``. This:

- Drops main.py's import count by 7 (one consolidated line vs. eight)
- Keeps the per-phase modules small and individually testable
- Documents the public phase surface in one place

Add new phase entry points here when you introduce one, and import the module rather than the
symbol from ``main.py``. Existing per-phase modules remain importable directly for tests that
need finer-grained patching.
"""
from __future__ import annotations

from ._phase_composite_discovery import run_composite_target_discovery
from ._phase_composite_post import run_composite_post_processing
from ._phase_config_setup import setup_configuration
from ._phase_finalize import finalize_suite
from ._phase_polars_fixes import apply_polars_categorical_fixes
from ._phase_recurrent import train_recurrent_models
from ._phase_temporal_audit import run_temporal_audit_batch
from ._phase_train_one_target import _train_one_target

__all__ = [
    "apply_polars_categorical_fixes",
    "finalize_suite",
    "run_composite_post_processing",
    "run_composite_target_discovery",
    "run_temporal_audit_batch",
    "setup_configuration",
    "train_recurrent_models",
    "_train_one_target",
]
