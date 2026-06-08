"""Shared module-scope state for the FE pair-search submodules: the module logger
and the cross-thread ``times_spent`` accumulator lock."""
from __future__ import annotations

import logging
import threading

# Module-level logger for module-scope helpers (e.g. _dispatch_batch_mi_with_noise_gate).
# ``check_prospective_fe_pairs`` still lazy-imports the parent's ``logger`` for its own body.
_module_logger = logging.getLogger("mlframe.feature_selection.filters._feature_engineering_pairs")


# Wave 27 P1 (2026-05-20): ``check_prospective_fe_pairs`` is dispatched via
# ``parallel_run`` from mrmr.py with backend='threading'. The function
# accumulates per-binary-transform timings into a shared ``times_spent``
# defaultdict via ``+=``. Python's ``+=`` on a float is load-add-store and
# NOT atomic even under the GIL between threads; concurrent workers can
# drop updates silently, under-reporting the diagnostic at mrmr.py:1691.
# This module-level lock serialises the increment; threading workers
# synchronise correctly. Under loky/spawn each worker gets its own
# defaultdict copy (no shared state); the lock has no effect there but
# also doesn't break.
_TIMES_SPENT_LOCK = threading.Lock()
