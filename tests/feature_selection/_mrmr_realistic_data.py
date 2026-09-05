"""Re-export shim for the realistic-case mRMR fixture builders.

The implementation now lives verbatim in
:mod:`tests.feature_selection._synth.mrmr_realistic_data`; this module exists only so that
the 4 existing importers under ``tests/`` keep working without an edit.

New tests should not import from this shim. The case families are slated to move into
production (``mlframe.data.datasets``); once they land there, import them from production.
Until then, import from ``tests.feature_selection._synth.mrmr_realistic_data`` directly.
"""

from tests.feature_selection._synth.mrmr_realistic_data import CaseMeta
from tests.feature_selection._synth.mrmr_realistic_data import make_realistic_case
from tests.feature_selection._synth.mrmr_realistic_data import default_fuzz_grid
from tests.feature_selection._synth.mrmr_realistic_data import _draw as _draw
from tests.feature_selection._synth.mrmr_realistic_data import _positive as _positive
from tests.feature_selection._synth.mrmr_realistic_data import _family_ratio_plus_trig as _family_ratio_plus_trig
from tests.feature_selection._synth.mrmr_realistic_data import _family_subsumed_plus_private as _family_subsumed_plus_private
from tests.feature_selection._synth.mrmr_realistic_data import _family_smooth_interaction as _family_smooth_interaction
from tests.feature_selection._synth.mrmr_realistic_data import _FAMILIES as _FAMILIES
from tests.feature_selection._synth.mrmr_realistic_data import _FAMILY_OPERANDS as _FAMILY_OPERANDS

# Literal list of string constants on purpose: vulture only honours ``__all__`` when it
# parses as an ``ast.Assign`` to a list/tuple of string constants, and its ``_ignore_import``
# exemption applies only to files literally named ``__init__.py``. Without this, every
# re-export above would fire as an unused import at confidence 90 in the blocking
# ``tests-lint-blocking`` vulture job.
__all__ = [
    "CaseMeta",
    "make_realistic_case",
    "default_fuzz_grid",
    "_draw",
    "_positive",
    "_family_ratio_plus_trig",
    "_family_subsumed_plus_private",
    "_family_smooth_interaction",
    "_FAMILIES",
    "_FAMILY_OPERANDS",
]
