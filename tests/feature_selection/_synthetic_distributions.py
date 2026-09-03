"""Re-export shim for the shared multi-distribution operand sampler.

The implementation now lives verbatim in
:mod:`tests.feature_selection._synth.distributions`; this module exists only so that the
5 existing importers under ``tests/`` keep working without an edit. Three of them do
``from tests.feature_selection import _synthetic_distributions as sd`` and then reach for
arbitrary attributes (``sd.DISTRIBUTIONS``, ``sd._POSITIVE_FAMILIES``, ...), so the whole
module-level surface -- private names included -- is re-exported here, not just the
publicly imported subset.

New tests should not import from this shim. ``DISTRIBUTIONS`` / ``with_outliers`` /
``_enforce_domain`` are slated to move into production (``mlframe.data.datasets``); once
they land there, import them from production. Until then, import from
``tests.feature_selection._synth.distributions`` directly.
"""

from tests.feature_selection._synth.distributions import DOMAIN_ANY
from tests.feature_selection._synth.distributions import DOMAIN_POSITIVE
from tests.feature_selection._synth.distributions import DOMAIN_DIVISOR
from tests.feature_selection._synth.distributions import POSITIVE_FLOOR
from tests.feature_selection._synth.distributions import DIVISOR_FLOOR
from tests.feature_selection._synth.distributions import DISTRIBUTIONS
from tests.feature_selection._synth.distributions import HEAVY_TAILED_FAMILIES
from tests.feature_selection._synth.distributions import PROFILES
from tests.feature_selection._synth.distributions import with_outliers
from tests.feature_selection._synth.distributions import family_for_operand
from tests.feature_selection._synth.distributions import sample_operand
from tests.feature_selection._synth.distributions import sample_operands
from tests.feature_selection._synth.distributions import available_profiles
from tests.feature_selection._synth.distributions import _uniform as _uniform
from tests.feature_selection._synth.distributions import _normal as _normal
from tests.feature_selection._synth.distributions import _lognormal as _lognormal
from tests.feature_selection._synth.distributions import _exponential as _exponential
from tests.feature_selection._synth.distributions import _gamma as _gamma
from tests.feature_selection._synth.distributions import _student_t as _student_t
from tests.feature_selection._synth.distributions import _pareto as _pareto
from tests.feature_selection._synth.distributions import _beta_u as _beta_u
from tests.feature_selection._synth.distributions import _bimodal as _bimodal
from tests.feature_selection._synth.distributions import _POSITIVE_FAMILIES as _POSITIVE_FAMILIES
from tests.feature_selection._synth.distributions import _enforce_domain as _enforce_domain
from tests.feature_selection._synth.distributions import _str_to_int as _str_to_int

# Literal list of string constants on purpose: vulture only honours ``__all__`` when it
# parses as an ``ast.Assign`` to a list/tuple of string constants, and its ``_ignore_import``
# exemption applies only to files literally named ``__init__.py``. Without this, every
# re-export above would fire as an unused import at confidence 90 in the blocking
# ``tests-lint-blocking`` vulture job.
__all__ = [
    "DOMAIN_ANY",
    "DOMAIN_POSITIVE",
    "DOMAIN_DIVISOR",
    "POSITIVE_FLOOR",
    "DIVISOR_FLOOR",
    "DISTRIBUTIONS",
    "HEAVY_TAILED_FAMILIES",
    "PROFILES",
    "with_outliers",
    "family_for_operand",
    "sample_operand",
    "sample_operands",
    "available_profiles",
    "_uniform",
    "_normal",
    "_lognormal",
    "_exponential",
    "_gamma",
    "_student_t",
    "_pareto",
    "_beta_u",
    "_bimodal",
    "_POSITIVE_FAMILIES",
    "_enforce_domain",
    "_str_to_int",
]
