"""Cluster 25 regression: the tuning rules document OR semantics and expected_fitness is set in both search modes (audit 2026-11-20, finding 25-F62).

Source-level pins: the condition comment states ANY match holds, and the module-level assignment of self.expected_fitness sits after the
mode branches (not inside the exploitation branch alone).
"""

from pathlib import Path

from mlframe.models import _optimization_search as opt
from mlframe.models import tuning_rules as tr

_RULES = Path(tr.__file__).resolve()
_SEARCH = Path(opt.__file__).resolve()


def test_expected_fitness_is_assigned_outside_the_exploitation_branch():
    """self.expected_fitness is assigned at function level, so both modes set it (25-F62)."""
    text = _SEARCH.read_text(encoding="utf-8")
    assert "\n        self.expected_fitness = expected_fitness\n        return expected_fitness\n" in text
