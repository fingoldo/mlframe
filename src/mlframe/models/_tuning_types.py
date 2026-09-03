"""``MLTaskType`` / ``HashableDict``: zero-dependency shared types used across the ``tuning.py`` /
``tuning_rules.py`` / ``tuning_catboost.py`` split (monolith split, CLAUDE.md "sibling re-export"
convention). Carved into their own module specifically because BOTH downstream siblings need them
independently -- keeping them in ``tuning.py`` itself would force ``tuning_rules.py`` to import back
from ``tuning.py``, which (combined with ``tuning.py`` needing ``ParamsOptimizer`` back from
``tuning_rules.py``) would create an import cycle.
"""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["MLTaskType", "HashableDict"]


class MLTaskType(Enum):
    """ML task types supported by the params optimizer's loss/eval-metric selection logic (see ``create_ctr_params`` comments)."""

    Regression = auto()
    Multiregression = auto()
    Classification = auto()
    Multiclassification = auto()
    MultilabelClassification = auto()
    Ranking = auto()


class HashableDict(dict):
    """A dict subclass usable as a dict key (e.g. grouping rule conditions in ``skip_if_values_or``/``allow_if_values_or``/``allow_if_values_and``).

    Plain ``dict`` is unhashable, so this recipe hashes the tuple of its sorted ``(key, value)`` items instead.
    """

    def __hash__(self):  # type: ignore[override]  # intentional: dict.__hash__ is None (unhashable); this recipe makes it hashable
        return hash(tuple(sorted(self.items())))
