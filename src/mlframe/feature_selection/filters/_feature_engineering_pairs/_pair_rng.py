"""Per-pair random generators of the external-validation factor picks."""

from __future__ import annotations

from typing import Any

import numpy as np


class PairSeededRng:
    """Source of the external-validation factor picks, one generator per operand pair.

    The picks used to come from one generator shared by the whole sweep, so the draws a pair saw depended on how many draws the pairs before it had made: pairs could not be scored in another
    order or at the same time without changing which factors validated them. Each pair now draws from ``default_rng([seed, a, b])`` (the instance seed and the pair's operand indices), a pure
    function of the pair, so the picks are the same in any order and on any thread."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)

    def for_pair(self, raw_vars_pair: Any) -> "np.random.Generator":
        """The generator of the pair ``raw_vars_pair`` (its operand indices)."""
        return np.random.default_rng([self._seed, *(int(v) for v in raw_vars_pair)])
