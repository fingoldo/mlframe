"""The dedup fast path keyed its cache by value, and in an object column equality is not identity.

`True == 1` and `hash(True) == hash(1)`, so a dict (and `pd.factorize`, and `pd.unique`) collapses them into a
single key. `mapping = {v: fcn(v) for v in pd.unique(s)}` therefore returned `fcn(True)` for every `1` in the
column. `Decimal(1) == 1` collapses identically.

The sharp edge is that this happened ONLY above the `4 * sample` row gate: below it the helper defers to the
row-wise `map` and is correct, so the same column produced different values depending on how many rows it had.

These tests assert the two paths agree, which is what a size-dependent defect requires.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.cleaning_helpers import map_elementwise_dedup

GATE = 80_000  # 4 * the default sample; above this the dedup path engages


def _type_name(v):
    """A function whose output distinguishes values that compare equal -- the whole point."""
    return type(v).__name__


class TestTheSameColumnGivesTheSameAnswerAtAnySize:
    """A helper that changes its answer with row count is the defect, independent of which answer is right."""

    @pytest.mark.parametrize("pair", [(True, 1), (False, 0), (1, 1.0), (Decimal(1), 1)])
    def test_equal_but_differently_typed_values_are_not_collapsed(self, pair):
        """Each pair is `==` and hash-equal, so a value-keyed cache merges them."""
        a, b = pair
        s = pd.Series([a, b] * (GATE // 2 + 10), dtype=object)
        assert map_elementwise_dedup(s, _type_name).equals(s.map(_type_name))

    def test_the_answer_does_not_change_across_the_gate(self):
        """Below the gate the row-wise path runs; above it the dedup path did. They must agree."""
        s = pd.Series([True, 1, False, 0] * (GATE // 4 + 10), dtype=object)
        above = map_elementwise_dedup(s, _type_name)
        below = map_elementwise_dedup(s.iloc[:1000], _type_name)
        assert set(above.unique()) == set(below.unique()) == {"bool", "int"}


class TestTheDocumentedFastPathIsUnchanged:
    """The optimisation exists for repeated string values; restricting it must not disable it there."""

    def test_a_string_column_still_deduplicates(self, monkeypatch):
        """Pinned by mechanism, not by a stopwatch: `fcn` must be called once per distinct value, not per row."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.choice(["ru", "us", "de", "fr"], GATE + 10).astype(object))
        calls = {"n": 0}

        def _counting(v):
            """Count how many times the elementwise function is actually invoked."""
            calls["n"] += 1
            return str(v).upper()

        out = map_elementwise_dedup(s, _counting)
        assert out.equals(s.map(str.upper))
        assert calls["n"] == 4, f"the fast path called fcn {calls['n']} times for 4 distinct values"

    def test_a_string_column_with_missing_values_still_deduplicates(self):
        """NaN is not a str but cannot hash-collide with an unequal-typed value either."""
        rng = np.random.default_rng(1)
        vals = rng.choice(["ru", "us", None], GATE + 10).astype(object)
        s = pd.Series(vals)
        assert map_elementwise_dedup(s, lambda v: "NA" if v is None else str(v).upper()).equals(s.map(lambda v: "NA" if v is None else str(v).upper()))

    def test_a_high_cardinality_column_still_falls_back(self):
        """The pre-existing gate must survive the new one."""
        s = pd.Series([f"id_{i}" for i in range(GATE + 10)], dtype=object)
        assert map_elementwise_dedup(s, str.upper).equals(s.map(str.upper))
