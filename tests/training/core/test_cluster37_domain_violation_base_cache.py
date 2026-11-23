"""Cluster 37 regression: the NaN-rows domain-violation check builds each full base column once, not once per spec (audit 2026-11-20, finding 37-F27).

Source-level pin: inside the per-spec loop, base columns go through the per-call cache ``_full_base`` and the expensive builder is called only from that cache.
"""

import re
from pathlib import Path

from mlframe.training.core import _phase_composite_discovery as mod


def _function_body() -> str:
    """Return the source of the domain-violation function up to the next top-level def."""
    text = Path(mod.__file__).read_text(encoding="utf-8")
    start = text.index("def _run_composite_targ_nan_rows_domain_violations(")
    end = text.index("\ndef ", start + 10)
    return text[start:end]


def test_base_columns_are_built_through_the_cache_only():
    """Only the cache helper calls the full-column builder inside this function (37-F27)."""
    body = _function_body()
    calls = re.findall(r"_build_full_column_from_splits\(", body)
    assert len(calls) == 1
    assert "def _full_base(" in body
    assert "_base_cache" in body
