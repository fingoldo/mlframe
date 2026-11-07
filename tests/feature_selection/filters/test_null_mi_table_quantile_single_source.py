"""The checked-in null-MI table is built at the quantile its docstring names (audit 2026-10-14, F03).

The lookup module documents a 95th-percentile floor. A table rebuilt at another quantile would silently change every search threshold
while the docstring still said 95, so the table metadata must match the docstring and the builder default.
"""

from pathlib import Path

import orjson

from mlframe.feature_selection.filters import _null_mi_threshold as nm

_REPO = Path(__file__).resolve().parents[3]
_BUILDER = _REPO / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "build_null_mi_table.py"


def _builder_quantile() -> float:
    """The QUANTILE constant from the table builder, read from its source without importing the benchmark package."""
    for line in _BUILDER.read_text(encoding="utf-8").splitlines():
        if line.startswith("QUANTILE ="):
            return float(line.split("=", 1)[1].strip())
    raise AssertionError("QUANTILE not found in the table builder")


def test_table_meta_quantile_matches_the_docstring():
    """The shipped table is at the 95th percentile, as the module docstring says."""
    data = orjson.loads(nm._TABLE_PATH.read_bytes())
    assert data["meta"]["quantile"] == 0.95


def test_table_meta_quantile_matches_the_builder_default():
    """The builder default and the shipped table agree, so a rebuild with defaults reproduces the table's quantile."""
    data = orjson.loads(nm._TABLE_PATH.read_bytes())
    assert data["meta"]["quantile"] == _builder_quantile()
