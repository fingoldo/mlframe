"""``ParamOracle._all_combos`` refuses a Cartesian product above the cap instead of materialising it.

The enumeration built every combination as a dict before anything could check the size, so an eight-knob grid of ten values each (1e8
combinations) exhausted memory. The size is now computed first and an oversized grid raises ValueError.
"""

from __future__ import annotations

import pytest

from mlframe.utils import _param_oracle as po


def test_oversized_grid_is_refused_before_it_is_materialised(tmp_path, monkeypatch):
    """A grid whose product exceeds the cap raises ValueError rather than building the full list."""
    monkeypatch.setattr(po, "_MAX_PARAM_COMBOS", 1000, raising=False)
    grid = {k: list(range(10)) for k in ("a", "b", "c", "d")}  # 10_000 combinations, above the patched cap of 1_000
    oracle = po.ParamOracle(str(tmp_path / "store.jsonl"), param_space=grid, minimize="elapsed_s")

    with pytest.raises(ValueError, match="combinations"):
        oracle._all_combos()
