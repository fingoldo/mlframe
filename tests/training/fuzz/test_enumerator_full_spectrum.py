"""The fuzz enumerator must realise every axis value on its gate set, and every axis-value pair on the wide sweep.

The gate (``target=150``) is CI-sized: every single axis value is realised there. The wide sweep
(``exhaustive_pairs=True``) keeps every combo the pairwise pass adds and covers every pair.
"""

from __future__ import annotations

import pytest


@pytest.mark.timeout(1500)
def test_gate_realises_every_single_axis_value_at_target_size():
    """The 150-combo gate realises every single axis value."""
    from tests.training._fuzz_combo import enumerator as e
    from tests.training._fuzz_combo.axes import AXES

    combos = e.enumerate_combos(target=150, master_seed=20260422)
    assert len(combos) == 150
    seen: dict[str, set] = {}
    for c in combos:
        for name in AXES:
            seen.setdefault(name, set()).add(e._hashable(getattr(c, name)))
    missing = [(n, v) for n, vals in AXES.items() for v in vals if e._hashable(v) not in seen.get(n, set())]
    assert not missing, f"axis values never realised on the gate: {missing[:10]}"


@pytest.mark.timeout(3000)
def test_wide_sweep_covers_every_axis_pair():
    """The exhaustive-pairs sweep realises every axis-value pair."""
    from tests.training._fuzz_combo import enumerator as e

    combos = e.enumerate_combos(target=150, master_seed=20260422, exhaustive_pairs=True)
    covered: set = set()
    for c in combos:
        covered |= e._combo_pairs(c)
    required = e._all_axis_pairs()
    assert covered >= required, f"{len(required - covered)} axis pairs never realised"
