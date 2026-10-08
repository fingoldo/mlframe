"""Quick-mode fuzz smoke for ``train_mlframe_models_suite``.

Companion to the full ``test_fuzz_suite`` (150 combos x FUZZ_SEED).
This module runs the same parametrized harness against a 10-combo
slice so plain ``pytest -m fast`` / ``pytest --fast`` runs and PR CI
still hit the suite end-to-end without paying the full sweep budget.

The slow 150-combo sweep moves to ``slow_only`` and only fires when
explicitly enabled (or when ``-m slow`` is requested).
"""

from __future__ import annotations

import os

import pytest

# Fuzz combos run hundreds of train_mlframe_models_suite iterations and are
# deselected from the default test run; pass pytest --run-fuzz to include.
pytestmark = pytest.mark.fuzz

# Reuse all the heavy plumbing (FuzzCombo dataclass, frame builder, the
# parametrized test body, xfail rules) from the full suite by importing
# the existing module. ``test_fuzz_train_mlframe_models_suite`` is the
# test function; we re-parametrize a thinner combo set against the same
# implementation.
from tests.training._fuzz_combo import enumerate_combos

# Quick slice: 10 combos, same master seed as the full suite so the
# selected combos are deterministic across CI runs. Increase / decrease
# via FUZZ_QUICK_COUNT.
_QUICK_COUNT = int(os.environ.get("FUZZ_QUICK_COUNT", "10"))
_QUICK_MASTER_SEED = int(os.environ.get("FUZZ_SEED", "20260422"))
QUICK_COMBOS = enumerate_combos(target=_QUICK_COUNT, master_seed=_QUICK_MASTER_SEED)


# Open product gaps found by running this slice: combo short id -> what fails.
_KNOWN_GAPS: dict[str, str] = {}


@pytest.mark.fast
@pytest.mark.timeout(900)
@pytest.mark.parametrize("combo", QUICK_COMBOS, ids=[c.pytest_id() for c in QUICK_COMBOS])
def test_fuzz_train_mlframe_models_suite_quick(combo, tmp_path):
    """Quick smoke; delegates to the full suite's combo runner, whose invariants (non-empty models, metadata keys, prediction sanity) all apply."""
    from py_ci_shared.pytest_known_gap import known_gap

    from .test_fuzz_suite import test_fuzz_train_mlframe_models_suite as _full

    assert combo in QUICK_COMBOS
    gap = _KNOWN_GAPS.get(combo.short_id())
    try:
        assert _full(combo, tmp_path) is None
    except Exception:
        if gap is not None:
            known_gap(gap, gap_closed=False)
        raise
    if gap is not None:
        known_gap(gap, gap_closed=True)
