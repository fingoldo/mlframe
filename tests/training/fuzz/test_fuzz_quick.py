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


@pytest.mark.fast
@pytest.mark.timeout(300)
@pytest.mark.parametrize("combo", QUICK_COMBOS, ids=[c.pytest_id() for c in QUICK_COMBOS])
def test_fuzz_train_mlframe_models_suite_quick(combo, tmp_path, request):
    """Quick smoke; delegates to the full suite's combo runner."""
    from .test_fuzz_suite import test_fuzz_train_mlframe_models_suite as _full
    _full(combo, tmp_path, request)
