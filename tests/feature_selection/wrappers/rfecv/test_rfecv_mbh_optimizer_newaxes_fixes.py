"""Regression tests for verified bugs in ``_rfecv_mbh_optimizer._build_mbh_optimizer``.

Finding [6][P2] (init_design_size, S9/S10 Wave 2): an explicit integer
``init_design_size=5`` was silently routed through the adaptive 'auto' branch
because the guard read ``_init_size_param == "auto" or _init_size_param == 5``.
On a small problem (p=15 -> auto K=3) the user's explicit request for 5 anchors
was overridden down to 3. The fix keys the adaptive branch only on the 'auto'
sentinel (or any non-int), so explicit integers fall through to the requested K.

These exercise the smallest callable (``_build_mbh_optimizer``) directly via a
lightweight stand-in for ``self``; no full RFECV fit / training run.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# ETR surrogate path needs sklearn (ExtraTreesRegressor).
pytest.importorskip("sklearn")

from mlframe.feature_selection.wrappers.rfecv._mbh_optimizer import _build_mbh_optimizer
from mlframe.feature_selection.wrappers._enums import OptimumSearch


def _make_self(init_design_size):
    """Minimal stand-in carrying only the attributes _build_mbh_optimizer reads."""
    return SimpleNamespace(
        optimizer_plotting=None,
        max_nfeatures=None,
        optimizer_config=None,
        mbh_adaptive_threshold=30,
        init_design_size=init_design_size,
    )


def _seeded_for(init_design_size, *, p, max_refits):
    original_features = list(range(p))
    opt = _build_mbh_optimizer(
        _make_self(init_design_size),
        original_features=original_features,
        max_refits=max_refits,
        top_predictors_search_method=OptimumSearch.ModelBasedHeuristic,
    )
    assert opt is not None
    return list(opt.seeded_inputs)


def test_explicit_init_design_size_5_honored_on_small_problem():
    """init_design_size=5 must yield 5 anchors even when auto would pick 3 (p=15)."""
    seeded = _seeded_for(5, p=15, max_refits=None)
    # Bug pre-fix: routed through 'auto' -> 10 < p <= 50 -> K=3 (3 anchors).
    # Post-fix: explicit int 5 falls through to _K = 5.
    assert len(seeded) == 5, f"expected 5 explicit anchors, got {seeded}"


def test_auto_still_resolves_to_3_on_same_small_problem():
    """Control: 'auto' on p=15 stays at K=3, proving explicit-5 now diverges from auto."""
    seeded = _seeded_for("auto", p=15, max_refits=None)
    assert len(seeded) == 3, f"expected 3 auto anchors on p=15, got {seeded}"


def test_explicit_5_differs_from_auto():
    """Direct contrast on the exact value that triggered the bug (the literal 5)."""
    explicit = _seeded_for(5, p=15, max_refits=None)
    auto = _seeded_for("auto", p=15, max_refits=None)
    assert len(explicit) > len(auto), f"explicit init_design_size=5 ({explicit}) must not collapse to auto ({auto})"


def test_explicit_int_other_than_5_unaffected():
    """Sanity: a non-5 explicit int (4) still routes to the requested K (regression guard)."""
    seeded = _seeded_for(4, p=15, max_refits=None)
    assert len(seeded) == 4, f"expected 4 explicit anchors, got {seeded}"
