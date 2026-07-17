"""Regression tests for the strict-parse fix of ``is_composite_target_name``
(audit 2026-06-10, item T20).

Pre-fix, ``is_composite_target_name`` used a substring scan over
``_COMPOSITE_NAME_FRAGMENTS`` (``f"-{alias}-"`` / ``f"-{full}-"``) plus a
legacy ``__{full}__`` substring loop. That scan false-positived on malformed
and empty-segment names that merely *contain* an alias fragment but do NOT
parse as a valid ``{target}-{alias}-{base}`` triple -- e.g. ``-linres-x``
(empty target), ``y-linres-`` (empty base), ``__linear_residual__x`` (empty
legacy target). Such a name is NOT a composite target and must label as MTTR
(raw mean), not MTRESID (residual mean).

The fix replaces the substring scan with a strict structural parse that
anchors the alias as a complete dash-/underscore-delimited token bracketed
by a non-empty target segment and a non-empty base segment.

Each ``test_*_pre_fix_false_positive`` below asserts ``False`` on a name the
OLD substring scan returned ``True`` for -- so the test FAILS on pre-fix code
and PASSES post-fix. The ``_preserved`` tests pin that every genuine
composite name (short-alias, legacy ``__``, multi-base, unary, dashed
target/base) still routes correctly.
"""

from __future__ import annotations

import pytest

from mlframe.training.composite.transforms.naming import (
    is_composite_target_name,
    compose_target_name,
    TRANSFORM_NAME_SHORT,
    _COMPOSITE_NAME_FRAGMENTS,
)


def _old_substring_scan(name: str) -> bool:
    """Reproduction of the PRE-FIX detection logic, used only to prove the
    regression cases below actually flipped (these names returned True on the
    old code)."""
    if not name:
        return False
    if any(frag in name for frag in _COMPOSITE_NAME_FRAGMENTS):
        return True
    for full in TRANSFORM_NAME_SHORT.keys():
        if f"__{full}__" in name:
            return True
    return False


# --- Names the OLD scan mis-classified as composite (regression: must be False now) ---

PRE_FIX_FALSE_POSITIVES = [
    "-linres-x",  # empty target segment
    "y-linres-",  # empty base segment
    "-diff-x",  # empty target, short alias
    "col-ratio-",  # empty base, short alias
    "mid-spline-",  # empty base
    "__linear_residual__x",  # empty legacy target
    "x__linear_residual__",  # empty legacy base
]


@pytest.mark.parametrize("name", PRE_FIX_FALSE_POSITIVES)
def test_strict_parse_rejects_malformed_names(name):
    # Sanity: confirm the OLD logic genuinely accepted this name, so the
    # assertion below is a real regression guard (fails on pre-fix code).
    assert _old_substring_scan(name) is True, f"test fixture stale: old scan no longer accepts {name!r}"
    # Post-fix: a malformed / empty-segment name is NOT a composite target.
    assert is_composite_target_name(name) is False, f"{name!r} is not a valid {{target}}-{{alias}}-{{base}} triple and must label MTTR, not MTRESID"


# --- Real composite names: must still be detected (no regression) ---

REAL_COMPOSITE_NAMES = [
    "y-linres-lag1",
    "TVT-linres-TVT_prev",
    "y-monres-base",
    "y-cbrtY-lag1",  # unary
    "y-linresM-lag1+lag2",  # multi-base ('+' join)
    "y-spline-x",
    "y-interact-x",
    "y__linear_residual__lag1",  # legacy double-underscore
    "y__monotonic_residual__base",  # legacy
    "quarterly-margin-ratio-of-revenue",  # dashed base, alias is internal token
]


@pytest.mark.parametrize("name", REAL_COMPOSITE_NAMES)
def test_strict_parse_accepts_real_composites(name):
    assert is_composite_target_name(name) is True, f"{name!r} is a canonical composite name and must label MTRESID"


# --- Plain user / raw target names: must NOT be detected ---

NON_COMPOSITE_NAMES = [
    "raw_y",
    "y",
    "",
    "net-ratiometric-index",  # 'ratio' only a substring of 'ratiometric'
    "super-diffusion-model",  # 'diff' only a substring of 'diffusion'
    "revenue-growth-rate-2024",
]


@pytest.mark.parametrize("name", NON_COMPOSITE_NAMES)
def test_non_composite_names_not_detected(name):
    assert is_composite_target_name(name) is False


def test_compose_roundtrips_through_strict_parse():
    """Every name emitted by ``compose_target_name`` (the production
    producer) must be recognised by ``is_composite_target_name`` (the
    production consumer) -- the two halves of the naming contract."""
    for transform_name in TRANSFORM_NAME_SHORT:
        name = compose_target_name("y", transform_name, "base")
        assert is_composite_target_name(name) is True, f"compose/parse contract broken for transform {transform_name!r} -> {name!r}"


def test_ambiguous_three_token_user_columns_documented():
    """``price-diff-7d`` / ``debt-ratio-q`` are structurally identical to a
    real composite (target-alias-base, alias is a registered token) so there
    is no signal to separate them without the target list. They remain
    detected; this test PINS that documented limitation so a future change
    that tries to 'fix' it does so deliberately."""
    assert is_composite_target_name("price-diff-7d") is True
    assert is_composite_target_name("debt-ratio-q") is True
