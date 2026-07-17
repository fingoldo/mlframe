"""Regression for finding #8: regex-based ensemble-label rebuild must not
inject backslashes / fail on backreference-like substrings inside model
tags.

Pre-fix: ``re.sub(pattern, repl_string, name)`` interpreted backreferences
in ``repl_string``; a model tag like ``"x\1y"`` or ``"x\\dy"`` produced
either an invalid-group-reference crash (caught by the broad except and
silently swallowed -- mislabelled result) or a literal-backslash in the
rebuilt label.

Post-fix: ``re.sub`` receives a CALLABLE that returns the label verbatim,
which skips escape interpretation entirely.
"""

from __future__ import annotations

import re


def _apply_label_substitution(ensemble_name: str, new_label: str) -> str:
    """Mirror of the production label-rebuild path in score_ensemble.

    The production code lives in the gated branch that requires running
    score_ensemble end-to-end; we extract the exact re.sub call so the
    regression is testable without paying that setup cost.
    """
    _label_value = new_label
    return re.sub(
        r"\[[^\]]+\]",
        lambda _m, _v=_label_value: _v,
        ensemble_name,
        count=1,
    )


def test_plain_label_substitutes_correctly():
    """Plain label substitutes correctly."""
    assert _apply_label_substitution("pre[a+b+c] suffix", "[a+b]") == "pre[a+b] suffix"


def test_label_with_literal_backslash_is_preserved_verbatim():
    # Hypothetical model class name with a backslash in it; the callable replacement
    # must pass it through unchanged.
    """Label with literal backslash is preserved verbatim."""
    out = _apply_label_substitution("pre[x] s", "[a\\b]")
    assert out == "pre[a\\b] s", f"got {out!r}"
    assert "\\" in out  # literal backslash retained, not duplicated


def test_label_with_backreference_token_does_not_crash():
    # \1 in a string replacement would normally raise "invalid group reference";
    # callable form must pass it through verbatim.
    """Label with backreference token does not crash."""
    out = _apply_label_substitution("pre[x] s", r"[a+\1+b]")
    assert out == "pre[a+\\1+b] s", f"got {out!r}"


def test_label_with_group_name_token_does_not_crash():
    """Label with group name token does not crash."""
    out = _apply_label_substitution("pre[x] s", "[a+\\g<name>+b]")
    assert out == "pre[a+\\g<name>+b] s"


def test_no_match_returns_input_unchanged():
    # When the ensemble_name has no [...] block, re.sub leaves it as-is.
    """No match returns input unchanged."""
    assert _apply_label_substitution("no brackets here", "[a+b]") == "no brackets here"
