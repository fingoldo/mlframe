"""Raw columns whose names hold spaces or hyphens are found as leaves of an engineered candidate name."""

from __future__ import annotations

from mlframe.feature_selection.filters._mrmr_fe_step._step_inert_operands import _raw_leaves, _special_raw_names


def test_plain_names_still_split_on_tokens():
    """The token path is unchanged for ordinary names."""
    raw = {"a", "b", "c"}
    assert _raw_leaves("max(rint(a),neg(b))", raw, _special_raw_names(raw)) == ["a", "b"]


def test_names_with_spaces_and_hyphens_are_leaves():
    """'my col' and 'x-1' were split into fragments that matched nothing, so the inert-leaf test was skipped for them."""
    raw = {"my col", "x-1", "e"}
    leaves = _raw_leaves("max(rint(my col),neg(x-1))", raw, _special_raw_names(raw))
    assert leaves == ["my col", "x-1"]


def test_special_name_is_not_found_inside_a_longer_name():
    """'a b' must not match inside 'za bz'."""
    raw = {"a b"}
    assert _raw_leaves("f(za bz)", raw, _special_raw_names(raw)) == []


def test_fragments_of_a_special_name_are_not_counted_separately():
    """With raw columns 'my col', 'my' and 'col', the candidate over 'my col' has the single leaf 'my col'."""
    raw = {"my col", "my", "col"}
    assert _raw_leaves("sqrt(my col)", raw, _special_raw_names(raw)) == ["my col"]
