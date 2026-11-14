"""The shared split of an engineered basis name ``{source}__{code}{degree}`` keeps a source that holds ``__`` whole."""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._basis_name_codes import BASIS_NAME_CODES, source_of_engineered_name, split_engineered_name

_SOURCES = [
    "a",
    "city__NY",
    "div(exp((a__ratio_log__b)),sign(opat__a_b_c))",
    "add(div(invsqrt(b),exp(a__LL2)),mul(log(c),sin(d)))",
    "x__He2",
]


@pytest.mark.parametrize("code", BASIS_NAME_CODES)
@pytest.mark.parametrize("degree", [1, 3, 12])
def test_every_source_shape_round_trips_with_every_basis_code(code, degree):
    """split(f"{source}__{code}{degree}") is (source, f"{code}{degree}") for plain, one-hot and engineered sources."""
    for source in _SOURCES:
        name = f"{source}__{code}{degree}"
        assert split_engineered_name(name) == (source, f"{code}{degree}"), name
        assert source_of_engineered_name(name) == source, name


def test_the_name_from_the_warm_fit_log_is_split_at_its_basis_suffix():
    """The column the univariate, jmim, cmim and ksg builders skipped with 'cannot parse basis/degree'."""
    name = "div(exp((a__ratio_log__b)),sign(opat__a_b_c))__T3"
    assert split_engineered_name(name) == ("div(exp((a__ratio_log__b)),sign(opat__a_b_c))", "T3")
    assert name.split("__", 1)[1] != "T3", "the first-'__' split the builders used cut this name inside its source"


def test_a_name_without_a_basis_suffix_keeps_the_legacy_split():
    """No trailing code and degree: the first '__' decides as before, and a name with no '__' fails to unpack exactly as before."""
    assert split_engineered_name("city__NY__tag") == ("city", "NY__tag")
    assert source_of_engineered_name("plain") == "plain"
    with pytest.raises(ValueError):
        _src, _suffix = split_engineered_name("plain")  # the two-value unpacking is the point
