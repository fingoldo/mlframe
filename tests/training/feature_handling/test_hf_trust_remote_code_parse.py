"""``trust_remote_code`` read from a ``hf://`` URL query is parsed as a flag, not truth-tested as a string (NET2-01).

The URL query arrives as the text ``"False"``. ``bool("False")`` is ``True``, which made ``hf://org/model?trust_remote_code=False``
execute the hub repo's modeling code. Text is now matched against explicit spellings, and an unknown value is refused.
"""

import pytest

from mlframe.training.feature_handling.hf_provider import parse_trust_remote_code


@pytest.mark.parametrize("text", ["False", "false", "FALSE", "0", "no", "off", ""])
def test_false_spellings_from_a_url_query_do_not_trust_remote_code(text):
    """A false spelling in the URL query must read as False, never as the truthy string it looks like."""
    assert parse_trust_remote_code(text) is False


@pytest.mark.parametrize("text", ["True", "true", "1", "yes", "on"])
def test_true_spellings_still_opt_in(text):
    """The opt-in spellings keep working so an explicit request still trusts the repo."""
    assert parse_trust_remote_code(text) is True


def test_python_bools_pass_through():
    """Booleans from Python code are used as given."""
    assert parse_trust_remote_code(False) is False
    assert parse_trust_remote_code(True) is True


def test_unknown_value_is_refused_instead_of_guessed():
    """A value that is neither a boolean nor a known spelling raises rather than enabling remote code."""
    with pytest.raises(ValueError, match="trust_remote_code"):
        parse_trust_remote_code("maybe")
