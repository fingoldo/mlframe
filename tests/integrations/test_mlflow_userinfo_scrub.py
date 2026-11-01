"""The MLflow userinfo scrubber redacts a whole password, even one that contains "@", and stays linear on long text (NET2-02, RDS2-1).

The old pattern stopped at the first "@", so ``https://user:pa@ss@host`` left ``ss@`` in the log. Its scheme could also start inside  # pragma: allowlist secret
a long run of letters, which made the scan quadratic.
"""

import time

import pytest

pytest.importorskip("mlflow")

from mlframe.integrations.mlflow import _strip_userinfo


def test_password_containing_an_at_sign_is_fully_redacted():
    """Nothing of the password survives the scrub, and the host stays readable."""
    out = _strip_userinfo("cannot reach https://user:pa@ss@tracking.example.com/api")  # pragma: allowlist secret
    assert "pa" not in out and "ss" not in out
    assert out == "cannot reach https://***@tracking.example.com/api"


def test_plain_userinfo_is_still_redacted():
    """The ordinary case keeps working."""
    assert _strip_userinfo("https://u:secret@host/x") == "https://***@host/x"  # pragma: allowlist secret


def test_text_without_userinfo_is_unchanged():
    """A URL with no credentials passes through untouched."""
    assert _strip_userinfo("see https://host:5000/path") == "see https://host:5000/path"


def test_long_run_of_letters_is_scanned_in_linear_time():
    """A long letter run with no scheme separator is not rescanned from every position."""
    text = "a" * 300_000
    start = time.perf_counter()
    assert _strip_userinfo(text) == text
    assert time.perf_counter() - start < 2.0
