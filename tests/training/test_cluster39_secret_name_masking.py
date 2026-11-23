"""Cluster 39 regression: identifier fields that merely contain the word token are not masked as secrets (audit 2026-11-20, finding 39-F42)."""

import pytest

from mlframe.training.feature_handling.providers import _is_secret_field


@pytest.mark.parametrize("name", ["pad_token", "eos_token", "cache_key", "token_type", "bos_token"])
def test_identifier_fields_are_not_secret(name):
    """Tokenizer role names and cache keys are identifiers and must not be masked (39-F42)."""
    assert _is_secret_field(name) is False


@pytest.mark.parametrize("name", ["api_key", "access_token", "password", "auth_token", "client_secret"])
def test_real_credentials_are_still_secret(name):
    """Control: genuine credential field names are still masked (39-F42)."""
    assert _is_secret_field(name) is True
