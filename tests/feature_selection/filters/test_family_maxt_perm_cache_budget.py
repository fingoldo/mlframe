"""The permutation-block cache budget of ``_family_maxt_gate`` comes from ``MLFRAME_PERM_BLOCK_CACHE_MB``, defaulting to 1024 MiB."""

import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as gate


@pytest.mark.parametrize("unset", [True])
def test_default_budget_is_1024_mib(monkeypatch, unset):
    """Without the env var the budget is 1024 MiB, one full permutation stream of a 100k-row family."""
    monkeypatch.delenv("MLFRAME_PERM_BLOCK_CACHE_MB", raising=False)
    assert gate._perm_block_cache_budget_bytes() == 1024 * 1024 * 1024


@pytest.mark.parametrize("mb", ["7", "0"])
def test_env_var_sets_the_budget_in_mib(monkeypatch, mb):
    """The env var is read as MiB, so 7 gives 7 MiB and 0 disables the cache."""
    monkeypatch.setenv("MLFRAME_PERM_BLOCK_CACHE_MB", mb)
    assert gate._perm_block_cache_budget_bytes() == int(mb) * 1024 * 1024
