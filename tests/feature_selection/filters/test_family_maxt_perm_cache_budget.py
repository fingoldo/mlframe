"""The permutation-block cache budget of ``_family_maxt_gate`` comes from ``MLFRAME_PERM_BLOCK_CACHE_MB``, defaulting to 1024 MiB.

The budget is read once at import, so each case imports the module in a fresh interpreter rather than reloading it in this process.
"""

import os
import subprocess
import sys

import pytest

_PROBE = "from mlframe.feature_selection.filters import _family_maxt_gate as g; print(g._PERM_BLOCK_CACHE_BYTES)"


def _budget_bytes(env_value):
    """Import the module in a child interpreter with the env var set (or unset when ``env_value`` is None) and return the byte budget."""
    env = {k: v for k, v in os.environ.items() if k != "MLFRAME_PERM_BLOCK_CACHE_MB"}
    if env_value is not None:
        env["MLFRAME_PERM_BLOCK_CACHE_MB"] = env_value
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True, timeout=600)
    return int(out.stdout.strip().splitlines()[-1])


def test_default_budget_is_1024_mib():
    """Without the env var the budget is 1024 MiB, one full permutation stream of a 100k-row family."""
    assert _budget_bytes(None) == 1024 * 1024 * 1024


@pytest.mark.parametrize("mb", ["7", "0"])
def test_env_var_sets_the_budget_in_mib(mb):
    """The env var is read as MiB, so 7 gives 7 MiB and 0 disables the cache."""
    assert _budget_bytes(mb) == int(mb) * 1024 * 1024
