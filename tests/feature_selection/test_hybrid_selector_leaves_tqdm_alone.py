"""Importing the hybrid selector does not switch off the process's progress bars; its fit silences them only for the call (STATE5-1)."""

import os
import subprocess
import sys

from mlframe.feature_selection.hybrid_selector import _tqdm_off_during_call


def test_import_does_not_set_tqdm_disable():
    """In a fresh interpreter with the variable unset, importing the module leaves it unset."""
    env = {k: v for k, v in os.environ.items() if k != "TQDM_DISABLE"}
    code = "import os; import mlframe.feature_selection.hybrid_selector; print('TQDM_DISABLE' in os.environ)"
    done = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, timeout=240, check=False, env=env)
    assert done.stdout.split()[-1] == "False", done.stdout + done.stderr


def test_the_decorator_sets_the_variable_for_the_call_and_restores_it(monkeypatch):
    """Inside the call TQDM_DISABLE is 1; afterwards the previous value (or absence) is back, also when the call raises."""
    seen = []

    @_tqdm_off_during_call
    def call(fail=False):
        """Record the variable and optionally fail."""
        seen.append(os.environ.get("TQDM_DISABLE"))
        if fail:
            raise RuntimeError("x")

    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    call()
    assert seen == ["1"] and "TQDM_DISABLE" not in os.environ
    monkeypatch.setenv("TQDM_DISABLE", "0")
    try:
        call(fail=True)
    except RuntimeError:
        pass
    assert os.environ["TQDM_DISABLE"] == "0"
