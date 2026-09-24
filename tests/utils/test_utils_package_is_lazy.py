"""Importing a leaf helper out of ``mlframe.utils`` must not drag the experiment-tracking stack in with it."""

import os
import subprocess
import sys
from pathlib import Path

import mlframe

# A fresh process per check: "was this module imported" is only answerable before anything else imports it. It must
# run against the same source tree as this session, not whatever copy happens to be installed.
_SRC = str(Path(mlframe.__file__).resolve().parents[1])


def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter against this source tree and return its stripped stdout."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([_SRC, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout.strip()


def test_importing_the_package_does_not_import_sqlalchemy():
    """The experiments module reaches SQLAlchemy through pyutilz.db; nothing that merely throttles a log should pay it.

    Only SQLAlchemy is asserted: ``redis`` arrives earlier and from elsewhere - ``mlframe/__init__`` imports
    ``pyutilz.system.gpu_dispatch`` to install the GPU-runtime trigger, and that package imports redis itself.
    """
    out = _run("import sys, mlframe.utils; print('sqlalchemy' in sys.modules)")
    assert out == "False", f"mlframe.utils pulled the experiments stack: {out}"


def test_the_log_throttle_name_is_the_function_not_its_module():
    """``log_throttle`` resolves to the callable, not to the submodule that shares its name."""
    out = _run("from mlframe.utils import log_throttle; print(callable(log_throttle))")
    assert out == "True"


def test_every_advertised_name_still_resolves():
    """Making the package lazy must not leave a name in ``__all__`` that no longer resolves."""
    out = _run("import mlframe.utils as u; missing = [n for n in u.__all__ if not hasattr(u, n)]; print(missing)")
    assert out == "[]", f"names in __all__ that no longer resolve: {out}"


def test_an_experiments_name_still_works_and_is_what_pulls_the_stack():
    """The negative control: asking for an experiments name still works, and is what imports the heavy stack."""
    out = _run("import sys; from mlframe.utils import create_experiment; print(callable(create_experiment), 'sqlalchemy' in sys.modules)")
    assert out == "True True"
