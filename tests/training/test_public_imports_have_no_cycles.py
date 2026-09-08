"""The public training import paths must work on a fresh interpreter.

``mlframe.training.trainer`` imported ``_trainer_train_and_evaluate`` at the bottom of the module, which
imported ``_calib_oof_outputs``, which imported ``training.core`` -- whose package ``__init__`` loads the
whole suite and reaches back into ``trainer`` while it is still executing. Every one of these raised
``ImportError: cannot import name ... from partially initialized module``:

    import mlframe.training.trainer
    from mlframe.training import train_and_evaluate_model

The suite never saw it, and that is the interesting part: pytest's collection happens to import
``training.core`` first, so by the time anything reaches ``trainer`` the cycle is already resolved. Only a
process whose FIRST mlframe import is one of these hits it -- which is exactly what a user does, and what
the repo's own wheel smoke-test command does. It was found on 2026-09-08 by a CI job importing the public
surface from a clean environment, having been present in the source for at least as long as git history
shows those three import lines unchanged.

Each case runs in a subprocess on purpose: within one pytest process the modules are already imported, so
an in-process assertion would pass no matter what.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs this interpreter on source from this repo, no external input
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every entry point a consumer might make their first mlframe import. The last one is the exact command
# ci.yml's wheel smoke-test runs.
FIRST_IMPORTS = [
    "import mlframe.training.trainer",
    "from mlframe.training import train_and_evaluate_model",
    "from mlframe.training.core import train_mlframe_models_suite",
    "from mlframe.training.trainer import train_and_evaluate_model",
    (
        "from mlframe.metrics.core import *; "
        "from mlframe.training import train_and_evaluate_model; "
        "from mlframe.models.ensembling import *; "
        "from mlframe.calibration.quality import *"
    ),
]


def _env_with_src_first() -> dict:
    """A copy of the environment with this repo's ``src`` ahead of anything already on PYTHONPATH."""
    src_root = str(REPO_ROOT / "src")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([src_root, existing]) if existing else src_root
    return env


@pytest.mark.parametrize("statement", FIRST_IMPORTS, ids=lambda s: s[:48])
def test_public_import_works_as_the_first_import_of_a_process(statement: str):
    """A cycle only shows up when the module in it is the first thing imported."""
    proc = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-c", statement + "\nprint('IMPORT_OK')"],
        capture_output=True,
        text=True,
        timeout=300,
        # The working directory is the SOURCE root, not the repo root. `python -c` puts the working
        # directory first on sys.path, which is what makes the subprocess import the tree under test
        # rather than whatever mlframe the machine has installed -- a different checkout entirely when
        # running from a git worktree, and on this machine an editable install whose path does not even
        # contain `mlframe.training`. PYTHONPATH is set as well, but cannot be relied on alone: a
        # .pth-installed editable finder is a meta-path hook and takes precedence over sys.path entries.
        cwd=str(REPO_ROOT / "src"),
        env=_env_with_src_first(),
    )
    assert proc.returncode == 0 and "IMPORT_OK" in proc.stdout, (
        f"`{statement}` fails as the first import of a fresh interpreter. This is an import cycle: it will "
        f"not reproduce inside the test session, because collection imports the package in a different "
        f"order.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    )
