"""Regression: the ``mlframe.signal`` subpackage must not break stdlib ``signal``.

When ``src/mlframe`` is on ``sys.path`` (e.g. a script whose cwd is inside the
package) a bare ``import signal`` from anywhere in the process -- including
stdlib ``multiprocessing.resource_tracker`` reached via joblib/loky during
``import sklearn`` -- resolves to this subpackage instead of the standard
library module. Historically that raised
``AttributeError: module 'signal' has no attribute 'SIGINT'`` and broke
``from mlframe.metrics import fast_calibration_report``.

The subpackage now transparently re-exports stdlib ``signal``'s public names, so
``signal.SIGINT`` keeps resolving even when this package wins the import.
"""

import os
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys

# repo_root/tests/signal_tests/this_file -> repo_root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_MLFRAME = os.path.join(_REPO_ROOT, "src", "mlframe")


def _run(code: str, cwd: str):
    """Returns ``subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env, capture_output=True, tex...`` (after 2 setup steps)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_metrics_import_with_cwd_inside_package():
    """The exact historical repro: cwd inside ``src/mlframe`` where the shadow bites."""
    proc = _run("from mlframe.metrics import fast_calibration_report; print('OK')", cwd=_SRC_MLFRAME)
    assert proc.returncode == 0, f"import failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert "OK" in proc.stdout
    assert "has no attribute 'SIGINT'" not in proc.stderr


def test_shadowing_signal_still_exposes_stdlib_names():
    """When the subpackage shadows stdlib ``signal``, ``SIGINT``/``SIGTERM`` must still resolve."""
    proc = _run("import signal; print(int(signal.SIGINT), int(signal.SIGTERM))", cwd=_SRC_MLFRAME)
    assert proc.returncode == 0, f"stdlib signal names missing:\nSTDERR:{proc.stderr}"
    assert proc.stdout.split(), proc.stdout
