"""Meta-test: a test must restore every ``os.environ`` write it makes.

A test function that assigns ``os.environ["X"] = "1"`` and never puts it back leaks the setting into every test that runs after it in the same
process. Which tests those are depends on ordering and on pytest-xdist's distribution, so the victim is a different innocent test each run and
it passes when run alone. Real case here: ``test_resident_gate_selection_identical_to_host`` set ``MLFRAME_FE_GPU_STRICT``,
``MLFRAME_FE_GPU_STRICT_RESIDENT`` and ``MLFRAME_CMI_GPU`` directly, so the "host" arm of ``test_conditional_perm_null_gpu_bit_identical_to_cpu``
ran in strict-GPU mode and failed 4 runs in 5.

``test_no_module_level_env_mutation_in_tests`` covers the import-time half; this covers the write inside a function. The scan is
``py_ci_shared.env_write_restore``. Fix a finding with ``monkeypatch.setenv`` / ``monkeypatch.delenv``, ``mock.patch.dict(os.environ, ...)``, a
yield-fixture that restores after the yield, or try/finally. Existing sites sit in the baseline; it only shrinks.

Refresh (shrink-only) with::

    PY_CI_SHARED_REFRESH=env_write_restore pytest tests/test_meta/test_tests_restore_environment_writes.py
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.env_write_restore import assert_env_write_restore

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_env_write_restore_baseline.json"


def test_no_new_unrestored_environment_write_in_tests(request):
    """No test function gains an unrestored ``os.environ`` write beyond the baseline, and no baseline entry is stale."""
    # ~3400 test modules today; a scan that parsed fewer than 1000 has stopped reaching the tree.
    assert_env_write_restore(_TESTS_DIR, baseline_path=_BASELINE_PATH, min_files=1000, request=request)
