"""Meta-test: a test module must not mutate ``os.environ`` at import time.

Under ``pytest-xdist`` one worker process imports many test modules and then runs tests from all of them.
A module-level ``os.environ[...] = ...`` therefore applies to every test that worker runs afterwards, not
just to the file that wrote it - and which files share a worker depends on the distribution, so the damage
lands on a different, innocent test on each run. The symptom is a cross-file failure that disappears the
moment the victim is run alone, which is the most expensive kind of failure to chase.

``os.environ.setdefault`` is the worst form: it looks defensive, is silent when the variable is already
set, and is never restored. ``tests/feature_selection/biz_val/test_biz_val_hybrid_cooccur_clusterrep.py``
blanked ``CUDA_VISIBLE_DEVICES`` this way at import, which disabled the GPU for every later test in the
same worker. That stayed invisible while no production code read the variable live; once the GPU opt-out
became a hard contract, it turned into ~70 spurious GPU-test failures.

The fix is always ``monkeypatch.setenv`` inside the test or a fixture, which pytest restores. If a variable
genuinely must be set before an import that happens at module scope, do it in ``conftest.py`` with an
autouse fixture, or move the import inside the test.

The scan is ``py_ci_shared.import_side_effects``. Its baseline keys a site by file and statement text rather
than line number, so a comment added above one does not re-report it, and an entry nothing reproduces fails:
the old copy printed a drained site and kept it, leaving room for the next one.

Refresh with::

    pytest tests/test_meta/test_no_module_level_env_mutation_in_tests.py --refresh-module-env-mutation-baseline
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.import_side_effects import assert_no_new_import_time_env_mutations

_TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_module_env_mutation_baseline.json"


def test_no_new_module_level_env_mutation():
    """No test module gains an import-time ``os.environ`` mutation beyond the baseline, and no entry is stale."""
    # ~3400 test modules today; a scan that parsed fewer than 1000 has stopped reaching the tree.
    assert_no_new_import_time_env_mutations(_TESTS_DIR, _BASELINE_PATH, min_files=1000, refresh_flag="--refresh-module-env-mutation-baseline")
