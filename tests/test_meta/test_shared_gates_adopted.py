"""py-ci-shared gates adopted with 1.17.0: RNG seed range, sentinel reads, stale citations, cache locking, swallowed failures,
xfails that park fixable work. Import cycles run through the shared gate in test_no_import_cycles.py.

The zero-tolerance gates pass no baseline: the tree has no finding, so any new one fails. The ratchet keeps the findings
the tree already had, each with its reason in the baseline file; a new one fails and a fixed one must leave the baseline.
Refresh the ratchet with PY_CI_SHARED_REFRESH=xfail and write every new note by hand.
File floors sit well under the measured counts (src/mlframe about 2,650 parsed files, tests/ about 4,000).
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.module_cache_thread_safety import assert_thread_safe_module_caches
from py_ci_shared.no_xfail_to_defer import assert_no_xfail_to_defer
from py_ci_shared.numba_seed_range import assert_numba_seeds_fit_int64
from py_ci_shared.sentinel_or_fallback import assert_no_sentinel_or_fallback
from py_ci_shared.stale_source_citations import assert_no_stale_source_citations
from py_ci_shared.swallowed_exceptions import assert_no_swallowed_exceptions

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mlframe"
HERE = Path(__file__).resolve().parent


def test_numba_seeds_fit_the_seed_argument():
    """An entropy seed handed to numba must fit its int64 argument; a restore that overflows is swallowed and does nothing."""
    assert_numba_seeds_fit_int64(SRC, min_files=1000)


def test_no_sentinel_setting_read_through_or():
    """`max_tokens or default` turns a meaningful 0 into the default; read the declared settings with an explicit None check."""
    assert_no_sentinel_or_fallback(SRC, min_files=1000)


def test_source_citations_point_at_real_lines():
    """A `file.py:NNN` citation past the end of a split file sends the reader nowhere; cite the symbol instead."""
    assert_no_stale_source_citations(SRC, min_files=1000)


def test_module_caches_are_locked():
    """A module-level cache that inserts and evicts in one function needs a lock under threaded joblib and DataLoader workers."""
    assert_thread_safe_module_caches(SRC, min_files=1000)


def test_no_swallowed_io_or_broad_failures():
    """An `except OSError: pass` in production code hides the failure the caller needed to see."""
    assert_no_swallowed_exceptions(SRC, min_files=1000)


def test_no_new_xfail_that_defers_a_fix():
    """An xfail parks a known defect; each existing one names its gap in the baseline, and a new one must be a fix instead."""
    assert_no_xfail_to_defer(REPO_ROOT / "tests", repo_root=REPO_ROOT, baseline_path=HERE / "_xfail_baseline.json", min_files=3000)
