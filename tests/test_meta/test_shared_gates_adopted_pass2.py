"""py-ci-shared gates adopted in the second adoption pass: atomic-write staging, day-boundary clocks, hash-key determinism, LF writes,
machine-specific paths, pickle state, plotly annotation loops, polars null equality, re-iterated iterables, stdlib json, stub parity,
vendored imports, wrapper protocols, runner labels, committed line endings, API floor, CI install coverage and addopts path runs.

Gates without a baseline are zero-tolerance: the tree has no finding, so any new one fails. `_benchmarks` trees are developer scripts that
write to scratch drives and load their own pickles, so the gates about shipped-library hygiene skip them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from py_ci_shared.api_floor import assert_api_floor
from py_ci_shared.atomic_write_staging import assert_atomic_write_staging
from py_ci_shared.ci_default_branch_never_cancelled import assert_ci_default_branch_never_cancelled
from py_ci_shared.ci_install_covers_conftest import assert_ci_install_covers_conftest
from py_ci_shared.clock_day_boundary import assert_no_clock_day_boundary
from py_ci_shared.committed_line_endings import assert_committed_line_endings
from py_ci_shared.coverage_config_parity import assert_coverage_config_parity
from py_ci_shared.drifted_duplicate_literals import RULE_SET, assert_no_drifted_duplicate_literals
from py_ci_shared.gate_population_canary import (
    assert_canary_is_matched,
    assert_every_gate_declares_its_population,
    assert_population_is_not_empty,
    gate_modules,
    gate_canaries,
)
from py_ci_shared.hash_key_determinism import assert_hash_keys_are_deterministic
from py_ci_shared.id_keyed_cache_validates_identity import assert_id_keyed_cache_validates_identity
from py_ci_shared.lf_file_writes import assert_no_crlf_writes
from py_ci_shared.machine_specific_paths import assert_no_machine_specific_paths
from py_ci_shared.persisted_negative_probe import assert_persisted_negative_probe
from py_ci_shared.pickle_state_completeness import assert_no_pickle_state_gaps
from py_ci_shared.plotly_annotation_loop import assert_no_plotly_annotation_loops
from py_ci_shared.polars_null_equality import assert_polars_null_equality
from py_ci_shared.pytest_addopts_path_runs import assert_path_runs_select_tests
from py_ci_shared.reiterated_iterable_params import assert_no_reiterated_iterable_params
from py_ci_shared.stdlib_json_ban import assert_no_stdlib_json
from py_ci_shared.stub_signature_parity import assert_stub_signature_parity
from py_ci_shared.unresolved_module_attributes import assert_no_unresolved_module_attributes
from py_ci_shared.vendored_internal_imports import assert_no_vendored_internal_imports
from py_ci_shared.workflow_runner_labels import assert_workflow_runner_labels_pinned
from py_ci_shared.wrapper_protocol_parity import assert_wrapper_protocol_parity

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SRC = SRC_ROOT / "mlframe"
TESTS = REPO_ROOT / "tests"
HERE = Path(__file__).resolve().parent

DEV_SCRIPT_DIRS = ("_benchmarks", "benchmarks", "profiling", "legacy", "explore", "research", "scripts")


def test_atomic_write_staging_is_unique_per_writer():
    """A fixed `.tmp` staging name lets two writers interleave into one file; the staging path must be unique per writer."""
    assert_atomic_write_staging(root=[SRC_ROOT], min_files=1000)


def test_no_clock_day_boundary_in_tests():
    """A test comparing against `today()` fails once a day at midnight; the clock must be injected."""
    assert_no_clock_day_boundary(tests_root=TESTS, min_files=1000)


def test_hash_keys_are_deterministic_across_processes():
    """A cache key built from `hash()` of a str or tuple changes with PYTHONHASHSEED and misses every run."""
    assert_hash_keys_are_deterministic(root=[SRC_ROOT], min_files=1000)


def test_files_are_written_with_lf_line_endings():
    """`open(..., "w")` on Windows writes CRLF into files that are later hashed or diffed; write bytes or pass newline."""
    assert_no_crlf_writes(root=SRC_ROOT)


def test_no_machine_specific_paths_in_the_repo():
    """A pasted `D:/...` or `C:\\Users\\...` location works on one machine only; resolve it at runtime."""
    assert_no_machine_specific_paths(root=REPO_ROOT, skip_dir_names=DEV_SCRIPT_DIRS, min_files=1000)


def test_pickle_state_is_complete():
    """A `__getstate__` that drops an attribute `__setstate__` never rebuilds leaves a half-restored object after a load."""
    assert_no_pickle_state_gaps(root=[SRC_ROOT], min_files=1000)


def test_plotly_annotations_are_batched():
    """`fig.add_annotation` in a loop re-validates the growing tuple each call (quadratic); assign the batch once."""
    assert_no_plotly_annotation_loops(root=SRC, skip_dir_names=("_benchmarks",), min_files=1000)


def test_polars_null_comparisons_use_eq_missing():
    """`==` on a polars column yields null for a null operand, which a filter then drops silently."""
    assert_polars_null_equality(root=SRC, advisory=False, min_files=1000)


def test_iterable_params_are_not_iterated_twice():
    """A parameter documented as an iterable that is walked twice yields nothing the second time for a generator."""
    assert_no_reiterated_iterable_params(root=SRC, min_files=1000)


# Each path is a file whose JSON shape is fixed by a stored digest or by an unavailable orjson feature.
STDLIB_JSON_ALLOWED = {
    "mlframe/data/datasets/spec.py": "the dataset spec digest is computed over stdlib's ensure_ascii form; orjson would change every stored id",
    "mlframe/training/composite/cache.py": "the composite cache key digest is computed over stdlib's default float and ascii form",
    "mlframe/training/composite/provenance.py": "the composite_id digest is computed over stdlib's default float and ascii form",
    "mlframe/training/_canonical_json.py": "the non-orjson branch of the canonical encoder must match stdlib's allow_nan=False output",
    "mlframe/training/cb/_cb_gpu_budget.py": "raw_decode reads one JSON object out of a binary snapshot blob; orjson has no partial decode",
}


def test_no_stdlib_json_outside_the_digest_paths():
    """Stdlib json is several times slower than orjson; only the files whose output feeds a stored digest may keep it."""
    assert_no_stdlib_json(root=SRC_ROOT, include_tests=False, skip_dir_names=("_benchmarks",), allow=STDLIB_JSON_ALLOWED, min_files=1000)


def test_stub_signatures_match_the_callables_they_replace():
    """A test stub that drops a parameter of the real callable keeps passing after the callable grows a caller of it."""
    assert_stub_signature_parity(root=TESTS, min_files=1000)


def test_no_vendored_copy_is_imported():
    """`joblib.externals.*` is a private copy of a standalone package; an import must name the standalone one or say why not."""
    assert_no_vendored_internal_imports(root=SRC, first_party=("mlframe",), min_files=1000)


def test_wrapper_protocols_forward_every_member():
    """A wrapper that implements a protocol must forward each member the wrapped object exposes."""
    assert_wrapper_protocol_parity(root=SRC, min_files=1000)


def test_workflow_runner_labels_are_pinned():
    """`ubuntu-latest` and `windows-latest` move when GitHub re-points them; pin the image a workflow runs on."""
    assert_workflow_runner_labels_pinned(REPO_ROOT, min_files=5)


@pytest.mark.slow
def test_api_floor_matches_requires_python():
    """An API newer than `requires-python` crashes on the oldest supported interpreter."""
    assert_api_floor(root=SRC_ROOT, pyproject=REPO_ROOT / "pyproject.toml", min_files=1000)


def test_coverage_config_is_consistent_across_ci():
    """The coverage `fail_under`, `source` and `omit` that CI applies must be the ones the config declares."""
    assert_coverage_config_parity(repo_root=REPO_ROOT)


# The GPU matrix installs from a run-time extras list that the gate cannot evaluate statically.
CI_INSTALL_ACKNOWLEDGED = {
    "gpu-matrix.yml::gpu-tests": "installs the [all,dev] extras plus a run-time CUDA extra and torch pin, so py-ci-shared arrives through [dev]",
}


def test_ci_install_covers_what_conftest_imports():
    """A CI job that runs pytest must install every distribution `conftest.py` imports at collection."""
    assert_ci_install_covers_conftest(REPO_ROOT, acknowledge=CI_INSTALL_ACKNOWLEDGED)


def test_ci_install_covers_what_entry_modules_import():
    """The extras a CI job installs must cover what the module it runs imports when it starts."""
    from py_ci_shared.ci_install_covers_entry_imports import assert_ci_install_covers_entry_imports

    assert_ci_install_covers_entry_imports(root=REPO_ROOT)


def test_default_branch_workflows_are_never_cancelled():
    """A push-to-master run that a newer push cancels leaves the earlier commit with no verdict."""
    assert_ci_default_branch_never_cancelled(REPO_ROOT)


def test_addopts_does_not_deselect_the_tests_a_hook_names():
    """`addopts` is prepended to every pytest run; a hook naming a test path must not have its `-m` deselect that path."""
    assert_path_runs_select_tests(REPO_ROOT, repo_root=REPO_ROOT, min_files=1000)


def test_id_keyed_caches_validate_identity():
    """A cache keyed by `id(obj)` returns another object's entry once the first is collected and its id reused."""
    assert_id_keyed_cache_validates_identity(root=SRC, min_files=1000)


def test_failed_probes_do_not_persist_a_negative_verdict():
    """A hardware probe that fails once must not write a verdict that outlives the process."""
    assert_persisted_negative_probe(root=SRC, min_files=1000)


def test_committed_files_keep_lf_line_endings():
    """A blob committed with CRLF diffs as a whole-file change on the next LF edit; the ratchet lists the ones that predate the gate."""
    assert_committed_line_endings(REPO_ROOT, baseline_path=HERE / "_committed_line_endings_baseline.json", min_files=1000)


def test_no_duplicated_literal_set_drifts_apart():
    """The same tuple of numbers written in several modules changes in one and not the others; each accepted group says why."""
    assert_no_drifted_duplicate_literals(
        root=SRC,
        rules=(RULE_SET,),
        skip_dir_names=("_benchmarks",),
        baseline_path=HERE / "_drifted_duplicate_literals_baseline.json",
        min_files=1000,
    )


def _is_case_insensitive_fs() -> bool:
    """Whether `Path` lookups ignore case here, which makes the gate resolve the class `MRMR` to the package directory `mrmr`."""
    return (SRC / "FEATURE_SELECTION").exists()


def test_module_attribute_reads_resolve():
    """`module.NAME` where the module no longer defines NAME fails only the code that reads it; the ratchet lists the stale readers.

    The test tree is scanned only on a case-sensitive file system: on a case-insensitive one the gate reads `MRMR._FIT_CACHE` as an
    attribute of the package directory `mrmr` and reports every use of the class.
    """
    roots = [SRC_ROOT] if _is_case_insensitive_fs() else [SRC_ROOT, TESTS]
    assert_no_unresolved_module_attributes(
        roots,
        resolve_roots=[SRC_ROOT],
        baseline_path=HERE / "_unresolved_module_attributes_baseline.json",
        min_files=1000,
    )


def test_every_baselined_gate_declares_its_population():
    """A gate that reports no offender must be able to show the files it scanned, or an empty scan reads as a clean tree."""
    assert_every_gate_declares_its_population(HERE)


OPTED_IN_GATES = [p for p in gate_modules(HERE) if "def _candidate_files" in p.read_text(encoding="utf-8")]


@pytest.mark.parametrize("path", OPTED_IN_GATES, ids=lambda p: p.name)
def test_each_gate_examined_something(path):
    """A gate whose candidate population is empty passes without looking at anything."""
    assert_population_is_not_empty(path)


@pytest.mark.parametrize(("path", "canary"), gate_canaries(HERE), ids=lambda v: getattr(v, "name", v))
def test_each_declared_canary_is_matched(path, canary):
    """A gate's positive control must still be matched by one of its patterns, or the pattern has gone inert."""
    assert_canary_is_matched(path, canary)
