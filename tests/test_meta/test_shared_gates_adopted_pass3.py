"""py-ci-shared gates adopted in the third adoption pass: the ML-correctness gates (NaN under fastmath, tied-rank scatter, group-blind splits,
uninitialised OOF buffers, estimator API contracts, unseeded generators, loop-invariant heavy calls, njit-unsupported calls, truncating
discretisation), the script and test hygiene gates (SystemExit in library code, definitions after an entry guard, duplicate test bodies,
state written before the work it gates, unwired optional collaborators, split CLI flag spellings) and the advisory caller-frame mutation scan.

Apart from the two baselined at the bottom, none has a baseline: the tree has no finding, so any new one fails. A deliberate exception carries the gate's own marker with a reason.
`_benchmarks` trees are developer scripts and are scanned by the gates whose defects are correctness bugs, not by the performance gate.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - reads this repo's own refs, fixed argv
import warnings
from pathlib import Path

import pytest

from py_ci_shared.caller_frame_mutation import assert_caller_frame_mutation
from py_ci_shared.checkout_resolution import assert_modules_resolve_to_checkout
from py_ci_shared.ci_pin_version_skew import assert_installed_matches_ci_pin
from py_ci_shared.cli_flag_spelling import assert_cli_flag_spelling
from py_ci_shared.comment_repo_paths import assert_comment_repo_paths
from py_ci_shared.commit_metadata import assert_commit_metadata
from py_ci_shared.definition_after_entry_guard import assert_definition_after_entry_guard
from py_ci_shared.duplicate_test_body import assert_no_duplicate_test_bodies
from py_ci_shared.git_child_env import assert_no_new_unscrubbed_git_spawns
from py_ci_shared.group_blind_split import assert_group_blind_split
from py_ci_shared.hardcoded_token_ceilings import assert_no_hardcoded_token_ceilings
from py_ci_shared.loop_invariant_cast import assert_loop_invariant_cast
from py_ci_shared.ml_api_contracts import assert_ml_api_contracts
from py_ci_shared.module_state_test_reset import assert_module_state_test_reset
from py_ci_shared.njit_fastmath_nan import assert_njit_fastmath_nan
from py_ci_shared.njit_unsupported_calls import assert_njit_unsupported_calls
from py_ci_shared.oof_buffer_zero_init import assert_oof_buffer_zero_init
from py_ci_shared.optional_collaborator_wired import assert_optional_collaborator_wired
from py_ci_shared.order_losing_filters import assert_no_order_losing_filters
from py_ci_shared.ordinal_rank_scatter import assert_ordinal_rank_scatter
from py_ci_shared.patch_target_read_at_call import assert_patch_target_read_at_call
from py_ci_shared.predictable_tmp_path import assert_predictable_tmp_path
from py_ci_shared.script_entry_points import assert_script_entry_points
from py_ci_shared.skip_as_pass import assert_no_new_skip_as_pass
from py_ci_shared.state_before_work import assert_state_before_work
from py_ci_shared.systemexit_in_library import assert_no_systemexit_in_library
from py_ci_shared.tracked_secret_shapes import assert_no_tracked_secret_shapes
from py_ci_shared.truncating_discretisation import assert_truncating_discretisation
from py_ci_shared.unseeded_rng_fallback import assert_unseeded_rng_fallback
from py_ci_shared.verified_then_reopened import assert_verified_then_reopened
from py_ci_shared.wallclock_test_waits import assert_no_new_wallclock_test_waits

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "mlframe"
TESTS = REPO_ROOT / "tests"
HERE = Path(__file__).resolve().parent

# Ceiling of the scanned population: a gate that sees fewer files than this has lost its root and would pass on nothing.
MIN_SRC_FILES = 1000
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}  # a hook's GIT_DIR must not steer the child git


def test_njit_fastmath_kernels_keep_nan_and_inf_visible():
    """`fastmath=True` lets LLVM assume no NaN/inf and fold `isfinite` / `v != v` to True; a kernel that tests or returns NaN needs a NaN-safe flag set."""
    assert_njit_fastmath_nan(SRC, min_files=MIN_SRC_FILES)


def test_ranks_are_not_built_by_ordinal_scatter():
    """`ranks[argsort(x)] = arange(n)` splits tied values by input order and ranks NaN highest; use `rankdata(method='average')`."""
    assert_ordinal_rank_scatter(SRC, min_files=MIN_SRC_FILES)


def test_functions_given_groups_do_not_split_without_them():
    """A function that receives group labels and then builds a KFold or train_test_split without them leaks a group across the split."""
    assert_group_blind_split(SRC, min_files=MIN_SRC_FILES)


def test_oof_buffers_are_not_zero_initialised():
    """An out-of-fold buffer filled per fold from a splitter that may skip rows must start as NaN, or the skipped rows read as predictions."""
    assert_oof_buffer_zero_init(SRC, min_files=MIN_SRC_FILES)


def test_estimator_api_contracts_hold():
    """`best_iteration` is an index not a count, `argmax` is a column not a label, and an `__init__` keeps its parameters as given."""
    assert_ml_api_contracts(SRC, min_files=MIN_SRC_FILES)


def test_missing_seed_is_not_turned_into_os_entropy():
    """A function that turns a missing seed into `default_rng()` makes the run irreproducible; derive it from a seed or say why entropy is wanted."""
    assert_unseeded_rng_fallback(SRC, min_files=MIN_SRC_FILES)


def test_no_loop_recomputes_a_loop_invariant_cast_or_sort():
    """A cast, copy, sort, unique, quantile or bincount whose operands never change inside the loop is hoisted above it."""
    assert_loop_invariant_cast(SRC, heavy=True, exclude=("_benchmarks/",), min_files=MIN_SRC_FILES)


def test_compiled_bodies_call_nothing_numba_or_torchscript_rejects():
    """`typing.cast`, `np.median`, logging and `warnings.warn` raise TypingError on another numba build, far from the edit."""
    assert_njit_unsupported_calls(SRC, min_files=MIN_SRC_FILES)


def test_distinct_count_guard_does_not_license_an_integer_cast():
    """`if nunique <= 32: y = y.astype(int)` collapses a fractional target to one class."""
    assert_truncating_discretisation(SRC, min_files=MIN_SRC_FILES)


def test_library_code_does_not_exit_the_interpreter():
    """`sys.exit()` inside a function that threads and importers reach kills the host process."""
    assert_no_systemexit_in_library(SRC, min_files=MIN_SRC_FILES)


def test_no_definition_follows_the_script_entry_guard():
    """A def after `if __name__ == '__main__'` does not exist yet when the guard body runs as a script."""
    assert_definition_after_entry_guard(SRC, min_files=MIN_SRC_FILES)


def test_no_two_tests_in_a_file_share_a_body():
    """A copy of a test under a new name pins nothing new; edit it to test the case its name promises."""
    assert_no_duplicate_test_bodies(TESTS, min_files=MIN_SRC_FILES)


def test_progress_state_is_written_after_the_work_it_gates():
    """A stamp or counter written before the fallible work it records claims progress that never happened."""
    assert_state_before_work(SRC, min_files=MIN_SRC_FILES)


def test_optional_collaborators_are_wired_by_a_production_caller():
    """A `*_memo`, `*_cache` or `*_fn` parameter that no production call site passes is an unreachable branch."""
    assert_optional_collaborator_wired(SRC, min_files=MIN_SRC_FILES)


def test_cli_flags_use_one_spelling():
    """The same flag registered as `--a-b` in one script and `--a_b` in another makes a copied command line fail."""
    assert_cli_flag_spelling(REPO_ROOT, min_files=MIN_SRC_FILES)


def test_caller_frame_mutation_scan_runs_and_only_warns():
    """Advisory: a function that mutates a DataFrame parameter in place warns, it never fails the suite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_caller_frame_mutation(SRC, advisory=True, min_files=MIN_SRC_FILES)


def test_predictable_tmp_paths_carry_a_per_user_component():
    """A fixed name under the shared temp directory can be pre-created by another local user."""
    assert_predictable_tmp_path(SRC, min_files=MIN_SRC_FILES)


def test_no_order_losing_filter_selects_rows_by_an_index_built_mask():
    """A row selection by an index-built mask returns rows in index order where its positional twin keeps the caller's order."""
    assert_no_order_losing_filters(sorted(SRC.rglob("*.py")), REPO_ROOT, min_files=MIN_SRC_FILES)


def test_scripts_register_no_module_and_do_not_put_the_cwd_on_sys_path():
    """A flat script imported by bare name registers itself in `sys.modules`, and a script that puts the cwd on `sys.path` imports whatever sits there."""
    assert_script_entry_points(REPO_ROOT, [REPO_ROOT / "scripts", REPO_ROOT / "tools"])


def test_comments_name_files_that_exist():
    """A comment that points at a `scripts/` or `tools/` path names a file that exists."""
    assert_comment_repo_paths(REPO_ROOT)


def test_skip_as_pass_scripts_do_not_report_success():
    """A script that prints SKIPPED and exits 0 reports a pass nobody earned."""
    assert_no_new_skip_as_pass(REPO_ROOT)


def test_tests_that_spawn_git_scrub_the_hook_environment():
    """Under a git hook `GIT_DIR` points at the committing repository; a test that spawns git without scrubbing it runs against that repository."""
    assert_no_new_unscrubbed_git_spawns(REPO_ROOT)


def test_a_verified_file_is_not_reopened_to_load_it():
    """Verifying a sidecar by path and then opening the path again lets a replaced file be loaded unverified; verify the bytes that get loaded."""
    assert_verified_then_reopened(SRC, min_files=MIN_SRC_FILES)


def test_tests_do_not_wait_on_the_wall_clock():
    """A real sleep makes the suite slower on every run and flaky under load; inject a clock or wait on the event the test is after."""
    assert_no_new_wallclock_test_waits(REPO_ROOT, python_mode="ast")


def test_no_output_token_ceiling_is_hardcoded():
    """An LLM output ceiling chosen by eye truncates a long answer silently."""
    assert_no_hardcoded_token_ceilings(SRC, min_files=MIN_SRC_FILES)


def test_ci_pin_version_matches_the_installed_py_ci_shared_release():
    """The release the workflows pin and the one the local hooks run must not drift apart."""
    assert_installed_matches_ci_pin(REPO_ROOT)


def test_the_mlframe_under_test_is_this_checkout():
    """The suite must examine the code in this checkout, not a copy installed somewhere else."""
    assert_modules_resolve_to_checkout(REPO_ROOT, ["mlframe"])


def test_no_tracked_file_holds_a_vendor_token_shape():
    """A vendor token committed once stays in every clone; the shapes are configured in `[tool.py_ci_shared.secret_shapes]`."""
    assert_no_tracked_secret_shapes(REPO_ROOT, config_path=REPO_ROOT / "pyproject.toml", min_files=1000)


def _unpushed_range() -> str | None:
    """`origin/master..HEAD` when that ref exists, else None (a shallow CI checkout has no remote-tracking master)."""
    done = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "origin/master"], cwd=REPO_ROOT, capture_output=True, text=True, env=_GIT_ENV)  # nosec B603 B607
    return "origin/master..HEAD" if done.returncode == 0 else None


def test_unpushed_commits_carry_no_forbidden_trailer_or_invisible_subject():
    """A commit about to be pushed must not carry a Co-Authored-By trailer or a subject that starts with a BOM or invisible character."""
    rev_range = _unpushed_range()
    if rev_range is None:
        pytest.skip("no origin/master ref to bound the unpushed range")
    assert_commit_metadata(REPO_ROOT, rev_range=rev_range, forbidden_trailers=["Co-Authored-By"])


# The two gates below carry a shrink-only baseline: each has more than fifty mechanical hits that are paid down by name in
# audits/2026-11-05/_TRACKER.md. A new name fails; a fixed name must be removed from the baseline.
def test_module_level_mutable_state_has_a_test_side_reset():
    """Process-lifetime module state that a function mutates and no test ever mentions has no reset between tests."""
    assert_module_state_test_reset(SRC, TESTS, baseline_path=HERE / "_module_state_test_reset_baseline.json", min_files=MIN_SRC_FILES)


def test_monkeypatches_target_the_binding_the_code_under_test_reads():
    """Patching a name a module only re-exports changes nothing: the code under test keeps its own binding."""
    assert_patch_target_read_at_call([TESTS], [SRC], baseline_path=HERE / "_patch_target_read_at_call_baseline.json", min_files=MIN_SRC_FILES)
