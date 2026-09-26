"""A module-level function with no production call site enforces nothing.

A guard, a validator, a cleanup hook that nothing calls looks exactly like one that does: it is
defined, it is tested, it appears in review. The only difference is that whatever it enforces is not
enforced, and that difference shows up as the bug it was written to prevent rather than as anything
pointing here. A test calling it is not a production call site, and neither is an `__all__` entry or
a doctest -- that is how this class of dead control hides.

Baselined rather than gated: this repo has a large existing set, and the number that matters is
whether a NEW one appears. The baseline can only shrink -- an entry that gains a caller is drained
on the next refresh.

`ignore` takes bare names for the shapes this check cannot judge and must not guess at: entry points
the interpreter or a framework calls, and the lazy-module protocol.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "_uncalled_functions_baseline.json"

#: Called by the interpreter, a framework or the import system rather than by name.
#: `__getattr__`/`__dir__` are PEP 562 lazy-module hooks; `main` is a console entry point;
#: `upgrade`/`downgrade` are alembic's.
_INTERPRETER_INVOKED = {"__getattr__", "__dir__", "main", "upgrade", "downgrade"}

#: Documented public API: users call these, nothing inside the package has to. Each is exported (``__all__`` or a package
#: re-export) and covered by tests of its own.
_PUBLIC_API = {
    "ewma", "trimmed_mean", "overlap", "mrr",  # core / metrics utilities
    "feature_matrix", "to_polars", "indicator", "inject_matched_probes", "probe_false_discovery_rate",  # mlframe.data
    "replay_report", "spec_hash_for", "assert_oracle_agrees_with_dit", "reference_crosscheck",  # benchmark-bed tooling
    "create_aggregated_features", "optimize_pipeline_by_gridsearch", "get_full_classifier_name",
    "predict_mlframe_models_suite", "serving_base_from_columns", "blend",  # blend is exported as pseudo_bma_blend
}

#: Kernel versions kept deliberately when a faster one replaced them (the repository keeps every version for A/B and
#: fallback): the typed-list DWT pair, the DFS MDLP twin of the BFS default, the CUDA DTW diagonal step, the numba column-
#: stats kernel the benchmark compares against.
_RETAINED_KERNELS = {
    "_wavedec_numba_typedlist", "_waverec_numba_typedlist", "_mdlp_recurse_validated", "_numba_cuda_diagonal_step",
    "_col_stats_float_numba_kernel",
}

#: Called, but the pinned py-ci-shared (v1.17.0) cannot see it: a lazy import with an ``except ImportError: f = None``
#: fallback reads as a local shadow there. Fixed upstream in 9b48122; drop this entry when the pin moves past it.
_SCANNER_LAG = {"_fill_bf_batch_njit"}


def _production_files() -> list[Path]:
    """Shipped modules only. Benchmarks (``_benchmarks/`` and ``_bench_*`` helpers) keep deliberately-uncalled baselines."""
    return sorted(p for p in (REPO_ROOT / "src").rglob("*.py") if "_benchmarks" not in p.as_posix() and not p.name.startswith("_bench_"))


def test_no_new_function_without_a_caller(request):
    """Kills: a new guard or validator that nothing ever calls."""
    from py_ci_shared.uncalled_functions import assert_no_new_uncalled_function

    files = _production_files()
    assert len(files) > 1000, f"only {len(files)} production files found -- the scan lost its subject"

    assert_no_new_uncalled_function(files, REPO_ROOT, BASELINE, ignore=_INTERPRETER_INVOKED | _PUBLIC_API | _RETAINED_KERNELS | _SCANNER_LAG)
