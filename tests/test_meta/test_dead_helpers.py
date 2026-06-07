"""Meta-test — internal-helper symbols inside ``mlframe/training/`` and
``mlframe/feature_selection/`` must be referenced by at least one *non-test*
consumer (another production module in the same sub-package, an
``__init__`` re-export, or a downstream module).

Catches the failure mode where a refactor leaves stale helpers behind:
the function still has a test (so coverage looks fine), but no production
caller — meaning the test is policing dead code. Symptoms include
slow CI on tests that don't gate any user-visible behaviour, and growing
helper-module bloat that makes new contributors mistake an abandoned
helper for the canonical one.

Scope is intentionally NARROW: the top-level mlframe modules
(``evaluation.py``, ``metrics.py``, ``FeatureEngineering.py`` etc.) are the
public-API surface — users import directly from those modules in notebooks
and downstream code, so static-grep would flag every public helper as "dead"
and the test would become noise. Only modules under ``training/`` and
``feature_selection/`` are *internal* by convention; helpers there should
be consumed internally or via an ``__init__`` re-export.

Heuristic: walk the AST of every in-scope module; collect every top-level
``def``/``class`` whose name does NOT start with ``_`` (private symbols
are intentionally module-internal); for each, grep the rest of the
non-test corpus for at least one reference. Misses are flagged.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

import mlframe
from pyutilz.dev.meta_test_utils import (
    consumer_corpus,
    public_top_level_symbols,
    strip_lineno,
)

MLFRAME_DIR = Path(mlframe.__file__).resolve().parent

# Only police modules under these sub-packages — they're the *internal*
# pipeline where helpers should be consumed by other production modules.
# The top-level ``mlframe/`` directory and other folders contain public-API
# modules that users import directly from notebooks.
_IN_SCOPE_DIRS = ("training", "feature_selection")

_SKIP_PATH_FRAGMENTS = (
    "tests",
    "__pycache__",
    "legacy",
    # ``_benchmarks`` holds standalone dev benches + bench-local test_* probes
    # (e.g. ``fs_hybrid/test_hybrid_tree_member.py``) that are test-adjacent by
    # design -- their ``def test_*`` / ``def bench_*`` bodies have no production
    # consumer because they ARE the consumers. Treated like ``tests`` so the
    # dead-helper sensor scopes to the real internal pipeline. Mirrors the
    # ``_benchmarks`` exemption in ``test_no_underscore_imports_cross_package``.
    "_benchmarks",
)
_SKIP_FILENAME_PREFIXES = ("bench_", "profile_", "_")
_SKIP_FILENAMES = {
    "__init__.py",
    "__main__.py",
    "version.py",
    "synthetic.py",  # data-gen for tests/notebooks; standalone by design
}

# Hard whitelist for symbols intentionally part of the public API but used
# only by external consumers (notebooks, downstream packages). Cite reason.
_PUBLIC_API_WHITELIST: set[str] = {
    # Tracked classes/functions re-exported from mlframe.training.__init__
    # are matched dynamically below; only add here for non-trainer modules.
}

# Helpers the maintainer surfaced and explicitly deferred deletion on
# (orphaned by static grep but kept around pending decision). Each entry
# is "module-relative-path:lineno::Name" — match what the failure message
# emits so it's a copy-paste from the failure into this set. Drain to zero
# over time. Lineno is stripped before the comparison so re-numbering due
# to nearby edits doesn't break the whitelist.
_USER_DEFERRED_DEAD_HELPERS: set[str] = {
    "feature_selection/filters.py::init_kernels",
    "feature_selection/filters.py::find_impactful_features",
    "feature_selection/filters.py::create_redundant_continuous_factor",
    "feature_selection/filters.py::discretize_sklearn",
    "feature_selection/mi.py::grok_mutual_information_old",
    "feature_selection/optbinning.py::get_binningprocess_featureselectors",
    "training/phases.py::record_phase",
    "training/phases.py::phase_snapshot",
    # Surfaced 2026-04-28 after the corpus heuristic switched to the
    # shared ``pyutilz.dev.meta_test_utils.consumer_corpus`` (which
    # correctly excludes ``tests/``). These two are called only from
    # ``tests/training/test_per_class_isotonic.py`` — either move them
    # into the test module as fixtures, or document as public API.
    "training/metrics_registry.py::unregister_metric",
    "training/metrics_registry.py::list_registered",
    # Surfaced 2026-05-10 by the filters/* package refactor's full-suite
    # gate. Both are pre-existing (unrelated to filters.py) — refactor PR
    # surfaces them but does not own them. Owners should either move into
    # test fixtures or document as public.
    "training/composite.py::report_to_markdown",
    "training/phases.py::phase_ram_snapshot",
    # 2026-05-15 — surfaced after the src/ migration normalised the corpus
    # search root (rglob now finds these in src/mlframe/* instead of root).
    # Same pattern as the older entries: orphaned helpers slated for either
    # public-API promotion or deletion.
    "feature_selection/boruta_shap.py::load_data",
    "feature_selection/importance.py::compute_permutation_importances",
    "feature_selection/importance.py::explain_top_feature_importances",
    "feature_selection/filters/composition.py::compose_pair_fe",
    "feature_selection/filters/estimators.py::ksg_mi_pair",
    "feature_selection/filters/feature_engineering.py::apply_gpu_unary_batched",
    "feature_selection/filters/feature_engineering.py::apply_gpu_binary_batched",
    "feature_selection/filters/hermite_fe.py::optimise_pair_multimode",
    "feature_selection/filters/screen.py::ScreenState",
    "feature_selection/filters/supervised_binning.py::apply_bin_edges",
    "training/baselines/dummy.py::plot_best_dummy_baseline_overlay",
    # 2026-05-21 — surfaced after the recent refactors split helpers
    # into new sibling modules. Pending owner decision (delete vs.
    # public-API re-export).
    "feature_selection/filters/hermite_fe.py::plugin_mi_classif_batch_dispatch",
    "training/extractors.py::FeaturesAndTargetsExtractorProtocol",
    "training/metrics_registry.py::get_metric_direction",
    "training/neural/base.py::suppress_lightning_workers_warning",
    # Placeholder translator paired with the populated sklearn-MLP one;
    # stays inert until ROBUST_RECURRENT_OVERRIDES_UNDER_DRIFT is filled
    # in (requires a sequence-DGP sweep that the MLP sweep does not
    # transfer to). Documented in its own docstring.
    "training/feature_drift_report.py::translate_sklearn_mlp_overrides_to_recurrent_config_kwargs",
    # 2026-05-30 -- Wave 8 (JMIM / BUR) intermediate dispatcher. Currently
    # only consumed by the JMIM aggregator switching in evaluate_gain via
    # thread-local; the helper exists for future SU-aware site callers that
    # cannot read the thread-local in the @njit kernel hot path. Surface as
    # public API once at least one external caller materialises.
    "feature_selection/filters/info_theory.py::mi_or_su",
    # 2026-06-05 -- empirical-null permutation kernels added alongside the
    # ``mi_direct(return_null_mean=True)`` path. The prange twin
    # (``parallel_mi_prange_with_null``) is the one ``mi_direct`` currently
    # dispatches; these two are the joblib-outer and Besag-Clifford twins kept
    # for parity so a future ``return_null_mean`` wiring of the ``outer`` / ``bc``
    # parallelism modes reuses the same accumulate-sum-of-perm-MI contract rather
    # than re-deriving it. Surface as consumed once those modes route null means.
    "feature_selection/filters/permutation.py::parallel_mi_with_null",
    "feature_selection/filters/permutation.py::parallel_mi_besag_clifford_with_null",
    # Bench-populate helper for the polyeval CPU-backend ParamOracle: callable on
    # demand to seed the oracle with measured njit-vs-njit_par crossovers; no
    # production caller by design (the oracle is consulted, not benched, in prod).
    "feature_selection/filters/hermite_fe.py::benchmark_polyeval_cpu_backends",
}


def _python_files() -> list[Path]:
    out: list[Path] = []
    for py in MLFRAME_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _SKIP_PATH_FRAGMENTS):
            continue
        # Only in-scope sub-packages.
        rel_parts = py.relative_to(MLFRAME_DIR).parts
        if not rel_parts or rel_parts[0] not in _IN_SCOPE_DIRS:
            continue
        if py.name in _SKIP_FILENAMES:
            continue
        if any(py.name.startswith(p) for p in _SKIP_FILENAME_PREFIXES):
            continue
        out.append(py)
    return out


# Symbol enumeration / corpus assembly / lineno-strip moved to
# ``pyutilz.dev.meta_test_utils`` so the same logic is shared with
# pyutilz's own meta-test suite.


def _reexport_set(init_path: Path) -> set[str]:
    """Names re-exported via ``__all__`` or simple ``from .X import Y`` lines
    in an ``__init__.py``. Counts as production consumption."""
    if not init_path.exists():
        return set()
    try:
        src = init_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    out: set[str] = set()
    # __all__ entries
    m = re.search(r"__all__\s*=\s*\[(.*?)\]", src, flags=re.DOTALL)
    if m:
        out.update(re.findall(r"['\"]([A-Za-z_]\w*)['\"]", m.group(1)))
    # ``from .X import Y, Z``
    for m in re.finditer(r"from\s+\.\S+\s+import\s+([^\n#]+)", src):
        for name in re.split(r"[,\s]+", m.group(1).strip()):
            name = name.strip("()")
            if name and name.isidentifier() and not name.startswith("_"):
                out.add(name)
    # mlframe.training/__init__.py uses a TYPE_CHECKING-aware lazy-import
    # mapping: ``'<exposed_name>': ('.<module>', '<actual_name>')``. Both
    # sides count as re-exported.
    for m in re.finditer(r"['\"]([A-Za-z_]\w*)['\"]\s*:\s*\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]([A-Za-z_]\w*)['\"]\s*\)", src):
        out.add(m.group(1))
        out.add(m.group(2))
    return out


def test_no_dead_public_helpers():
    init_paths = list(MLFRAME_DIR.rglob("__init__.py"))
    reexports: set[str] = set()
    for init in init_paths:
        reexports.update(_reexport_set(init))

    files = _python_files()
    assert files, "no production .py files found — package layout broken?"

    dead: list[str] = []
    total = 0
    corpus = consumer_corpus(MLFRAME_DIR)

    # Tokenise once, lookup per symbol. Replaces an O(N_symbols * len(corpus))
    # regex sweep -- previously took ~60s and tripped pytest's per-test timeout
    # on Windows. Identifier grammar matches Python's: a leading letter or
    # underscore followed by word chars. ``\bname\b`` boundaries are subsumed.
    token_counts = collections.Counter(re.findall(r"[A-Za-z_]\w*", corpus))

    for path in files:
        symbols = public_top_level_symbols(path)
        if not symbols:
            continue
        for name, lineno in symbols:
            total += 1
            if name in _PUBLIC_API_WHITELIST or name in reexports:
                continue
            # The definition line itself contributes 1 reference; ≥ 2 means
            # "called by something somewhere" (own-module callers, other
            # modules, or recursion all count). Single-occurrence ⇒ defined
            # but never called.
            if token_counts.get(name, 0) >= 2:
                continue
            rel = path.relative_to(MLFRAME_DIR)
            entry = f"{rel}:{lineno}::{name}"
            if strip_lineno(entry) in _USER_DEFERRED_DEAD_HELPERS:
                continue
            dead.append(entry)

    assert total > 30, (
        f"only {total} public symbols audited — class/function discovery broken?"
    )
    if dead:
        pytest.fail(
            f"{len(dead)} public helper(s) with no non-test consumer "
            f"(neither another production module nor an __init__ re-export). "
            f"Either delete the helper, OR re-export it via __init__.py if "
            f"it's part of the public API, OR whitelist via "
            f"_PUBLIC_API_WHITELIST with reasoning:\n  " + "\n  ".join(dead[:50])
            + (f"\n  ... and {len(dead) - 50} more" if len(dead) > 50 else "")
        )
