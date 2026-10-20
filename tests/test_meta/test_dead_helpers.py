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
    # Vendored infonet demo, run by hand as a sanity check of the sliced-MI estimator on independent samples.
    "example_highd",
}

# Helpers the maintainer surfaced and explicitly deferred deletion on
# (orphaned by static grep but kept around pending decision). Each entry
# is "module-relative-path:lineno::Name" — match what the failure message
# emits so it's a copy-paste from the failure into this set. Drain to zero
# over time. Lineno is stripped before the comparison so re-numbering due
# to nearby edits doesn't break the whitelist.
_USER_DEFERRED_DEAD_HELPERS: set[str] = {
    # GPU appliers for the GPU-compatible transform names; agreement with the CPU registry is pinned in
    # tests/feature_selection/gpu/test_apply_gpu_batched_vs_cpu_registry.py. Production builds candidates with the fused resident kernels.
    "feature_selection/filters/feature_engineering.py::apply_gpu_unary_batched",
    "feature_selection/filters/feature_engineering.py::apply_gpu_binary_batched",
    # Joblib-outer and Besag-Clifford twins of the empirical-null permutation kernel. The prange twin (``parallel_mi_prange_with_null``) is the
    # one ``mi_direct`` dispatches; these stay for parity and are exercised by tests/feature_selection/biz_val/test_biz_val_filters_permutation.py.
    "feature_selection/filters/permutation.py::parallel_mi_with_null",
    "feature_selection/filters/permutation.py::parallel_mi_besag_clifford_with_null",
    # Naive per-cell reference of grok_mutual_information; parity pinned in tests/feature_selection/test_grok_mi_old_parity.py.
    "feature_selection/mi.py::grok_mutual_information_old",
    # Placeholder translator paired with the populated sklearn-MLP one; inert until ROBUST_RECURRENT_OVERRIDES_UNDER_DRIFT is filled in
    # (requires a sequence-DGP sweep that the MLP sweep does not transfer to). Documented in its own docstring.
    "training/feature_drift_report.py::translate_sklearn_mlp_overrides_to_recurrent_config_kwargs",
}


def _python_files() -> list[Path]:
    """Returns ``out`` (after 2 setup steps)."""
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
    """No dead public helpers."""
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

    assert total > 30, f"only {total} public symbols audited — class/function discovery broken?"
    if dead:
        pytest.fail(
            f"{len(dead)} public helper(s) with no non-test consumer "
            f"(neither another production module nor an __init__ re-export). "
            f"Either delete the helper, OR re-export it via __init__.py if "
            f"it's part of the public API, OR whitelist via "
            f"_PUBLIC_API_WHITELIST with reasoning:\n  " + "\n  ".join(dead[:50]) + (f"\n  ... and {len(dead) - 50} more" if len(dead) > 50 else "")
        )
