"""No production code imports another package's underscore-prefixed module.

Underscore-prefixed modules under any ``src/mlframe/<pkg>/`` directory are internal; the public surface is
whatever the package exposes under public names. Sibling files inside the same package may import each
other freely; production code elsewhere must not reach into ``other_pkg._private``, which couples two
subsystems to an implementation detail that breaks silently when either side moves a helper.

A helper other packages need is exposed by the owning package's ``shared.py`` (its cross-package API) under a
public name; the implementation stays in its home module and consumers import it from ``shared``, aliased back
to their local name where they used the private one.

`py_ci_shared.private_imports` does the scan: the owning package is the dotted path up to the first
underscore segment, an importer inside it or below it is a sibling, and ``_benchmarks/``, ``_profile_*`` and
``_bench_*`` files are test-adjacent and exempt. The narrower ``training.core``-only version of this check is
a special case of it and no longer exists separately.

An allowlist entry that stops occurring fails too, so the list only shrinks.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.private_imports import assert_no_private_cross_package_imports

REPO_ROOT = Path(__file__).resolve().parents[2]

# Add an entry ONLY when exposing the name through the owning package's ``shared`` module is genuinely infeasible.
ALLOWLIST: set[tuple[str, str]] = set()


def test_no_new_underscore_imports_cross_package() -> None:
    """No production import reaches into a foreign package's underscore module outside the allowlist."""
    assert_no_private_cross_package_imports(REPO_ROOT / "src" / "mlframe", "mlframe", REPO_ROOT, allowlist=ALLOWLIST)
