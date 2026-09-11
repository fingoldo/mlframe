"""No production code imports another package's underscore-prefixed module.

Underscore-prefixed modules under any ``src/mlframe/<pkg>/`` directory are internal; the public surface is
whatever the package re-exports from its own ``__init__.py``. Sibling files inside the same package may
import each other freely; production code elsewhere must not reach into ``other_pkg._private``, which
couples two subsystems to an implementation detail that breaks silently when either side moves a helper.

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

# Add an entry ONLY when promotion to the owning package's public ``__init__.py`` is genuinely infeasible.
#
# _bootstrap_fused_binary_bundle.py imports _fast_brier_score_loss_seq / _fast_log_loss_binary_seq to call
# them from INSIDE its own @numba.njit(parallel=True) bootstrap kernel: the public dispatchers pick seq vs
# parallel at plain-Python runtime and are not njit-callable, so the private sequential variant is the only
# usable one there, and making it public would suggest it is meant for general use.
ALLOWLIST: set[tuple[str, str]] = {
    ("src/mlframe/evaluation/_bootstrap_fused_binary_bundle.py", "mlframe.metrics._core_auc_brier"),
    ("src/mlframe/evaluation/_bootstrap_fused_binary_bundle.py", "mlframe.metrics._log_loss_and_separation"),
}


def test_no_new_underscore_imports_cross_package() -> None:
    """No production import reaches into a foreign package's underscore module outside the allowlist."""
    assert_no_private_cross_package_imports(REPO_ROOT / "src" / "mlframe", "mlframe", REPO_ROOT, allowlist=ALLOWLIST)
