"""A GPU branch must be seeded exactly as the CPU branch beside it.

Both `mi_direct_gpu` and `mi_direct` take `base_seed`, and both document it as what makes the permutation
shuffle reproducible. Two call sites forwarded it on the CPU side and dropped it on the GPU side, so
`cp.random.default_rng(None)` seeded from OS entropy. The permutation `null_mean` is subtracted from the
score that drives selection, which made the SELECTED FEATURE SET non-deterministic on a CUDA host with the
caller's `random_seed` pinned -- reproducible on CPU, run-varying on GPU, and silent either way.

Asserted STATICALLY, on the call sites, for two reasons: the defect is a missing argument rather than a
numeric difference, so the source is where it is visible; and a runtime check would only run on a CUDA host,
which is exactly the machine class the bug hides from everywhere else.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"

# (module, the GPU callee whose call sites must carry a seed). Both take `base_seed` and document it.
SEEDED_GPU_CALLEES = (
    ("feature_selection/filters/evaluation.py", "mi_direct_gpu"),
    ("feature_selection/filters/permutation.py", "mi_direct_gpu"),
)

# Names any of these call sites may legitimately use for the seed they forward.
_SEED_KWARGS = frozenset({"base_seed", "random_seed", "seed", "random_state"})


def _calls_to(tree: ast.Module, callee: str) -> list[ast.Call]:
    """Every call to ``callee`` in the module, by plain name or attribute."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name == callee:
            out.append(node)
    return out


@pytest.mark.parametrize(("rel", "callee"), SEEDED_GPU_CALLEES, ids=[f"{r.split('/')[-1]}::{c}" for r, c in SEEDED_GPU_CALLEES])
def test_every_gpu_call_forwards_a_seed(rel: str, callee: str):
    """A call to the GPU kernel that omits the seed leaves the permutation null drawn from OS entropy."""
    path = SRC / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = _calls_to(tree, callee)

    assert calls, f"no call to {callee} found in {rel} -- this guard is pinned to a call site that moved, and is now checking nothing"

    unseeded = [c.lineno for c in calls if not any(kw.arg in _SEED_KWARGS for kw in c.keywords if kw.arg)]
    assert not unseeded, (
        f"{rel}: {callee} called without a seed at line(s) {unseeded}. The CPU branch beside it forwards one, "
        "so the same public call is reproducible on CPU and run-varying on GPU, and the selected feature set "
        "moves between runs with random_seed pinned."
    )


def test_the_cpu_branches_still_forward_their_seed():
    """Pins the other half of the parity, so a future edit cannot 'fix' the mismatch by dropping both.

    Without this, deleting `base_seed` from the CPU branches would make the test above pass while making
    reproducibility strictly worse.
    """
    path = SRC / "feature_selection/filters/evaluation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cpu_calls = _calls_to(tree, "mi_direct")

    assert cpu_calls, "no CPU-side mi_direct call found in evaluation.py; the parity this pins no longer exists"
    seeded = [c for c in cpu_calls if any(kw.arg in _SEED_KWARGS for kw in c.keywords if kw.arg)]
    assert seeded, "no CPU-side mi_direct call forwards a seed any more -- reproducibility was removed rather than extended to the GPU"
