"""``DISCOVERY_ALGO_VERSION`` must be bumped whenever the discovery or transform sources change.

The discovery disk cache keyed on ``mlframe.__version__``, which does not change inside a release, so a warm cache kept
replaying specs selected by code that had since been fixed. The version constant is in the cache key; this gate hashes the
sources that decide selection and fails when they change and the constant does not (a bump re-pins the hash).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import mlframe

py_ci_shared = pytest.importorskip("py_ci_shared", reason="py-ci-shared is a dev-only git dependency")

_COMPOSITE = Path(mlframe.__file__).resolve().parent / "training" / "composite"
_BASELINE = Path(__file__).resolve().parent / "_discovery_algo_version_baseline.json"


def _selection_sources() -> list[Path]:
    """Every discovery and transform module (benchmarks excluded), sorted for a stable hash."""
    files = [p for sub in ("discovery", "transforms") for p in (_COMPOSITE / sub).rglob("*.py") if "_benchmarks" not in p.parts]
    assert len(files) >= 60, f"only {len(files)} selection sources found; the scope no longer matches the tree"
    return sorted(files, key=lambda p: p.relative_to(_COMPOSITE).as_posix())  # OS-independent order (WindowsPath compares case-folded)


def test_discovery_algo_version_is_bumped_with_its_sources():
    """A change under composite/discovery or composite/transforms needs a DISCOVERY_ALGO_VERSION bump."""
    from py_ci_shared.content_hash_version_bump_gate import assert_version_bumped_with_content

    from mlframe.training.composite.discovery._algo_version import DISCOVERY_ALGO_VERSION

    assert_version_bumped_with_content(files=_selection_sources(), version=DISCOVERY_ALGO_VERSION, baseline_path=_BASELINE)
