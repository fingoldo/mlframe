"""Smoke tests for the --fast mode plumbing in tests/conftest.py."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import fast_n_estimators, fast_subset, is_fast_mode


def test_fast_subset_identity_outside_fast(monkeypatch):
    monkeypatch.delenv("MLFRAME_FAST", raising=False)
    assert fast_subset([1, 2, 3]) == [1, 2, 3]


def test_fast_subset_trims_in_fast(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    assert fast_subset([1, 2, 3]) == [1]
    assert fast_subset([1, 2, 3], keep=2) == [1, 2]


def test_fast_subset_representative(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    assert fast_subset(["a", "b", "c"], representative="b") == ["b"]


def test_fast_subset_representative_missing_falls_back(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    assert fast_subset([1, 2, 3], representative=99) == [1]


def test_fast_subset_handles_pytest_param(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    entries = [pytest.param(1, id="one"), pytest.param(2, id="two")]
    out = fast_subset(entries, representative=2)
    assert len(out) == 1
    assert out[0].values == (2,)


def test_fast_n_estimators_identity_outside_fast(monkeypatch):
    monkeypatch.delenv("MLFRAME_FAST", raising=False)
    assert fast_n_estimators(300) == 300
    assert fast_n_estimators(5) == 5


def test_fast_n_estimators_shrinks_in_fast(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    assert fast_n_estimators(300) == 40
    assert fast_n_estimators(300, fast=20) == 20


def test_fast_n_estimators_never_exceeds_full(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    # a budget already smaller than the fast target stays put, not inflated to 40
    assert fast_n_estimators(10) == 10
    assert fast_n_estimators(1) == 1


def test_fast_n_estimators_respects_floor(monkeypatch):
    monkeypatch.setenv("MLFRAME_FAST", "1")
    assert fast_n_estimators(300, fast=0, floor=1) == 1


def test_is_fast_mode_rejects_falsey(monkeypatch):
    for v in ("", "0", "false", "False"):
        monkeypatch.setenv("MLFRAME_FAST", v)
        assert is_fast_mode() is False


def test_is_fast_mode_accepts_truthy(monkeypatch):
    for v in ("1", "yes", "on", "true"):
        monkeypatch.setenv("MLFRAME_FAST", v)
        assert is_fast_mode() is True


def test_slow_marker_skipped_in_fast_subprocess(tmp_path):
    """End-to-end: --fast actually skips @pytest.mark.slow tests."""
    test_file = tmp_path / "test_slow_skip_demo.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.mark.slow\n"
        "def test_heavy(): assert True\n"
        "def test_light(): assert True\n"
    )
    conftest = tmp_path / "conftest.py"
    conftest.write_text(
        "import os\n"
        "import pytest\n"
        "def pytest_configure(config):\n"
        "    config.addinivalue_line('markers', 'slow: slow test')\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    if os.environ.get('MLFRAME_FAST', '') in ('', '0', 'false'): return\n"
        "    skip = pytest.mark.skip(reason='fast')\n"
        "    for it in items:\n"
        "        if 'slow' in it.keywords: it.add_marker(skip)\n"
    )
    env = {**os.environ, "MLFRAME_FAST": "1"}
    result = subprocess.run(
        # ``-p no:randomly``: the spawned subprocess inherits whatever
        # random-seeder plugins are installed in the ambient env. On boxes with
        # spacy/thinc present, pytest_randomly's reseed hook calls thinc's
        # ``fix_random_seed`` -> ``numpy.random.seed(seed)`` which raises
        # ``ValueError: Seed must be between 0 and 2**32 - 1`` on some
        # numpy/thinc combos (observed prod box 2026-05-27), erroring the demo
        # tests before they can report pass/skip. This test is about --fast
        # marker skipping, NOT random ordering, so disable the plugin in the
        # controlled subprocess to isolate it from that env incompatibility.
        [sys.executable, "-m", "pytest", str(test_file),
         "-p", "no:cacheprovider", "-p", "no:randomly",
         "--no-cov", "-q", "-o", "addopts="],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert "1 passed" in result.stdout and "1 skipped" in result.stdout, result.stdout + result.stderr
