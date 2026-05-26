"""Smoke tests for mlframe.calibration.post.

Imports the module behind importorskip for optional calibration deps,
then verifies public helpers operate on a tiny synthetic binary task.
"""
from __future__ import annotations

import numpy as np
import pytest

post = pytest.importorskip("mlframe.calibration.post")


# B2#32 DRY: the 6 optional calibration libs were re-importorskip'd in every test body. Hoisting to a single
# module-scope tuple + a tiny helper keeps the skip reason readable and removes the 18+ redundant import probes
# that ran at fit time on every test.
_OPTIONAL_CALIB_DEPS = ("netcal", "pycalib", "ml_insights", "betacal", "venn_abers", "calibration")


def _require_optional_calib_deps():
    """Skip the calling test when ANY of the heavy optional calibration backends is missing. ``calibration`` is the
    ``verified_calibration`` alias -- the package is imported as ``calibration`` even though it ships on PyPI under
    the ``verified-calibration`` name."""
    for mod in _OPTIONAL_CALIB_DEPS:
        pytest.importorskip(mod)


@pytest.fixture
def tiny_binary():
    """Tiny well-separated binary task with deterministic RNG."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    # Probabilities loosely correlated with y to give calibration something to learn.
    base = rng.uniform(0.1, 0.9, size=n)
    probs = np.clip(0.5 * base + 0.5 * y.astype(float), 0.0, 1.0)
    return probs, y


@pytest.mark.fast
def test_should_run_include_skip_filters():
    """include=None & skip=None means run; explicit include narrows; skip excludes."""
    assert post.should_run("CalibratedClassifierCV", include=None, skip=None) is True
    assert post.should_run("Foo", include=[r"^Foo$"], skip=None) is True
    assert post.should_run("Bar", include=[r"^Foo$"], skip=None) is False
    assert post.should_run("Bar", include=None, skip=[r"^Bar$"]) is False


@pytest.mark.fast
def test_get_postcalibrators_returns_nonempty_list(tiny_binary):
    """get_postcalibrators must yield a non-empty calibrator list for a binary task.

    Skips on boxes lacking the heavy optional calibration deps (netcal / pycalib /
    ml_insights / betacal / venn_abers) -- they are lazy-imported inside the
    function so the module loads but the call raises ModuleNotFoundError.
    """
    _require_optional_calib_deps()

    probs, y = tiny_binary
    cals = post.get_postcalibrators(calib_target=y, num_bins=5)
    assert isinstance(cals, list)
    assert len(cals) > 0
    # NamedCalibrator wrappers expose .name and .lib.
    first = cals[0]
    assert hasattr(first, "name")
    assert hasattr(first, "lib")
