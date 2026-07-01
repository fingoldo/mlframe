"""Regression: ``mlframe.feature_engineering.numerical`` must import
without raising AttributeError when astropy is wedged by a transitive
numpy-API deprecation.

Pre-fix path (iter-55 300k seed=31 retry):
- numpy 2.x removed ``np.in1d``; astropy 7.x still references it at
  module load time.
- ``mlframe/feature_engineering/numerical.py`` did
  ``from astropy.stats import histogram`` unconditionally at module top.
- Any caller importing mlframe -> feature_engineering -> numerical
  hit ``AttributeError: module 'numpy' has no attribute 'in1d'``,
  taking down the whole training surface (the suite entry point goes
  through feature_engineering.basic which imports numerical
  transitively).

Post-fix: the astropy import is wrapped in try/except and a
``histogram`` shim falls back to ``np.histogram`` (signature-
compatible for the ``bins="scott"`` / string-rule cases the module
uses). Module stays importable; ``cont_entropy`` produces a finite
number either way.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_numerical_module_imports_even_when_astropy_broken() -> None:
    """Module-level import must not raise. This is the iter-55 regression
    sensor — fails if a future change makes the astropy import
    unconditional again.

    Run the fresh-import probe in a SUBPROCESS: an in-process
    ``importlib.reload`` rebinds this module's globals in the parent
    interpreter (the sys.modules restore does not undo the reload), which
    would pollute every later consumer of
    ``mlframe.feature_engineering.numerical``. A subprocess keeps the parent
    interpreter pristine while still exercising the from-scratch import path.
    """
    import os
    import subprocess
    import sys
    import textwrap

    import mlframe as _mlframe_pkg
    _src_root = os.path.dirname(os.path.dirname(_mlframe_pkg.__file__))
    _env = {**os.environ, "PYTHONPATH": _src_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    _probe = textwrap.dedent("""
        import sys
        import mlframe.feature_engineering.numerical as mod
        ok = hasattr(mod, "cont_entropy") and hasattr(mod, "histogram")
        sys.stdout.write("NUMERICAL_IMPORT_OK=" + str(ok))
    """)
    _res = subprocess.run(
        [sys.executable, "-c", _probe],
        capture_output=True, text=True, timeout=180, env=_env,
    )
    assert _res.returncode == 0, f"probe subprocess failed: {_res.stderr}"
    assert "NUMERICAL_IMPORT_OK=True" in _res.stdout, (
        f"mlframe.feature_engineering.numerical failed the fresh-import "
        f"contract (cont_entropy / histogram missing) -- probe printed "
        f"{_res.stdout!r}, stderr {_res.stderr!r}"
    )


def test_histogram_shim_matches_np_histogram_signature() -> None:
    """The shim must accept the same call shapes (a, bins, **kwargs)
    that astropy.stats.histogram accepts in this module's usage."""
    from mlframe.feature_engineering.numerical import histogram
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(500)
    hist, edges = histogram(arr, bins="scott")
    assert hist.ndim == 1
    assert edges.ndim == 1
    assert edges.size == hist.size + 1
    # Integer bins also accepted.
    hist2, edges2 = histogram(arr, bins=10)
    assert hist2.size == 10
    assert edges2.size == 11


def test_cont_entropy_finite_on_normal_distribution() -> None:
    """End-to-end: cont_entropy still works (the original astropy
    consumer call site)."""
    from mlframe.feature_engineering.numerical import cont_entropy
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(1000)
    h = cont_entropy(arr)
    # cont_entropy(unnormalised counts) gives a finite (possibly
    # negative on dense bins) float; just lock the finite + scalar
    # contract.
    assert isinstance(h, float)
    assert np.isfinite(h)


def test_simulated_astropy_broken_uses_numpy_fallback(monkeypatch) -> None:
    """Force the numpy-fallback path by monkeypatching the module's
    private astropy reference to None, then verify histogram() still
    returns sensible output. Locks the "astropy unavailable" branch."""
    import mlframe.feature_engineering.numerical as mod
    monkeypatch.setattr(mod, "_astropy_histogram", None)
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(500)
    hist, edges = mod.histogram(arr, bins="scott")
    assert hist.ndim == 1
    # Numpy returns a numpy.ndarray; ensure no astropy-specific types leaked through.
    assert isinstance(hist, np.ndarray)
    assert isinstance(edges, np.ndarray)
