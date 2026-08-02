"""Regression: a transient (non-ImportError) failure in ``_select_mi_backend``'s trial import must NOT
permanently downgrade the process to the sklearn reference loop.

BUG FOUND AND FIXED (2026-08-02, incidental to a profiling cycle): ``_select_mi_backend`` caches its
numba-vs-sklearn decision once per process into the module-level ``_MI_BACKEND`` -- a bare
``except Exception`` around the trial import treated ANY exception (including a transient device/driver
fault surfaced while importing ``hermite_fe``, which probes CUDA availability at import time) as "numba
genuinely unavailable", permanently downgrading the whole process to ``_mi_classif_batch_sklearn`` (the
~100x-slower reference loop) for its entire lifetime. Confirmed live via a 2M-row cProfile (combo
``c0094_5637be0a``) that caught this fallback costing 161.8s cumtime / 19 calls on a heavily
GPU-contended run, while a clean process resolves ``_MI_BACKEND`` to ``"numba"`` correctly. Only a
genuine ``ImportError`` (numba dispatcher truly absent, e.g. a stripped-down install) should trigger the
sklearn fallback; any other exception is unrelated to whether the CPU njit path works and must not
downgrade the decision.
"""

from __future__ import annotations

from unittest.mock import patch


def test_import_error_falls_back_to_sklearn():
    """A genuine ImportError (numba dispatcher truly absent) still selects the sklearn backend."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as m

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        """Raise ImportError for the hermite_fe trial import, delegate everything else."""
        if name.endswith("hermite_fe") or (len(args) >= 4 and args[3] and any("plugin_mi_classif_batch_dispatch" in str(f) for f in args[3])):
            raise ImportError("simulated: numba genuinely absent")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert m._select_mi_backend() == "sklearn"


def test_transient_non_import_error_still_selects_numba():
    """A transient non-ImportError failure (e.g. a device/driver fault) during the trial import must NOT
    downgrade the backend selection -- it defaults to 'numba' regardless."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as m

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        """Raise a non-ImportError RuntimeError for the hermite_fe trial import, delegate everything else."""
        if name.endswith("hermite_fe") or (len(args) >= 4 and args[3] and any("plugin_mi_classif_batch_dispatch" in str(f) for f in args[3])):
            raise RuntimeError("simulated: transient CUDA driver fault")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        assert m._select_mi_backend() == "numba"


def test_env_override_still_takes_priority_over_trial_import():
    """MLFRAME_NUMBA_MI env overrides bypass the trial import entirely, in both directions."""
    import os

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_mi_backends as m

    orig = os.environ.get("MLFRAME_NUMBA_MI")
    try:
        os.environ["MLFRAME_NUMBA_MI"] = "0"
        assert m._select_mi_backend() == "sklearn"
        os.environ["MLFRAME_NUMBA_MI"] = "1"
        assert m._select_mi_backend() == "numba"
    finally:
        if orig is None:
            os.environ.pop("MLFRAME_NUMBA_MI", None)
        else:
            os.environ["MLFRAME_NUMBA_MI"] = orig
