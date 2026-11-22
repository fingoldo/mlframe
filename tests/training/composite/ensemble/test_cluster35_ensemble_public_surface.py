"""Audit 2026-11-20 finding 35-F48: the ensemble package's __all__ names the public surface and every name resolves.

The unused Ridge import is gone, and a star-import exposes exactly the public names, not typing names or private helpers.
"""

import mlframe.training.composite.ensemble as ens


def test_every_all_name_resolves():
    """Each name in __all__ is an attribute of the package."""
    missing = [n for n in ens.__all__ if not hasattr(ens, n)]
    assert missing == []


def test_public_names_are_exported_and_typing_names_are_not():
    """The intended public names are listed; typing helpers and the removed Ridge names are not."""
    for name in ("derive_seeds", "detect_gpu_in_use", "env_signature", "compute_oof_holdout_predictions", "CompositeCrossTargetEnsemble"):
        assert name in ens.__all__
    for name in ("Any", "Dict", "List", "Optional", "Ridge", "RidgeCV"):
        assert name not in ens.__all__
