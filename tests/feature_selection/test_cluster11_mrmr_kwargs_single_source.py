"""Cluster 11 regression: every MRMR arm is built from one keyword set (audit 2026-11-20, finding 11-F19).

The base arm, the wrapper's bare MRMR and the variant arm all draw from bare_mrmr_kwargs. These tests capture the constructor
keywords and pin that the base and wrapper agree, and that a variant overrides only what it declares.
"""

from mlframe.feature_selection import filters
from mlframe.feature_selection._benchmarks.fs_hybrid import _arms, _arms_it_family


class _Capture:
    """Stand-in MRMR that records its constructor keywords."""

    last = None

    def __init__(self, **kwargs):
        """Store the keywords."""
        _Capture.last = kwargs


def test_base_arm_and_bare_wrapper_share_keywords(monkeypatch):
    """The mrmr arm (fe off) and the wrapper's bare MRMR are built with identical keywords (11-F19)."""
    monkeypatch.setattr(filters, "MRMR", _Capture)
    _arms.MRMRArm(fe=False, max_runtime_mins=2.0, random_seed=7)._build_model()
    base = dict(_Capture.last)
    _arms_it_family._bare_mrmr(2.0, 7)
    assert _Capture.last == base
    assert base["fe_max_steps"] == 0


def test_variant_overrides_only_its_declared_keyword(monkeypatch):
    """A variant arm keeps the shared keywords and changes only what it declares (11-F19)."""
    monkeypatch.setattr(filters, "MRMR", _Capture)
    _arms_it_family.MRMRVariantArm("v", {"max_runtime_mins": 9.0}, max_runtime_mins=2.0, random_seed=7)._build_model()
    kwargs = _Capture.last
    assert kwargs["max_runtime_mins"] == 9.0
    assert kwargs["fe_max_steps"] == 0 and kwargs["n_jobs"] == -1 and kwargs["random_seed"] == 7


def test_helper_returns_the_documented_keys():
    """The helper's keyword set is the one the arms rely on (11-F19)."""
    assert _arms.bare_mrmr_kwargs(1.5, 3) == {"verbose": 0, "fe_max_steps": 0, "n_jobs": -1, "random_seed": 3, "max_runtime_mins": 1.5}
