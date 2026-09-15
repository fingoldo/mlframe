"""Guards and probes whose failure fallback silently disabled them (mrmr_audit_2026-09-14 NUM-22..NUM-31).

Each site caught an exception at debug level and substituted a value that switched off the very check it belonged to:

* NUM-22: an unmeasurable clean-form correlation (-1.0) disabled the demotion of a distorted prewarp form;
* NUM-23: an entropy failure disabled the MI-ceiling sanity bound without a trace;
* NUM-24: a missing marginal MI became 0.0, inflating interaction information and routing the pair as synergy;
* NUM-25 / NUM-26: a failing VRAM probe answered "the upload fits";
* NUM-27: a failing RAM probe disabled the OOM headroom guard without a trace;
* NUM-28: a constant target of unorderable values skipped the constant-y guard;
* NUM-29: an unreadable target was guessed to be single-output;
* NUM-30: a failed defaults lookup silently turned the fast-search profile into a no-op;
* NUM-31: one failure disabled the usability admission route for every pair, at debug.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


def _warned(caplog, *needles):
    """True when some WARNING-or-higher record contains every needle."""
    return any(r.levelno >= logging.WARNING and all(n in r.getMessage() for n in needles) for r in caplog.records)


# ---------------------------------------------------------------------------------------------------------------- NUM-22


@pytest.mark.parametrize(
    "pw_corr, clean_corr, expected",
    [
        (0.5, None, True),  # clean form unmeasurable -> fall back to the simpler form
        (None, 0.5, True),  # prewarp form unmeasurable -> it cannot justify its distortion
        (0.9, 0.5, False),  # prewarp meaningfully more usable -> keep it
        (0.5, 0.49, True),  # prewarp within 5% of the clean form -> demote
        (0.5, -1.0, False),  # clean config genuinely unrecoverable -> nothing to demote to
    ],
)
def test_prewarp_demotion_decision(pw_corr, clean_corr, expected):
    """An unmeasurable correlation must never switch the clean-form demotion off."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score import _should_demote_prewarp

    assert _should_demote_prewarp(pw_corr, clean_corr) is expected


# ---------------------------------------------------------------------------------------------------------------- NUM-23


def test_target_entropy_failure_is_warned(caplog):
    """An H(y) that cannot be computed disables the MI-ceiling bound; that must be visible."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _target_entropy_nats

    y = np.array([{"a": 1}, {"b": 2}, {"a": 1}], dtype=object)  # unorderable: np.unique raises
    with caplog.at_level(logging.WARNING):
        assert _target_entropy_nats(y) == 0.0
    assert _warned(caplog, "H(y)"), "the disabled MI-ceiling bound was not reported"


# ---------------------------------------------------------------------------------------------------------------- NUM-24


def test_missing_marginal_mi_does_not_route_synergy(caplog):
    """A pair with no cached marginal MI must be kept unrouted, never promoted to synergy by a 0.0 substitute."""
    from mlframe.feature_selection.filters._interaction_information import ROUTE_SYNERGY, route_prospective_pairs

    key = ((0, 1), 0.5)
    with caplog.at_level(logging.WARNING):
        kept, routes, _ii = route_prospective_pairs(
            {key: 1.0}, cached_MIs={(1,): 0.01}, nbins=np.array([4, 4]), nbins_y=2, n=1000, ii_floor=0.01,
        )
    assert routes.get((0, 1)) != ROUTE_SYNERGY, "a cache miss manufactured a synergy route"
    assert key in kept, "an unroutable pair must be kept for the search, not dropped"
    assert _warned(caplog, "marginal"), "the missing marginal MI was not reported"


# ---------------------------------------------------------------------------------------------------------------- NUM-25 / NUM-26


def test_gpu_upload_guard_fails_closed_when_the_vram_probe_raises(monkeypatch, caplog):
    """If free VRAM cannot be read, the pair-MI upload guard must refuse, not assume the upload fits."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import _gpu_upload_fits

    def _boom():
        """Stand in for a failing memGetInfo."""
        raise RuntimeError("synthetic memGetInfo failure")

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", _boom)
    with caplog.at_level(logging.WARNING):
        assert _gpu_upload_fits(1024) is False
    assert _warned(caplog, "memGetInfo")


def test_cmi_cuda_guard_fails_closed_when_the_vram_probe_raises(monkeypatch, caplog):
    """The CMI CUDA gate must not fall through to CUDA when its VRAM probe fails."""
    cp = pytest.importorskip("cupy")
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cc

    def _boom():
        """Stand in for a failing memGetInfo."""
        raise RuntimeError("synthetic memGetInfo failure")

    monkeypatch.setattr(cc, "_CMI_GPU_FAILED", False)
    monkeypatch.setattr(cc, "cupy_available", lambda: True)
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", _boom)
    with caplog.at_level(logging.WARNING):
        assert cc._should_use_cuda(n=1000, p=10, joint_size=16) is False
    assert _warned(caplog, "VRAM")


# ---------------------------------------------------------------------------------------------------------------- NUM-27 / NUM-28


def _frame(n=200, seed=0):
    """A small numeric frame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(3)})


def test_ram_headroom_probe_failure_is_warned(monkeypatch, caplog):
    """A failing psutil probe disables the OOM headroom guard; the missing protection must be visible."""
    import psutil

    from mlframe.feature_selection.filters.mrmr import MRMR

    def _boom():
        """Stand in for a sandbox where virtual_memory() errors."""
        raise RuntimeError("synthetic virtual_memory failure")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    X = _frame()
    with caplog.at_level(logging.WARNING):
        MRMR()._validate_inputs(X, (X["f0"] > 0).astype(int))
    assert _warned(caplog, "headroom")


def test_constant_target_of_unorderable_values_still_raises():
    """A constant target whose values cannot be sorted must still hit the constant-y guard."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X = _frame()
    y = np.array([None] * len(X), dtype=object)
    with pytest.raises(ValueError, match="only 1 unique value"):
        MRMR()._validate_inputs(X, y)


# ---------------------------------------------------------------------------------------------------------------- NUM-29


def test_unreadable_target_is_not_guessed_single_output():
    """If y cannot be read as an array, the fit path must not be guessed."""
    from mlframe.feature_selection.filters.mrmr._mrmr_class import _mrmr_y_is_multioutput

    class _Unreadable:
        """A target container numpy cannot convert."""

        def __array__(self, *args, **kwargs):
            raise RuntimeError("synthetic conversion failure")

    with pytest.raises(TypeError, match="_Unreadable"):
        _mrmr_y_is_multioutput(_Unreadable())


# ---------------------------------------------------------------------------------------------------------------- NUM-30


def test_fast_search_profile_failure_is_warned(monkeypatch, caplog):
    """A fast-search profile that could not be applied must say so, not silently leave every knob untouched."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    def _boom():
        """Stand in for a failing constructor-defaults introspection."""
        raise RuntimeError("synthetic defaults failure")

    monkeypatch.setattr(MRMR, "_ctor_defaults", staticmethod(_boom))
    with caplog.at_level(logging.WARNING):
        MRMR(fe_fast_search=True)._apply_fast_search_profile()
    assert _warned(caplog, "fast_search", "NOT applied")


# ---------------------------------------------------------------------------------------------------------------- NUM-31


def test_usability_batch_failure_is_warned(caplog):
    """One failure removes the usability admission route for every pair in the step; it must be visible."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _batch_usability_admission_verdicts

    def _boom(idx):
        """Stand in for a failing operand fetch."""
        raise RuntimeError("synthetic operand failure")

    with caplog.at_level(logging.WARNING):
        out = _batch_usability_admission_verdicts(
            SimpleNamespace(), need_usability=[(0, 1), (0, 2)], y_continuous=np.arange(100.0), cached_operand=_boom, cached_single_corr=lambda i: 0.1,
        )
    assert out == {}
    assert _warned(caplog, "usability", "2")
