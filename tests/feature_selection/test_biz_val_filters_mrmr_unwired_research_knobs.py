"""Activation pins for three MRMR research-extension constructor knobs that are currently UNWIRED: ``relaxmrmr_alpha`` (RelaxMRMR 3-D MI, Vinh 2016),
``pid_synergy_bonus`` (PID I_ccs synergy, Ince 2017) and ``cmi_perm_stop`` (+``cmi_perm_alpha`` / ``cmi_perm_n_permutations``; CMI-permutation stop, Yu-Principe 2019).

Finding (B2 audit): each is accepted, validated and stored on the estimator (``MRMR.__init__``) and appears in the ``recommend_enabled_fe`` introspection list, but
NONE is read anywhere in ``fit()`` -- ``grep self.relaxmrmr_alpha / self.pid_synergy_bonus / self.cmi_perm_stop`` over ``src/`` finds zero consumption sites. The
standalone kernels (``relax_mrmr_score``, ``pid_decomposition``, ``cmi_permutation_stop``) exist and work, but are not dispatched into the greedy selection loop, so
setting these knobs to extreme values leaves the selected support BIT-IDENTICAL to the default. This is the same dead-knob shape that ``mi_correction`` had before it
was wired (see ``test_biz_val_filters_mrmr_mi_correction.py::test_mi_correction_knob_activates_and_resets_thread_local``).

These tests do two things: (a) prove each underlying kernel is genuinely functional (not vaporware), and (b) pin the current no-op behaviour with ``xfail(strict=True)``
so that the day any of these knobs is wired into ``fit()`` the selection-changes assertion starts PASSING -> the strict-xfail becomes an XPASS failure -> whoever wires it
is forced to replace the pin with a real biz_value win assertion + update the disposition. The pins are the regression sensor; they must not be deleted while the knobs
remain dead.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _patch_no_ktc_sweep():
    """Force the kernel-tuning cache into in-memory fallback so a fit is not gated on the cross-process tuning sweep / disk lock (keeps each fit < timeout)."""
    try:
        import pyutilz.performance.kernel_tuning.cache as _M

        _M.KernelTuningCache.get_or_tune = lambda self, k, *, dims, tuner, axes, fallback, **kw: (fallback() if callable(fallback) else fallback)
        _im = _M.KernelTuningCache(in_memory=True)
        _M.KernelTuningCache.load_or_create = classmethod(lambda cls: _im)
    except Exception:
        pass


_patch_no_ktc_sweep()


def _synth(seed: int = 0, n: int = 700):
    """Additive-signal classification synthetic with redundant + synergistic structure: the regime where a wired relax/pid/cmi knob would plausibly move selection."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = x0 + 0.25 * rng.standard_normal(n)  # redundant with x0
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    xor = (a ^ b).astype(float)  # synergy pair (a, b)
    noise = rng.standard_normal((n, 3))
    X = pd.DataFrame(
        np.column_stack([x0, x1, x2, a.astype(float), b.astype(float), xor, noise]),
        columns=["x0", "x1", "x2red", "a", "b", "xor", "n0", "n1", "n2"],
    )
    y = ((x0 + x1 + (a ^ b)) > 0.0).astype(int).values
    return X, y


def _fit_support(**knobs):
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _synth()
    base = dict(verbose=0, random_seed=42, n_workers=1, use_simple_mode=True, quantization_nbins=8, max_runtime_mins=1, fe_max_steps=0)
    base.update(knobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(**base).fit(X, y)
    return m.support_.tolist()


# ---- (a) the standalone kernels are functional (genuine capability behind each dead knob) ----


def test_relax_mrmr_kernel_runs_and_is_finite():
    from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score

    rng = np.random.default_rng(1)
    n = 400
    xc = rng.integers(0, 8, n)
    z = [rng.integers(0, 8, n), rng.integers(0, 8, n)]
    y = rng.integers(0, 2, n)
    s = relax_mrmr_score(xc, z, y, 8, [8, 8], 2, alpha=1.0)
    assert np.isfinite(s), "relax_mrmr_score must return a finite scalar score"


def test_cmi_permutation_stop_kernel_discriminates_signal_from_noise():
    from mlframe.feature_selection.filters._cmi_perm_stop import cmi_permutation_stop

    rng = np.random.default_rng(2)
    n = 600
    y = rng.integers(0, 2, n)
    x_sig = (y + rng.integers(0, 2, n)) % 4  # depends on y
    x_noise = rng.integers(0, 4, n)
    sig_is_sig, _, sig_p = cmi_permutation_stop(x_sig, y, [], 4, 2, [], n_permutations=60, alpha=0.05, seed=0)
    noise_is_sig, _, noise_p = cmi_permutation_stop(x_noise, y, [], 4, 2, [], n_permutations=60, alpha=0.05, seed=0)
    assert sig_p < noise_p, f"CMI-perm p-value must be smaller for the real signal (sig_p={sig_p:.3f} vs noise_p={noise_p:.3f})"


def test_pid_decomposition_kernel_finds_synergy_on_xor():
    from mlframe.feature_selection.filters._pid_decomposition import pid_decomposition

    rng = np.random.default_rng(3)
    n = 800
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    y = (a ^ b).astype(np.int64)
    res = pid_decomposition(a.astype(np.int64), b.astype(np.int64), y, 2, 2, 2)
    syn = res["synergistic"]
    assert syn > 0.05, f"PID must report positive synergy on a pure-XOR target; got synergy={syn:.4f}"


# ---- (b) strict-xfail pins: the constructor knobs are currently NO-OPS at the selection level ----
# When any knob below is wired into fit(), its selection-changes assertion starts passing -> the strict xfail becomes an XPASS failure -> replace the pin with a real
# biz_value win test and update the audit disposition. Do NOT delete these pins while the knobs remain unconsumed in fit().


@pytest.mark.xfail(reason="relaxmrmr_alpha is stored on MRMR but never read in fit(); extreme values do not change selection (B2 audit)", strict=True)
def test_relaxmrmr_alpha_changes_selection():
    base = _fit_support(relaxmrmr_alpha=0.0)
    relaxed = _fit_support(relaxmrmr_alpha=9.0)
    assert relaxed != base, "EXPECTED (once wired): a large relaxmrmr_alpha 3-D redundancy term should change the selected support vs alpha=0"


@pytest.mark.xfail(reason="pid_synergy_bonus is stored on MRMR but never read in fit(); extreme values do not change selection (B2 audit)", strict=True)
def test_pid_synergy_bonus_changes_selection():
    base = _fit_support(pid_synergy_bonus=0.0)
    bonus = _fit_support(pid_synergy_bonus=9.0)
    assert bonus != base, "EXPECTED (once wired): a large pid_synergy_bonus should promote the synergistic (a,b) pair and change the selected support vs 0.0"


@pytest.mark.xfail(reason="cmi_perm_stop/cmi_perm_alpha/cmi_perm_n_permutations are stored on MRMR but never read in fit(); enabling does not change selection (B2 audit)", strict=True)
def test_cmi_perm_stop_changes_selection():
    base = _fit_support(cmi_perm_stop=False)
    stopped = _fit_support(cmi_perm_stop=True, cmi_perm_alpha=0.5, cmi_perm_n_permutations=20)
    assert stopped != base, "EXPECTED (once wired): an aggressive CMI-permutation stop (alpha=0.5) should prune trailing redundant/noise picks vs the default stop"
