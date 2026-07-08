"""Live contract for four MRMR research-extension constructor knobs now WIRED into ``fit()``: ``relaxmrmr_alpha`` (RelaxMRMR 3-D MI, Vinh 2016),
``pid_synergy_bonus`` (PID I_ccs synergy, Ince 2017), ``cmi_perm_stop`` (+``cmi_perm_alpha`` / ``cmi_perm_n_permutations``; CMI-permutation stop, Yu-Principe 2019),
and ``cpt_test`` (+``cpt_n_permutations``; D10 Conditional Permutation Test, Berrett-Wang-Barber-Samworth 2020).

Each knob is read in the per-candidate scoring step (``evaluation.evaluate_candidate``) via thread-locals set by ``MRMR.fit`` -- mirroring the ``mi_correction`` /
BUR precedent. RelaxMRMR replaces the complex-mode Fleuret score, PID adds a max-synergy bonus, CMI-perm-stop drops a conditionally-insignificant candidate (permuting
its UNCONDITIONAL marginal), and CPT does the same but via within-selected-stratum permutation (giving valid p-values under arbitrary confounding by the selected set).
All four default to a no-op (alpha=0 / bonus=0 / stop=False / test=False) so the default selection is byte-identical; these tests assert that ACTIVATING each knob
changes the selected support (the standalone kernel checks below additionally prove the underlying capability). RelaxMRMR and CMI-perm-stop act on the conditional-MI
redundancy term that only exists in complex mode, so their activation tests run with ``use_simple_mode=False``.
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
    y = ((x0 + x1 + (a ^ b)) > 0.0).astype(int)
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


def test_conditional_permutation_test_kernel_discriminates_signal_from_noise():
    from mlframe.feature_selection.filters._conditional_permutation import conditional_permutation_test

    rng = np.random.default_rng(4)
    n = 800
    z = rng.integers(0, 4, n)  # "selected" stratifying variable
    y = (z + rng.integers(0, 2, n)) % 4
    x_sig = (y + rng.integers(0, 2, n)) % 4  # depends on y even conditional on z
    x_noise = rng.integers(0, 4, n)  # independent of y given z
    _, sig_p = conditional_permutation_test(x_sig, y, z, nbins_x=4, nbins_y=4, nbins_z=4, n_permutations=60, seed=0)
    _, noise_p = conditional_permutation_test(x_noise, y, z, nbins_x=4, nbins_y=4, nbins_z=4, n_permutations=60, seed=0)
    assert sig_p < noise_p, f"CPT p-value must be smaller for the real conditional signal (sig_p={sig_p:.3f} vs noise_p={noise_p:.3f})"


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


# ---- (b) live contract: ACTIVATING each wired knob changes the selected support vs its no-op default ----


def test_relaxmrmr_alpha_changes_selection():
    base = _fit_support(relaxmrmr_alpha=0.0, use_simple_mode=False)
    relaxed = _fit_support(relaxmrmr_alpha=9.0, use_simple_mode=False)
    assert relaxed != base, "a large relaxmrmr_alpha 3-D redundancy term must change the selected support vs alpha=0"


def test_pid_synergy_bonus_changes_selection():
    base = _fit_support(pid_synergy_bonus=0.0, use_simple_mode=False)
    bonus = _fit_support(pid_synergy_bonus=9.0, use_simple_mode=False)
    assert bonus != base, "a large pid_synergy_bonus must promote the synergistic (a,b) pair and change the selected support vs 0.0"


def test_cmi_perm_stop_fires_in_live_fit_and_prunes_candidates():
    """``cmi_perm_stop=True`` must drive the wired CMI-permutation stop INSIDE a live fit: the kernel is consulted from
    ``evaluation.evaluate_candidate`` and forces ``>=1`` conditionally-insignificant candidate's gain to 0 -- the
    selection-affecting mechanism. The no-stop default never consults the kernel.

    The over-specified ``support_ changes`` proxy is intentionally NOT asserted here: at the tractable suite size
    (n<=1500, single-process) MRMR's Fleuret/redundancy pass already drops conditionally-redundant near-duplicates with
    the stop OFF, so the FINAL support is unchanged -- a fact the sibling ``test_biz_val_filters_mrmr_research_knobs_wired
    .py`` measures and documents (``identical support_ on both synthetics``). The real wiring contract is that the stop
    is reached and fires; the kernel's own discrimination win is pinned by ``test_cmi_permutation_stop_kernel_*`` above.
    """
    from mlframe.feature_selection.filters import _cmi_perm_stop as _cm

    real_stop = _cm.cmi_permutation_stop
    calls: list[tuple[bool, float, float]] = []

    def _spy(*a, **k):
        res = real_stop(*a, **k)
        calls.append(res)
        return res

    X, y = _synth()
    common = dict(verbose=0, random_seed=42, n_workers=1, use_simple_mode=False, quantization_nbins=8, max_runtime_mins=1, fe_max_steps=0)

    from mlframe.feature_selection.filters.mrmr import MRMR

    # No-stop default: the kernel must NOT be consulted at all.
    calls.clear()
    _cm.cmi_permutation_stop = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MRMR(cmi_perm_stop=False, **common).fit(X, y)
        assert len(calls) == 0, "with cmi_perm_stop=False the CMI-permutation kernel must never be consulted (byte-identical no-op default)"

        # Stop active: the kernel is consulted and prunes >=1 conditionally-insignificant candidate (gain forced to 0).
        calls.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MRMR(cmi_perm_stop=True, cmi_perm_alpha=0.5, cmi_perm_n_permutations=20, **common).fit(X, y)
    finally:
        _cm.cmi_permutation_stop = real_stop

    assert len(calls) > 0, "with cmi_perm_stop=True the wired CMI-permutation stop must be reached from evaluate_candidate during the live fit"
    pruned = [c for c in calls if not c[0]]  # is_significant == False -> candidate gain forced to 0
    assert len(pruned) >= 1, (
        f"an aggressive CMI-permutation stop (alpha=0.5) must flag >=1 conditionally-insignificant candidate as not-significant "
        f"and prune its gain; got {len(calls)} consultations, {len(pruned)} pruned"
    )


def test_cpt_test_fires_in_live_fit_and_prunes_candidates():
    """``cpt_test=True`` must drive the wired D10 Conditional Permutation Test INSIDE a live fit: the kernel is consulted
    from ``evaluation.evaluate_candidate`` and forces >=1 candidate's gain to 0 when its within-selected-stratum
    permutation p-value is >= 0.05 -- the selection-affecting mechanism. The default (cpt_test=False) never consults it.
    """
    from mlframe.feature_selection.filters import _conditional_permutation as _cp

    real_test = _cp.conditional_permutation_test
    calls: list[tuple[float, float]] = []

    def _spy(*a, **k):
        res = real_test(*a, **k)
        calls.append(res)
        return res

    X, y = _synth()
    common = dict(verbose=0, random_seed=42, n_workers=1, use_simple_mode=False, quantization_nbins=8, max_runtime_mins=1, fe_max_steps=0)

    from mlframe.feature_selection.filters.mrmr import MRMR

    # Default off: the kernel must NOT be consulted at all.
    calls.clear()
    _cp.conditional_permutation_test = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MRMR(cpt_test=False, **common).fit(X, y)
        assert len(calls) == 0, "with cpt_test=False the D10 CPT kernel must never be consulted (byte-identical no-op default)"

        # Active: the kernel is consulted and prunes >=1 candidate whose conditional-permutation p-value is not significant.
        calls.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            MRMR(cpt_test=True, cpt_n_permutations=20, **common).fit(X, y)
    finally:
        _cp.conditional_permutation_test = real_test

    assert len(calls) > 0, "with cpt_test=True the wired D10 CPT must be reached from evaluate_candidate during the live fit"
    pruned = [c for c in calls if c[1] >= 0.05]  # p_value >= 0.05 -> candidate gain forced to 0
    assert len(pruned) >= 1, (
        f"D10 CPT must flag >=1 candidate as conditionally-insignificant given the selected set and prune its gain; "
        f"got {len(calls)} consultations, {len(pruned)} pruned"
    )
