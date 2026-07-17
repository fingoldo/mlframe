"""biz_value: Miller-Madow bias correction for binned KL / JS drift divergence (qual-6).

The binned plug-in KL(P||Q) and JS(P,Q) are positively biased in finite samples and stay clearly above 0 even when
P and Q are the SAME distribution (true divergence 0). ``bias_correction=True`` (the default) subtracts the
Miller-Madow MI/entropy floor and must measurably shrink the |estimate - truth| gap on the same-distribution case
without erasing detection on a genuine shift. Floors set ~15% below the measured win so seed noise doesn't trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._drift import kl_divergence, js_divergence


def _self_gap(fn, n: int, nbins: int, seeds: int, bias_correction: bool) -> float:
    """Mean |divergence - 0| for two equal-size samples drawn from the SAME standard normal (true divergence 0)."""
    errs = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        errs.append(abs(fn(b, a, nbins=nbins, bias_correction=bias_correction)))
    return float(np.mean(errs))


@pytest.mark.parametrize("fn", [kl_divergence, js_divergence])
def test_biz_val_divergence_bias_correction_shrinks_self_divergence(fn):
    """On same-distribution samples (true=0) the corrected default must cut the |bias| gap by >=2x vs plug-in.

    Measured at n=200 nbins=50: KL plug-in ~0.225 -> corrected ~0.021 (10.5x); JS ~0.064 -> ~0.008 (7.7x).
    Floor 2x is well below the smallest measured ratio; a regressed correction (no subtraction) trips it.
    """
    n, nbins, seeds = 200, 50, 8
    plug = _self_gap(fn, n, nbins, seeds, bias_correction=False)
    corr = _self_gap(fn, n, nbins, seeds, bias_correction=True)
    assert corr < plug / 2.0, f"{fn.__name__}: corrected gap {corr:.5f} should be <= half plug-in {plug:.5f}"


@pytest.mark.parametrize("fn", [kl_divergence, js_divergence])
def test_biz_val_divergence_bias_correction_preserves_shift_detection(fn):
    """The correction must NOT erase a genuine distribution shift: a 1-sigma shift stays clearly above the
    corrected self-divergence floor. Measured separation is large (KL ~0.40, JS ~0.10 above the self floor)."""
    n, nbins, seeds = 200, 50, 8
    self_floor = _self_gap(fn, n, nbins, seeds, bias_correction=True)
    shifted = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        a = rng.normal(1.0, 1.0, n)
        b = rng.normal(0.0, 1.0, n)
        shifted.append(fn(b, a, nbins=nbins, bias_correction=True))
    assert np.mean(shifted) > self_floor + 0.05, f"{fn.__name__}: shifted divergence {np.mean(shifted):.4f} must clear self floor {self_floor:.4f} + 0.05"


@pytest.mark.parametrize("fn", [kl_divergence, js_divergence])
def test_biz_val_divergence_bias_correction_clamps_nonnegative(fn):
    """Subtracting the bias floor must never push the divergence below 0 (clamp at 0)."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    assert fn(b, a, nbins=40, bias_correction=True) >= 0.0


@pytest.mark.parametrize("fn", [kl_divergence, js_divergence])
def test_biz_val_divergence_pre_binned_unaffected_by_correction(fn):
    """The pre_binned path has no counts/sample-sizes, so the correction is a no-op there (identical output)."""
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.4, 0.1, 0.5])
    on = fn(q, p, pre_binned=True, bias_correction=True)
    off = fn(q, p, pre_binned=True, bias_correction=False)
    assert on == off
