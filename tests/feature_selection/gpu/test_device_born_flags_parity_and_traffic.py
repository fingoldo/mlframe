"""residency_audit()-based parity+traffic tests for the 5 under-tested DEFAULT-ON device_born_* flags
(mrmr_audit_2026-07-20 gpu_residency.md #3 + #4).

Prior to this file, grepping the repo for MLFRAME_FE_GPU_DEVICE_BORN_BINAGG / _DISPERSION /
_DUAL_UPLIFT / _WAVELET returned zero hits anywhere -- none of these DEFAULT-ON flags had a test that
actually flips the env var and compares host vs device-born output, even though each carries a specific
bit-identity/selection-equivalence claim in its own docstring. This file adds that toggle + compare for
all five (the four named above, plus an explicit toggle for the previously-untoggled EXTRA_BASIS test),
using ``tests/feature_selection/gpu/test_resident_311_residual_parity.py``'s pattern (monkeypatch env
var + before/after selection comparison).

``test_wavelet_batched_mi_parity`` here also resolves gpu_residency.md #4: the wavelet device-born
docstring (``_gpu_strict_fe/_entry.py::fe_gpu_device_born_wavelet_enabled``) and
``_wavelet_basis_fe_batched.py`` both cite a test of exactly this name as the thing that "pins" the
partition-equivalence claim; before this file no such test existed anywhere in the repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is available (used to skip the module when it is not)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _strict(monkeypatch):
    """Turn on STRICT-residency (the umbrella gate every device_born_* flag also requires)."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")


# ---------------------------------------------------------------------------
# BINAGG
# ---------------------------------------------------------------------------
def test_binagg_device_born_parity(monkeypatch):
    """DEVICE_BORN_BINAGG toggled off vs on must select the SAME binned_numeric_agg survivor columns
    (the device gate scores an on-device-rebuilt OOF from the SAME recipes, never reading host OOF
    values, so the survivor set is byte-identical to fitting every candidate on the host)."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

    rng = np.random.default_rng(11)
    n = 6000
    g1 = rng.integers(0, 6, n).astype(np.float64)
    g2 = rng.integers(0, 6, n).astype(np.float64)
    a1 = rng.standard_normal(n)
    a2 = rng.standard_normal(n) * (1.0 + g1)  # group-dependent scale -> genuine per-cell signal
    X = pd.DataFrame({"g1": g1, "g2": g2, "a1": a1, "a2": a2})
    y = (a2 > np.median(a2)).astype(np.int64)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_BINAGG", "0")
    _, appended_host, _ = binned_numeric_agg_with_recipes(X, y, n_folds=3, max_pairs=8, max_group_cols=2, max_agg_cols=2)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_BINAGG", "1")
    with residency_audit() as rep:
        _, appended_dev, _ = binned_numeric_agg_with_recipes(X, y, n_folds=3, max_pairs=8, max_group_cols=2, max_agg_cols=2)

    assert appended_host == appended_dev, f"BINAGG selection differs: host {appended_host} vs device-born {appended_dev}"
    # the ~120MB+ host OOF matrix documented as the H2D this flag collapses must not appear as bulk D2H/H2D
    # in the audited (flag-on) call.
    assert not rep.bulk_h2d or all(b < 20 * n for b in rep.bulk_h2d), f"unexpected large H2D under BINAGG device-born: {rep.summary()}"


# ---------------------------------------------------------------------------
# DISPERSION + DUAL_UPLIFT (both gate inside hybrid_conditional_dispersion_fe)
# ---------------------------------------------------------------------------
def _hetero_fixture(seed: int = 0, n: int = 5000):
    """Two-bin heteroscedastic fixture (shared with test_conditional_dispersion_fe.py): bin A x_i~N(0,1),
    bin B x_i~N(0,5); y flags rows anomalous for their OWN conditional spread. Decoy noise columns
    (g1, g2) give a non-degenerate raw noise floor -- a bare 2-col raw frame yields a degenerate MAD
    floor that gates everything out (mirrors test_conditional_dispersion_fe.py's own fixture note)."""
    rng = np.random.default_rng(seed)
    xj = rng.random(n)
    binB = xj > 0.5
    sd = np.where(binB, 5.0, 1.0)
    xi = rng.standard_normal(n) * sd
    y = (np.abs(xi) > 2.0 * sd).astype(int)
    rng2 = np.random.default_rng(seed + 99)
    X = pd.DataFrame({"xi": xi, "xj": xj, "g1": rng2.standard_normal(n), "g2": rng2.standard_normal(n)})
    return X, y


def test_dispersion_device_born_parity(monkeypatch):
    """DEVICE_BORN_DISPERSION toggled off vs on must admit the SAME conditional-dispersion columns (the
    device gate rebuilds the candidate matrix from the frozen recipes; only the per-row f64 divide
    differs at ULP per the flag's own docstring, well below the binning edges)."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._extra_fe_families_dispersion import hybrid_conditional_dispersion_fe
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

    X, y = _hetero_fixture()

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION", "0")
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DUAL_UPLIFT", "0")
    _, appended_host, _, _ = hybrid_conditional_dispersion_fe(X, y, n_bins=10, top_k=10, max_pair_cols=6)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION", "1")
    with residency_audit() as rep:
        _, appended_dev, _, _ = hybrid_conditional_dispersion_fe(X, y, n_bins=10, top_k=10, max_pair_cols=6)

    assert set(appended_host) == set(appended_dev), f"DISPERSION selection differs: host {sorted(appended_host)} vs device-born {sorted(appended_dev)}"
    assert appended_dev, "the heteroscedastic fixture must admit at least one dispersion column (else this test can't tell parity from both-empty)"
    print("DISPERSION device-born residency: " + rep.summary())


def test_dual_uplift_device_born_parity(monkeypatch):
    """DEVICE_BORN_DUAL_UPLIFT toggled off vs on must admit the SAME columns (the sibling |residual| MI
    comparison is additive on the SAME resident plug-in estimator both cand/raw/sibling MI already use
    under STRICT, so there is no uplift-ratio/baseline-mismatch flip surface)."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._extra_fe_families_dispersion import hybrid_conditional_dispersion_fe

    X, y = _hetero_fixture(seed=3)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION", "0")
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DUAL_UPLIFT", "0")
    _, appended_host, _, _ = hybrid_conditional_dispersion_fe(X, y, n_bins=10, top_k=10, max_pair_cols=6)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DISPERSION", "0")  # isolate: dual_uplift alone
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_DUAL_UPLIFT", "1")
    _, appended_dev, _, _ = hybrid_conditional_dispersion_fe(X, y, n_bins=10, top_k=10, max_pair_cols=6)

    assert set(appended_host) == set(appended_dev), f"DUAL_UPLIFT selection differs: host {sorted(appended_host)} vs device-born {sorted(appended_dev)}"
    assert appended_dev, "the heteroscedastic fixture must admit at least one dispersion column (else this test can't tell parity from both-empty)"


# ---------------------------------------------------------------------------
# WAVELET -- pins the partition-equivalence claim cited (but previously nonexistent) as
# test_wavelet_batched_mi_parity by _wavelet_basis_fe_batched.py and _gpu_strict_fe/_entry.py.
# ---------------------------------------------------------------------------
def test_wavelet_batched_mi_parity(monkeypatch):
    """The DEVICE-BORN batched wavelet leg-rank MI (select_wavelet_legs_batched, gated by
    fe_gpu_device_born_wavelet_enabled) must admit the SAME (j, k) legs as the exact host per-leg path --
    the partition-equivalence claim both _wavelet_basis_fe_batched.py's module docstring and
    _gpu_strict_fe/_entry.py's fe_gpu_device_born_wavelet_enabled docstring cite this exact test name for."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._wavelet_basis_fe_batched import select_wavelet_legs_batched

    rng = np.random.default_rng(5)
    n = 6000
    x = rng.uniform(0.0, 1.0, n)
    # a genuine localized bump: y jumps only inside a narrow sub-window of x (Haar-leg-detectable,
    # invisible to a coarse/smooth binning) -- the scenario the wavelet family exists to catch.
    y = np.where((x > 0.6) & (x < 0.68), 1.0, 0.0) + 0.05 * rng.standard_normal(n)
    lo, span = float(x.min()), float(x.max() - x.min())

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_WAVELET", "0")
    legs_host = select_wavelet_legs_batched(x, y, lo, span, max_scale=4, max_legs=8, scale_sigma=1.0)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_WAVELET", "1")
    legs_dev = select_wavelet_legs_batched(x, y, lo, span, max_scale=4, max_legs=8, scale_sigma=1.0)

    assert legs_host, "the localized-bump fixture must admit at least one Haar leg on the host path"
    assert set(legs_host) == set(legs_dev), f"wavelet leg admission differs: host {sorted(legs_host)} vs device-born {sorted(legs_dev)}"


# ---------------------------------------------------------------------------
# EXTRA_BASIS -- an explicit on/off toggle comparison (the existing
# test_extra_basis_device_born_parity.py exercises the device builder function directly but never
# flips the env var to compare against the actual gated caller's host fallback).
# ---------------------------------------------------------------------------
def test_extra_basis_device_born_toggle_matches_host_uplift_mi(monkeypatch):
    """extra_basis_eng_mi_resident (gated by DEVICE_BORN_EXTRA_BASIS) must return the SAME MI values
    (within the documented ~1e-5 relaxed-dtype tolerance) as the exact host mi_classif_batch scorer over
    the same engineered matrix, when the flag is explicitly toggled on vs off."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import generate_extra_basis_features
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_resident import extra_basis_eng_mi_resident
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    rng = np.random.default_rng(0)
    n = 6000
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(0.1, 5, n)
    d = rng.uniform(0, 2 * np.pi, n)
    X = pd.DataFrame({"a": a, "b": b, "d": d})
    y = np.sin(d) + np.sin(2.0 * ((a - a.mean()) / a.std()) ** 2) + (b > 2.5).astype(float) + 0.05 * rng.standard_normal(n)
    y_bin = (y > np.median(y)).astype(np.int64)

    eng, meta = generate_extra_basis_features(X, cols=["a", "b", "d"], extra_bases=("spline", "fourier", "wavelet"), y=y, fourier_adaptive=True, fourier_chirp=True)
    assert eng.shape[1] > 0

    host_mi = np.asarray(_mi_classif_batch(eng.to_numpy(dtype=np.float64), y_bin, nbins=10), dtype=np.float64)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_EXTRA_BASIS", "0")
    off = extra_basis_eng_mi_resident(X, eng, y_bin, meta, nbins=10)
    assert off is None, "the flag must gate the device path off; the caller should fall back to the host scorer"

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_EXTRA_BASIS", "1")
    dev_mi = extra_basis_eng_mi_resident(X, eng, y_bin, meta, nbins=10)
    assert dev_mi is not None, "the device-born path should engage under STRICT-residency with the flag on"
    np.testing.assert_allclose(dev_mi, host_mi, rtol=1e-5, atol=1e-5)
