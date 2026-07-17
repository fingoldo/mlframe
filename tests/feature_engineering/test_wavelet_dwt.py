"""Tests for ``mlframe.feature_engineering.wavelet_dwt`` -- numba DWT
correctness vs pywt, batched-vs-single agreement, denoise contract."""

from __future__ import annotations

import numpy as np
import pytest

pywt = pytest.importorskip("pywt")


@pytest.fixture(scope="module")
def signal_set():
    rng = np.random.default_rng(0)
    return {
        "len32": rng.normal(0, 1, 32),
        "len64": rng.normal(0, 1, 64),
        "len100": rng.normal(0, 1, 100),
        "len256": rng.normal(0, 1, 256),
    }


WAVELETS = ["haar", "db4", "db6", "coif3", "sym4"]


class TestWavedecMatchesPywt:
    @pytest.mark.parametrize("wavelet", WAVELETS)
    @pytest.mark.parametrize("sig_key", ["len32", "len64", "len100", "len256"])
    def test_wavedec_matches_pywt(self, signal_set, wavelet, sig_key):
        from mlframe.feature_engineering.wavelet_dwt import wavedec

        sig = signal_set[sig_key]
        max_level = min(4, pywt.dwt_max_level(len(sig), wavelet))
        ref = pywt.wavedec(sig, wavelet, level=max_level, mode="symmetric")
        got = wavedec(sig, wavelet, max_level)
        assert len(ref) == len(got)
        for c_ref, c_got in zip(ref, got):
            assert c_ref.shape == c_got.shape
            np.testing.assert_allclose(c_ref, c_got, atol=1e-10)


class TestWavedecBatched:
    def test_batched_matches_single_loop(self, signal_set):
        pytest.importorskip("numba")
        from mlframe.feature_engineering.wavelet_dwt import (
            wavedec,
            wavedec_batched,
        )

        # Build (N=64, T=100) batch
        rng = np.random.default_rng(7)
        sigs = rng.normal(0, 1, (64, 100))
        max_level = 3
        wavelet = "db4"
        # Per-signal loop reference
        loop_coeffs = [wavedec(sigs[i], wavelet, max_level) for i in range(64)]
        # Batched
        approx, details_flat, lengths = wavedec_batched(sigs, wavelet, max_level)
        # Check shapes
        assert approx.shape == (64, lengths[-1])
        assert details_flat.shape == (64, int(lengths[1:].sum()))
        # Verify each signal: batched approx must match loop's [0] coeff;
        # batched details (level-1 first in flat layout) must match
        # loop coeffs reversed (loop is [approx, det_N, ..., det_1]).
        for i in range(64):
            # Loop format: [approx, det_max, ..., det_1] (pywt order).
            # Batched: approx + concatenated details level-1-first.
            np.testing.assert_allclose(
                approx[i],
                loop_coeffs[i][0],
                atol=1e-10,
                err_msg=f"approx mismatch at signal {i}",
            )
            offset = 0
            for level in range(max_level):
                lvl_len = int(lengths[level + 1])
                batched_lvl = details_flat[i, offset : offset + lvl_len]
                loop_lvl = loop_coeffs[i][-(level + 1)]  # level-1 is last
                np.testing.assert_allclose(
                    batched_lvl,
                    loop_lvl,
                    atol=1e-10,
                    err_msg=f"detail L{level + 1} mismatch at signal {i}",
                )
                offset += lvl_len


class TestWaverecRoundTrip:
    @pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
    def test_wavedec_then_waverec_recovers_signal(self, wavelet):
        from mlframe.feature_engineering.wavelet_dwt import wavedec, waverec

        rng = np.random.default_rng(11)
        sig = rng.normal(0, 1, 128)
        coeffs = wavedec(sig, wavelet, 3)
        rec = waverec(coeffs, wavelet)
        # Reconstruction may be 1 sample longer/shorter than original
        # (pywt convention with odd boundary cases); compare overlap.
        n = min(len(sig), len(rec))
        np.testing.assert_allclose(rec[:n], sig[:n], atol=1e-9)

    @pytest.mark.parametrize("wavelet", ["db4", "sym4"])
    def test_waverec_matches_pywt(self, wavelet):
        from mlframe.feature_engineering.wavelet_dwt import wavedec, waverec

        rng = np.random.default_rng(13)
        sig = rng.normal(0, 1, 128)
        coeffs = wavedec(sig, wavelet, 3)
        rec_our = waverec(coeffs, wavelet)
        rec_pywt = pywt.waverec(coeffs, wavelet, mode="symmetric")
        n = min(len(rec_our), len(rec_pywt))
        np.testing.assert_allclose(rec_our[:n], rec_pywt[:n], atol=1e-9)


class TestWaveletDenoise:
    def test_denoise_reduces_noise_variance(self):
        from mlframe.feature_engineering.wavelet_dwt import wavelet_denoise

        rng = np.random.default_rng(17)
        n = 1024
        t = np.arange(n) / 50.0
        clean = np.sin(t)
        noise = 0.5 * rng.normal(0, 1, n)
        noisy = clean + noise
        denoised = wavelet_denoise(noisy, wavelet="db4", level=3)
        # Denoised signal must be MEASURABLY closer to clean than the
        # noisy input.
        mse_noisy = float(np.mean((noisy - clean) ** 2))
        mse_denoised = float(np.mean((denoised[:n] - clean) ** 2))
        assert mse_denoised < mse_noisy * 0.6, f"denoise did not reduce noise enough: noisy MSE={mse_noisy:.4f}, denoised MSE={mse_denoised:.4f}"

    def test_denoise_hard_vs_soft(self):
        """Hard threshold preserves above-threshold coefficients verbatim;
        soft shrinks them by the threshold. On a noisy signal both
        should reduce variance, but they produce different outputs."""
        from mlframe.feature_engineering.wavelet_dwt import wavelet_denoise

        rng = np.random.default_rng(19)
        sig = rng.normal(0, 1, 512)
        soft = wavelet_denoise(sig, wavelet="db4", level=2, threshold=0.5, mode="soft")
        hard = wavelet_denoise(sig, wavelet="db4", level=2, threshold=0.5, mode="hard")
        assert not np.allclose(soft, hard)


class TestFilterCache:
    def test_filters_cached_on_repeat_call(self):
        from mlframe.feature_engineering.wavelet_dwt import (
            get_wavelet_filters,
            _FILTER_CACHE,
        )

        _FILTER_CACHE.clear()
        _ = get_wavelet_filters("db4")
        assert "db4" in _FILTER_CACHE
        # Mutate cache entry; second call must return the mutated value
        # (proves caching, not re-fetch).
        sentinel = (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))
        _FILTER_CACHE["db4"] = sentinel
        got = get_wavelet_filters("db4")
        assert got is sentinel
