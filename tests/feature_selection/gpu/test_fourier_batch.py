"""The batched Fourier detector returns the same frequency lists as the single-column resident detector."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_batch_gpu import detect_fourier_freqs_batch_gpu
from mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_detect_gpu_resident import detect_fourier_freqs_for_col_gpu

GRID = [round(0.25 * k, 3) for k in range(1, 17)]
N = 6000
PARAMS = dict(min_val_corr=0.15, min_rows=800, max_freqs=4, fourier_detect_max_n=0)


def _jobs(seed: int = 0):
    """Columns with one tone, two tones, a tone of another frequency, pure noise and a constant, each with its own target."""
    r = np.random.default_rng(seed)
    z = [r.random(N) for _ in range(5)]
    def noise():
        """A fresh noise draw per target."""
        return 0.3 * r.standard_normal(N)

    ys = [
        np.sin(2 * np.pi * 3.0 * z[0]) + noise(),
        np.sin(2 * np.pi * 2.0 * z[1]) + 0.8 * np.sin(2 * np.pi * 5.5 * z[1]) + noise(),
        np.cos(2 * np.pi * 4.25 * z[2]) + noise(),
        r.standard_normal(N),
        np.sin(2 * np.pi * 3.0 * z[4]) + noise(),
    ]
    z[4] = np.full(N, 0.5)  # constant column: rejected by the guards
    return [(z[k], ys[k], GRID) for k in range(5)]


def _single(z, y, g):
    """The single-column resident detector with the shared parameters."""
    return detect_fourier_freqs_for_col_gpu(z, y, f_grid=g, **PARAMS)


def test_batch_equals_single_column_detector():
    """Frequency lists agree column by column (tones, a two-tone column, noise, a rejected constant column)."""
    jobs = _jobs()
    got = detect_fourier_freqs_batch_gpu(jobs, single=_single, **PARAMS)
    want = [_single(*j) for j in jobs]
    assert len(got) == len(want)
    for k, (g, w) in enumerate(zip(got, want)):
        assert g == pytest.approx(w, abs=1e-9), f"column {k}: batch {g} single {w}"
    assert any(len(w) for w in want) and want[4] == [] and want[3] == []


def test_columns_with_different_grids_and_a_batch_of_one():
    """Per-column grids of different lengths are padded and ignored correctly; a batch of one equals the single call."""
    jobs = _jobs(1)
    z0, y0, _ = jobs[0]
    mixed = [(z0, y0, GRID), (jobs[1][0], jobs[1][1], GRID[:10]), (jobs[2][0], jobs[2][1], GRID[4:])]
    got = detect_fourier_freqs_batch_gpu(mixed, single=_single, **PARAMS)
    for (z, y, g), res in zip(mixed, got):
        assert res == pytest.approx(_single(z, y, g), abs=1e-9)
    one = detect_fourier_freqs_batch_gpu([jobs[0]], single=_single, **PARAMS)
    assert one[0] == pytest.approx(_single(*jobs[0]), abs=1e-9)


def test_dispatcher_falls_back_to_the_loop_off_the_resident_mode(monkeypatch):
    """Without the strict-resident switch the dispatcher runs the single-column detector per job."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_batch as disp

    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    jobs = _jobs()
    got = disp.detect_fourier_freqs_batch(jobs, min_val_corr=0.15, min_rows=800, max_freqs=4)
    assert len(got) == len(jobs)
    assert got[3] == [] and got[4] == []
