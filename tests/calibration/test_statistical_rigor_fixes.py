"""Regression tests for the calibration statistical-rigor fixes.

Covers:
  1. ``_ece_score`` validates + normalises non-{0,1} labels ({-1,+1}, {1,2}) instead of
     silently computing a wrong per-bin accuracy.
  2. ``_stratified_inner_folds`` raises on >2 classes instead of a global (un-stratified) shuffle.
  3. ``_heldout_ece_ci`` uses a Student-t quantile -> wider interval than the old normal z at k=5.
  7. ``generate_probs_from_outcomes`` is deterministic per random_state under concurrent threads
     (per-call Generator, no njit-global RNG race).
  8. ``_build_resample_indices`` raises MemoryError above the RAM ceiling, not below.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mlframe.calibration.policy import (
    _ece_score,
    _stratified_inner_folds,
    _heldout_ece_ci,
    _build_resample_indices,
)


# ---------------------------------------------------------------------------
# Finding 1: non-{0,1} label guard / normalisation in _ece_score
# ---------------------------------------------------------------------------
def test_ece_score_normalizes_pm1_labels():
    """Ece score normalizes pm1 labels."""
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y01 = (rng.random(200) < p).astype(np.int64)
    # {-1,+1} encoding with the SAME positive class must give the SAME ECE as {0,1}.
    y_pm1 = np.where(y01 == 1, 1, -1).astype(np.int64)
    ece01 = _ece_score(y01, p)
    ece_pm1 = _ece_score(y_pm1, p)
    assert np.isfinite(ece01)
    assert ece01 == pytest.approx(ece_pm1, abs=1e-12)


def test_ece_score_normalizes_12_labels():
    """Ece score normalizes 12 labels."""
    rng = np.random.default_rng(1)
    p = rng.random(200)
    y01 = (rng.random(200) < p).astype(np.int64)
    y_12 = (y01 + 1).astype(np.int64)  # {1,2}, positive class == 2
    ece01 = _ece_score(y01, p)
    ece_12 = _ece_score(y_12, p)
    assert ece01 == pytest.approx(ece_12, abs=1e-12)


def test_ece_score_non_float64_fastpath_also_normalizes():
    # Exercise the coercion (non-fast) path with a float32 prob array and {1,2} labels.
    """Ece score non float64 fastpath also normalizes."""
    rng = np.random.default_rng(2)
    p = rng.random(150).astype(np.float32)
    y01 = (rng.random(150) < 0.5).astype(np.int64)
    y_12 = (y01 + 1).astype(np.int64)
    assert _ece_score(y01, p) == pytest.approx(_ece_score(y_12, p), abs=1e-12)


def test_ece_score_raises_on_non_binary():
    """Ece score raises on non binary."""
    p = np.linspace(0.01, 0.99, 30)
    y_multi = np.array([0, 1, 2] * 10)
    with pytest.raises(ValueError, match="exactly 2 distinct"):
        _ece_score(y_multi, p)


# ---------------------------------------------------------------------------
# Finding 2: multiclass raise in _stratified_inner_folds
# ---------------------------------------------------------------------------
def test_stratified_inner_folds_binary_ok():
    """Stratified inner folds binary ok."""
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    folds = _stratified_inner_folds(y, n_splits=3, random_state=0)
    assert len(folds) == 3
    assert sum(f.size for f in folds) == y.size


def test_stratified_inner_folds_raises_multiclass():
    """Stratified inner folds raises multiclass."""
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        _stratified_inner_folds(y, n_splits=3, random_state=0)


# ---------------------------------------------------------------------------
# Finding 3: Student-t CI wider than normal-z at k=5
# ---------------------------------------------------------------------------
def test_heldout_ece_ci_student_t_wider_than_normal_z():
    """Heldout ece ci student t wider than normal z."""
    from scipy.stats import norm

    fold_eces = [0.010, 0.012, 0.009, 0.014, 0.011]
    k = len(fold_eces)
    alpha = 0.05
    lo, hi = _heldout_ece_ci(float(np.mean(fold_eces)), fold_eces, alpha)
    half_t = (hi - lo) / 2.0

    arr = np.asarray(fold_eces, dtype=np.float64)
    se = float(np.std(arr, ddof=1)) / np.sqrt(k)
    half_z = float(norm.ppf(1.0 - alpha / 2.0)) * se

    # t_{k-1} quantile > z quantile, so the reported half-width is strictly larger.
    assert half_t > half_z
    # centred on the mean
    assert (lo + hi) / 2.0 == pytest.approx(float(np.mean(arr)), abs=1e-12)


def test_heldout_ece_ci_single_fold_degenerate():
    """Heldout ece ci single fold degenerate."""
    lo, hi = _heldout_ece_ci(0.02, [0.02], 0.05)
    assert lo == hi == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Finding 7: generate_probs_from_outcomes deterministic under threads
# ---------------------------------------------------------------------------
def test_generate_probs_deterministic_under_threads():
    """Generate probs deterministic under threads."""
    from mlframe.calibration.probabilities import generate_probs_from_outcomes

    outcomes = (np.arange(500) % 2).astype(np.int64)
    ref = generate_probs_from_outcomes(outcomes, random_state=7)

    # Fire many concurrent calls with the SAME seed AND interleave a DIFFERENT seed.
    # A process/njit-global RNG seed inside the kernel would let concurrent seed=9 calls
    # clobber the seed=7 stream -> non-deterministic ref reproduction.
    def _same():
        """Helper that same."""
        return generate_probs_from_outcomes(outcomes, random_state=7)

    def _other():
        """Helper that other."""
        return generate_probs_from_outcomes(outcomes, random_state=9)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_same if i % 2 == 0 else _other) for i in range(32)]
        results = [(i, f.result()) for i, f in enumerate(futs)]

    for i, r in results:
        if i % 2 == 0:
            assert np.array_equal(r, ref), "seed=7 output changed under concurrency"


def test_generate_probs_seed_determinism_and_global_untouched():
    """Generate probs seed determinism and global untouched."""
    from mlframe.calibration.probabilities import generate_probs_from_outcomes

    outcomes = (np.arange(500) % 2).astype(np.int64)
    st = np.random.get_state()
    a = generate_probs_from_outcomes(outcomes, random_state=5)
    b = generate_probs_from_outcomes(outcomes, random_state=5)
    c = generate_probs_from_outcomes(outcomes, random_state=6)
    after = np.random.get_state()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # global numpy RNG untouched
    assert st[0] == after[0] and np.array_equal(st[1], after[1]) and st[2:] == after[2:]


# ---------------------------------------------------------------------------
# Finding 8: RAM ceiling guard for _build_resample_indices
# ---------------------------------------------------------------------------
def test_build_resample_indices_raises_above_ceiling(monkeypatch):
    # Set a tiny ceiling so a modest matrix trips it. 100 bootstraps x 100 rows x 8 = 80_000 bytes.
    """Build resample indices raises above ceiling."""
    monkeypatch.setenv("MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES", "10000")
    with pytest.raises(MemoryError, match="exceeding"):
        _build_resample_indices(n=100, n_bootstrap=100, stratify=None, random_state=0)


def test_build_resample_indices_ok_below_ceiling(monkeypatch):
    """Build resample indices ok below ceiling."""
    monkeypatch.setenv("MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES", str(1 << 30))
    out = _build_resample_indices(n=100, n_bootstrap=100, stratify=None, random_state=0)
    assert out.shape == (100, 100)


def test_build_resample_indices_default_ceiling_allows_typical(monkeypatch):
    """Build resample indices default ceiling allows typical."""
    monkeypatch.delenv("MLFRAME_CALIBRATION_RESAMPLE_MAX_BYTES", raising=False)
    # Typical calibration set: 5000 rows x 1000 bootstraps x 8 = 40 MB << 1 GiB default.
    out = _build_resample_indices(n=5000, n_bootstrap=1000, stratify=None, random_state=0)
    assert out.shape == (1000, 5000)
