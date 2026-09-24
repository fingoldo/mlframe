"""Discovery memory budgets, as ratios to the data they process.

The leak-corr filter held every candidate column and then stacked a second copy, then the numpy correlation made centred
and float64 copies of that: 3.25x one sample matrix at 200k x 50 on pandas, 2.6x on polars. It now gathers into one block
and runs the correlation kernel on it in place (1.25x: the block plus a boolean non-finite mask).

After the tiny rerank the only discovery data that should stay resident is what later phases or later fits read: the MI
screen's bin codes, which the prebin cache keeps for a re-discovery on the same sample, and the base values of the base
pool. The per-base matrices of the screen and the rerank, and the shared fold caches built on them, must be gone.
"""

from __future__ import annotations

import gc
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytestmark = pytest.mark.slow  # three discovery fits at 30k-200k rows (~3 min)


class _Stop(Exception):
    """Ends a fit right after the phase under test."""


def _frame(n: int, f: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(f)})
    X["y"] = 2 * X["f0"] + np.sin(X["f1"]) + rng.normal(0, 0.3, n)
    return X


def _filter_peak(frame, n: int, f: int, monkeypatch) -> tuple[int, int]:
    """(traced peak of ``_filter_features``, rows of its leak-corr sample)."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.composite.discovery._filter import _leak_corr_sample_rows
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    real = CompositeTargetDiscovery._filter_features
    out = {}

    def spy(self, df, feature_cols, y_train, train_idx):
        real(self, df, feature_cols, y_train, train_idx)  # imports and kernel compilation stay out of the measured call
        gc.collect()
        tracemalloc.start()
        try:
            real(self, df, feature_cols, y_train, train_idx)
            out["peak"] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        sample = _leak_corr_sample_rows(int(np.asarray(train_idx).size))
        out["rows"] = int(np.asarray(train_idx).size) if sample is None else int(sample.size)
        raise _Stop

    monkeypatch.setattr(CompositeTargetDiscovery, "_filter_features", spy)
    with pytest.raises(_Stop):
        CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True)).fit(frame, "y", [f"f{i}" for i in range(f)], np.arange(n))
    return out["peak"], out["rows"]


@pytest.mark.parametrize("carrier", ["pandas", "polars"])
def test_the_leak_corr_filter_peaks_at_one_sample_matrix_and_its_mask(carrier, monkeypatch):
    n, f = 200_000, 50
    frame = _frame(n, f)
    if carrier == "polars":
        pl = pytest.importorskip("polars")
        frame = pl.from_pandas(frame)
    peak, rows = _filter_peak(frame, n, f, monkeypatch)
    sample_matrix = rows * f * 4
    assert peak <= 1.3 * sample_matrix, f"{carrier}: filter peak {peak / 1e6:.1f} MB = {peak / sample_matrix:.2f}x the {sample_matrix / 1e6:.1f} MB sample matrix"


def _resident_after_rerank(n: int, f: int, bases: int, monkeypatch) -> tuple[int, int]:
    """(traced bytes held at the rerank checkpoint beyond the fit's entry, screen rows)."""
    import mlframe.training.composite.discovery._fit as fit_mod
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    at = {}
    real_prebin = fit_mod._prebin_feature_columns_cached

    def prebin(matrix, **k):
        at["screen_rows"] = int(matrix.shape[0])
        return real_prebin(matrix, **k)

    def report(state, tag):
        if tag in ("entry", "tiny_model_rerank_done"):
            gc.collect()
            at[tag] = tracemalloc.get_traced_memory()[0]

    def fit(frame, screen_n):
        cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, mi_sample_n=screen_n, base_candidates=[f"f{i}" for i in range(bases)],
                                             transforms=["diff", "linear_residual"], interaction_base_discovery_enabled=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            CompositeTargetDiscovery(cfg).fit(frame, "y", [f"f{i}" for i in range(f)], np.arange(len(frame)))

    fit(_frame(3000, f, seed=1), 2000)  # imports, kernel compilation and one-time caches stay out of the measured fit
    monkeypatch.setattr(fit_mod, "_phase_ram_report", report)
    monkeypatch.setattr(fit_mod, "_prebin_feature_columns_cached", prebin)
    monkeypatch.setenv("MLFRAME_DISCOVERY_RAM_PROFILER", "1")
    tracemalloc.start()
    try:
        fit(_frame(n, f).astype({f"f{i}": np.float32 for i in range(f)}), n)
    finally:
        tracemalloc.stop()
    assert "tiny_model_rerank_done" in at, "the rerank checkpoint was not reached"
    return at["tiny_model_rerank_done"] - at["entry"], at["screen_rows"]


def test_after_the_rerank_only_the_bin_codes_the_base_pool_and_one_fold_scheme_stay_resident(monkeypatch):
    """The Ridge fold factors of the rerank's per-base matrices (F x F each, up to 64) used to outlive them."""
    n, f, bases = 30_000, 100, 6
    extra, screen_rows = _resident_after_rerank(n, f, bases, monkeypatch)
    codes = screen_rows * f * 2  # int16 bin codes, kept by the prebin cache for a re-discovery on the same sample
    base_pool = bases * screen_rows * 4  # float32 base values per base candidate
    folds = 5 * screen_rows * 8  # the cached shuffled-KFold index arrays of the screen (train + validation per fold)
    vectors = 12 * screen_rows * 8  # the fit's own row vectors: y, the train / screen / holdout indices, masks
    budget = codes + base_pool + folds + vectors
    assert extra <= budget, (
        f"{extra / 1e6:.2f} MB resident after the rerank, over codes {codes / 1e6:.2f} + base pool {base_pool / 1e6:.2f} + "
        f"folds {folds / 1e6:.2f} + row vectors {vectors / 1e6:.2f} MB"
    )
