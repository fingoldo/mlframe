"""The auto scorer scores the column-separable scorers (plug-in, KSG, copula, dCor, HSIC, xi) in one batched call per bootstrap resample; the output must equal the per-column path."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._orth_auto_scorer_fe as auto


def _frame(n: int = 1200, p: int = 6, seed: int = 0, ties: bool = False) -> pd.DataFrame:
    """A numeric frame of ``p`` standard-normal columns, optionally rounded so the columns carry ties."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    return frame.round(1) if ties else frame


@pytest.mark.parametrize("target", ["binary", "multiclass"])
@pytest.mark.parametrize("ties", [False, True])
def test_batched_bootstrap_scores_equal_the_per_column_path(monkeypatch, target, ties):
    """Same winners, same LCBs and same uplift with the batch on and off."""
    X = _frame(ties=ties)
    rng = np.random.default_rng(1)
    signal = X["x0"].to_numpy() * X["x1"].to_numpy() + 0.5 * rng.normal(size=len(X))
    y = (signal > 0).astype(int) if target == "binary" else np.digitize(signal, [-0.5, 0.5]).astype(int)
    batched = auto.hybrid_orth_mi_auto_scorer_fe(X, y)
    monkeypatch.setattr(auto, "_batched_bootstrap_values", lambda *a, **k: None)
    per_column = auto.hybrid_orth_mi_auto_scorer_fe(X, y)
    got = batched[0] if isinstance(batched, tuple) else batched
    ref = per_column[0] if isinstance(per_column, tuple) else per_column
    assert got.equals(ref)


def test_batch_is_skipped_for_a_continuous_target():
    """A float target is left to the per-column scorers (the batch declines it)."""
    cols = [np.random.default_rng(2).normal(size=300)]
    y = np.random.default_rng(3).normal(size=300)
    boot = [np.arange(300)]
    hyper = {"nbins": 10, "n_neighbors": 3, "copula_n_bins": 20, "dcor_n_sample": 100, "n_jobs": 1}
    assert auto._batched_bootstrap_values(cols, y, boot, 0, hyper) is None
