"""The benchmark refit is scored only on rows the selector never saw (audit 2026-11-20, finding 10-F1)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.bench_shapproxied_adaptive_guards import _fit_holdout_auc


def test_holdout_auc_ignores_the_fit_rows():
    """The fit rows carry the signal, the holdout labels are random: an honest holdout AUC is near chance. Re-splitting the
    concatenated frame scored the fit rows too and reported a high AUC."""
    rng = np.random.default_rng(0)
    n = 800
    Xfit = pd.DataFrame(rng.standard_normal((n, 2)), columns=["a", "b"])
    yfit = pd.Series((Xfit["a"] > 0).astype(int).to_numpy())
    Xhold = pd.DataFrame(rng.standard_normal((400, 2)), columns=["a", "b"])
    yhold = pd.Series(rng.integers(0, 2, 400))
    auc = _fit_holdout_auc(Xfit, yfit, Xhold, yhold, ["a", "b"], seed=0)
    assert 0.35 < auc < 0.65, auc
