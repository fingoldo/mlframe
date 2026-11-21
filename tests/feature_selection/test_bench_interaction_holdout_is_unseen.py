"""The interaction-proxy bench scores a refit only on holdout rows the selector never saw (audit 2026-11-20, finding 10-F2)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks.bench_shap_interaction_proxy import honest_auc


def test_holdout_auc_is_chance_when_only_the_search_rows_carry_signal():
    """Signal lives in the search rows, holdout labels are random: a holdout score near chance proves the search rows were not scored."""
    rng = np.random.default_rng(0)
    Xs = pd.DataFrame(rng.standard_normal((800, 2)), columns=["a", "b"])
    ys = (Xs["a"] > 0).astype(int).to_numpy()
    Xh = pd.DataFrame(rng.standard_normal((400, 2)), columns=["a", "b"])
    yh = rng.integers(0, 2, 400)
    auc = honest_auc(Xs, ys, Xh, yh, ["a", "b"], seed=0)
    assert 0.35 < auc < 0.65, auc
