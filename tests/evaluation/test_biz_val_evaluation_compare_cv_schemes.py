"""biz_value test for ``evaluation.compare_cv_schemes``.

The win: reproduces the source's counter-intuitive finding -- when the true out-of-time scenario has the
SAME entities continuing into the future (a realistic "existing customers keep transacting" production
setup), GroupKFold's held-out-entity simulation is an artificially pessimistic proxy (it tests a cold-start
scenario that never actually happens), while plain KFold (which lets the model learn each entity's fixed
effect from its other rows) tracks the true out-of-time score far more closely. ``compare_cv_schemes`` should
correctly pick KFold as the best scheme here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold

from mlframe.evaluation.compare_cv_schemes import compare_cv_schemes


def _rmse(y_true, y_pred):
    """Returns ``float(np.sqrt(mean_squared_error(y_true, y_pred)))``."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_biz_val_compare_cv_schemes_picks_kfold_when_entities_persist_into_the_future():
    """Compare cv schemes picks kfold when entities persist into the future."""
    rng = np.random.default_rng(0)
    n_entities = 60
    rows_hist, rows_future = 40, 10
    entity_offset = rng.normal(0, 3, n_entities)

    rows = []
    for e in range(n_entities):
        for t in range(rows_hist + rows_future):
            rows.append({"entity": e, "t": t, "y": 10 + 0.05 * t + entity_offset[e] + rng.normal(0, 0.5)})
    df = pd.DataFrame(rows)
    X = df[["entity", "t"]]
    y = df["y"].to_numpy()

    is_hist = df["t"].to_numpy() < rows_hist
    hist_idx = np.flatnonzero(is_hist)
    future_idx = np.flatnonzero(~is_hist)

    kfold_splits = [(hist_idx[tr], hist_idx[te]) for tr, te in KFold(5, shuffle=True, random_state=0).split(hist_idx)]
    groupkfold_splits = [(hist_idx[tr], hist_idx[te]) for tr, te in GroupKFold(5).split(hist_idx, groups=df["entity"].to_numpy()[hist_idx])]

    result = compare_cv_schemes(
        X,
        y,
        schemes={"kfold": kfold_splits, "groupkfold": groupkfold_splits},
        ooo_time_idx=(hist_idx, future_idx),
        model_factory=lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0),
        metric_fn=_rmse,
    )

    assert result["best_scheme"] == "kfold", result
    assert result["scheme_scores"]["kfold"]["gap_to_ooo_time"] < result["scheme_scores"]["groupkfold"]["gap_to_ooo_time"] * 0.5


def test_biz_val_compare_cv_schemes_significance_check_rejects_noise_level_win():
    """Two CV schemes whose point-estimate gaps differ only by fold-resampling noise: a naive point-estimate
    comparison always crowns a "winner" (whichever gap happens to be marginally smaller on this particular
    fold split), but ``significance_alpha`` should correctly report that the win doesn't clear the noise band.
    """
    rng = np.random.default_rng(1)
    n_rows = 400
    X = rng.normal(0, 1, (n_rows, 3))
    y = X[:, 0] * 2 + rng.normal(0, 1.0, n_rows)

    hist_idx = np.arange(int(n_rows * 0.8))
    future_idx = np.arange(int(n_rows * 0.8), n_rows)

    # Two KFold splitters differing only by random_state -- same scheme, same noise-generating process,
    # so any observed gap difference between them is pure resampling noise, not a real methodological edge.
    splits_a = [(hist_idx[tr], hist_idx[te]) for tr, te in KFold(5, shuffle=True, random_state=0).split(hist_idx)]
    splits_b = [(hist_idx[tr], hist_idx[te]) for tr, te in KFold(5, shuffle=True, random_state=1).split(hist_idx)]

    result = compare_cv_schemes(
        X,
        y,
        schemes={"kfold_a": splits_a, "kfold_b": splits_b},
        ooo_time_idx=(hist_idx, future_idx),
        model_factory=lambda: RandomForestRegressor(n_estimators=50, max_depth=4, random_state=0),
        metric_fn=_rmse,
        significance_alpha=0.05,
    )

    # A naive point-estimate comparison always declares a winner -- gaps are almost never exactly tied.
    gap_a = result["scheme_scores"]["kfold_a"]["gap_to_ooo_time"]
    gap_b = result["scheme_scores"]["kfold_b"]["gap_to_ooo_time"]
    assert gap_a != gap_b, "fixture must produce a nonzero point-estimate gap difference for the test to be meaningful"

    # The significance check must correctly flag that "winner" as not statistically distinguishable from noise.
    assert result["best_scheme_significant"] is False
    other = "kfold_b" if result["best_scheme"] == "kfold_a" else "kfold_a"
    assert result["significance"][other]["actionable"] is False


def test_compare_cv_schemes_empty_schemes_returns_none_best():
    """Compare cv schemes empty schemes returns none best."""
    X = np.zeros((10, 1))
    y = np.zeros(10)
    result = compare_cv_schemes(
        X,
        y,
        schemes={},
        ooo_time_idx=(np.arange(5), np.arange(5, 10)),
        model_factory=lambda: RandomForestRegressor(n_estimators=5),
        metric_fn=_rmse,
    )
    assert result["best_scheme"] is None
    assert result["scheme_scores"] == {}
