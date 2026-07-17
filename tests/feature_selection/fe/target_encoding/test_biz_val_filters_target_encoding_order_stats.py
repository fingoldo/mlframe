"""biz_value: robust order-statistic target encodings beat the plain-mean encoder on a HEAVY-TAILED per-category target.

Synthetic: each category has a true center (the signal); the OBSERVED training y adds heavy-tailed (Student-t, df=1.5)
contamination + outliers. A held-out set is scored against the CLEAN centers -- what a robust encoder should recover.
The mean encoder is dragged around by the fat tails; the median / trimmed-mean encoders recover the center, so their
OOS R^2 clears the mean encoder's by a wide margin. A regression that silently breaks the order-stat path (wrong stat,
lost floor, leaked full-cell) collapses the delta and trips this test.

Measured (seed 42, n=2000, K=25): mean R^2 0.532, median 0.964 (delta 0.432), trimmed_mean 0.922 (delta 0.390).
Assertion floors sit ~30% below the measured deltas to absorb seed noise while still catching a real regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mlframe.feature_selection.filters._target_encoding_fe import apply_target_encoding, kfold_target_encode_fit


def _oos_r2(stat: str, Xtr, y_tr, Xte, y_te_clean) -> float:
    _, rec = kfold_target_encode_fit(Xtr, y_tr, ["c"], stats=(stat,), n_folds=5, random_state=0)
    sl = rec["c"]["stat_lookups"][stat]
    gm = rec["c"]["global_stats"][stat]
    enc = apply_target_encoding(Xte, "c", {"lookup": sl, "global_mean": gm}).reshape(-1, 1)
    model = LinearRegression().fit(enc, y_te_clean)
    return float(r2_score(y_te_clean, model.predict(enc)))


@pytest.fixture(scope="module")
def _heavy_tailed_te_data():
    rng = np.random.default_rng(42)
    n, K = 2000, 25
    centers = rng.normal(0.0, 3.0, K)
    cat_tr = rng.integers(0, K, n)
    cat_te = rng.integers(0, K, n)

    def make_y(cat):
        return centers[cat] + rng.standard_t(1.5, size=len(cat)) * 4.0  # heavy-tailed contamination

    return (
        pd.DataFrame({"c": cat_tr}),
        make_y(cat_tr),
        pd.DataFrame({"c": cat_te}),
        centers[cat_te],  # clean OOS target = true centers
    )


def test_biz_val_te_median_beats_mean_on_heavy_tailed_target(_heavy_tailed_te_data):
    Xtr, y_tr, Xte, y_te_clean = _heavy_tailed_te_data
    r2_mean = _oos_r2("mean", Xtr, y_tr, Xte, y_te_clean)
    r2_median = _oos_r2("median", Xtr, y_tr, Xte, y_te_clean)
    assert r2_median >= r2_mean + 0.30, f"median TE should beat mean by >=0.30 R^2; got {r2_median:.3f} vs {r2_mean:.3f}"


def test_biz_val_te_trimmed_mean_beats_mean_on_heavy_tailed_target(_heavy_tailed_te_data):
    Xtr, y_tr, Xte, y_te_clean = _heavy_tailed_te_data
    r2_mean = _oos_r2("mean", Xtr, y_tr, Xte, y_te_clean)
    r2_trim = _oos_r2("trimmed_mean", Xtr, y_tr, Xte, y_te_clean)
    assert r2_trim >= r2_mean + 0.27, f"trimmed-mean TE should beat mean by >=0.27 R^2; got {r2_trim:.3f} vs {r2_mean:.3f}"
