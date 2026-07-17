"""biz_value: the distribution-driven (E3) TailComposite estimator wins on heavy-tail targets.

The analyzer recommends ``TailCompositeEstimator`` for heavy-tail targets; its GPD-extrapolated upper-tail
quantile tracks the nominal extreme coverage far better than a Gaussian-tail assumption (which mis-estimates
the deep tail of a Student-t target). Measured at q=0.999: GPD |coverage-nominal| ~0.0004 vs Gaussian ~0.0059
(~14x closer). Floor is set well inside that margin so the win is detected but seed noise does not trip it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_biz_val_tail_composite_extreme_quantile_beats_gaussian():
    from scipy.stats import norm
    from sklearn.linear_model import LinearRegression

    from mlframe.training.composite._estimator_dispatch import (
        instantiate_recommended_estimator,
        recommend_composite_estimator,
    )

    rec = recommend_composite_estimator(["heavy_tail(excess_kurt=18.0)"])
    assert rec["estimator"] == "TailCompositeEstimator"

    q = 0.999
    gpd_errs, gauss_errs = [], []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        n = 8000
        f0 = rng.normal(size=n)
        y = 2.0 * f0 + rng.standard_t(2.5, size=n)  # heavy tail (excess kurtosis ~ unbounded for df<4)
        X = pd.DataFrame({"f0": f0})
        tr, te = slice(0, 4000), slice(4000, n)

        est = instantiate_recommended_estimator(rec, base_estimator=LinearRegression(), transform_name="diff", base_column="f0", threshold_pct=0.85)
        est.fit(X.iloc[tr], y[tr])

        gpd = est.predict_tail_quantile(X.iloc[te], q)
        gpd_errs.append(abs(float(np.mean(y[te] <= gpd)) - q))

        mu_te = est.predict(X.iloc[te])
        resid_std = float(np.std(y[tr] - est.predict(X.iloc[tr])))
        gauss = mu_te + norm.ppf(q) * resid_std
        gauss_errs.append(abs(float(np.mean(y[te] <= gauss)) - q))

    gpd_err = float(np.mean(gpd_errs))
    gauss_err = float(np.mean(gauss_errs))
    assert gpd_err < 0.0025, f"GPD extreme-quantile coverage error too large: {gpd_err:.4f}"
    assert gauss_err > gpd_err + 0.002, f"GPD should beat Gaussian tail clearly: gpd={gpd_err:.4f} gauss={gauss_err:.4f}"
