"""qual-10: adjusted R^2 vs plain R^2 -- known-ground-truth bias bench.

Plain R^2 on the fitting sample is upward-biased: it never decreases when predictors are added, so it overstates the
true explained variance whenever the predictor count ``p`` is non-trivial relative to ``n``. Adjusted R^2 (Wherry /
Ezekiel) penalises by model degrees of freedom: ``1 - (1 - R^2) * (n - 1) / (n - p - 1)``.

Ground truth: ``y = x0 + noise`` with x0 ~ N(0,1) and noise ~ N(0,1) gives Var(signal)=1, Var(y)=2, so the TRUE
population R^2 of the x0-only model is exactly 0.5. We then fit an OLS on x0 PLUS (p-1) pure-noise predictors. The
extra predictors contribute nothing in the population, so the honest goodness-of-fit stays 0.5 -- but plain training
R^2 inflates with each junk feature. We measure absolute error to the true 0.5 and tally which estimator is closer.

Verdict (run on this machine, 7 seeds):
- p/n >= 0.1 (the regime adjusted R^2 exists for): adjusted wins 27/35 cells, mean |estimate-true| 0.159 -> 0.109.
- p/n small (<0.05): wash (adjusted 9/21, plain 12/21), both ~0.022 -- adjusted converges back to plain, no harm.

Conclusion: adjusted R^2 is the correct goodness-of-fit when ``p`` matters relative to ``n``; shipped as the NEW
``fast_adjusted_r2_score`` (it needs ``p``, which plain ``fast_r2_score`` does not take). The default-FLIP hypothesis
(make adjusted R^2 the default ``fast_r2_score``) is REJECTED: the two answer different questions, adjusted requires a
predictor count the plain signature has no way to supply, and at small p/n there is no win to justify a breaking change.

Run: python -m mlframe.metrics._benchmarks.bench_adjusted_r2
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

TRUE_R2 = 0.5  # y = x0 + noise, both N(0,1)


def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
    return 1.0 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def _adj_r2(yt: np.ndarray, yp: np.ndarray, p: int) -> float:
    n = len(yt)
    return 1.0 - (1.0 - _r2(yt, yp)) * (n - 1) / (n - p - 1)


def _fit(n: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    y = x0 + rng.normal(0, 1, n)
    X = np.column_stack([x0] + [rng.normal(0, 1, n) for _ in range(p - 1)])
    yp = LinearRegression().fit(X, y).predict(X)
    return y, yp


def run() -> None:
    for label, combos in [
        ("p/n >= 0.1 (adjusted-R2 regime)", [(40, 8), (50, 10), (50, 20), (100, 20), (100, 30)]),
        ("p/n small (<0.05)", [(500, 5), (1000, 10), (2000, 5)]),
    ]:
        pe_all, ae_all, adj_win, plain_win, cells = [], [], 0, 0, 0
        for (n, p) in combos:
            for seed in range(7):
                y, yp = _fit(n, p, seed)
                pe = abs(_r2(y, yp) - TRUE_R2)
                ae = abs(_adj_r2(y, yp, p) - TRUE_R2)
                pe_all.append(pe)
                ae_all.append(ae)
                cells += 1
                if ae < pe:
                    adj_win += 1
                elif pe < ae:
                    plain_win += 1
        print(
            f"{label}: cells={cells} mean|plain-true|={np.mean(pe_all):.4f} "
            f"mean|adj-true|={np.mean(ae_all):.4f} adj_wins={adj_win}/{cells} plain_wins={plain_win}/{cells}"
        )


if __name__ == "__main__":
    run()
