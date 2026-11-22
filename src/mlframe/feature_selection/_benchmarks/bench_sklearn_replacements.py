"""Warm A/B timing of every sklearn piece that MRMR replaced by its own code, against the sklearn original, at the sizes the fits use.

Usage: ``bench_sklearn_replacements.py [quick|full]``. Each row is the median of several warm calls (the first call compiles the numba kernels and is not counted); ``x`` is the sklearn time over the
replacement's time (above 1 the replacement is faster). A replacement that is slower prints a ``SLOWER`` flag. Nothing here changes a result: the equality of the values is pinned by the tests named in the
tracker row MRMR-SKLEARN-REPLACEMENTS-EVIDENCE-PLAN.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

REPEATS = 5


def _median_ms(fn, repeats: int = REPEATS) -> float:
    """Median wall time of ``fn()`` in milliseconds over ``repeats`` warm calls."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


ROWS: list = []


def _row(group: str, case: str, ours_ms: float, sk_ms: float) -> None:
    """Record and print one comparison."""
    ratio = sk_ms / ours_ms if ours_ms > 0 else float("inf")
    flag = "" if ratio >= 1.0 else "  SLOWER"
    ROWS.append((group, case, ours_ms, sk_ms, ratio))
    print(f"{group:28s} {case:34s} ours {ours_ms:10.2f} ms   sklearn {sk_ms:10.2f} ms   x{ratio:7.2f}{flag}", flush=True)


def bench_ksg(sizes) -> None:
    """KSG MI per column, discrete and continuous target."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    from mlframe.feature_selection.filters._ksg_dispatch import ksg_mi_columns

    rng = np.random.default_rng(0)
    for n, p in sizes:
        X = rng.standard_normal((n, p))
        yc = (X[:, 0] + rng.standard_normal(n) > 0).astype(np.int64)
        yr = X[:, 0] + 0.5 * rng.standard_normal(n)
        _row("ksg discrete target", f"n={n} p={p}", _median_ms(lambda: ksg_mi_columns(X, yc, discrete_target=True, n_neighbors=3, random_state=0)),
             _median_ms(lambda: mutual_info_classif(X, yc, discrete_features=False, random_state=0), 3))
        _row("ksg continuous target", f"n={n} p={p}", _median_ms(lambda: ksg_mi_columns(X, yr, discrete_target=False, n_neighbors=3, random_state=0)),
             _median_ms(lambda: mutual_info_regression(X, yr, discrete_features=False, random_state=0), 3))


def bench_linear(sizes) -> None:
    """Scaled OLS / ridge / logistic probes against ``make_pipeline(StandardScaler(), model)``."""
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from mlframe.feature_selection.filters._linear_probes import ScaledLogistic, ScaledOLS, ScaledRidge

    rng = np.random.default_rng(1)
    for n, p in sizes:
        X = rng.standard_normal((n, p)) * rng.uniform(0.5, 4.0, size=p)
        y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
        yb = (y > 0).astype(np.int64)
        case = f"n={n} p={p}"
        _row("ScaledOLS fit+predict", case, _median_ms(lambda: ScaledOLS().fit(X, y).predict(X)), _median_ms(lambda: make_pipeline(StandardScaler(), LinearRegression()).fit(X, y).predict(X)))
        _row("ScaledRidge fit+predict", case, _median_ms(lambda: ScaledRidge(1.0).fit(X, y).predict(X)), _median_ms(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y).predict(X)))
        _row("ScaledLogistic fit+proba", case, _median_ms(lambda: ScaledLogistic().fit(X, yb).predict_proba(X), 3),
             _median_ms(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(X, yb).predict_proba(X), 3))


def bench_elastic_net(sizes) -> None:
    """Coordinate descent Lasso / ElasticNet against sklearn."""
    from sklearn.linear_model import ElasticNet, Lasso

    from mlframe.feature_selection.filters._elastic_net_cd import elastic_net_coef

    rng = np.random.default_rng(2)
    for n, p in sizes:
        X = rng.standard_normal((n, p))
        y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
        case = f"n={n} p={p}"
        _row("Lasso (alpha 0.05)", case, _median_ms(lambda: elastic_net_coef(X, y, 0.05, 1.0)), _median_ms(lambda: Lasso(alpha=0.05, max_iter=5000, tol=1e-9).fit(X, y), 3))
        _row("ElasticNet (0.05, l1 0.5)", case, _median_ms(lambda: elastic_net_coef(X, y, 0.05, 0.5)), _median_ms(lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, tol=1e-9).fit(X, y), 3))


def bench_ledoit_wolf(sizes) -> None:
    """Ledoit-Wolf shrunk covariance against sklearn."""
    from sklearn.covariance import LedoitWolf

    from mlframe.feature_selection.filters._ledoit_wolf import ledoit_wolf_fit

    rng = np.random.default_rng(3)
    for n, p in sizes:
        X = rng.standard_normal((n, p)) @ rng.standard_normal((p, p))
        _row("LedoitWolf fit", f"n={n} p={p}", _median_ms(lambda: ledoit_wolf_fit(X)), _median_ms(lambda: LedoitWolf().fit(X)))


def bench_ridge_closed_form(sizes) -> None:
    """Closed-form ridge with unpenalised intercept against sklearn ``Ridge``."""
    from sklearn.linear_model import Ridge

    from mlframe.feature_selection.filters._fe_probe_ridge import fit_ridge_closed_form

    rng = np.random.default_rng(4)
    for n, p in sizes:
        X = rng.standard_normal((n, p))
        y = X[:, 0] + 0.3 * rng.standard_normal(n)
        _row("fit_ridge_closed_form", f"n={n} p={p}", _median_ms(lambda: fit_ridge_closed_form(X, y, 1.0)), _median_ms(lambda: Ridge(alpha=1.0).fit(X, y)))


def bench_knn(sizes) -> None:
    """Chebyshev 2-D kNN against ``KDTree``."""
    from sklearn.neighbors import KDTree

    from mlframe.feature_selection.filters._ksg_cc_fast import knn_chebyshev_2d

    rng = np.random.default_rng(5)
    for n, k in sizes:
        pts = rng.standard_normal((n, 2))
        _row("knn_chebyshev_2d", f"n={n} k={k}", _median_ms(lambda: knn_chebyshev_2d(pts, k), 3), _median_ms(lambda: KDTree(pts, metric="chebyshev").query(pts, k=k + 1), 3))


def bench_contingency(sizes) -> None:
    """Contingency-table MI against ``sklearn.metrics.mutual_info_score``."""
    from sklearn.metrics import mutual_info_score

    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import contingency_mi_nats

    rng = np.random.default_rng(6)
    for n, k in sizes:
        a = rng.integers(0, k, size=n)
        b = (a + rng.integers(0, 3, size=n)) % k
        _row("contingency_mi_nats", f"n={n} bins={k}", _median_ms(lambda: contingency_mi_nats(a, b)), _median_ms(lambda: mutual_info_score(a, b)))


def bench_outside(sizes) -> None:
    """The two call sites outside MRMR that now use the KSG kernels."""
    from sklearn.feature_selection import mutual_info_regression

    from mlframe.calibration.quality import mutual_information_score
    from mlframe.feature_engineering.numerical import compute_mutual_info_regression

    rng = np.random.default_rng(7)
    for n in sizes:
        y = rng.standard_normal(n)
        pred = y + 0.5 * rng.standard_normal(n)
        _row("calibration MI score", f"n={n}", _median_ms(lambda: mutual_information_score(y, pred)), _median_ms(lambda: mutual_info_regression(y.reshape(-1, 1), pred, n_neighbors=2), 3))
        _row("compute_mutual_info_regression", f"n={n}", _median_ms(lambda: compute_mutual_info_regression(y, pred)), _median_ms(lambda: mutual_info_regression(pred.reshape(-1, 1), y, n_neighbors=2), 3))


def main() -> None:
    """Run every group at the quick or the full sizes."""
    full = len(sys.argv) > 1 and sys.argv[1] == "full"
    bench_ksg([(20000, 1), (20000, 20), (100000, 20)] + ([(100000, 100)] if full else []))
    bench_linear([(20000, 10), (100000, 50)] + ([(100000, 200)] if full else []))
    bench_elastic_net([(20000, 20), (100000, 100)])
    bench_ledoit_wolf([(5000, 30), (20000, 100)] + ([(100000, 300)] if full else []))
    bench_ridge_closed_form([(20000, 20), (100000, 100)])
    bench_knn([(20000, 3), (100000, 3)])
    bench_contingency([(100000, 16), (1000000, 32)])
    bench_outside([2000, 20000])
    slower = [r for r in ROWS if r[4] < 1.0]
    print(f"\n{len(ROWS)} comparisons, {len(slower)} where the replacement is SLOWER: " + "; ".join(f"{g} {c} x{r:.2f}" for g, c, _o, _s, r in slower))


if __name__ == "__main__":
    main()
