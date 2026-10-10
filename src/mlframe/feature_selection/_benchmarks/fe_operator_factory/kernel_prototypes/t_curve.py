"""MI as a function of the offset t for the case-2 form, the OLS shift estimate (raw y and rank y), the null inflation of an 8-point offset grid and the 5k-subsample choice of t.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes.t_curve``."""

import warnings

warnings.simplefilter("ignore")
import numpy as np

from mlframe.feature_selection.filters._fe_cpu_batch import cpu_fe_batch_mi


def main(n=30000) -> None:
    """Print MI(t) of the offset product on the case-2 data, the OLS shift estimate, the null inflation of an 8-point grid and the subsample choice of t."""
    rng = np.random.default_rng(0)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)

    def yb(y):
        """Decile codes of ``y``."""
        return np.searchsorted(np.quantile(y, np.linspace(0, 1, 11)[1:-1]), y).astype(np.int64)

    yc = yb(y)
    u, v = np.log(c), np.sin(d)
    ts = np.linspace(-0.5, 1.5, 21)
    M = np.stack([(u + t) * v for t in ts], 1)
    mi = cpu_fe_batch_mi(M, yc, 10)
    for t, m in zip(ts, mi):
        print(f"t={t:+.2f} MI={m:.4f}")
    # closed form: OLS y ~ [u*v, v, 1] -> t = coef_v/coef_uv
    A = np.stack([u * v, v, np.ones(n)], 1)
    co = np.linalg.lstsq(A, y, rcond=None)[0]
    t_ols = co[1] / co[0]
    print("OLS t =", t_ols, " (truth ln2=0.693)  MI at t_ols:", cpu_fe_batch_mi(((u + t_ols) * v)[:, None], yc, 10)[0])
    # robustness: OLS on rank-transformed y
    from scipy.stats import rankdata

    yr = rankdata(y) / n
    co = np.linalg.lstsq(A, yr, rcond=None)[0]
    t2 = co[1] / co[0]
    print("OLS(rank y) t =", t2, "MI:", cpu_fe_batch_mi(((u + t2) * v)[:, None], yc, 10)[0])
    # null inflation: permuted y, max over G=8 grid vs t=0; many perms
    Gts = np.linspace(-0.5, 1.5, 8)
    infl, base = [], []
    for s in range(60):
        yp = yc[np.random.default_rng(s).permutation(n)]
        X = np.stack([(u + t) * v for t in Gts] + [u * v], 1)
        m = cpu_fe_batch_mi(X, yp, 10)
        infl.append(m[:8].max() - m[8])
        base.append(m[8])
    print(f"null: MI(t=0) mean {np.mean(base):.5f}; max-over-8-offsets minus t0: mean {np.mean(infl):.5f} q95 {np.quantile(infl, 0.95):.5f}")
    # golden/subsample: pick t on 5k subsample, evaluate full
    sub = rng.choice(n, 5000, replace=False)
    mis = cpu_fe_batch_mi(np.stack([(u[sub] + t) * v[sub] for t in ts], 1), yc[sub], 10)
    tb = ts[mis.argmax()]
    print("subsample(5k) grid argmax t =", tb, "full MI:", cpu_fe_batch_mi(((u + tb) * v)[:, None], yc, 10)[0])


if __name__ == "__main__":
    main()
