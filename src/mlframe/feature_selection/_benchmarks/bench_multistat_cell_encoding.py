"""Does encoding a cell by MORE target statistics (std/skew/kurtosis) beat the mean-only target encoder?

Honest train/test measurement of two distinct ideas, because they have opposite theory:

  (1) MULTI-STAT TARGET encoding -- encode cell by [mean, std, skew, kurt] of the TARGET y.
      Theory: to predict y, E[y|cell]=mean is the Bayes point estimate; higher moments of y|cell do not
      shift the conditional mean, and for BINARY y the higher moments are deterministic functions of the
      mean (Bernoulli: std=sqrt(p(1-p)), skew=(1-2p)/std) -> add literally nothing. Expect ~0 gain.

  (2) MULTI-STAT FEATURE aggregation -- encode cell by [mean, std, skew, kurt] of a NUMERIC FEATURE within
      the cell. Theory: when the label depends on the WITHIN-CELL SPREAD of an auxiliary feature (not its
      mean), std/skew genuinely separate cells the mean cannot. Expect a real win. (This is the regime
      ``compute_numerical_aggregates_numba`` is built for -- aggregating feature values, not the target.)

Leak-safe: OOF on train, train cell-stats replayed on the held-out test fold.

Run:  python -m mlframe.feature_selection._benchmarks.bench_multistat_cell_encoding
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold

SEEDS = (0, 1, 2, 3, 4)
STATS = ("mean", "std", "skew", "kurt")


def _cell_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0}
    sd = float(np.std(values))
    sk = float(skew(values)) if (values.size > 2 and sd > 1e-12) else 0.0
    ku = float(kurtosis(values)) if (values.size > 3 and sd > 1e-12) else 0.0
    return {
        "mean": float(np.mean(values)),
        "std": sd,
        "skew": sk if np.isfinite(sk) else 0.0,
        "kurt": ku if np.isfinite(ku) else 0.0,
    }


def _oof_multistat_encode(codes_tr, src_tr, n_folds, seed, which_stats, global_src):
    """OOF per-cell stats of ``src`` (target or feature), returning (train_features, replay_table)."""
    n = len(codes_tr)
    feats = np.zeros((n, len(which_stats)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_folds, n)
    gstats = _cell_stats(global_src)
    for f in range(n_folds):
        oof = folds == f
        infold = ~oof
        df = pd.DataFrame({"c": codes_tr[infold], "v": src_tr[infold]})
        lut = {c: _cell_stats(g["v"].to_numpy()) for c, g in df.groupby("c")}
        for i in np.where(oof)[0]:
            s = lut.get(int(codes_tr[i]), gstats)
            feats[i] = [s[k] for k in which_stats]
    # Full-train replay table
    df = pd.DataFrame({"c": codes_tr, "v": src_tr})
    table = {int(c): _cell_stats(g["v"].to_numpy()) for c, g in df.groupby("c")}
    return feats, table, gstats


def _apply_table(codes, table, gstats, which_stats):
    return np.array([[table.get(int(c), gstats)[k] for k in which_stats] for c in codes], dtype=np.float64)


def _bin(x, edges):
    return np.searchsorted(edges, x, side="right")


def _make_cross_codes(x0, x1, nbins, e0=None, e1=None):
    if e0 is None:
        qs = np.linspace(0, 1, nbins + 1)[1:-1]
        e0, e1 = np.unique(np.quantile(x0, qs)), np.unique(np.quantile(x1, qs))
    c0, c1 = _bin(x0, e0), _bin(x1, e1)
    nb0 = c0.max() + 1
    return (c0 + c1 * nb0), e0, e1, int(nb0)


# Scenarios return: (x0, x1, y, src_to_encode, task, raw_extra) where raw_extra is an (n, k) array of
# RAW features ALSO fed to the downstream alongside the TE columns (None -> TE-only). Target encoding always
# encodes ``src_to_encode`` (the target y for the target-encoder scenarios).

def scenario_target_binary(seed):
    """(1a) binary y, cell-driven P(y=1), TE-only downstream. Control: target moments are functions of the mean."""
    rng = np.random.default_rng(seed)
    n = 8000
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    logit = 3.0 * np.sign((x0 - 0.5) * (x1 - 0.5))
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(float)
    return x0, x1, y, y, "clf", None


def scenario_target_regression(seed):
    """(1b) continuous y = base(cell) + homoscedastic noise, TE-only. Mean is Bayes -> expect no gain."""
    rng = np.random.default_rng(seed)
    n = 8000
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    base = 2.0 * np.sign((x0 - 0.5) * (x1 - 0.5))
    y = base + rng.normal(0, 1, n)
    return x0, x1, y, y, "reg", None


def scenario_varying_slope(seed):
    """(1c) REGRESSION with cell-varying slope: y = a(cell) + b(cell)*x_raw + noise.

    mean(y|cell) ~ a(cell) (b averages out over x_raw); std(y|cell) ~ |b(cell)|*std(x_raw) carries the SLOPE
    magnitude that the mean cannot. Fed to the model WITH x_raw, std(y|cell) lets it recover the b(cell)*x_raw
    interaction. The fair regime for multi-stat TARGET encoding the simple scenarios missed.
    """
    rng = np.random.default_rng(seed)
    n = 12000
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    x_raw = rng.normal(0, 1, n)
    a = 1.5 * np.sign((x0 - 0.5) * (x1 - 0.5))
    b = 2.0 * (np.abs(x0 - 0.5) + np.abs(x1 - 0.5))  # slope magnitude varies smoothly by cell
    y = a + b * x_raw + rng.normal(0, 0.3, n)
    return x0, x1, y, y, "reg", x_raw.reshape(-1, 1)


def scenario_varying_slope_signed(seed):
    """(1d) like 1c but the slope SIGN varies by cell too: std(y|cell) gives |b|, not sign -> partial signal.

    Tests whether the magnitude proxy still helps when the sign is unidentifiable from a symmetric spread.
    """
    rng = np.random.default_rng(seed)
    n = 12000
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    x_raw = rng.normal(0, 1, n)
    a = 1.0 * np.sign(x0 - 0.5)
    b = 3.0 * np.sign((x0 - 0.5) * (x1 - 0.5)) * (0.5 + np.abs(x1 - 0.5))
    y = a + b * x_raw + rng.normal(0, 0.3, n)
    return x0, x1, y, y, "reg", x_raw.reshape(-1, 1)


def scenario_feature_spread(seed):
    """(2) regression target T = sigma(cell) -- the WITHIN-CELL SPREAD of an aux feature, not its mean.

    aux ~ N(0, sigma(cell)) where sigma is a smooth function of the (x0, x1) cell. We observe only ``aux``
    and predict T. mean(aux|cell) ~ 0 everywhere -> R2 ~ 0; std(aux|cell) -> recovers sigma(cell) -> R2 ~ 1.
    This is exactly the regime ``compute_numerical_aggregates_numba`` targets: aggregating FEATURE values.
    """
    rng = np.random.default_rng(seed)
    n = 8000
    x0, x1 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    sigma = 0.5 + 2.0 * (np.abs(x0 - 0.5) + np.abs(x1 - 0.5))  # smooth, cell-varying spread
    aux = rng.normal(0, sigma, n)
    T = sigma  # the quantity to predict is the per-cell spread itself
    return x0, x1, T, aux, "reg", None  # NB: src_to_encode = aux (a FEATURE), not the target T


def _eval(x0, x1, y, src, task, raw_extra, seed):
    n = len(y)
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    codes_tr, e0, e1, _ = _make_cross_codes(x0[tr], x1[tr], 8)
    codes_te, *_ = _make_cross_codes(x0[te], x1[te], 8, e0, e1)
    raw_tr = raw_extra[tr] if raw_extra is not None else np.empty((len(tr), 0))
    raw_te = raw_extra[te] if raw_extra is not None else np.empty((len(te), 0))
    out = {}
    for label, which in (("mean_only", ("mean",)),
                         ("+std+skew+kurt", STATS)):
        f_tr, table, g = _oof_multistat_encode(codes_tr, src[tr], 5, seed, which, src[tr])
        f_te = _apply_table(codes_te, table, g, which)
        Xtr = np.hstack([raw_tr, f_tr])
        Xte = np.hstack([raw_te, f_te])
        if task == "clf":
            m = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(Xtr, y[tr])
            out[label] = roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])
        else:
            m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(Xtr, y[tr])
            out[label] = r2_score(y[te], m.predict(Xte))
    return out


def main():
    scenarios = [
        ("(1a) binary y, TE-only             ", scenario_target_binary, "AUC"),
        ("(1b) regression y, TE-only         ", scenario_target_regression, "R2 "),
        ("(1c) varying-slope reg, TE + x_raw ", scenario_varying_slope, "R2 "),
        ("(1d) signed-slope reg, TE + x_raw  ", scenario_varying_slope_signed, "R2 "),
    ]
    print("\nMulti-stat TARGET encoding -- honest OOS (mean-only vs +std+skew+kurt of y per cell), 5 seeds")
    print("downstream = GBM on [raw features | TE columns]\n")
    for name, fn, metric in scenarios:
        rows = [_eval(*fn(s), s) for s in SEEDS]
        mo = np.mean([r["mean_only"] for r in rows])
        ms = np.mean([r["+std+skew+kurt"] for r in rows])
        print(f"{name} {metric}  mean_only={mo:.4f}  +std+skew+kurt={ms:.4f}  (d {ms-mo:+.4f})")

    # Complementary: multi-stat aggregation of a FEATURE per cell (grouped-agg FE regime, NOT target encoding).
    # Kept because the user wants every accuracy lever visible -- here std of a feature recovers a spread signal
    # the mean cannot, the strongest illustration of why higher moments matter.
    print("\n(2) FEATURE aggregation per cell (encodes a FEATURE's stats, not the target) -- separate lever:")
    rows = [_eval(*scenario_feature_spread(s), s) for s in SEEDS]
    mo = np.mean([r["mean_only"] for r in rows])
    ms = np.mean([r["+std+skew+kurt"] for r in rows])
    print(f"(2) feature-spread aux -> sigma     R2   mean_only={mo:.4f}  +std+skew+kurt={ms:.4f}  (d {ms-mo:+.4f})")
    print()


if __name__ == "__main__":
    main()
