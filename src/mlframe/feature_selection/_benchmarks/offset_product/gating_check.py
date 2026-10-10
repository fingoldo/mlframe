"""The offset-product family inside a whole MRMR fit: what survives the gating and the cross-stage dedup, and what it does to the downstream error.

Four datasets (n = 20000, a 25% hold-out): ``sign`` (the interaction the family exists for), ``plain`` (a ratio plus a product with no sign change: the family must change nothing),
``cat`` (a categorical column with a target-encoding signal next to a sign-crossing numeric pair: overlap with ``kfold_target_encoded``), ``smooth`` (a smooth nonlinear additive target: overlap
with the spline / Fourier / wavelet bases). Each is fitted with ``fe_offset_product_enable`` off and on; the selected features and the hold-out MAE and RMSE (ridge and gradient boosting on the
transformed frame) are printed side by side. Run: ``python -m mlframe.feature_selection._benchmarks.offset_product.gating_check [dataset ...]``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters.mrmr import MRMR

warnings.simplefilter("ignore")
N_ROWS = 20_000
TEST_FRACTION = 0.25
CAT_LEVELS = 8


def _sign(rng: np.random.Generator):
    """Ratio term plus the sign-crossing interaction ``ln(2c) * sin(d/3)``; the columns live in [0.1, 1.1) so that ``a**2 / b`` has no heavy tail that would dominate every RMSE."""
    a, b, c, d, e, f = (rng.random(N_ROWS) + 0.1 for _ in range(6))
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), 0.2 * a**2 / b + f / 5.0 + np.log(2 * c) * np.sin(d / 3)


def _plain(rng: np.random.Generator):
    """Ratio term plus a product without a sign change."""
    a, b, c, d, e, f = (rng.random(N_ROWS) + 0.1 for _ in range(6))
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), 0.2 * a**2 / b + f / 5.0 + c * np.sin(d / 3)


def _cat(rng: np.random.Generator):
    """A categorical effect (target-encoding signal) next to the sign-crossing pair ``(x - 0.5) * z``."""
    x, z, w, u = (rng.random(N_ROWS) for _ in range(4))
    g = rng.integers(0, CAT_LEVELS, N_ROWS)
    effect = rng.standard_normal(CAT_LEVELS)[g]
    df = pd.DataFrame({"x": x, "z": z, "w": w, "u": u, "g": pd.Categorical(g)})
    return df, effect + 3.0 * (x - 0.5) * z + 0.3 * rng.standard_normal(N_ROWS)


def _smooth(rng: np.random.Generator):
    """Smooth nonlinear additive target: the spline / Fourier / wavelet territory, no interaction."""
    x1, x2, x3, x4, x5 = (rng.random(N_ROWS) for _ in range(5))
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}), np.sin(6 * x1) + 2 * (x2 - 0.5) ** 2 + 0.5 * np.exp(x3) + 0.2 * rng.standard_normal(N_ROWS)


DATASETS = {"sign": _sign, "plain": _plain, "cat": _cat, "smooth": _smooth}


def _errors(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> dict:
    """Hold-out MAE and RMSE of ridge and gradient boosting."""
    out = {}
    for name, model in (("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))), ("hgb", HistGradientBoostingRegressor(max_iter=150, early_stopping=False, random_state=0))):
        resid = yte - model.fit(Xtr, ytr).predict(Xte)
        out[name] = (float(np.abs(resid).mean()), float(np.sqrt((resid**2).mean())))
    return out


def run(name: str, seed: int = 0) -> None:
    """Fit the dataset with the family off and on and print the comparison."""
    rng = np.random.default_rng(seed)
    df, y = DATASETS[name](rng)
    cut = int(len(df) * (1 - TEST_FRACTION))
    dtr, dte, ytr, yte = df.iloc[:cut], df.iloc[cut:], y[:cut], y[cut:]
    rows = {}
    for flag in (False, True):
        fs = MRMR(verbose=0, fe_max_steps=2, n_jobs=1, random_seed=seed, fe_offset_product_enable=flag).fit(dtr, pd.Series(ytr, name="y"))
        Ztr, Zte = np.asarray(fs.transform(dtr), dtype=float), np.asarray(fs.transform(dte), dtype=float)
        rows[flag] = (list(map(str, fs.get_feature_names_out())), list(getattr(fs, "offset_product_features_", [])), _errors(Ztr, ytr, Zte, yte))
    print(f"== {name}")
    for flag, (names, roster, err) in rows.items():
        print(f"  offset_product={flag!s:5} selected={names} offset roster={roster}")
        print("     " + "  ".join(f"{m}: MAE {e[0]:.4f} RMSE {e[1]:.4f}" for m, e in err.items()))
    off, on = rows[False][2], rows[True][2]
    print("  relative improvement on vs off (positive = on is better): " + "  ".join(f"{m} MAE {(off[m][0] - on[m][0]) / off[m][0]:+.2%} RMSE {(off[m][1] - on[m][1]) / off[m][1]:+.2%}" for m in off), flush=True)


def main() -> None:
    """Run the named datasets (default: all); ``name@seed`` picks the seed (default 0)."""
    for arg in sys.argv[1:] or list(DATASETS):
        name, _, seed = arg.partition("@")
        run(name, int(seed or 0))


if __name__ == "__main__":
    main()
