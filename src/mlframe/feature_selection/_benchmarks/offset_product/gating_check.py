"""The offset-product family inside a whole MRMR fit: what survives the gating and the cross-stage dedup, and what it does to the downstream error.

Four datasets (n = 20000, a 25% hold-out): ``sign`` (the interaction the family exists for), ``plain`` (a ratio plus a product with no sign change: the family must change nothing),
``cat`` (a categorical column with a target-encoding signal next to a sign-crossing numeric pair: overlap with ``kfold_target_encoded``), ``smooth`` (a smooth nonlinear additive target: overlap
with the spline / Fourier / wavelet bases). Each is fitted with ``fe_offset_product_enable`` off and on; the selected features and the hold-out MAE and RMSE (ridge and gradient boosting on the
consumer-specific feature lists) are printed side by side. The repository routes tree models to ``transform`` (the MI list) and linear models to ``transform_usability("linear")`` (the
usability-aware list, ``usability_aware_lists=True``), so each model is scored on its own list; ridge on the MI list is printed for reference. Run: ``python -m mlframe.feature_selection._benchmarks.offset_product.gating_check [dataset ...]``.
"""

from __future__ import annotations

import sys
import warnings
from unittest import mock

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


ARMS = {"off": (False, False), "on": (True, False), "on+pool": (True, True)}  # arm -> (fe_offset_product_enable, offset products also offered to the usability pool)
DATASETS = {"sign": _sign, "plain": _plain, "cat": _cat, "smooth": _smooth}


def _mae_rmse(model, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> "tuple[float, float]":
    """Hold-out MAE and RMSE of ``model`` fitted on the train part."""
    resid = yte - model.fit(Xtr, ytr).predict(Xte)
    return float(np.abs(resid).mean()), float(np.sqrt((resid**2).mean()))


def _errors(mi_lists: "tuple[np.ndarray, np.ndarray]", lin_lists: "tuple[np.ndarray, np.ndarray]", ytr: np.ndarray, yte: np.ndarray) -> dict:
    """Hold-out errors per consumer: gradient boosting on the MI list, ridge on the linear list (``ridge_mi``: ridge on the MI list, reference)."""
    ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))  # noqa: E731 - a fresh estimator per call
    hgb = HistGradientBoostingRegressor(max_iter=150, early_stopping=False, random_state=0)
    return {
        "ridge": _mae_rmse(ridge(), *lin_lists[:1], ytr, lin_lists[1], yte),
        "hgb": _mae_rmse(hgb, mi_lists[0], ytr, mi_lists[1], yte),
        "ridge_mi": _mae_rmse(ridge(), mi_lists[0], ytr, mi_lists[1], yte),
    }


def run(name: str, seed: int = 0) -> None:
    """Fit the dataset with the family off and on and print the comparison."""
    rng = np.random.default_rng(seed)
    df, y = DATASETS[name](rng)
    cut = int(len(df) * (1 - TEST_FRACTION))
    dtr, dte, ytr, yte = df.iloc[:cut], df.iloc[cut:], y[:cut], y[cut:]
    rows = {}
    pool_target = "mlframe.feature_selection.filters._usability_offset_pool.offset_product_candidates"
    for arm, (flag, in_pool) in ARMS.items():
        MRMR.clear_fit_cache()  # the fit memo is keyed by data and constructor parameters, not by the patch below: without this the second "on" arm would be a cache hit
        with mock.patch(pool_target, side_effect=lambda *a, **k: []) if not in_pool else mock.patch.dict({}):
            fs = MRMR(verbose=0, fe_max_steps=2, n_jobs=1, random_seed=seed, fe_offset_product_enable=flag, usability_aware_lists=True).fit(dtr, pd.Series(ytr, name="y"))
        mi = (np.asarray(fs.transform(dtr), dtype=float), np.asarray(fs.transform(dte), dtype=float))
        lin = (np.asarray(fs.transform_usability(dtr, which="linear"), dtype=float), np.asarray(fs.transform_usability(dte, which="linear"), dtype=float))
        rows[arm] = (list(map(str, fs.get_feature_names_out())), [str(c.name) for c in (getattr(fs, "support_linear_", None) or [])], _errors(mi, lin, ytr, yte))
    print(f"== {name}")
    for arm, (names, roster, err) in rows.items():
        print(f"  [{arm}] MI list={names}")
        print(f"                       linear list={roster}")
        print("     " + "  ".join(f"{m}: MAE {e[0]:.4f} RMSE {e[1]:.4f}" for m, e in err.items()))
    off = rows["off"][2]
    for arm in ("on", "on+pool"):
        on = rows[arm][2]
        print(f"  [{arm}] vs [off] (positive = better): " + "  ".join(f"{m} MAE {(off[m][0] - on[m][0]) / off[m][0]:+.2%} RMSE {(off[m][1] - on[m][1]) / off[m][1]:+.2%}" for m in off if m != "ridge_mi"), flush=True)


def main() -> None:
    """Run the named datasets (default: all); ``name@seed`` picks the seed (default 0)."""
    for arg in sys.argv[1:] or list(DATASETS):
        name, _, seed = arg.partition("@")
        run(name, int(seed or 0))


if __name__ == "__main__":
    main()
