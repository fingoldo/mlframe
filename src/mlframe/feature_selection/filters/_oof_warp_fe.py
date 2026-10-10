"""Out-of-fold warp operator: replace a numeric column by the calibrated ``E[rank(y) | x]`` when that exposes structure the raw column hides from a 10-bin MI and from a linear model.

A column that matters through a non-monotone effect (a sine, a bump, a U shape) has a low plug-in MI when the oscillation is finer than the bins, and a linear model cannot use it at all. The
warp (``_oof_warp_service``) maps the column onto the target's own scale with cross-fitted bin means, so the effect becomes monotone in the target.

Acceptance has no hand-set margin: the warp's held-out MI must exceed the raw column's by more than ``SIGNIFICANCE_Z`` standard errors of the paired difference
(``_fe_gain_stats.paired_gain_se``), and by at least the practical effect ``min_relative_gain`` of the raw MI (a constructor knob). A monotone relation gives a warp that bins like the raw
column and is rejected here (the linear-downstream pool is offered the warps separately, see ``_usability_warp_pool``). Columns with at most ``FEW_CLASSES_MAX`` distinct values are skipped: a
warp of a nominal-like column is a target encoding, which other families already cover.

Replay (kind ``oof_warp1d``) stores the full table, the fill and the output range; it is a pure function of the source column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._fe_gain_stats import held_out_codes, paired_gain_se, plugin_mi_of_codes
from ._oof_warp_service import apply_warp1d, fit_oof_warp1d
from ._y_encoding import FEW_CLASSES_MAX

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = ["hybrid_oof_warp_fe", "apply_oof_warp1d_recipe", "build_oof_warp1d_recipe", "warp_candidates"]

SIGNIFICANCE_Z = 2.0  # one-sided normal critical value applied to the standard error of the MI gain
DEFAULT_MIN_RELATIVE_GAIN = 0.05  # default of the practical-effect knob
DEFAULT_SCAN_ROWS = 100_000  # the warp is fitted and scored on at most this many evenly spaced rows; rows beyond them get the full table
N_MI_BINS = 10
_MIN_ROWS = 400


def build_oof_warp1d_recipe(*, name: str, src: str, cx: np.ndarray, cy: np.ndarray, fill: float, lo: float, hi: float) -> "EngineeredRecipe":
    """Frozen recipe of one warp column: the full table, the fill for a non-finite value and the output range."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="oof_warp1d",
        src_names=(str(src),),
        extra={"cx": np.asarray(cx, dtype=np.float64).copy(), "cy": np.asarray(cy, dtype=np.float64).copy(), "fill": float(fill), "lo": float(lo), "hi": float(hi)},
    )


def apply_oof_warp1d_recipe(recipe, X) -> np.ndarray:
    """Replay one warp column from the stored table; a pure function of the source column."""
    from .engineered_recipes.shared import extract_column

    ex = recipe.extra
    x = np.asarray(extract_column(X, recipe.src_names[0]), dtype=np.float64)
    return np.clip(apply_warp1d(x, ex["cx"], ex["cy"], ex["fill"]), ex["lo"], ex["hi"])


def _rank_scaled(y: np.ndarray) -> np.ndarray:
    """Average ranks of ``y`` scaled into ``(0, 1]`` (a bounded target: bin means are robust to heavy tails)."""
    from scipy.stats import rankdata

    return rankdata(y, method="average") / float(len(y))


def _scan_index(n: int, scan_rows: int) -> np.ndarray:
    """Evenly spaced row indices of the fitting / scoring sample (all rows when ``n`` is small)."""
    return np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)


def warp_candidates(X: "pd.DataFrame", y: np.ndarray, cols: Sequence[str], *, scan_rows: int = DEFAULT_SCAN_ROWS, seed: int = 0) -> "list[dict]":
    """Fit the warp of each numeric column of ``cols`` and score it against the raw column on the odd rows of the scan sample.

    Returns one dict per usable column: ``col``, ``x`` (the full column), ``fit`` (the service result on the scan rows), ``rows`` (the scan index), ``mi_raw``, ``mi_warp``, ``gain``, ``se``.
    Columns that are constant, nominal-like (at most ``FEW_CLASSES_MAX`` distinct values) or too short are skipped."""
    from ._y_encoding import encode_y_for_classif_mi

    n = len(X)
    if n < _MIN_ROWS:
        return []
    y_arr = np.asarray(y).ravel()
    rows = _scan_index(n, scan_rows)
    codes = np.asarray(encode_y_for_classif_mi(y_arr[rows]), dtype=np.int64)
    ky = int(codes.max()) + 1
    yr = _rank_scaled(y_arr[rows])
    even = np.arange(len(rows)) % 2 == 0
    y_odd = codes[~even]
    out = []
    for c in cols:
        x = np.asarray(X[c].to_numpy(), dtype=np.float64)
        xr = x[rows]
        finite = np.isfinite(xr)
        if finite.sum() < _MIN_ROWS or np.unique(xr[finite]).size <= FEW_CLASSES_MAX:
            continue
        fit = fit_oof_warp1d(xr, yr, seed=seed)
        cw = held_out_codes(fit["oof"], even, ~even, N_MI_BINS)
        cr = held_out_codes(np.where(finite, xr, np.min(xr[finite])), even, ~even, N_MI_BINS)
        mi_w, mi_r = plugin_mi_of_codes(cw, N_MI_BINS, y_odd, ky), plugin_mi_of_codes(cr, N_MI_BINS, y_odd, ky)
        out.append({"col": c, "x": x, "fit": fit, "rows": rows, "mi_raw": mi_r, "mi_warp": mi_w, "gain": mi_w - mi_r, "se": paired_gain_se(cw, cr, y_odd, ky, N_MI_BINS)})
    return out


def training_column(cand: dict) -> np.ndarray:
    """The warp column over every row: the cross-fitted value on the fitting rows, the full table on the others."""
    fit = cand["fit"]
    col = apply_warp1d(cand["x"], fit["cx"], fit["cy"], fit["fill"])
    col[cand["rows"]] = fit["oof"]
    return col


def hybrid_oof_warp_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    top_k: int = 5,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    min_relative_gain: float = DEFAULT_MIN_RELATIVE_GAIN,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """Warp every usable numeric column and keep at most ``top_k`` whose held-out MI beats the raw column significantly (see the module docstring).

    Returns ``(X_aug, appended, recipes, enc_df)``; ``y`` only steers the fit and the acceptance, recipes carry the table, never ``y``."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_oof_warp_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    empty = (X, [], [], pd.DataFrame())
    cols = [c for c in (num_cols if num_cols else X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if not cols or y is None:
        return empty
    accepted = []
    for cand in warp_candidates(X, y, cols, scan_rows=scan_rows):
        if cand["gain"] > SIGNIFICANCE_Z * cand["se"] and cand["gain"] > float(min_relative_gain) * max(cand["mi_raw"], 0.0):
            accepted.append(cand)
        elif reject_sink is not None:
            reject_sink(
                gate="oof_warp_gain", candidate=f"oofwarp({cand['col']})", operand_names=str(cand["col"]), operator="oof_warp1d", observed=cand["gain"],
                threshold=SIGNIFICANCE_Z * cand["se"], reason="held-out MI gain over the raw column not significant or below the practical effect",
            )
    accepted.sort(key=lambda c: -c["gain"])
    new_cols, recipes = {}, []
    for cand in accepted[: int(top_k)]:
        name = f"oofwarp({cand['col']})"
        if name in X.columns or name in new_cols:
            continue
        col = training_column(cand)
        new_cols[name] = col
        fit = cand["fit"]
        recipes.append(build_oof_warp1d_recipe(name=name, src=cand["col"], cx=fit["cx"], cy=fit["cy"], fill=fit["fill"], lo=float(col.min()), hi=float(col.max())))
    if not new_cols:
        return empty
    enc_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, enc_df], axis=1), list(new_cols), recipes, enc_df
