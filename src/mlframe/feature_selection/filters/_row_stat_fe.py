"""Row-statistic operator: min, max, median, range, std or a soft max / soft min of a learned subset of columns.

A target that depends on the spread or the extreme of several columns jointly (the range of four sensors, the worst of five tolerances) is invisible to every pair form and to the marginal MI of the
columns. The search standardises up to ``MAX_POOL`` numeric columns, scores EVERY pair under each of the seven statistics in one parallel kernel (``_row_stat_kernels.eval_candidates``: bin edges
from a sample of the even rows, MI on the even rows for the selection and on the odd rows held out), grows the best subset of each statistic one column at a time while the even-row MI rises by
more than the noise scale of a plug-in MI (``Z * sqrt(2 * df) / (2 n)``, ``df = (bins - 1)(classes - 1)``, from the chi-square law of ``2 n MI`` under independence), and keeps the finalists
whose HELD-OUT MI beats the best raw column's by more than ``SIGNIFICANCE_Z`` standard errors of the paired difference (``_fe_gain_stats.paired_gain_se``) and by at least the practical effect
``min_relative_gain``. The baseline is the better of the best raw column and the best LINEAR combination of the finalist's own columns (least-squares weights fitted on the even rows against
the rank target): a median or soft max of two columns is close to their mean, and a linear mix is what the sum forms and the linear models already cover. The plain mean is not a candidate statistic, and a subset of two columns is left to the pair forms (``MIN_SUBSET`` = 3).

Replay (kind ``row_stat``) stores the source columns, the frozen standardisation, the statistic and the output range; it is a pure function of the source columns.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._fe_gain_stats import held_out_codes, paired_gain_se, plugin_mi_of_codes, quantile_codes
from ._row_stat_kernels import MAX_SUBSET, STAT_NAMES, eval_candidates, stat_column
from ._y_encoding import FEW_CLASSES_MAX

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = ["hybrid_row_stat_fe", "apply_row_stat_recipe", "build_row_stat_recipe", "row_stat_candidates"]

MAX_POOL = 16  # numeric columns the search ranges over
MIN_SUBSET = 3  # a statistic of two columns is a pair form (max, min, sum, difference ...) the preset already has; the operator exists for three or more
SIGNIFICANCE_Z = 2.0  # one-sided normal critical value applied to the standard error of the MI gain and to the growth noise scale
DEFAULT_MIN_RELATIVE_GAIN = 0.05  # default of the practical-effect knob
DEFAULT_SCAN_ROWS = 20_000  # rows the subset search runs on (a pair table and a growth round are n * k per candidate)
N_MI_BINS = 10
_MIN_ROWS = 800
_OUTPUT_CLIP_Q = (0.001, 0.999)


def build_row_stat_recipe(*, name: str, src: Sequence[str], stat: str, mu: np.ndarray, sd: np.ndarray, lo: float, hi: float) -> "EngineeredRecipe":
    """Frozen recipe of one row-statistic column: source columns, statistic, frozen standardisation and output range."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="row_stat",
        src_names=tuple(str(c) for c in src),
        extra={"stat": str(stat), "mu": np.asarray(mu, dtype=np.float64).copy(), "sd": np.asarray(sd, dtype=np.float64).copy(), "lo": float(lo), "hi": float(hi)},
    )


def _standardised_block(A: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """``(p, n)`` C-contiguous block of the standardised columns of the ``(n, p)`` array ``A``; a non-finite value becomes 0 (the training mean)."""
    Z = (A - mu) / sd
    return np.ascontiguousarray(np.where(np.isfinite(Z), Z, 0.0).T)


def apply_row_stat_recipe(recipe, X, _block_cache: Optional[dict] = None) -> np.ndarray:
    """Replay one row-statistic column from the stored standardisation; a pure function of the source columns.

    ``_block_cache`` (fit time only): statistics over the same source columns share one standardised ``(p, n)`` block instead of rebuilding it."""
    from .engineered_recipes.shared import extract_column

    ex = recipe.extra
    key = (tuple(recipe.src_names), ex["mu"].tobytes(), ex["sd"].tobytes())
    Zt = _block_cache.get(key) if _block_cache is not None else None
    if Zt is None:
        A = np.column_stack([np.asarray(extract_column(X, c), dtype=np.float64) for c in recipe.src_names])
        Zt = _standardised_block(A, ex["mu"], ex["sd"])
        if _block_cache is not None:
            _block_cache[key] = Zt
    k = Zt.shape[0]
    subset = np.arange(k, dtype=np.int64)
    out = stat_column(Zt, subset, k, STAT_NAMES.index(ex["stat"]))
    return np.clip(out, ex["lo"], ex["hi"])


def _rank_scaled(y: np.ndarray) -> np.ndarray:
    """Average ranks of ``y`` scaled into ``(0, 1]``."""
    from scipy.stats import rankdata

    return rankdata(y, method="average") / float(len(y))


def _scan_index(n: int, scan_rows: int) -> np.ndarray:
    """Evenly spaced row indices of the search sample (all rows when ``n`` is small)."""
    return np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)


def _noise_scale(n_even: int, nb: int, ky: int) -> float:
    """Standard deviation of a plug-in MI under independence, ``sqrt(2 df) / (2 n)`` with ``df = (nb - 1)(ky - 1)`` (``2 n MI`` is chi-square distributed): the growth step must beat ``Z`` of them."""
    return float(np.sqrt(2.0 * (nb - 1) * (ky - 1)) / (2.0 * n_even))


def _pool_columns(X: "pd.DataFrame", cols: Sequence[str], rows: np.ndarray, codes: np.ndarray, ky: int) -> "list[str]":
    """Up to ``MAX_POOL`` numeric columns, ranked by the larger of the MI of the column and the MI of its distance from the median (a spread effect is symmetric and invisible to the first)."""
    scored = []
    for c in cols:
        x = np.asarray(X[c].to_numpy(), dtype=np.float64)[rows]
        finite = np.isfinite(x)
        if finite.sum() < _MIN_ROWS or np.unique(x[finite]).size <= FEW_CLASSES_MAX:
            continue
        x = np.where(finite, x, np.median(x[finite]))
        mi = max(plugin_mi_of_codes(quantile_codes(x, N_MI_BINS), N_MI_BINS, codes, ky), plugin_mi_of_codes(quantile_codes(np.abs(x - np.median(x)), N_MI_BINS), N_MI_BINS, codes, ky))
        scored.append((mi, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:MAX_POOL]]


def _subset_matrix(subsets: "list[list[int]]") -> "tuple[np.ndarray, np.ndarray]":
    """``(B, MAX_SUBSET)`` padded index matrix and the sizes of the candidate subsets."""
    mat = np.zeros((len(subsets), MAX_SUBSET), dtype=np.int64)
    for b, s in enumerate(subsets):
        mat[b, : len(s)] = s
    return mat, np.array([len(s) for s in subsets], dtype=np.int64)


def _search(Zt: np.ndarray, codes: np.ndarray, ky: int) -> "list[tuple[float, int, list[int]]]":
    """Per statistic, the best subset (all pairs, then greedy growth); returns ``(even-row MI, statistic id, subset)`` sorted by MI, best first."""
    p, n = Zt.shape
    n_stats = len(STAT_NAMES)
    pairs = list(itertools.combinations(range(p), 2))
    cand = [[a, b] for _ in range(n_stats) for a, b in pairs]
    stats = np.repeat(np.arange(n_stats, dtype=np.int64), len(pairs))
    mat, sizes = _subset_matrix(cand)
    ev, od = np.empty(len(cand)), np.empty(len(cand))
    eval_candidates(Zt, mat, sizes, stats, codes, ky, N_MI_BINS, ev, od)
    best = []
    for s in range(n_stats):
        sl = slice(s * len(pairs), (s + 1) * len(pairs))
        j = int(np.argmax(ev[sl]))
        best.append([float(ev[sl][j]), s, list(cand[s * len(pairs) + j])])
    margin = SIGNIFICANCE_Z * _noise_scale((n + 1) // 2, N_MI_BINS, ky)
    for _ in range(MAX_SUBSET - 2):
        grow, owners = [], []
        for s, (_mi, _s, sub) in enumerate(best):
            if len(sub) < min(MAX_SUBSET, p):
                for j in range(p):
                    if j not in sub:
                        grow.append([*sub, j])
                        owners.append(s)
        if not grow:
            break
        mat, sizes = _subset_matrix(grow)
        st = np.array([best[s][1] for s in owners], dtype=np.int64)
        ev, od = np.empty(len(grow)), np.empty(len(grow))
        eval_candidates(Zt, mat, sizes, st, codes, ky, N_MI_BINS, ev, od)
        grew = False
        for s in set(owners):
            idx = [i for i, o in enumerate(owners) if o == s]
            j = idx[int(np.argmax(ev[idx]))]
            if ev[j] > best[s][0] + margin:
                best[s] = [float(ev[j]), s, grow[j]]
                grew = True
        if not grew:
            break
    best.sort(key=lambda t: -t[0])
    return [(mi, s, sorted(sub)) for mi, s, sub in best]


def row_stat_candidates(X: "pd.DataFrame", y: np.ndarray, cols: Sequence[str], *, scan_rows: int = DEFAULT_SCAN_ROWS) -> "list[dict]":
    """The finalists of the subset search with their held-out evidence.

    One dict per statistic whose best subset has at least two columns: ``stat``, ``src`` (column names), ``mu`` / ``sd`` (frozen standardisation), ``rows`` (the search sample), ``gain`` and ``se``
    (held-out MI gain over the best raw column of the pool and its standard error), ``mi_base``. Empty when fewer than two usable columns or too few rows."""
    from ._y_encoding import encode_y_for_classif_mi

    n = len(X)
    if n < _MIN_ROWS or len(cols) < 2:
        return []
    y_arr = np.asarray(y).ravel()
    rows = _scan_index(n, scan_rows)
    codes = np.asarray(encode_y_for_classif_mi(y_arr[rows]), dtype=np.int64)
    ky = int(codes.max()) + 1
    pool = _pool_columns(X, cols, rows, codes, ky)
    if len(pool) < 2:
        return []
    A = np.column_stack([np.asarray(X[c].to_numpy(), dtype=np.float64)[rows] for c in pool])  # the scan rows only: the full columns are touched for the accepted statistics alone
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Zt = _standardised_block(A, mu, sd)
    even = np.arange(len(rows)) % 2 == 0
    y_odd = codes[~even]
    raw_codes = [held_out_codes(Zt[j], even, ~even, N_MI_BINS) for j in range(len(pool))]
    raw_mi = [plugin_mi_of_codes(c, N_MI_BINS, y_odd, ky) for c in raw_codes]
    jb = int(np.argmax(raw_mi))
    yr = _rank_scaled(y_arr[rows])
    out = []
    for mi_train, s, sub in _search(Zt, codes, ky):
        if len(sub) < MIN_SUBSET:
            continue
        feat = stat_column(Zt, np.array(sub, dtype=np.int64), len(sub), s)
        cc = held_out_codes(feat, even, ~even, N_MI_BINS)
        mi_c = plugin_mi_of_codes(cc, N_MI_BINS, y_odd, ky)
        w = np.linalg.lstsq(Zt[sub][:, even].T, yr[even] - yr[even].mean(), rcond=None)[0]
        lin_codes = held_out_codes(w @ Zt[sub], even, ~even, N_MI_BINS)
        mi_lin = plugin_mi_of_codes(lin_codes, N_MI_BINS, y_odd, ky)
        base_codes, mi_base, base_col = (lin_codes, mi_lin, "linear mix") if mi_lin > raw_mi[jb] else (raw_codes[jb], raw_mi[jb], pool[jb])
        out.append(
            {
                "stat": STAT_NAMES[s], "src": [pool[j] for j in sub], "mu": mu[sub], "sd": sd[sub], "rows": rows, "mi_train": mi_train, "mi_cand": mi_c, "mi_base": mi_base,
                "gain": mi_c - mi_base, "se": paired_gain_se(cc, base_codes, y_odd, ky, N_MI_BINS), "base_col": base_col,
            }
        )
    return out


def hybrid_row_stat_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    top_k: int = 2,
    scan_rows: int = DEFAULT_SCAN_ROWS,
    min_relative_gain: float = DEFAULT_MIN_RELATIVE_GAIN,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """Search the row statistics, keep at most ``top_k`` finalists whose held-out MI beats the best raw column significantly (see the module docstring).

    Returns ``(X_aug, appended, recipes, enc_df)``; ``y`` only steers the search and the acceptance, recipes carry the standardisation and the statistic, never ``y``."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_row_stat_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    empty = (X, [], [], pd.DataFrame())
    cols = [c for c in (num_cols if num_cols else X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(cols) < 2 or y is None:
        return empty
    accepted = []
    for cand in row_stat_candidates(X, y, cols, scan_rows=scan_rows):
        if cand["gain"] > SIGNIFICANCE_Z * cand["se"] and cand["gain"] > float(min_relative_gain) * max(cand["mi_base"], 0.0):
            accepted.append(cand)
        elif reject_sink is not None:
            reject_sink(
                gate="row_stat_gain", candidate=f"rowstat_{cand['stat']}({','.join(map(str, cand['src']))})", operand_names=",".join(map(str, cand["src"])), operator=cand["stat"],
                observed=cand["gain"], threshold=SIGNIFICANCE_Z * cand["se"], reason="held-out MI gain over the best raw column not significant or below the practical effect",
            )
    accepted.sort(key=lambda c: -c["gain"])
    new_cols, recipes, blocks = {}, [], {}
    for cand in accepted[: int(top_k)]:
        name = f"rowstat_{cand['stat']}({','.join(map(str, cand['src']))})"
        if name in X.columns or name in new_cols:
            continue
        rec = build_row_stat_recipe(name=name, src=cand["src"], stat=cand["stat"], mu=cand["mu"], sd=cand["sd"], lo=-np.inf, hi=np.inf)
        col = apply_row_stat_recipe(rec, X, blocks)
        lo, hi = np.quantile(col[cand["rows"]], _OUTPUT_CLIP_Q)
        if not hi > lo:
            continue
        rec = build_row_stat_recipe(name=name, src=cand["src"], stat=cand["stat"], mu=cand["mu"], sd=cand["sd"], lo=lo, hi=hi)
        new_cols[name] = np.clip(col, lo, hi)
        recipes.append(rec)
    if not new_cols:
        return empty
    enc_df = pd.DataFrame(new_cols, index=X.index)
    return pd.concat([X, enc_df], axis=1), list(new_cols), recipes, enc_df
