"""FAMILY D -- Cross-feature conditional DISPERSION / 2nd-moment (NUM x NUM)

Backlog #12 (2026-06-09). Family B (``_extra_fe_families.conditional_residual``)
models conditional **location** -- ``x_i - E[x_i | bin(x_j)]`` -- but NOTHING in
the L21-L104 catalog models conditional **scale**. A small mean-residual can be a
huge outlier vs a TIGHT conditional spread (a fraud amount anomalous FOR that
merchant, a latency anomalous FOR that payload size, a volatility regime). Family
D fills that gap.

Mechanism
---------
Bin ``x_j`` into ``n_bins`` quantile bins (the SAME edges Family B uses); per bin
store BOTH the conditional mean ``mu_hat_bin`` AND the conditional std
``sigma_hat_bin`` of ``x_i``. Emit the conditional z-score and dispersion anomaly:

* ``z   = (x_i - mu_hat_bin) / sigma_hat_bin``  (signed, scale-normalised
  deviation);
* ``|z| = abs(z)``                              (the dispersion ANOMALY -- how
  many conditional std's out, regardless of direction);
* ``z2  = z**2``                                (squared anomaly -- emphasises
  the tails / extreme outliers).

The signed ``z`` carries the SAME monotone ordering as Family B's raw residual
(divided by a per-bin constant), so on its own it is a near-duplicate the dedup
drops; the load-bearing emissions are ``|z|`` and ``z2``, which are NON-monotone
folds of the residual and so carry signal a marginal / location feature cannot
express (the dispersion-anomaly target ``y = 1[|x_i| unusually far from its
x_j-conditional spread]``).

Why this is MI-gateable (UNLIKE the hinge / isotonic monotone reshapes)
-----------------------------------------------------------------------
``|z|`` is a genuinely NON-monotone transform of ``x_i`` (it folds the residual
about its conditional mean), so on a heteroscedastic target it carries MUTUAL
INFORMATION the raw column and the Family-B mean-residual do NOT -- it clears the
NORMAL MI-uplift gate (the same gate Families A/B route through). This is the key
contrast with backlog #11/#14: a single relu leg / isotonic fit is MONOTONE ->
MI-invariant by the data-processing inequality, so they need the
linear-usability admission path; the dispersion ``|z|`` does NOT, it is a true
MI gain on the right target.

Self-limiting on homoscedastic data
-----------------------------------
When ``x_i``'s conditional spread is CONSTANT across ``x_j`` bins
(``sigma_hat_bin ~= sigma_const``), ``z = residual / sigma_const`` is just a
SCALED copy of the Family-B mean-residual and ``|z|`` is a scaled ``|residual|``.
The cross-stage Spearman dedup that already protects the catalog then drops the
dispersion column as a near-duplicate of its mean-residual sibling -- so on
homoscedastic data Family D adds NOTHING (it only earns its keep where the
conditional spread genuinely varies). Pure noise -> no dispersion column clears
the MI-uplift gate. Both are verified in the biz-value benchmark.

Leak-safe replay
----------------
The recipe (kind ``"conditional_dispersion"``) extends Family B's
``conditional_residual`` payload with the per-bin std: it stores the ``x_j``
quantile EDGES + the per-bin ``(mu_hat, sigma_hat)`` of ``x_i`` (NO y). Replay
digitises ``x_j`` with the stored edges, looks up the per-bin ``(mu_hat,
sigma_hat)``, and computes ``z = (x_i - mu_hat) / sigma_hat`` closed-form --
reads only X, so ``transform`` is leakage-free by construction (exactly Family
B's contract, plus the stored std).
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# NOTE: the binning helpers ``_digitize_with_edges`` / ``_quantile_edges`` /
# ``_top_mi_num_cols`` live in the PARENT ``_extra_fe_families`` module, which
# re-imports THIS sibling at its bottom (the Family-D re-export). A top-level
# ``from ._extra_fe_families import ...`` here therefore closes an import cycle
# (parent <-> sibling) that ``test_no_import_cycles`` flags. Import them LAZILY
# in-body in the three consumers below (mirroring the already-lazy
# ``generate_conditional_residual_features`` / ``engineered_name_conditional_residual``
# imports in this module) so the parent finishes initialising before the sibling
# resolves the names. 2026-06-10.

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_conditional_dispersion",
    "generate_conditional_dispersion_features",
    "apply_conditional_dispersion",
    "build_conditional_dispersion_recipe",
    "_apply_conditional_dispersion_recipe",
    "hybrid_conditional_dispersion_fe",
]

# Dispersion-anomaly emission kinds. ``z`` (signed) is kept for completeness but
# is a near-duplicate of the Family-B residual under dedup; ``absz`` / ``z2`` are
# the load-bearing NON-monotone folds that carry heteroscedastic signal.
_DISPERSION_KINDS = ("z", "absz", "z2")

# Floor on a per-bin std before it is usable as a divisor. A bin with < this many
# rows, or a (near-)constant ``x_i`` within the bin, has an unreliable sigma_hat;
# we fall back to the GLOBAL std of the (finite) residual so the z-score is still
# scale-normalised and replay-stable rather than dividing by ~0 (which would
# manufacture a spurious huge |z|).
_DISPERSION_MIN_BIN_ROWS: int = 2
_DISPERSION_SIGMA_FLOOR: float = 1e-9


def engineered_name_conditional_dispersion(x_i: str, x_j: str, kind: str) -> str:
    """``(x_i - E[x_i|bin(x_j)]) / Std[x_i|bin(x_j)]`` reads as "x_i's deviation
    from its x_j-conditional expectation, in units of its x_j-conditional
    spread". ``kind`` in ``{z, absz, z2}`` selects signed / anomaly / squared."""
    suffix = {"z": "zscore", "absz": "absz", "z2": "z2"}.get(kind, kind)
    return f"{x_i}__{suffix}_by__{x_j}"


def _per_bin_mean_std(
    xi: np.ndarray,
    codes_j: np.ndarray,
    n_bins_eff: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Per-bin (mean, std) of the FINITE ``x_i`` values grouped by ``codes_j``.

    Returns ``(bin_mean, bin_std, global_mean, global_std)``. A bin with too few
    rows or a (near-)constant ``x_i`` gets ``global_std`` as its std (an honest
    scale-normaliser instead of a ~0 divisor that would fabricate a giant |z|).
    Uses the numerically-stable two-pass mean-then-centred-SS reduction over the
    grouped values (one ``np.add.at`` for the sum, one for the centred SS)."""
    finite_i = np.isfinite(xi)
    if not finite_i.any():
        z = np.zeros(n_bins_eff, dtype=np.float64)
        return z.copy(), np.ones(n_bins_eff, dtype=np.float64), 0.0, 1.0
    xi_f = xi[finite_i]
    codes_f = codes_j[finite_i]
    global_mean = float(xi_f.mean())
    global_std = float(xi_f.std())
    if not np.isfinite(global_std) or global_std < _DISPERSION_SIGMA_FLOOR:
        global_std = 1.0  # degenerate x_i -> unit normaliser (residual ~0 anyway)

    bin_sum = np.zeros(n_bins_eff, dtype=np.float64)
    bin_cnt = np.zeros(n_bins_eff, dtype=np.float64)
    np.add.at(bin_sum, codes_f, xi_f)
    np.add.at(bin_cnt, codes_f, 1.0)
    bin_mean = np.where(
        bin_cnt > 0.0, bin_sum / np.maximum(bin_cnt, 1.0), global_mean,
    )
    # Centred sum of squares per bin: sum((x_i - bin_mean[bin])^2). Subtract each
    # row's bin mean, square, scatter-add -- one extra O(n) pass, no per-bin loop.
    centred = xi_f - bin_mean[codes_f]
    bin_css = np.zeros(n_bins_eff, dtype=np.float64)
    np.add.at(bin_css, codes_f, centred * centred)
    # Population std per bin (ddof=0): sqrt(CSS / count). Bins below the min-row
    # floor or with a (near-)constant x_i fall back to the global std.
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_std = np.sqrt(np.where(bin_cnt > 0.0, bin_css / np.maximum(bin_cnt, 1.0), 0.0))
    usable = (bin_cnt >= float(_DISPERSION_MIN_BIN_ROWS)) & (
        bin_std >= _DISPERSION_SIGMA_FLOOR
    )
    bin_std = np.where(usable, bin_std, global_std)
    return (
        bin_mean.astype(np.float64),
        bin_std.astype(np.float64),
        float(global_mean),
        float(global_std),
    )


def _zscore_from_bins(
    xi: np.ndarray,
    codes_j: np.ndarray,
    bin_mean: np.ndarray,
    bin_std: np.ndarray,
) -> np.ndarray:
    """Closed-form conditional z-score ``(x_i - mu_hat_bin) / sigma_hat_bin``.
    NaN ``x_i`` rows emit 0.0 (no deviation information). Std is already floored
    to ``global_std`` in degenerate bins, so the divide is always finite."""
    codes_j = np.clip(codes_j, 0, bin_mean.size - 1)
    per_row_mean = bin_mean[codes_j]
    per_row_std = bin_std[codes_j]
    per_row_std = np.where(
        per_row_std >= _DISPERSION_SIGMA_FLOOR, per_row_std, 1.0,
    )
    finite_i = np.isfinite(xi)
    z = np.zeros_like(xi, dtype=np.float64)
    z[finite_i] = (xi[finite_i] - per_row_mean[finite_i]) / per_row_std[finite_i]
    return z


def _emit_kind(z: np.ndarray, kind: str) -> np.ndarray:
    """Map the signed z-score to the requested emission kind."""
    if kind == "z":
        return z
    if kind == "absz":
        return np.abs(z)
    if kind == "z2":
        return z * z
    raise ValueError(f"conditional_dispersion: unknown kind {kind!r}")


def generate_conditional_dispersion_features(
    X: pd.DataFrame,
    num_cols: Sequence[str],
    *,
    n_bins: int = 10,
    kinds: Sequence[str] = ("absz", "z2"),
):
    """For every ordered pair ``(x_i, x_j)`` from ``num_cols`` emit the
    conditional dispersion-anomaly columns selected by ``kinds`` (default the two
    NON-monotone folds ``|z|`` and ``z**2`` -- the signed ``z`` is a near-
    duplicate of the Family-B mean-residual under dedup).

    Returns ``(enc_df, raw_recipes)``. Each payload stores the ``x_j`` quantile
    edges + per-bin ``(mu_hat, sigma_hat)`` of ``x_i`` (extends Family B's
    ``bin_mean`` with ``bin_std``); replay digitises ``x_j`` with the stored edges
    and computes ``z = (x_i - mu_hat) / sigma_hat`` closed-form (no ``y``,
    leak-safe).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"generate_conditional_dispersion_features: X must be a pandas "
            f"DataFrame; got {type(X).__name__}"
        )
    if len(X) == 0:
        raise ValueError("generate_conditional_dispersion_features: X is empty")
    kinds = [k for k in kinds if k in _DISPERSION_KINDS]
    if not kinds:
        kinds = ["absz", "z2"]
    num_cols = [
        c for c in num_cols
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if len(num_cols) < 2:
        return pd.DataFrame(index=X.index), raw_recipes

    from ._extra_fe_families import _digitize_with_edges, _quantile_edges  # lazy: break parent<->sibling cycle
    col_vals = {c: np.asarray(X[c].to_numpy(), dtype=np.float64) for c in num_cols}
    for x_j in num_cols:
        xj = col_vals[x_j]
        edges = _quantile_edges(xj, n_bins)
        codes_j = _digitize_with_edges(xj, edges)
        n_bins_eff = edges.size - 1
        for x_i in num_cols:
            if x_i == x_j:
                continue
            xi = col_vals[x_i]
            bin_mean, bin_std, global_mean, global_std = _per_bin_mean_std(
                xi, codes_j, n_bins_eff,
            )
            z = _zscore_from_bins(xi, codes_j, bin_mean, bin_std)
            payload = {
                "x_i": x_i,
                "x_j": x_j,
                "edges": edges,
                "bin_mean": bin_mean,
                "bin_std": bin_std,
                "global_mean": global_mean,
                "global_std": global_std,
            }
            for kind in kinds:
                vals = _emit_kind(z, kind)
                # Skip a degenerate constant emission (e.g. |z| all 0) -- it
                # carries no information and only burdens the screen.
                if float(np.std(vals)) <= 1e-12:
                    continue
                name = engineered_name_conditional_dispersion(x_i, x_j, kind)
                encoded[name] = vals.astype(np.float64)
                raw_recipes[name] = {**payload, "kind": kind}

    return pd.DataFrame(encoded, index=X.index), raw_recipes


def apply_conditional_dispersion(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay one conditional-dispersion column from the stored x_j edges +
    per-bin ``(mu_hat, sigma_hat)``. NaN ``x_i`` rows emit 0.0. Reads only X."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_conditional_dispersion: X_test must be a DataFrame; got "
            f"{type(X_test).__name__}"
        )
    x_i = recipe["x_i"]
    x_j = recipe["x_j"]
    if x_i not in X_test.columns or x_j not in X_test.columns:
        raise KeyError(
            f"apply_conditional_dispersion: missing column(s) {x_i!r}/{x_j!r} "
            f"from X_test"
        )
    edges = np.asarray(recipe["edges"], dtype=np.float64)
    bin_mean = np.asarray(recipe["bin_mean"], dtype=np.float64)
    bin_std = np.asarray(recipe["bin_std"], dtype=np.float64)
    kind = str(recipe.get("kind", "absz"))
    from ._extra_fe_families import _digitize_with_edges  # lazy: break parent<->sibling cycle
    xi = np.asarray(X_test[x_i].to_numpy(), dtype=np.float64)
    xj = np.asarray(X_test[x_j].to_numpy(), dtype=np.float64)
    codes_j = _digitize_with_edges(xj, edges)
    z = _zscore_from_bins(xi, codes_j, bin_mean, bin_std)
    return _emit_kind(z, kind).astype(np.float64)


def build_conditional_dispersion_recipe(
    *, name: str, x_i: str, x_j: str, edges: np.ndarray, bin_mean: np.ndarray,
    bin_std: np.ndarray, global_mean: float, global_std: float, kind: str,
):
    """Frozen recipe for one NUM x NUM conditional-dispersion column. Extends the
    Family-B ``conditional_residual`` payload with the per-bin std: stores the
    ``x_j`` quantile edges + per-bin ``(mu_hat, sigma_hat)`` of ``x_i`` + the
    emission ``kind``; replay reads only X (no y), so transform() is
    leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    if kind not in _DISPERSION_KINDS:
        raise ValueError(
            f"build_conditional_dispersion_recipe: kind must be one of "
            f"{_DISPERSION_KINDS}; got {kind!r}"
        )
    return EngineeredRecipe(
        name=name,
        kind="conditional_dispersion",
        src_names=(str(x_i), str(x_j)),
        extra={
            "x_i": str(x_i),
            "x_j": str(x_j),
            "edges": np.asarray(edges, dtype=np.float64).copy(),
            "bin_mean": np.asarray(bin_mean, dtype=np.float64).copy(),
            "bin_std": np.asarray(bin_std, dtype=np.float64).copy(),
            "global_mean": float(global_mean),
            "global_std": float(global_std),
            "disp_kind": str(kind),
        },
    )


def _apply_conditional_dispersion_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    x_i = str(recipe.extra["x_i"])
    x_j = str(recipe.extra["x_j"])
    if isinstance(X, pd.DataFrame):
        X_view = X
    else:
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                X_view = pd.DataFrame(
                    {x_i: X[x_i].to_numpy(), x_j: X[x_j].to_numpy()}
                )
            else:
                raise TypeError
        except (ImportError, TypeError):
            if isinstance(X, np.ndarray) and X.dtype.names is not None:
                X_view = pd.DataFrame({x_i: X[x_i], x_j: X[x_j]})
            else:
                raise TypeError(
                    f"conditional_dispersion recipe '{recipe.name}': cannot "
                    f"extract {x_i!r}/{x_j!r} from X of type {type(X).__name__}."
                )
    return apply_conditional_dispersion(
        X_view,
        {
            "x_i": x_i,
            "x_j": x_j,
            "edges": np.asarray(recipe.extra["edges"], dtype=np.float64),
            "bin_mean": np.asarray(recipe.extra["bin_mean"], dtype=np.float64),
            "bin_std": np.asarray(recipe.extra["bin_std"], dtype=np.float64),
            "kind": str(recipe.extra.get("disp_kind", "absz")),
        },
    )


# Minimum MI uplift a dispersion column must show over BOTH its raw source and
# its Family-B mean-residual sibling to be admitted. A genuine heteroscedastic
# fixture lifts MI(|z|) tens of percent over the location features (measured
# ~0.04-0.05 absolute on the two-bin fixture); on homoscedastic data the uplift
# is ~0 (|z| is a scaled |residual|) so the column is dropped. 0.01 absolute MI
# is a conservative margin: well below the genuine uplift, well above the
# finite-sample MI noise on a null pair.
_DISPERSION_MIN_MI_UPLIFT: float = 0.01


def _dual_uplift_filter(
    X: pd.DataFrame,
    y,
    enc_df: pd.DataFrame,
    winners: Sequence[str],
    raw_recipes: dict,
    *,
    n_bins: int,
    min_uplift: float = _DISPERSION_MIN_MI_UPLIFT,
) -> list[str]:
    """Keep only dispersion columns whose ``MI(col; y)`` exceeds BOTH the raw
    source ``x_i`` and the Family-B mean-residual sibling by ``min_uplift``.

    The mean-residual sibling for a dispersion column built on ``(x_i, x_j)`` is
    ``x_i - E[x_i | bin(x_j)]`` (Family B with the SAME edges). We materialise
    those siblings once per distinct ``(x_i, x_j)`` pair in the winner set and
    compare batched MI. Columns that do not beat the cheaper location features are
    dropped -- the self-limit that keeps homoscedastic / canonical fixtures from
    accreting a redundant dispersion column."""
    if not winners:
        return []
    from ._orthogonal_univariate_fe import _mi_classif_batch
    from ._unified_fe_gate import _coerce_y_classes
    from ._extra_fe_families import generate_conditional_residual_features

    y_bin = _coerce_y_classes(y)
    # MI of the candidate dispersion columns.
    cand_arr = enc_df[list(winners)].to_numpy(dtype=np.float64)
    cand_mi = np.asarray(_mi_classif_batch(cand_arr, y_bin, nbins=n_bins), dtype=np.float64)
    cand_mi = {nm: float(cand_mi[j]) for j, nm in enumerate(winners)}

    # Distinct (x_i, x_j) pairs among winners -> raw x_i MI + mean-residual sibling MI.
    pairs = {}
    for nm in winners:
        rec = raw_recipes.get(nm, {})
        pairs[(rec.get("x_i"), rec.get("x_j"))] = None
    raw_cols = sorted({xi for (xi, _xj) in pairs if xi is not None and xi in X.columns})
    raw_mi = {}
    if raw_cols:
        raw_arr = X[raw_cols].to_numpy(dtype=np.float64)
        rmi = np.asarray(_mi_classif_batch(raw_arr, y_bin, nbins=n_bins), dtype=np.float64)
        raw_mi = {c: float(rmi[j]) for j, c in enumerate(raw_cols)}

    # Family-B mean-residual sibling MI per distinct pair.
    sib_cols = [xi for (xi, xj) in pairs if xi is not None and xj is not None]
    sib_mi = {}
    if sib_cols:
        uniq_num = sorted({c for p in pairs for c in p if c is not None and c in X.columns})
        try:
            enc_b, _ = generate_conditional_residual_features(
                X[uniq_num], uniq_num, n_bins=n_bins,
            )
        except Exception:
            enc_b = pd.DataFrame(index=X.index)
        from ._extra_fe_families import engineered_name_conditional_residual
        sib_names = {}
        for (xi, xj) in pairs:
            if xi is None or xj is None:
                continue
            sn = engineered_name_conditional_residual(xi, xj)
            if sn in enc_b.columns:
                sib_names[(xi, xj)] = sn
        if sib_names:
            # Compare against the |mean-residual| (the ABSOLUTE Family-B residual),
            # NOT the signed residual. The load-bearing dispersion emissions are
            # FOLDS (|z|, z^2); the fair "cheaper location feature it could be
            # replacing" is the matching FOLD of the location residual. On
            # homoscedastic data |z| = |resid| / sigma_const is rank-identical to
            # |resid|, so MI(|z|) == MI(|resid|) and the uplift is ~0 -> dropped
            # (self-limiting). On heteroscedastic data the per-bin scale-norm makes
            # |z| carry MI the unnormalised |resid| cannot -> uplift clears.
            sib_abs = np.abs(
                enc_b[list(sib_names.values())].to_numpy(dtype=np.float64)
            )
            smi = np.asarray(_mi_classif_batch(sib_abs, y_bin, nbins=n_bins), dtype=np.float64)
            sib_mi = {pair: float(smi[j]) for j, pair in enumerate(sib_names.keys())}

    kept = []
    for nm in winners:
        rec = raw_recipes.get(nm, {})
        xi, xj = rec.get("x_i"), rec.get("x_j")
        m = cand_mi.get(nm, 0.0)
        base_raw = raw_mi.get(xi, 0.0)
        base_sib = sib_mi.get((xi, xj), 0.0)
        if not np.isfinite(m):
            continue
        if m >= base_raw + min_uplift and m >= base_sib + min_uplift:
            kept.append(nm)
    return kept


def hybrid_conditional_dispersion_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    kinds: Sequence[str] = ("absz", "z2"),
    top_k: int = 10,
    max_pair_cols: int = 6,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
):
    """End-to-end conditional-dispersion FE: bound the column set by top raw-MI
    (cardinality bound on the O(p^2) pair pool), materialise the dispersion-
    anomaly columns, MI-gate against the raw-baseline floor (Layer 91), keep top
    ``top_k``.

    The MI gate is the SELF-LIMITER: on homoscedastic data ``|z|`` is a scaled
    ``|residual|`` and clears no uplift over the raw / Family-B sibling, so it is
    dropped; on pure noise nothing clears the floor; only a genuinely
    heteroscedastic pair (the conditional spread varies across ``x_j`` bins)
    survives. Returns ``(X_aug, appended, recipes, scores)``. ``y`` is consumed
    only by the column-ranking + MI gate; recipes carry no ``y`` -> leak-safe.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"hybrid_conditional_dispersion_fe: X must be a pandas DataFrame; "
            f"got {type(X).__name__}"
        )
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [
            c for c in num_cols
            if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
        ]
    if len(num_cols) < 2:
        return X.copy(), [], [], pd.DataFrame()

    if y is not None:
        from ._extra_fe_families import _top_mi_num_cols  # lazy: break parent<->sibling cycle
        num_cols = _top_mi_num_cols(X, num_cols, y, max_pair_cols)
    else:
        num_cols = list(num_cols)[: int(max_pair_cols)]
    if len(num_cols) < 2:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_conditional_dispersion_features(
        X, num_cols, n_bins=n_bins, kinds=kinds,
    )
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate
        winners = local_mi_gate(
            enc_df, y, raw_X=X,
            top_k=int(mi_gate_top_k) if mi_gate_top_k else int(top_k),
        )
        # DUAL-UPLIFT GATE (backlog #12): a dispersion column earns its keep ONLY
        # if it carries MI BEYOND both (a) its raw source x_i and (b) its
        # Family-B mean-residual sibling x_i - E[x_i|bin(x_j)]. On homoscedastic
        # data |z| is a scaled |residual| -> no uplift over the sibling -> dropped
        # (self-limiting); a genuine heteroscedastic pair clears both. This is the
        # discriminating gate the MI floor alone cannot enforce (the floor only
        # tests "above noise", not "above the cheaper location feature").
        winners = _dual_uplift_filter(
            X, y, enc_df, winners, raw_recipes, n_bins=n_bins,
        )
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_conditional_dispersion_recipe(
            name=name,
            x_i=raw_recipes[name]["x_i"],
            x_j=raw_recipes[name]["x_j"],
            edges=raw_recipes[name]["edges"],
            bin_mean=raw_recipes[name]["bin_mean"],
            bin_std=raw_recipes[name]["bin_std"],
            global_mean=raw_recipes[name]["global_mean"],
            global_std=raw_recipes[name]["global_std"],
            kind=raw_recipes[name]["kind"],
        )
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df
