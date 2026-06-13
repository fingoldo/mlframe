"""Usability-aware feature selection -- a LINEAR-downstream selection list alongside
MRMR's model-agnostic MI selection (see tests/feature_selection/MRMR_USABILITY_AWARE_SELECTION_DESIGN.md).

MRMR's Fleuret objective ranks features by MI, which is rank-based and BLIND to linear
usability: on a magnitude-carrying target it picks a high-MI monotone warp a linear model
cannot use over a lower-MI form that is the linearly-aligned interaction. This module runs a
SEPARATE greedy whose relevance blends MI with the HELD-OUT |partial correlation of the
candidate's CONTINUOUS values with the RESIDUAL after the already-selected features|. The
residual is the key: on a heavy-tailed target the dominant term (e.g. ``a**2/b``) swamps a
raw-y correlation / R^2, but once it is selected and removed the residual is bounded and the
weak interaction's linear correlation is visible (measured: F2 linear MAE 0.092 -> 0.052).

The candidate pool is generated here (not reused from the main FE) so it is RICH ENOUGH to
contain the linearly-usable interaction forms the main FE's admission/one-best-per-pair prune
out -- the residual-partial-corr admission keeps a genuine ``mul(log(c),sin(d))`` and rejects an
additive cross-mix ``(a,c)``. Each selected feature is a replayable ``EngineeredRecipe`` (or a
raw column), so ``transform()`` can reproduce the linear feature space on test data.

Self-contained: it does NOT modify ``screen_predictors`` / the MI greedy; the caller runs it as
a second pass and exposes its selection (``support_linear_``) for the suite to route to linear
models.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


def _scrub(v: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    return np.nan_to_num(np.asarray(v, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)


def _f64(v: np.ndarray) -> np.ndarray:
    """Upcast a stored (possibly float32) candidate column to float64 for MI / correlation /
    recipe-edge computation where the heavy-tail precision matters (transient; not stored)."""
    return np.asarray(v, dtype=np.float64)


def _abscorr(u: np.ndarray, v: np.ndarray) -> float:
    u = _f64(u); v = _f64(v)  # precision for the heavy-tail correlation
    if u.size == 0 or float(np.std(u)) < 1e-12 or float(np.std(v)) < 1e-12:
        return 0.0
    r = np.corrcoef(u, v)[0, 1]
    return abs(float(r)) if np.isfinite(r) else 0.0


@dataclass
class UsableCandidate:
    name: str
    values: np.ndarray          # continuous, full-n, scrubbed
    mi: float                   # binned MI with y
    recipe: Any = None          # EngineeredRecipe for a pair form; None for a raw column
    src: tuple = ()             # (op_a, op_b) raw names, for diagnostics
    ops: tuple = ()             # (unary_a, unary_b, binary) names, for the recipe builder


def _binned_mi(x: np.ndarray, y_codes: np.ndarray, nbins: int) -> float:
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
    return float(_cmi_from_binned(_quantile_bin(_f64(x), nbins), y_codes, None))


def build_usability_candidate_pool(
    X_df: "Any",
    y_cont: np.ndarray,
    base_names: Sequence[str],
    *,
    unary_preset: str = "medium",
    binary_preset: str = "minimal",
    quantization_nbins: int = 10,
    quantization_method: str = "quantile",
    quantization_dtype: Any = np.int32,
    feature_dtype: Any = np.float32,
    mi_floor: float = 0.02,
    max_per_pair: int = 12,
    diversity_corr: float = 0.97,
    max_pairs: int = 60,
) -> list[UsableCandidate]:
    """Enumerate raw + unary/binary-product candidates (continuous, replayable) for pairs among
    ``base_names``. Per pair, keep up to ``max_per_pair`` DISTINCT forms clearing ``mi_floor``
    (greedy by MI, dropping any |corr|>``diversity_corr`` near-duplicate). For wide p, restrict to
    the ``max_pairs`` highest-marginal-MI pairs. Each pair form is a replayable
    ``EngineeredRecipe`` (so ``transform`` can reproduce it)."""
    import pandas as pd
    from .feature_engineering import create_unary_transformations, create_binary_transformations
    from .engineered_recipes import build_unary_binary_recipe, apply_recipe
    from ._mi_greedy_cmi_fe import _quantile_bin

    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(np.asarray(X_df))
    y_cont = _scrub(y_cont)
    y_codes = _quantile_bin(y_cont, quantization_nbins)

    unary = create_unary_transformations(preset=unary_preset)
    binary = create_binary_transformations(preset=binary_preset)
    base_names = [b for b in base_names if b in X_df.columns]

    pool: list[UsableCandidate] = []
    # raw columns are always candidates (a linear model often wants a raw operand too).
    for nm in base_names:
        col = _scrub(X_df[nm].to_numpy(), feature_dtype)
        if float(np.std(col)) <= 1e-9:
            continue
        pool.append(UsableCandidate(nm, col, _binned_mi(col, y_codes, quantization_nbins), None, (nm,), ()))

    # rank pairs by marginal-MI sum so a wide-p sweep keeps the most promising first.
    marg = {nm: _binned_mi(_scrub(X_df[nm].to_numpy()), y_codes, quantization_nbins) for nm in base_names}
    pairs = list(itertools.combinations(base_names, 2))
    pairs.sort(key=lambda p: marg[p[0]] + marg[p[1]], reverse=True)
    pairs = pairs[:max_pairs]

    for n1, n2 in pairs:
        x1 = _f64(_scrub(X_df[n1].to_numpy()))
        x2 = _f64(_scrub(X_df[n2].to_numpy()))
        cand_here: list[UsableCandidate] = []
        for ua, ub in itertools.product(unary, unary):
            try:
                ta, tb = unary[ua](x1), unary[ub](x2)
            except Exception:
                continue
            for bn, bf in binary.items():
                try:
                    val = _scrub(bf(ta, tb), feature_dtype)
                except Exception:
                    continue
                if float(np.std(val)) <= 1e-9:
                    continue
                m = _binned_mi(val, y_codes, quantization_nbins)
                if m < mi_floor:
                    continue
                name = f"{bn}({ua}({n1}),{ub}({n2}))"
                cand_here.append(UsableCandidate(name, val, m, None, (n1, n2), (ua, ub, bn)))
        # keep diverse top-MI forms for this pair.
        cand_here.sort(key=lambda c: c.mi, reverse=True)
        kept: list[UsableCandidate] = []
        for c in cand_here:
            if len(kept) >= max_per_pair:
                break
            if any(_abscorr(c.values, k.values) > diversity_corr for k in kept):
                continue
            kept.append(c)
        # build replayable recipes only for the kept forms (cheap: bounded count). Use the stored
        # (ua, ub, bn) ops directly -- never re-parse the display name.
        for c in kept:
            ua, ub, bn = c.ops
            try:
                recipe = build_unary_binary_recipe(
                    name=c.name, src_a_name=c.src[0], src_b_name=c.src[1],
                    unary_a_name=ua, unary_b_name=ub,
                    binary_name=bn, unary_preset=unary_preset, binary_preset=binary_preset,
                    quantization_nbins=quantization_nbins, quantization_method=quantization_method,
                    quantization_dtype=quantization_dtype,
                    fit_values_for_edges=_f64(c.values),  # edges need float64 precision
                )
                # verify the recipe replays to the same continuous values (else drop -- not replayable).
                replay = _scrub(apply_recipe(recipe, X_df), feature_dtype)
                if replay.shape == c.values.shape and np.allclose(_f64(replay), _f64(c.values), atol=1e-4, equal_nan=True):
                    c.recipe = recipe
                    pool.append(c)
            except Exception:
                continue
    return pool


def usability_greedy(
    pool: list[UsableCandidate],
    y_cont: np.ndarray,
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    n_folds: int = 4,
    mae_improve_rel: float = 0.01,
    shortlist: int = 40,
) -> list[UsableCandidate]:
    """CROSS-VALIDATED forward selection for the LINEAR downstream: greedily add the candidate that
    most reduces the K-fold CV mean-absolute-error of a linear model on the selected set, stopping
    when no candidate improves it by ``mae_improve_rel`` (relative). This is the gold-standard
    linear wrapper -- it directly optimises the deployed objective, so it inherently (a) prefers the
    LINEARLY-usable form over a high-MI monotone warp a linear model cannot use, (b) drops redundant
    forms (no CV gain), and (c) is robust to OVERFITTING a single held-out slice: a feature that
    helps one fold by chance but drags in a noise operand (e.g. ``min(log(d),e)``) does NOT lower
    the AVERAGE CV MAE, so it is rejected (a single-split gate let it through and regressed F2 n=80k
    MAE 0.054 -> 0.102; CV keeps it ~0.055). The stop replaces a hardcoded feature count.

    A cheap usability pre-rank (``MI + |corr with the post-dominant residual|``) shortlists the pool
    to ``shortlist`` candidates so the per-step CV cost is bounded; ``w`` weights the MI vs the
    residual-corr in that pre-rank only (the COMMIT decision is always the CV-MAE improvement)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # bench-attempt-rejected (2026-06-13): an audit flagged OLS min-norm as fragile on a singular/wide
    # selected set and suggested a small Ridge in the CV. Benched OLS vs Ridge(alpha=1) on F2: n=8000
    # IDENTICAL (0.0554 per-seed -- the design is never singular here, K<=8 features vs folds of 1000s
    # of rows), n=500 marginally WORSE (0.0670 -> 0.0672). The singular regime needs per-fold rows <
    # selected-feature count, which cannot happen given the shortlist + the small K, so Ridge buys
    # nothing. Kept OLS (the gold-standard wrapper that matches the deployed objective). Do not re-add.
    def _mk():
        return make_pipeline(StandardScaler(), LinearRegression())

    if not pool:
        return []
    y_cont = _scrub(y_cont)
    n = y_cont.shape[0]
    if n < 2:
        return []  # cannot cross-validate a usability greedy on < 2 rows
    rng = np.random.default_rng(int(seed))
    # BALANCED PARTITION (audit fix, 2026-06-13): a random ``rng.integers(0, n_folds)`` multinomial
    # assignment can leave a fold EMPTY at small n / large n_folds -> an empty TRAIN fold crashes
    # ``fit`` and an empty TEST fold yields a NaN MAE that poisons the per-fold consistency gate. A
    # shuffled ``arange(n) % k`` partition guarantees every fold has floor/ceil(n/k) >= 1 rows.
    n_folds = max(2, min(int(n_folds), n))
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    mi_max = max((c.mi for c in pool), default=1.0) or 1.0

    # PERF TODO (2026-06-13): this refits a full StandardScaler+LinearRegression for EVERY
    # (candidate, fold) -- O(shortlist * n_folds * K) least-squares solves. A custom incremental
    # solver would reuse almost all of it: the selected set's Gram matrix ``G = Xs.T @ Xs`` and
    # ``Xs.T @ y`` are FIXED across candidates within a step, so adding one candidate column is a
    # rank-1 border (one extra row/col of G + one dot with y); a Cholesky/QR downdate per fold then
    # solves in O(k^2) instead of refitting O(n k^2). Standardisation stats are also reusable
    # (column means/stds computed once per fold). Implement once the approach is proven to help.
    def _cv_per_fold(sel_idx) -> np.ndarray:
        if not sel_idx:
            return np.array([
                float(np.mean(np.abs(y_cont[folds == fo] - float(np.mean(y_cont[folds != fo])))))
                for fo in range(n_folds)
            ])
        Xs = np.column_stack([_f64(pool[i].values) for i in sel_idx])
        errs = []
        for fo in range(n_folds):
            trm, vam = folds != fo, folds == fo
            m = _mk().fit(Xs[trm], y_cont[trm])
            errs.append(float(np.mean(np.abs(y_cont[vam] - m.predict(Xs[vam])))))
        return np.asarray(errs, dtype=np.float64)

    # cheap residual-aware pre-rank to a bounded shortlist (so per-step CV stays cheap).
    def _shortlist(sel_idx) -> list[int]:
        # HELD-OUT residual (audit fix, 2026-06-13): fit on the fold-0-out train rows but score the
        # candidate correlation on the HELD-OUT fold-0 residual only -- the prior code predicted over
        # ALL rows (in-sample for the ~(k-1)/k training rows), which is the leakage the module's
        # "held-out residual" design explicitly avoids. The no-selection case uses the mean residual
        # over all rows (no model fit -> no leakage).
        if sel_idx:
            Xs = np.column_stack([_f64(pool[i].values) for i in sel_idx])
            ho = folds == 0
            tr = ~ho
            m = _mk().fit(Xs[tr], y_cont[tr])
            resid = y_cont[ho] - m.predict(Xs[ho])
            rows = ho
        else:
            resid = y_cont - float(np.mean(y_cont))
            rows = slice(None)
        scored = []
        for i in range(len(pool)):
            if i in sel_idx:
                continue
            use = _abscorr(pool[i].values[rows], resid)
            scored.append((i, (1.0 - w) * (pool[i].mi / mi_max) + w * use))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [i for i, _ in scored[: max(1, shortlist)]]

    import math
    # a committed feature must improve a MAJORITY of folds (>=75%), not just the mean -- a noise-
    # contaminated feature lowers some folds by chance and raises others (net ~0); requiring
    # consistency across folds rejects it and stops the greedy at the genuinely useful set.
    min_improving_folds = max(1, int(math.ceil(0.75 * n_folds)))
    selected: list[int] = []
    folds_cur = _cv_per_fold(selected)
    mae_cur = float(folds_cur.mean())
    for _ in range(min(K, len(pool))):
        cand_idx = _shortlist(selected)
        best_i, best_mean, best_folds = -1, mae_cur, folds_cur
        for i in cand_idx:
            mf = _cv_per_fold(selected + [i])
            if int(np.sum(mf < folds_cur)) < min_improving_folds:
                continue  # not a consistent improvement across folds
            if float(mf.mean()) < best_mean:
                best_mean, best_i, best_folds = float(mf.mean()), i, mf
        if best_i < 0 or best_mean >= mae_cur * (1.0 - mae_improve_rel):
            break
        selected.append(best_i)
        folds_cur, mae_cur = best_folds, best_mean
    return [pool[i] for i in selected]


def select_usability_aware_features(
    X_df: "Any",
    y_cont: np.ndarray,
    base_names: Sequence[str],
    *,
    w: float = 0.7,
    K: int = 8,
    seed: int = 0,
    pool_kwargs: Optional[dict] = None,
    greedy_kwargs: Optional[dict] = None,
) -> list[UsableCandidate]:
    """End-to-end: build the replayable candidate pool, then run the usability greedy. Returns the
    selected ``UsableCandidate`` list (each with a replayable ``recipe`` for pair forms, ``None``
    for raw columns), in selection order."""
    pool = build_usability_candidate_pool(X_df, y_cont, base_names, **(pool_kwargs or {}))
    return usability_greedy(pool, y_cont, w=w, K=K, seed=seed, **(greedy_kwargs or {}))
