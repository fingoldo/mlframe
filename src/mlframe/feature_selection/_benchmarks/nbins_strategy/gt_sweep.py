"""Ground-truth MRMR harness of the ``nbins_strategy`` comparison: which columns MRMR SELECTS (recall / precision / false-positive rate / F1 against known relevant, redundant and irrelevant roles) and how a
linear model and CatBoost do on them, per binning method, on a synthetic multi-column problem and a real wellbore frame.

Moved here from ``filters/_benchmarks/bench_mdlp_validated_split_suite.py`` (which re-exports every name, so its callers and tests are unchanged) to sit next to ``linear_oof.py``, the out-of-fold comparison
of the same strategies; ``README.md`` of this directory explains how the two fit together. The dated result files of the sweep are in ``results/``.
"""

from __future__ import annotations

import json  # noqa: F401
import logging
import os  # noqa: F401
import sys  # noqa: F401
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlframe.feature_engineering._safe_ratio import safe_div

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ground-truth MRMR recall/precision/F1 harness.
#
# The RMSE bench above measures binning quality in isolation (one column, one
# OOS-MSE fit). It does NOT tell us whether a binning method's per-column edge
# quality actually changes what MRMR.fit() SELECTS on a multi-column problem
# with known relevant / redundant / irrelevant roles -- MRMR's own MI-based
# relevance gate and Fleuret-style conditional-MI redundancy check could easily
# absorb or amplify a binning difference. This section runs the real MRMR
# estimator (``mrmr._mrmr_class.MRMR.fit``), not an isolated column fit.
# -----------------------------------------------------------------------------


@dataclass
class MulticolumnGT:
    """Ground-truth column roles for ``scen_multicolumn``."""

    relevant: list = field(default_factory=list)
    irrelevant: list = field(default_factory=list)
    redundant: list = field(default_factory=list)
    redundant_source: dict = field(default_factory=dict)  # redundant col name -> source relevant col name
    categorical_cols: list = field(default_factory=list)  # integer-coded categorical columns, for CatBoost cat_features


def scen_multicolumn(n: int, n_relevant: int, n_irrelevant: int, n_redundant: int, seed: int = 0):
    """Multi-column synthetic with KNOWN ground truth, deliberately non-uniform: relevant columns
    alternate continuous-at-varying-scale / uniform / low-card categorical-coded / high-card
    categorical-coded; irrelevant columns are pure noise in the same style mix; redundant columns
    are derived from a relevant column via one of 3 transforms (additive-noise, affine, signed-log)
    at a randomized correlation strength so the redundant SET has varied correlation structure, not
    one fixed rho. ~30% of columns get NaNs injected (1-15% of rows) to stress the real NaN contract.
    Returns (X: pd.DataFrame, y: np.ndarray, gt: MulticolumnGT)."""
    rng = np.random.default_rng(seed)
    relevant_names = [f"rel_{i}" for i in range(n_relevant)]
    relevant_data: dict = {}
    categorical_cols: list = []
    weights = rng.uniform(0.5, 2.0, n_relevant) * rng.choice([-1.0, 1.0], n_relevant)
    y = np.zeros(n)
    for i, name in enumerate(relevant_names):
        kind = i % 4
        if kind == 0:
            x = rng.standard_normal(n) * (10.0 ** rng.uniform(-1, 3))
        elif kind == 1:
            x = rng.uniform(0.0, 1.0, n) * 100.0
        elif kind == 2:
            x = rng.integers(0, 5, n).astype(np.float64)  # low-cardinality categorical
            categorical_cols.append(name)
        else:
            x = rng.integers(0, 500, n).astype(np.float64)  # high-cardinality categorical
            categorical_cols.append(name)
        relevant_data[name] = x
        xz = safe_div(x - x.mean(), x.std())
        y += weights[i] * xz
    y += rng.standard_normal(n) * 0.5

    cols: dict = dict(relevant_data)

    irrelevant_names = [f"noise_{i}" for i in range(n_irrelevant)]
    for i, name in enumerate(irrelevant_names):
        kind = i % 3
        if kind == 0:
            cols[name] = rng.standard_normal(n) * (10.0 ** rng.uniform(-2, 4))
        elif kind == 1:
            cols[name] = rng.integers(0, 8, n).astype(np.float64)
            categorical_cols.append(name)
        else:
            cols[name] = rng.integers(0, 1000, n).astype(np.float64)
            categorical_cols.append(name)

    redundant_names = []
    redundant_source: dict = {}
    for i in range(n_redundant):
        src = relevant_names[i % n_relevant]
        base = relevant_data[src]
        base_std = float(base.std()) + 1e-9
        name = f"redund_{i}_of_{src}"
        corr_strength = rng.uniform(0.5, 0.98)  # varied correlation structure across the redundant set
        noise_scale = base_std * (1.0 - corr_strength) * 3.0
        transform = i % 3
        if transform == 0:
            derived = base + rng.standard_normal(n) * noise_scale
        elif transform == 1:
            derived = base * 2.5 + 7.0 + rng.standard_normal(n) * noise_scale
        else:
            derived = np.sign(base) * np.log1p(np.abs(base)) + rng.standard_normal(n) * noise_scale
        cols[name] = derived
        redundant_names.append(name)
        redundant_source[name] = src

    for name in list(cols.keys()):
        if rng.random() < 0.3:
            frac = rng.uniform(0.01, 0.15)
            mask = rng.random(n) < frac
            arr = cols[name].astype(np.float64).copy()
            arr[mask] = np.nan
            cols[name] = arr

    X = pd.DataFrame(cols, copy=False)
    gt = MulticolumnGT(
        relevant=relevant_names, irrelevant=irrelevant_names, redundant=redundant_names,
        redundant_source=redundant_source, categorical_cols=categorical_cols,
    )
    return X, y, gt


_WELLBORE_DATA_PATH = r"C:\Users\Admin\Machine learning\data\Competitions\ROGII - Wellbore Geology Prediction\train_df.parquet"


def scen_wellbore100k(n_relevant: int, n_redundant: int, seed: int = 0, n: int = 100_000):
    """Real-data ground-truth scenario: the IRRELEVANT/noise pool is REAL wellbore log columns
    (real distributions, scales, cross-correlations, missingness patterns -- not synthesized), while
    the RELEVANT signal is purely synthetic columns injected on top with a KNOWN effect on ``y``.
    Ground truth stays exact (the real columns carry no synthetic signal by construction) while the
    noise/redundancy structure is realistic rather than idealized i.i.d. synthetic noise. Falls back
    to a warning + smaller synthetic-noise-only frame if the wellbore parquet isn't available on
    this machine (keeps the harness runnable off this specific dataset, at reduced realism).

    Returns ``(X, y, gt)`` in the same ``MulticolumnGT`` contract as ``scen_multicolumn``.
    """
    rng = np.random.default_rng(seed)
    try:
        real = pd.read_parquet(_WELLBORE_DATA_PATH)
        drop_cols = [c for c in ("TVT_input", "TVT", "well_id") if c in real.columns]
        real = real.drop(columns=drop_cols)
        if len(real) > n:
            real = real.iloc[rng.choice(len(real), size=n, replace=False)].reset_index(drop=True)
        n_rows = len(real)
        irrelevant_names = list(real.columns)
        categorical_cols = [c for c in irrelevant_names if real[c].dtype.kind in ("i", "u") or str(real[c].dtype) in ("category", "object")]
        cols = {}
        for c in irrelevant_names:
            _v = real[c].to_numpy()
            # Real logs carry a few +/-inf from upstream FE; convert to NaN so the imputer/nbins path
            # treats them as ordinary missing values instead of MRMR.fit rejecting the whole frame.
            if _v.dtype.kind == "f":
                _bad = np.isinf(_v)
                if _bad.any():
                    _v = _v.copy()
                    _v[_bad] = np.nan
            cols[c] = _v
    except Exception:
        logger.warning("scen_wellbore100k: wellbore parquet unavailable at %s -- falling back to a smaller synthetic-noise-only frame", _WELLBORE_DATA_PATH, exc_info=True)
        n_rows = min(n, 20_000)
        irrelevant_names = [f"noise_{i}" for i in range(20)]
        categorical_cols = []
        cols = {name: rng.standard_normal(n_rows) * (10.0 ** rng.uniform(-2, 4)) for name in irrelevant_names}

    relevant_names = [f"rel_{i}" for i in range(n_relevant)]
    relevant_data: dict = {}
    weights = rng.uniform(0.5, 2.0, n_relevant) * rng.choice([-1.0, 1.0], n_relevant)
    y = np.zeros(n_rows)
    for i, name in enumerate(relevant_names):
        x = rng.standard_normal(n_rows) * (10.0 ** rng.uniform(-1, 2))
        relevant_data[name] = x
        xz = safe_div(x - x.mean(), x.std())
        y += weights[i] * xz
    y += rng.standard_normal(n_rows) * 0.5
    cols.update(relevant_data)

    redundant_names = []
    redundant_source: dict = {}
    for i in range(n_redundant):
        src = relevant_names[i % n_relevant] if n_relevant else None
        if src is None:
            break
        base = relevant_data[src]
        base_std = float(base.std()) + 1e-9
        name = f"redund_{i}_of_{src}"
        corr_strength = rng.uniform(0.5, 0.98)
        noise_scale = base_std * (1.0 - corr_strength) * 3.0
        cols[name] = base + rng.standard_normal(n_rows) * noise_scale
        redundant_names.append(name)
        redundant_source[name] = src

    X = pd.DataFrame(cols, copy=False)
    gt = MulticolumnGT(
        relevant=relevant_names, irrelevant=irrelevant_names, redundant=redundant_names,
        redundant_source=redundant_source, categorical_cols=categorical_cols,
    )
    return X, y, gt


# nbins_strategy / nbins_strategy_kwargs to feed MRMR() per binning method under test (mirrors the
# ``method`` values used by ``run_one`` above, minus ``oos_validated`` -- the OOS-validated variant
# is not wired into ``supervised_binning.mdlp_bin_edges`` / MRMR's ``nbins_strategy`` dispatch, it is
# a standalone alternative function, so it has no MRMR-selectable analogue here).
# NO ALIASES: every key here is the EXACT, resolved (alias-free) nbins_strategy name (verified
# against _adaptive_nbins.py's _METHOD_ALIASES resolution table -- e.g. "quantile" is NOT used here
# because it is itself an alias that resolves to "qs", not a distinct method). For every strategy
# that has an independent count-FORMULA (sturges/freedman_diaconis/knuth/optimal_joint) crossed with
# an independent edge-PLACEMENT choice (quantile = equi-frequency, uniform = equi-width, via the
# `base` kwarg / `knuth_edge_type` for knuth), both combinations are listed as SEPARATE, explicitly-
# named entries -- "formula_placement" -- rather than picking one placement silently. Strategies that
# determine edges DIRECTLY (bayesian_blocks, fayyad_irani*, mah, qs) have no placement choice at all
# and are listed once under their bare resolved name.
MRMR_BINNING_METHODS: "dict[str, dict]" = {
    # Legacy non-adaptive path: MRMR(nbins_strategy=None, quantization_nbins=N) bypasses
    # per_feature_edges entirely (fixed-N equi-frequency binning, the pre-adaptive-binning default).
    "legacy_fixed_quantile10": {"nbins_strategy": None, "quantization_nbins": 10},
    # count-formula x placement, both combinations
    "sturges_quantile": {"nbins_strategy": "sturges", "nbins_strategy_kwargs": {"base": "quantile"}},
    "sturges_uniform": {"nbins_strategy": "sturges", "nbins_strategy_kwargs": {"base": "uniform"}},
    "freedman_diaconis_quantile": {"nbins_strategy": "freedman_diaconis", "nbins_strategy_kwargs": {"base": "quantile"}},
    "freedman_diaconis_uniform": {"nbins_strategy": "freedman_diaconis", "nbins_strategy_kwargs": {"base": "uniform"}},
    "optimal_joint_quantile": {"nbins_strategy": "optimal_joint", "nbins_strategy_kwargs": {"base": "quantile"}},
    "optimal_joint_uniform": {"nbins_strategy": "optimal_joint", "nbins_strategy_kwargs": {"base": "uniform"}},
    # Demoted (AccuracyWarning) formula -- included (both placements) specifically because the
    # mega-bench v3 numbers cited in its AccuracyWarning were measured in isolation, not through
    # this ground-truth MRMR-selection harness; kept in to verify (not assume) it stays weak once
    # real redundancy screening / MI relevance gating is in the loop too.
    "knuth_quantile": {"nbins_strategy": "knuth", "nbins_strategy_kwargs": {"knuth_edge_type": "quantile"}},
    "knuth_uniform": {"nbins_strategy": "knuth", "nbins_strategy_kwargs": {"knuth_edge_type": "uniform"}},
    # direct edge-determination methods (no separate count-formula / placement split)
    "qs_gupta": {"nbins_strategy": "quantile_spacing"},
    # one name since 2026-11-22: "bayesian_blocks" is accepted by MRMR and by the dispatcher alike.
    "bayesian_blocks": {"nbins_strategy": "bayesian_blocks"},
    # row labels are the historical ones (the 2026-07-21 results use them): fayyad_irani_* is ``mdlp``; "insample_validated" is the DEFAULT ``mdlp`` (mdlp_fast_mode False), "oos_validated" is ``mdlp_validated``.
    "fayyad_irani_fast": {"nbins_strategy": "mdlp", "nbins_strategy_kwargs": {"mdlp_fast_mode": True}},
    "fayyad_irani_insample_validated": {"nbins_strategy": "mdlp", "nbins_strategy_kwargs": {"mdlp_fast_mode": False}},
    "fayyad_irani_oos_validated": {"nbins_strategy": "mdlp_validated"},
    "mah_sci": {"nbins_strategy": "mah"},  # demoted (AccuracyWarning), kept in for the same reason as knuth above
}

# Kept off/minimal so the harness measures the BINNING method's effect on selection, not FE/cluster-
# aggregate machinery layered on top; ``use_simple_mode=False`` (the MRMR default) is kept ON because
# disabling it skips the conditional-MI redundancy check entirely, which would make the redundant-
# column false-positive-rate metric meaningless (everything redundant would pass through unchecked).
_MRMR_HARNESS_FIXED_KWARGS = dict(
    cluster_aggregate_enable=False,
    dcd_enable=False,
    fe_unary_preset="minimal",
    fe_binary_preset="minimal",
    fe_max_steps=1,
    verbose=0,
    n_jobs=1,
)


def run_mrmr_selection(X: pd.DataFrame, y: np.ndarray, method: str, random_seed: int = 0):
    """Fit the real ``mrmr._mrmr_class.MRMR`` (not an isolated column-level binning call) with the
    given binning method wired in, and return the set of ORIGINAL (non-engineered) column names it
    selected. Engineered/composite feature names from FE are excluded from the returned set (FE is
    held at 'minimal' + 1 step precisely so it stays a minor factor, not the object under test) --
    they are neither counted as a ground-truth hit nor a false positive."""
    from typing import Any

    from mlframe.feature_selection.filters.mrmr import MRMR

    if method not in MRMR_BINNING_METHODS:
        raise ValueError(method)
    kwargs: "dict[str, Any]" = dict(_MRMR_HARNESS_FIXED_KWARGS)
    kwargs.update(MRMR_BINNING_METHODS[method])
    kwargs["random_seed"] = random_seed
    model = MRMR(**kwargs)
    model.fit(X, y)
    selected = set(model.get_feature_names_out())
    return selected & set(X.columns)


@dataclass
class MrmrGTResult:
    config: str
    method: str
    seed: int
    recall: float
    precision: float
    fpr_noise_redundant: float
    f1: float
    n_selected: int
    n_relevant: int = 0  # denominator for recall, as an absolute count (report as "n_relevant_hit/n_relevant")
    n_relevant_hit: int = 0  # numerator for recall
    n_total_cols: int = 0  # denominator for "how many of ALL columns got selected" (report as "n_selected/n_total_cols")
    fit_time_s: float = float("nan")  # MRMR.fit wall-time for this (config, method, seed)
    downstream: "DownstreamQuality" = field(default_factory=lambda: DownstreamQuality())


def _prf(selected: set, gt: MulticolumnGT) -> tuple:
    relevant = set(gt.relevant)
    noise_and_redundant = set(gt.irrelevant) | set(gt.redundant)
    hits = selected & relevant
    recall = len(hits) / len(relevant) if relevant else float("nan")
    precision = len(hits) / len(selected) if selected else 0.0
    fpr = len(selected & noise_and_redundant) / len(noise_and_redundant) if noise_and_redundant else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, fpr, f1


@dataclass
class DownstreamQuality:
    """Downstream model-quality metrics computed on the SELECTED columns only -- answers "how well
    does a simple linear model do vs a powerful nonlinear (CatBoost) model on what got selected",
    a real accuracy signal beyond pure column-identity recall/precision. RMSE (not R2 -- R2 is a
    normalized, harder-to-compare-across-scenarios restatement of the same MSE) for the regression
    target; logloss (not AUC -- AUC ignores calibration and is threshold-free in a way that hides
    a model being confidently wrong) for a binarized (>median) version of the same target."""

    linear_rmse: float = float("nan")
    catboost_rmse: float = float("nan")
    linear_logloss: float = float("nan")
    catboost_logloss: float = float("nan")
    rmse_gap: float = float("nan")  # catboost_rmse - linear_rmse; negative = catboost wins
    logloss_gap: float = float("nan")  # catboost_logloss - linear_logloss; negative = catboost wins
    downstream_time_s: float = float("nan")


def _downstream_quality(X: pd.DataFrame, y: np.ndarray, selected: set, gt: MulticolumnGT, seed: int) -> DownstreamQuality:
    """Fit a linear model and CatBoost on the SELECTED columns (median-imputed + standardized for
    the linear model via an explicit Pipeline; raw + native NaN/categorical handling for CatBoost),
    score both regression (RMSE) and classification (logloss on a >median binarization of y) on a
    held-out 25% split. Returns all-NaN metrics (not a raise) when nothing was selected."""
    t0 = time.perf_counter()
    result = DownstreamQuality()
    cols = [c for c in X.columns if c in selected]
    if not cols:
        return result
    Xs = X[cols]
    cat_cols = [c for c in cols if c in set(gt.categorical_cols)]

    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_test = max(1, round(0.25 * n))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, X_test = Xs.iloc[train_idx], Xs.iloc[test_idx]
    # CatBoost requires cat_features columns to be int/string dtype (real-number float64 raises
    # CatBoostError: "cat_features must be integer or string ... NaN values should be converted to
    # string") -- our synthetic categorical columns are stored as float64 codes like every other
    # numeric column, so build a CatBoost-only view with those columns stringified. A LITERAL
    # np.nan still raises the SAME error even in an object-dtype column (verified: "bad object for
    # id: nan" / "NaN values should be converted to string") -- CatBoost wants missingness in a
    # categorical column represented as its OWN string sentinel, not a float NaN passthrough.
    X_train_cb, X_test_cb = X_train.copy(), X_test.copy()
    for c in cat_cols:
        X_train_cb[c] = X_train_cb[c].map(lambda v: str(int(v)) if pd.notna(v) else "__missing__")
        X_test_cb[c] = X_test_cb[c].map(lambda v: str(int(v)) if pd.notna(v) else "__missing__")
    y_train, y_test = y[train_idx], y[test_idx]
    y_thresh = float(np.median(y_train))
    yb_train_arr = (y_train > y_thresh).astype(np.int64)
    yb_test_arr = (y_test > y_thresh).astype(np.int64)
    degenerate = len(np.unique(yb_train_arr)) < 2 or len(np.unique(yb_test_arr)) < 2
    # degenerate split (e.g. all-identical y) -> skip classification metrics
    yb_train: "np.ndarray | None" = None if degenerate else yb_train_arr
    yb_test: "np.ndarray | None" = None if degenerate else yb_test_arr

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import log_loss, mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        lin_reg = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        lin_reg.fit(X_train, y_train)
        result.linear_rmse = float(np.sqrt(mean_squared_error(y_test, lin_reg.predict(X_test))))
    except Exception:
        logger.debug("linear regression downstream fit failed", exc_info=True)
    if yb_train is not None:
        try:
            lin_clf = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=200))])
            lin_clf.fit(X_train, yb_train)
            result.linear_logloss = float(log_loss(yb_test, lin_clf.predict_proba(X_test)[:, 1], labels=[0, 1]))
        except Exception:
            logger.debug("linear classification downstream fit failed", exc_info=True)

    try:
        from catboost import CatBoostRegressor
        cb_reg = CatBoostRegressor(iterations=150, depth=4, learning_rate=0.1, allow_writing_files=False, verbose=False, thread_count=1, cat_features=cat_cols or None)
        cb_reg.fit(X_train_cb, y_train)
        result.catboost_rmse = float(np.sqrt(mean_squared_error(y_test, cb_reg.predict(X_test_cb))))
    except Exception:
        logger.debug("catboost regression downstream fit failed", exc_info=True)
    if yb_train is not None:
        try:
            from catboost import CatBoostClassifier
            cb_clf = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.1, allow_writing_files=False, verbose=False, thread_count=1, cat_features=cat_cols or None)
            cb_clf.fit(X_train_cb, yb_train)
            result.catboost_logloss = float(log_loss(yb_test, cb_clf.predict_proba(X_test_cb)[:, 1], labels=[0, 1]))
        except Exception:
            logger.debug("catboost classification downstream fit failed", exc_info=True)

    if np.isfinite(result.catboost_rmse) and np.isfinite(result.linear_rmse):
        result.rmse_gap = result.catboost_rmse - result.linear_rmse
    if np.isfinite(result.catboost_logloss) and np.isfinite(result.linear_logloss):
        result.logloss_gap = result.catboost_logloss - result.linear_logloss
    result.downstream_time_s = time.perf_counter() - t0
    return result


def _append_jsonl_result(path: str, result) -> None:
    """Append one sweep result to ``path`` as a single JSON line, flushed to disk immediately.

    One line per completed step (not one rewritten document) so a sweep killed mid-run leaves every
    finished step readable, and a partially-written final line never corrupts the earlier ones.
    """
    import dataclasses
    import orjson

    with open(path, "ab") as f:
        f.write(orjson.dumps(dataclasses.asdict(result), option=orjson.OPT_SERIALIZE_NUMPY))
        f.write(b"\n")
        f.flush()


def load_jsonl_results(path: str) -> list:
    """Read back a checkpoint written by ``_append_jsonl_result`` as ``MrmrGTResult`` objects.

    A trailing partial line (the sweep was killed mid-write) is skipped rather than raising, so a
    checkpoint is always readable up to the last COMPLETE step.
    """
    import orjson

    out: list = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue  # truncated final line from an interrupted sweep
            dq = payload.pop("downstream", None)
            out.append(MrmrGTResult(**payload, downstream=DownstreamQuality(**dq) if dq else DownstreamQuality()))
    return out

def run_mrmr_gt_config(n: int, n_relevant: int, n_irrelevant: int, n_redundant: int, methods, seeds, config_label: str, compute_downstream: bool = True, checkpoint_path: str | None = None) -> list:
    results = []
    for seed in seeds:
        X, y, gt = scen_multicolumn(n, n_relevant, n_irrelevant, n_redundant, seed=seed)
        for method in methods:
            t0 = time.perf_counter()
            selected = run_mrmr_selection(X, y, method, random_seed=seed)
            fit_time_s = time.perf_counter() - t0
            recall, precision, fpr, f1 = _prf(selected, gt)
            dq = _downstream_quality(X, y, selected, gt, seed) if compute_downstream else DownstreamQuality()
            _res = MrmrGTResult(
                config_label, method, seed, recall, precision, fpr, f1, len(selected),
                n_relevant=len(gt.relevant), n_relevant_hit=round(recall * len(gt.relevant)) if gt.relevant else 0,
                n_total_cols=X.shape[1], fit_time_s=fit_time_s, downstream=dq,
            )
            results.append(_res)
            if checkpoint_path:
                # Persist BEFORE the next (potentially long) step, so an interrupted sweep keeps its work.
                _append_jsonl_result(checkpoint_path, _res)
    return results


def run_mrmr_gt_wellbore_config(n_relevant: int, n_redundant: int, methods, seeds, config_label: str, n: int = 100_000, compute_downstream: bool = True) -> list:
    """Same result contract as ``run_mrmr_gt_config``, but scored on ``scen_wellbore100k`` (real
    wellbore log columns as the noise/redundancy pool, synthetic injected signal) instead of the
    fully-synthetic ``scen_multicolumn``."""
    results = []
    for seed in seeds:
        X, y, gt = scen_wellbore100k(n_relevant, n_redundant, seed=seed, n=n)
        for method in methods:
            t0 = time.perf_counter()
            selected = run_mrmr_selection(X, y, method, random_seed=seed)
            fit_time_s = time.perf_counter() - t0
            recall, precision, fpr, f1 = _prf(selected, gt)
            dq = _downstream_quality(X, y, selected, gt, seed) if compute_downstream else DownstreamQuality()
            results.append(
                MrmrGTResult(
                    config_label, method, seed, recall, precision, fpr, f1, len(selected),
                    n_relevant=len(gt.relevant), n_relevant_hit=round(recall * len(gt.relevant)) if gt.relevant else 0,
                    n_total_cols=X.shape[1], fit_time_s=fit_time_s, downstream=dq,
                )
            )
    return results


# Representative subset for the fast (pytest-collected) sweep -- the 3 original methods plus
# A representative 5-of-15 subset to keep fast_subset's wall-time budget -- the legacy fixed-quantile
# baseline, both fayyad_irani in-sample variants, the OOS-validated variant, and one formula+placement
# combo (freedman_diaconis_uniform). The remaining methods are exercised only in run_mrmr_full_sweep.
_MRMR_FAST_SUBSET_METHODS = (
    "legacy_fixed_quantile10", "fayyad_irani_fast", "fayyad_irani_insample_validated",
    "fayyad_irani_oos_validated", "freedman_diaconis_uniform",
)


def run_mrmr_fast_subset() -> list:
    """Small-n, few-seed MRMR ground-truth sweep -- a few seconds total. Used by the pytest fast
    test. Covers 2/4/8 relevant columns (16 is exercised only in ``run_mrmr_full_sweep``, it needs a
    bigger n to keep any signal detectable through the noise floor at this row count)."""
    results = []
    for n_relevant in (2, 4, 8):
        results.extend(
            run_mrmr_gt_config(
                n=800,
                n_relevant=n_relevant,
                n_irrelevant=4,
                n_redundant=n_relevant,
                methods=_MRMR_FAST_SUBSET_METHODS,
                seeds=range(5),
                config_label=f"rel{n_relevant}",
            )
        )
    return results


def run_mrmr_full_sweep() -> list:
    """Thorough MRMR ground-truth sweep: 2/4/8/16 relevant columns, larger n, 3 seeds/config,
    ALL registered binning methods (``MRMR_BINNING_METHODS``) -- several minutes, NOT run by pytest."""
    results = []
    for n_relevant in (2, 4, 8, 16):
        n = 4000 if n_relevant <= 8 else 12000
        results.extend(
            run_mrmr_gt_config(
                n=n,
                n_relevant=n_relevant,
                n_irrelevant=max(8, n_relevant),
                n_redundant=n_relevant,
                methods=tuple(MRMR_BINNING_METHODS),
                seeds=range(3),
                config_label=f"rel{n_relevant}",
            )
        )
    return results


def print_mrmr_gt_report(results: list) -> None:
    """Report per (config, method): recall/precision/fpr/F1 as before, PLUS the absolute counts
    behind recall ("n_relevant_hit/n_relevant") and total-selected fraction ("n_selected/n_total_cols")
    the raw percentages hide, PLUS fit wall-time and downstream linear-vs-CatBoost RMSE/logloss
    (with the sign of the gap: negative = CatBoost wins) when ``compute_downstream=True`` was used."""
    from collections import defaultdict

    by_key: dict = defaultdict(list)
    for r in results:
        by_key[(r.config, r.method)].append(r)
    print(
        f"{'config':10s} {'method':16s} {'n':>3s} {'recall':>18s} {'precision':>10s} {'fpr':>10s} {'f1':>10s} "
        f"{'selected':>12s} {'fit_s':>8s} {'lin_rmse':>10s} {'cb_rmse':>10s} {'rmse_gap':>10s} "
        f"{'lin_ll':>8s} {'cb_ll':>8s} {'ll_gap':>8s}"
    )
    for (config, method), rows in by_key.items():
        n_relevant = rows[0].n_relevant
        n_total_cols = rows[0].n_total_cols
        mean_hit = float(np.mean([r.n_relevant_hit for r in rows]))
        mean_selected = float(np.mean([r.n_selected for r in rows]))
        recalls = np.array([r.recall for r in rows])
        precisions = np.array([r.precision for r in rows])
        fprs = np.array([r.fpr_noise_redundant for r in rows])
        f1s = np.array([r.f1 for r in rows])
        fit_times = np.array([r.fit_time_s for r in rows])
        lin_rmse = np.array([r.downstream.linear_rmse for r in rows])
        cb_rmse = np.array([r.downstream.catboost_rmse for r in rows])
        rmse_gap = np.array([r.downstream.rmse_gap for r in rows])
        lin_ll = np.array([r.downstream.linear_logloss for r in rows])
        cb_ll = np.array([r.downstream.catboost_logloss for r in rows])
        ll_gap = np.array([r.downstream.logloss_gap for r in rows])
        recall_abs = f"{mean_hit:.1f}/{n_relevant}={recalls.mean():.0%}"
        selected_abs = f"{mean_selected:.1f}/{n_total_cols}={(mean_selected / n_total_cols if n_total_cols else float('nan')):.0%}"
        print(
            f"{config:10s} {method:16s} {len(rows):3d} {recall_abs:>18s} "
            f"{precisions.mean():10.3f} {fprs.mean():10.3f} {f1s.mean():10.3f} "
            f"{selected_abs:>12s} {fit_times.mean():8.2f} "
            f"{np.nanmean(lin_rmse):10.3f} {np.nanmean(cb_rmse):10.3f} {np.nanmean(rmse_gap):10.3f} "
            f"{np.nanmean(lin_ll):8.3f} {np.nanmean(cb_ll):8.3f} {np.nanmean(ll_gap):8.3f}"
        )
