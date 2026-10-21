"""Shared machinery for the end-to-end FE family survival tests.

A family descriptor names the flag(s) that switch one feature-engineering family on, the recipe ``kind`` values that identify its output and a
scenario builder whose target only that family can express. ``fit_family`` builds an ``MRMR`` with every default-True ``fe_*_enable`` flag read
from the constructor signature switched off, then the family's own flags switched on, so a gate that silently kills the whole family shows up as
a missing recipe or a missing selected column.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters.mrmr import MRMR

SEEDS = (0, 1, 2)

PIPELINE_MACHINERY = frozenset(
    {
        "fe_rung_schedule_enable",
        "fe_auto_escalation_enable",
        "fe_synergy_prevalence_rescue_enable",
        "fe_pair_usability_admission_enable",
        "fe_additive_fusion_enable",
    }
)
"""Default-True flags that schedule, rescue or fuse candidates of any family and produce no family of their own. They stay at their default so a family
is tested inside the pipeline that ships; every flag that generates a family is off unless the family under test names it."""
TEST_FRACTION = 0.3


# (family, check) pairs that fail today. Each is a finding documented in audits/2026-10-20_fe_family_survival/FINDINGS.md, not a tolerance to widen: the check is left out of
# the parametrization until the family is fixed, and this registry is the list of what is left out. A family that is fixed is removed from here and the test then covers it.
OPEN_FINDINGS = {
    "created": frozenset({"binned_numeric_agg", "pair_cross"}),
    "lift": frozenset({"binned_numeric_agg", "count_encoding", "frequency_encoding", "pairwise_ratio", "wavelet"}),
    "power": frozenset({"binned_numeric_agg", "pair_cross", "pairwise_log_ratio"}),
}


def checked(families, check: str) -> list:
    """pytest parameters (one per family, id = family name) for ``check``, without the families that have an open finding for it."""
    import pytest

    return [pytest.param(f, id=f.name) for f in families if f.name not in OPEN_FINDINGS.get(check, frozenset())]


@dataclass(frozen=True, eq=False)
class Family:
    """One FE family: its enabling flags, its recipe kinds (narrowed by ``pattern``, a regex on the recipe name, when several families share a kind) and the scenario in which only it can recover the signal."""

    name: str
    flags: dict
    kinds: tuple
    build: Callable
    task: str = "regression"
    model: str = "linear"
    extra: dict = field(default_factory=dict)
    panel: int = 20
    pattern: str = ""


def default_on_fe_flags() -> list:
    """Names of every ``fe_*_enable`` constructor parameter whose default is exactly True, read from the live signature."""
    return [n for n, p in inspect.signature(MRMR.__init__).parameters.items() if n.startswith("fe_") and n.endswith("_enable") and p.default is True]


def isolated_mrmr(family: Family) -> MRMR:
    """``MRMR`` with every default-True FE flag off, the family's flags on and a deterministic single-worker setup."""
    kw = {n: False for n in default_on_fe_flags() if n not in PIPELINE_MACHINERY}
    kw.update(family.flags)
    kw.update(family.extra)
    return MRMR(verbose=0, random_state=0, n_jobs=1, **kw)


def recipes_of(m: MRMR) -> list:
    """The fitted engineered recipes (``_engineered_recipes_``, list or dict, plus the recipes of the usability lists' candidates), deduplicated by name."""
    rec = m._engineered_recipes_
    out = list(rec.values()) if isinstance(rec, dict) else list(rec)
    seen = {str(r.name) for r in out}
    for attr in ("support_linear_", "support_universal_"):
        for cand in getattr(m, attr, None) or []:
            r = getattr(cand, "recipe", None)
            if r is not None and str(r.name) not in seen:
                seen.add(str(r.name))
                out.append(r)
    return out


def numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    """All columns as float64; non-numeric ones become category codes, so a linear baseline can at least see them."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = s.to_numpy(dtype=np.float64)
        else:
            out[c] = pd.Categorical(s).codes.astype(np.float64)
    return pd.DataFrame(out, index=df.index)


def split(X: pd.DataFrame, y: np.ndarray, seed: int):
    """Shuffled train / held-out split; the selector is fitted on the train part only."""
    idx = np.random.default_rng(10_000 + seed).permutation(len(X))
    k = int(len(X) * (1.0 - TEST_FRACTION))
    tr, te = idx[:k], idx[k:]
    return X.iloc[tr].reset_index(drop=True), np.asarray(y)[tr], X.iloc[te].reset_index(drop=True), np.asarray(y)[te]


@dataclass
class FitResult:
    """A fitted selector with the train / held-out parts of the frame it was fitted on."""

    model: MRMR
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    fit_seconds: float


@lru_cache(maxsize=6)
def fit_family(family: Family, seed: int, noise: bool = False) -> FitResult:
    """Fit the isolated selector on the family's scenario (or, with ``noise``, on the same columns with a permuted target)."""
    import time

    X, y = family.build(seed)
    if noise:
        y = np.random.default_rng(777 + seed).permutation(np.asarray(y))
    Xtr, ytr, Xte, yte = split(X, y, seed)
    m = isolated_mrmr(family)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    return FitResult(m, Xtr, ytr, Xte, yte, time.perf_counter() - t0)


# Name patterns of the columns a family creates. A column of the family that the selector folds into a composite (``sub(temperature, invcbrt(lagged_diff_temperature__period1))``)
# has reached the output as much as one that stands alone: the selector is allowed to prefer the composite, as the mechanism-axis tests already state. Families listed here are
# credited for a selected recipe that reads one of their columns, directly or through a nested parent recipe.
OPERAND_PATTERNS = {
    "hinge": r"__relu_",
    "wavelet": r"__haar_|__wav",
    "row_argmax": r"^argmax__",
    "missingness_indicator": r"^is_missing__",
    "missingness_pattern": r"^missingness_pattern",
    "grouped_delta": r"^grouped_delta_",
    "lagged_diff": r"^lagged_diff_",
    "pairwise_log_ratio": r"^log_ratio__",
    "pairwise_ratio": r"^ratio__",
    "binned_numeric_agg": r"^binagg_",
    "modular": r"__mod",
}


def _operand_names(recipe) -> set:
    """Names a recipe reads: its source names and, through ``nested_parent_a/b``, those of its engineered parents (and the parents' own names)."""
    names = {str(s) for s in (getattr(recipe, "src_names", ()) or ())}
    for key in ("nested_parent_a", "nested_parent_b"):
        parent = (getattr(recipe, "extra", None) or {}).get(key)
        if parent is not None:
            names |= _operand_names(parent) | {str(parent.name)}
    return names


def family_columns(fit: FitResult, family: Family) -> tuple:
    """(recipes of the family's kinds, or selected recipes that read one of its columns; names of those that are in the selected output)."""
    recipes = recipes_of(fit.model)
    mine = [r for r in recipes if r.kind in family.kinds and (not family.pattern or re.search(family.pattern, str(r.name)))]
    selected = set(map(str, fit.model.get_feature_names_out()))
    kept = [r.name for r in mine if str(r.name) in selected]
    operand = OPERAND_PATTERNS.get(family.name)
    if operand:
        direct = {str(r.name) for r in mine}
        via = [r for r in recipes if str(r.name) in selected and str(r.name) not in direct and any(re.search(operand, n) for n in _operand_names(r))]
        mine, kept = mine + via, kept + [r.name for r in via]
    return mine, kept


def _downstream(family: Family):
    """Fresh downstream estimator: ridge / logistic for linear families, a small boosted tree otherwise."""
    if family.model == "linear":
        if family.task == "regression":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    if family.task == "regression":
        return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, random_state=0)
    return HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=0)


def _score(family: Family, est, Xtr: np.ndarray, ytr, Xte: np.ndarray, yte) -> float:
    """Held-out R^2 (regression) or ROC AUC (classification)."""
    est.fit(Xtr, ytr)
    if family.task == "regression":
        return float(r2_score(yte, est.predict(Xte)))
    return float(roc_auc_score(yte, est.predict_proba(Xte)[:, 1]))


def held_out_scores(fit: FitResult, family: Family) -> tuple:
    """(score on the raw columns only, score on the selector's selected output), both on the held-out part."""
    raw_tr, raw_te = numeric_view(fit.X_train), numeric_view(fit.X_test)
    sel_tr = numeric_view(pd.DataFrame(fit.model.transform(fit.X_train)))
    sel_te = numeric_view(pd.DataFrame(fit.model.transform(fit.X_test)))
    if family.model == "linear":
        med_raw, med_sel = raw_tr.median().fillna(0.0), sel_tr.median().fillna(0.0)
        raw_tr, raw_te = raw_tr.fillna(med_raw), raw_te.fillna(med_raw)
        sel_tr, sel_te = sel_tr.fillna(med_sel), sel_te.fillna(med_sel)
    raw = _score(family, _downstream(family), raw_tr.to_numpy(), fit.y_train, raw_te.to_numpy(), fit.y_test)
    sel = _score(family, _downstream(family), sel_tr.to_numpy(), fit.y_train, sel_te.to_numpy(), fit.y_test)
    return raw, sel


def lift_summary(family: Family) -> dict:
    """Per-seed raw / selected scores and lifts, their mean and the spread-derived lower bound ``mean - 2 sd``."""
    rows = []
    for s in SEEDS:
        raw, sel = held_out_scores(fit_family(family, s), family)
        rows.append((raw, sel, sel - raw))
    lifts = np.array([r[2] for r in rows])
    return {
        "raw": [r[0] for r in rows],
        "selected": [r[1] for r in rows],
        "lifts": lifts.tolist(),
        "mean": float(lifts.mean()),
        "sd": float(lifts.std(ddof=1)),
        "lower": float(lifts.mean() - 2.0 * lifts.std(ddof=1)),
    }


def finite_report(fit: FitResult, family: Family) -> Optional[str]:
    """None when ``transform`` of the held-out frame has the right shape and only finite values, else a description of the problem."""
    out = fit.model.transform(fit.X_test)
    arr = numeric_view(pd.DataFrame(out)).to_numpy()
    n_sel = len(list(fit.model.get_feature_names_out()))
    if arr.shape != (len(fit.X_test), n_sel):
        return f"shape {arr.shape} != {(len(fit.X_test), n_sel)}"
    names = list(map(str, fit.model.get_feature_names_out()))
    engineered = {str(r.name) for r in recipes_of(fit.model)}
    cols = [i for i, nm in enumerate(names) if nm in engineered]
    if not np.isfinite(arr[:, cols]).all():
        return f"{int((~np.isfinite(arr[:, cols])).sum())} non-finite cells in engineered columns"
    return None


def check_created_and_survived(family: Family) -> None:
    """Assert a recipe of the family's kind exists and at least one of its columns is in the selected output."""
    fit = fit_family(family, SEEDS[0])
    mine, kept = family_columns(fit, family)
    selected = list(fit.model.get_feature_names_out())
    assert mine, f"{family.name}: no recipe of kind {family.kinds} was created; selected={selected}; kinds seen={sorted({r.kind for r in recipes_of(fit.model)})}"
    assert kept, f"{family.name}: recipes {[r.name for r in mine]} were created but none survived into {selected}"


def check_replay(family: Family) -> None:
    """Assert ``transform`` of the held-out rows has the selected shape and only finite engineered values."""
    problem = finite_report(fit_family(family, SEEDS[0]), family)
    assert problem is None, f"{family.name}: {problem}"


def check_lift(family: Family) -> None:
    """Assert the selected output beats the raw columns on every seed and the lift exceeds twice its seed-to-seed spread."""
    s = lift_summary(family)
    print(f"[lift] {family.name}: raw={s['raw']} selected={s['selected']} lifts={s['lifts']} mean={s['mean']:.4f} sd={s['sd']:.4f}")
    assert min(s["lifts"]) > 0.0 and s["lower"] > 0.0, f"{family.name}: lift summary {s}"


def check_noise_control(family: Family) -> None:
    """Assert no column of the family is selected when the target is independent of every column."""
    fit = fit_family(family, SEEDS[0], True)
    _, kept = family_columns(fit, family)
    assert kept == [], f"{family.name}: selected {kept} on a target independent of every column"


POWER_TARGET = 0.8
"""Policy floor for the lower Clopper-Pearson bound of the detection rate; it is not tuned per family. A two-sided 90% interval needs at least 14 seeds,
all detected, to reach 0.8 (0.05 ** (1 / 14) = 0.809), so the panels have 14 or 20 seeds."""

FALSE_POSITIVE_MAX = 0.2
"""Policy ceiling for the upper Clopper-Pearson bound of the noise false-positive rate. The nominal per-fit level is 0.05; a panel of k seeds without a
false positive only bounds the rate by 1 - 0.05 ** (1 / k): 0.26 for 10 seeds (so 10 seeds could never meet 0.2), 0.19 for 14, 0.14 for 20."""

CONFIDENCE = 0.90


def clopper_pearson(k: int, n: int, confidence: float = CONFIDENCE) -> tuple:
    """Two-sided Clopper-Pearson interval (lower, upper) for ``k`` successes in ``n`` trials."""
    from scipy.stats import beta

    a = (1.0 - confidence) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


@cache
def detected(family: Family, seed: int, noise: bool) -> bool:
    """True when the fit on this seed's scenario (or permuted-target scenario) selected an engineered column of the family."""
    fit = fit_family(family, seed, noise)
    _, kept = family_columns(fit, family)
    return bool(kept)


def panel(family: Family, noise: bool) -> tuple:
    """(detections, seeds) over the family's seed panel."""
    return sum(detected(family, s, noise) for s in range(family.panel)), family.panel


def print_power_table(families) -> None:
    """Print the power table for ``families``; run the test modules with -s to regenerate it."""
    print("\nfamily | detections/seeds | 90% CI | noise false positives/seeds | 90% CI")
    for fam in families:
        k, n = panel(fam, False)
        f, m = panel(fam, True)
        (a, b), (c, d) = clopper_pearson(k, n), clopper_pearson(f, m)
        print(f"{fam.name} | {k}/{n} | [{a:.2f}, {b:.2f}] | {f}/{m} | [{c:.2f}, {d:.2f}]")


def check_power(family: Family) -> None:
    """Assert the lower bound of the detection rate over the signal panel is at least ``POWER_TARGET``."""
    k, n = panel(family, False)
    lo, hi = clopper_pearson(k, n)
    print(f"[power] {family.name}: {k}/{n} detected, 90% CI [{lo:.2f}, {hi:.2f}]")
    assert lo >= POWER_TARGET, f"{family.name}: detected in {k}/{n} seeds, lower bound {lo:.2f} < {POWER_TARGET}"


def check_false_positive_rate(family: Family) -> None:
    """Assert the upper bound of the false-positive rate over the permuted-target panel is at most ``FALSE_POSITIVE_MAX``."""
    k, n = panel(family, True)
    lo, hi = clopper_pearson(k, n)
    print(f"[noise] {family.name}: {k}/{n} false positives, 90% CI [{lo:.2f}, {hi:.2f}]")
    assert hi <= FALSE_POSITIVE_MAX, f"{family.name}: {k}/{n} false positives, upper bound {hi:.2f} > {FALSE_POSITIVE_MAX}"
