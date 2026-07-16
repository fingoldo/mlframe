"""Composite-target DISCOVERY for classification (base-margin auto-selection).

The regression ``CompositeTargetDiscovery`` auto-finds ``(base, transform)`` pairs.
This is its classification twin: it auto-finds which column of ``X`` is worth
anchoring a :class:`CompositeClassificationEstimator` on, i.e. which single
feature carries enough log-odds signal that handing it to the inner booster as a
BASE MARGIN (via a cheap univariate logistic model) beats training the booster
flat. Nothing existed on this side before: the estimator required the caller to
already know the dominant column.

Pipeline (mirrors the regression discovery discipline):

1. **Honest holdout carve** -- a stratified ``holdout_frac`` slice is removed
   BEFORE any screening and only used for the final re-score, so the reported
   gain is not winner's-curse-inflated.
2. **Stage-1 margin screen** -- for every numeric candidate column, K-fold OOS
   log-loss of a univariate ``LogisticRegression`` on that column alone vs the
   class-prior baseline log-loss. Columns that near-perfectly separate ``y``
   (OOS log-loss below ``leak_logloss_floor``) are rejected as suspected label
   leaks, mirroring the regression base-target-leakage gate. Deliberately an
   OOS predictive-error screen, NOT a mutual-information screen: MI is
   monotone-invariant and bias-inflated, and the regression pipeline's MI-first
   design is exactly what its honest-holdout gate has to correct for.
3. **Stage-2 paired composite-vs-plain CV** -- for the ``top_k`` stage-1
   survivors, a tiny inner booster wrapped in
   :class:`CompositeClassificationEstimator` (base margin = univariate logistic
   on the candidate) is compared against the SAME tiny booster trained flat, on
   the same stratified folds. A candidate survives only if the composite wins
   the majority of folds AND the mean log-loss gain exceeds ``min_cv_gain``.
4. **Honest re-score** -- stage-2 survivors are refit on the full screening
   split and re-scored on the untouched holdout; the best candidate must beat
   the plain model there too (``honest_gain > 0``) or discovery returns "no
   composite recommended" (empty result, plain model is the right call).

Binary and multiclass both work (log-loss and the univariate logistic handle
K >= 2; the estimator's multiclass base-margin path is used automatically).

Public surface: :class:`CompositeClassificationDiscovery`,
:class:`ClassificationDiscoveryResult`, :func:`discover_and_wrap_classification`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

from .classification import CompositeClassificationEstimator

logger = logging.getLogger(__name__)

# Univariate logistic settings for both the stage-1 screen and the base-margin
# model handed to the estimator. lbfgs + moderate iter cap: the problem is 1D.
_UNIVARIATE_LR_MAX_ITER = 200

# Default tiny inner booster budget: enough capacity to expose a margin-anchoring
# win, small enough that stage-2 stays cheap on wide candidate sets.
_TINY_INNER_N_ESTIMATORS = 60
_TINY_INNER_NUM_LEAVES = 15


def _default_tiny_inner(random_state: int) -> Any:
    """Small LightGBM classifier used for the stage-2 paired screen."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=_TINY_INNER_N_ESTIMATORS,
        num_leaves=_TINY_INNER_NUM_LEAVES,
        learning_rate=0.1,
        random_state=random_state,
        verbose=-1,
        n_jobs=1,
    )


def _to_numpy_2d(X: Any) -> tuple[np.ndarray, list[str]]:
    """``(values, column_names)`` from polars / pandas / ndarray input."""
    cols = getattr(X, "columns", None)
    if cols is not None:
        names = [str(c) for c in cols]
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
        names = [f"col_{i}" for i in range(arr.shape[1])]
    return np.asarray(arr, dtype=object if arr.dtype == object else None), names


def _numeric_candidate_mask(values: np.ndarray, names: list[str], forbidden: tuple[str, ...]) -> list[int]:
    """Indices of usable candidate columns: numeric, non-constant, name not forbidden."""
    out: list[int] = []
    lowered = [n.lower() for n in names]
    for j in range(values.shape[1]):
        if any(pat in lowered[j] for pat in forbidden):
            continue
        col = values[:, j]
        try:
            col_f = np.asarray(col, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        finite = col_f[np.isfinite(col_f)]
        if finite.size < col_f.size * 0.5:
            continue
        if finite.size == 0 or float(np.nanstd(col_f)) == 0.0:
            continue
        out.append(j)
    return out


def _impute_column(col: np.ndarray) -> np.ndarray:
    """Float64 copy with non-finite entries replaced by the finite median (LR cannot take NaN)."""
    col_f = np.asarray(col, dtype=np.float64).reshape(-1, 1).copy()
    bad = ~np.isfinite(col_f[:, 0])
    if bad.any():
        med = float(np.nanmedian(np.where(np.isfinite(col_f[:, 0]), col_f[:, 0], np.nan)))
        col_f[bad, 0] = med
    return np.asarray(col_f, dtype=np.float64)


@dataclass
class ClassificationDiscoveryResult:
    """Everything the screen learned, ranked; ``best`` is None when the plain model won."""

    baseline_logloss: float
    candidates: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    holdout_plain_logloss: float | None = None
    holdout_composite_logloss: float | None = None

    @property
    def honest_gain(self) -> float | None:
        """Holdout log-loss improvement of the composite over the plain model (positive = composite wins)."""
        if self.holdout_plain_logloss is None or self.holdout_composite_logloss is None:
            return None
        return self.holdout_plain_logloss - self.holdout_composite_logloss


class CompositeClassificationDiscovery:
    """Auto-select the base-margin column for :class:`CompositeClassificationEstimator`.

    Parameters
    ----------
    inner_estimator
        Booster used in the stage-2 paired screen and the final wrapped model.
        Must accept a base margin (LightGBM / XGBoost / CatBoost). Default: a
        tiny LightGBM classifier.
    candidate_columns
        Explicit candidate column names. Default None: every usable numeric
        column of ``X`` (capped at ``max_candidates`` by stage-1 gain).
    top_k
        Stage-1 survivors carried into the expensive stage-2 paired screen.
    cv_folds
        Stratified folds for both screening stages.
    min_margin_gain
        Stage-1 gate: univariate OOS log-loss must beat the prior baseline by
        at least this much for a column to be considered at all.
    min_cv_gain
        Stage-2 gate: mean paired log-loss gain of composite over plain.
    holdout_frac
        Stratified honest-holdout fraction carved before screening.
    leak_logloss_floor
        Stage-1 leak guard: a univariate OOS log-loss BELOW this is too good
        for a single honest feature and the column is rejected as a suspected
        label leak.
    forbidden_patterns
        Case-insensitive substrings excluding columns from candidacy.
    """

    def __init__(
        self,
        inner_estimator: Any = None,
        candidate_columns: list[str] | None = None,
        top_k: int = 3,
        cv_folds: int = 4,
        min_margin_gain: float = 1e-3,
        min_cv_gain: float = 0.0,
        holdout_frac: float = 0.2,
        leak_logloss_floor: float = 0.05,
        max_candidates: int = 100,
        forbidden_patterns: tuple[str, ...] = ("target", "label", "y_true"),
        random_state: int = 42,
    ) -> None:
        self.inner_estimator = inner_estimator
        self.candidate_columns = candidate_columns
        self.top_k = top_k
        self.cv_folds = cv_folds
        self.min_margin_gain = min_margin_gain
        self.min_cv_gain = min_cv_gain
        self.holdout_frac = holdout_frac
        self.leak_logloss_floor = leak_logloss_floor
        self.max_candidates = max_candidates
        self.forbidden_patterns = forbidden_patterns
        self.random_state = random_state

    # -- internals ---------------------------------------------------------
    def _univariate_margin_model(self) -> LogisticRegression:
        """Construct the logistic-regression model used to score a single candidate column."""
        return LogisticRegression(max_iter=_UNIVARIATE_LR_MAX_ITER)

    def _stage1_screen(self, values: np.ndarray, names: list[str], y: np.ndarray, cand_idx: list[int]) -> list[dict[str, Any]]:
        """Per-column univariate OOS log-loss vs the prior baseline; leak-guarded."""
        classes = np.unique(y)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        splits = list(skf.split(np.zeros(y.size), y))
        # Prior baseline: per-fold class frequencies scored on the fold's val rows.
        base_ll: list[float] = []
        for tr, va in splits:
            freq = np.array([(y[tr] == c).mean() for c in classes], dtype=np.float64)
            freq = np.clip(freq, 1e-9, 1.0)
            freq /= freq.sum()
            proba = np.tile(freq, (va.size, 1))
            base_ll.append(float(log_loss(y[va], proba, labels=classes)))
        baseline = float(np.mean(base_ll))

        rows: list[dict[str, Any]] = []
        for j in cand_idx:
            col = _impute_column(values[:, j])
            lls: list[float] = []
            ok = True
            for tr, va in splits:
                if np.unique(y[tr]).size < classes.size:
                    ok = False
                    break
                lr = self._univariate_margin_model()
                try:
                    lr.fit(col[tr], y[tr])
                    lls.append(float(log_loss(y[va], lr.predict_proba(col[va]), labels=classes)))
                except (ValueError, FloatingPointError):
                    ok = False
                    break
            if not ok or not lls:
                continue
            ll = float(np.mean(lls))
            row = {
                "column": names[j], "column_index": j,
                "univariate_logloss": ll, "margin_gain": baseline - ll,
                "suspected_leak": ll < self.leak_logloss_floor,
            }
            rows.append(row)
        rows.sort(key=lambda r: -r["margin_gain"])
        self._baseline_logloss = baseline
        return rows

    def _stage2_paired(self, X_screen: Any, y: np.ndarray, col_name: str) -> dict[str, Any]:
        """Paired composite-vs-plain CV log-loss on identical stratified folds."""
        classes = np.unique(y)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        comp_ll: list[float] = []
        plain_ll: list[float] = []
        col_full = _impute_column(_extract_named_column(X_screen, col_name))
        for tr, va in skf.split(np.zeros(y.size), y):
            inner = self.inner_estimator if self.inner_estimator is not None else _default_tiny_inner(self.random_state)
            plain = clone(inner)
            X_tr, X_va = _take_rows(X_screen, tr), _take_rows(X_screen, va)
            plain.fit(X_tr, y[tr])
            plain_ll.append(float(log_loss(y[va], plain.predict_proba(X_va), labels=classes)))
            margin_lr = self._univariate_margin_model().fit(col_full[tr], y[tr])
            est = CompositeClassificationEstimator(
                base_estimator=clone(inner),
                base_margin_estimator=_UnivariateColumnMargin(margin_lr, col_name),
            )
            est.fit(X_tr, y[tr])
            comp_ll.append(float(log_loss(y[va], est.predict_proba(X_va), labels=classes)))
        comp = np.asarray(comp_ll)
        plain = np.asarray(plain_ll)
        wins = int((comp < plain).sum())
        return {
            "cv_composite_logloss": float(comp.mean()),
            "cv_plain_logloss": float(plain.mean()),
            "cv_gain": float((plain - comp).mean()),
            "fold_wins": wins,
            "n_folds": int(comp.size),
        }

    # -- public API ----------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "CompositeClassificationDiscovery":
        """Run the 4-phase screen; results land in ``result_`` / ``best_estimator_``."""
        y_arr = np.asarray(y).reshape(-1)
        if np.unique(y_arr).size < 2:
            raise ValueError("CompositeClassificationDiscovery needs >= 2 classes.")
        idx_screen, idx_hold = train_test_split(
            np.arange(y_arr.size), test_size=self.holdout_frac,
            stratify=y_arr, random_state=self.random_state,
        )
        X_screen, X_hold = _take_rows(X, idx_screen), _take_rows(X, idx_hold)
        y_screen, y_hold = y_arr[idx_screen], y_arr[idx_hold]

        values, names = _to_numpy_2d(X_screen)
        cand_idx = _numeric_candidate_mask(values, names, tuple(p.lower() for p in self.forbidden_patterns))
        if self.candidate_columns is not None:
            wanted = set(self.candidate_columns)
            cand_idx = [j for j in cand_idx if names[j] in wanted]
        stage1 = self._stage1_screen(values, names, y_screen, cand_idx[: self.max_candidates])
        survivors = [r for r in stage1 if not r["suspected_leak"] and r["margin_gain"] > self.min_margin_gain]
        for r in stage1:
            if r["suspected_leak"]:
                logger.warning("classification discovery: column %r rejected as suspected label leak (univariate log-loss %.2g)", r["column"], r["univariate_logloss"])

        candidates: list[dict[str, Any]] = []
        for r in survivors[: self.top_k]:
            metrics = self._stage2_paired(X_screen, y_screen, r["column"])
            row = {**r, **metrics}
            row["accepted"] = metrics["cv_gain"] > self.min_cv_gain and metrics["fold_wins"] * 2 > metrics["n_folds"]
            candidates.append(row)
        candidates.sort(key=lambda r: -r["cv_gain"])
        accepted = [r for r in candidates if r["accepted"]]

        result = ClassificationDiscoveryResult(baseline_logloss=self._baseline_logloss, candidates=stage1)
        for row in candidates:  # stamp stage-2 metrics back onto the stage-1 table
            for s1 in result.candidates:
                if s1["column"] == row["column"]:
                    s1.update({k: v for k, v in row.items() if k not in s1})
        self.best_estimator_: CompositeClassificationEstimator | None = None
        if accepted:
            best = accepted[0]
            classes = np.unique(y_screen)
            inner = self.inner_estimator if self.inner_estimator is not None else _default_tiny_inner(self.random_state)
            col = _impute_column(_extract_named_column(X_screen, best["column"]))
            margin_lr = self._univariate_margin_model().fit(col, y_screen)
            est = CompositeClassificationEstimator(
                base_estimator=clone(inner),
                base_margin_estimator=_UnivariateColumnMargin(margin_lr, best["column"]),
            )
            est.fit(X_screen, y_screen)
            plain = clone(inner)
            plain.fit(X_screen, y_screen)
            result.holdout_composite_logloss = float(log_loss(y_hold, est.predict_proba(X_hold), labels=classes))
            result.holdout_plain_logloss = float(log_loss(y_hold, plain.predict_proba(X_hold), labels=classes))
            gain = result.honest_gain
            if gain is not None and gain > 0.0:
                result.best = best
                self.best_estimator_ = est
            else:
                logger.info("classification discovery: best candidate %r failed the honest holdout (gain %.4g <= 0); plain model recommended", best["column"], gain)
        self.result_ = result
        return self

    def recommend(self) -> dict[str, Any] | None:
        """The winning candidate row, or None when the plain model is the right call."""
        return self.result_.best


class _UnivariateColumnMargin:
    """Base-margin adapter: a fitted univariate logistic applied to one named column of X.

    sklearn-clone friendly: ``fit`` refits the inner logistic on the SAME single
    column extracted from whatever X the estimator passes, so the adapter never
    stores training data.
    """

    def __init__(self, lr: LogisticRegression, column: str) -> None:
        self.lr = lr
        self.column = column

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Return the constructor parameters, for sklearn clone/grid-search compatibility."""
        return {"lr": self.lr, "column": self.column}

    def set_params(self, **params: Any) -> "_UnivariateColumnMargin":
        """Set constructor parameters in place, for sklearn clone/grid-search compatibility."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _col(self, X: Any) -> np.ndarray:
        """Extract and impute this adapter's single named column from ``X``."""
        return _impute_column(_extract_named_column(X, self.column))

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "_UnivariateColumnMargin":
        """Fit a fresh clone of the inner logistic regression on this adapter's single column."""
        self.lr = clone(self.lr)
        if sample_weight is not None:
            self.lr.fit(self._col(X), y, sample_weight=sample_weight)
        else:
            self.lr.fit(self._col(X), y)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        """Return the inner logistic regression's decision function on this adapter's single column."""
        return np.asarray(self.lr.decision_function(self._col(X)))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return the inner logistic regression's predicted class probabilities on this adapter's single column."""
        return np.asarray(self.lr.predict_proba(self._col(X)))


def _extract_named_column(X: Any, name: str) -> np.ndarray:
    """One named column of polars / pandas / ndarray X as a flat float array."""
    if hasattr(X, "get_column"):  # polars
        return np.asarray(X.get_column(name).to_numpy(), dtype=np.float64).reshape(-1)
    if hasattr(X, "columns"):  # pandas
        return np.asarray(X[name].to_numpy(), dtype=np.float64).reshape(-1)
    j = int(name.split("_")[-1])  # ndarray fallback: our own col_<j> naming
    return np.asarray(X[:, j], dtype=np.float64).reshape(-1)


def _take_rows(X: Any, idx: np.ndarray) -> Any:
    """Row subset for polars / pandas / ndarray."""
    if hasattr(X, "get_column"):  # polars
        return X[idx.tolist()]
    if hasattr(X, "iloc"):  # pandas
        return X.iloc[idx]
    return X[idx]


def discover_and_wrap_classification(X: Any, y: Any, inner_estimator: Any = None, **discovery_kwargs: Any) -> tuple[Any, ClassificationDiscoveryResult]:
    """One-call convenience: run discovery, return ``(fitted_model, result)``.

    The fitted model is the composite when discovery found an honest win, else
    the plain inner booster fitted on all of ``X`` -- callers always get a
    usable classifier plus the audit trail explaining the choice.
    """
    disc = CompositeClassificationDiscovery(inner_estimator=inner_estimator, **discovery_kwargs)
    disc.fit(X, y)
    if disc.best_estimator_ is not None:
        return disc.best_estimator_, disc.result_
    inner = inner_estimator if inner_estimator is not None else _default_tiny_inner(int(discovery_kwargs.get("random_state", 42)))
    plain = clone(inner)
    plain.fit(X, np.asarray(y).reshape(-1))
    return plain, disc.result_
