"""ACE - Artificial Contrasts with Ensembles (Tuv, Borisov, Runger, Torkkola 2009).

A model-agnostic feature-significance filter that judges each real feature against a pool of
ARTIFICIAL CONTRASTS (column-permuted copies of the data whose relationship to y is destroyed).
For each of ``n_replicates`` independent runs it fits the estimator on ``[X | contrasts]`` and reads
its importances, then tests whether a real feature's importance is statistically larger than the
alpha-percentile of the pooled contrast importances via a one-sample t-test over the replicates. This
gives a per-feature p-value + accept/reject verdict, which the single-shot ``boruta`` importance getter
and the cross-model ``heterogeneous_relevance_vote`` do not: Boruta bins each trial into a binary hit and
runs a binomial test on the counts; ACE keeps the CONTINUOUS importance margin and tests it parametrically,
which is more powerful at small replicate counts.

ACE also implements the masking-removal loop from the paper: after a pass, accepted features are removed
and the procedure repeats on the remainder, so a weak-but-real feature masked by a stronger correlated
one gets a second chance to clear the (now lower) contrast bar. Set ``n_masking_rounds=1`` to disable.

Complements the lecture's Boruta slide (Дьяконов, "Важности признаков в ансамблях деревьев", slide 24:
"ACE - аналогично, но удаляются хорошие признаки"). Model-agnostic: works with any estimator exposing
``feature_importances_`` or ``coef_`` (RF/ET/GBM/linear); pass ``importance="permutation"`` to use held-out
permutation importance instead of the in-bag native importance (removes the impurity high-cardinality bias
Дьяконов warns about on slide 11).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Default contrast-pool percentile the real importance must beat. 100 = strongest single contrast
# (the Boruta MAX-shadow gate); Tuv 2009 uses a high percentile (95-100) to keep accepted-noise low.
_DEFAULT_CONTRAST_PERCENTILE = 100.0


@dataclass
class ACEResult:
    """Per-feature ACE verdict. Arrays are aligned to ``feature_names``."""

    feature_names: list
    importances_mean: np.ndarray  # mean real-feature importance over replicates
    contrast_threshold: np.ndarray  # per-feature contrast-percentile bar it was tested against
    p_values: np.ndarray  # one-sample t-test p-value: importance > contrast bar
    accepted: np.ndarray  # bool mask, p_value < alpha (Benjamini-Hochberg-corrected)
    selected_features: list = field(default_factory=list)  # names where accepted is True

    def support(self) -> np.ndarray:
        """sklearn-style boolean support mask."""
        return self.accepted


def _read_importances(model, importance: str, X, y, n_repeats: int, random_state: int) -> np.ndarray:
    """Return a 1-D importance vector for a FITTED model over all columns of X.

    ``importance='native'`` reads ``feature_importances_`` / ``|coef_|``; ``'permutation'`` runs sklearn's
    held-out permutation importance (Дьяконов's PFI - unbiased vs the impurity high-cardinality skew)."""
    if importance == "permutation":
        from sklearn.inspection import permutation_importance

        pi = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        # Negative permutation importance = "noise"; clip so it ties contrasts rather than flipping sign.
        return np.clip(np.asarray(pi.importances_mean, dtype=np.float64), 0.0, None)

    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64)
    if hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=np.float64))
        return coef.max(axis=0) if coef.ndim > 1 else coef
    raise AttributeError(
        f"ACE importance='native' needs feature_importances_ or coef_ on the fitted " f"{type(model).__name__}; got neither. Use importance='permutation'."
    )


def _benjamini_hochberg_reject(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """BH step-up: return a bool mask of rejected (accepted-as-relevant) hypotheses at FDR ``alpha``.

    ACE runs one test per feature, so without multiplicity control the per-feature alpha admits ~alpha*p
    false positives. BH controls the false-discovery rate across the whole feature battery."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresholds
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.nonzero(below)[0].max()  # largest rank passing the BH line; all lower ranks reject too
        reject[order[: k + 1]] = True
    return reject


def _make_contrasts(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Column-permuted copy of X: each column keeps its marginal (value multiset) but loses its row
    alignment with y and every other column, so a contrast carries NO genuine relationship to the target."""
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = rng.permutation(X[:, j])
    return out


def _one_replicate_importances(
    fit_predict_model, X: np.ndarray, y: np.ndarray, importance: str, n_perm_repeats: int, rng: np.random.Generator, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one estimator on ``[X | contrasts]`` and split its importances into (real, contrast) halves."""
    from sklearn.base import clone

    p = X.shape[1]
    contrasts = _make_contrasts(X, rng)
    X_joint = np.hstack([X, contrasts])
    model = clone(fit_predict_model)
    model.fit(X_joint, y)
    imps = _read_importances(model, importance, X_joint, y, n_repeats=n_perm_repeats, random_state=seed)
    if imps.shape[0] != 2 * p:
        raise ValueError(f"ACE expected {2 * p} importances from the joint fit, got {imps.shape[0]}.")
    return imps[:p], imps[p:]


def ace_select(
    X,
    y,
    estimator=None,
    *,
    n_replicates: int = 20,
    contrast_percentile: float = _DEFAULT_CONTRAST_PERCENTILE,
    alpha: float = 0.05,
    importance: str = "native",
    n_masking_rounds: int = 3,
    n_perm_repeats: int = 5,
    fdr_control: bool = True,
    feature_names: Sequence | None = None,
    random_state: int = 0,
) -> ACEResult:
    """Select relevant features by Artificial Contrasts with Ensembles (Tuv et al. 2009).

    Parameters
    ----------
    X, y : the design matrix (2-D array / DataFrame) and target.
    estimator : any sklearn estimator with ``feature_importances_`` or ``coef_`` (default: a
        RandomForest sized to the task). Cloned per replicate; never mutated.
    n_replicates : independent contrast draws whose importances feed the per-feature t-test. Tuv uses
        ~20-50; the t-test needs >= 2 (>= 5 for a stable variance estimate).
    contrast_percentile : the pooled-contrast percentile a real importance must exceed (default 100 = the
        Boruta MAX-contrast gate). Lower it (e.g. 95) for higher recall.
    alpha : significance level for the (optionally BH-corrected) per-feature test.
    importance : 'native' (in-bag impurity / |coef|) or 'permutation' (held-out PFI, unbiased vs impurity's
        high-cardinality skew - Дьяконов slide 11).
    n_masking_rounds : masking-removal passes. After each pass accepted features are removed and the
        procedure repeats on the remainder so features masked by a stronger correlate get another chance.
        1 disables the loop.
    fdr_control : Benjamini-Hochberg FDR correction across the feature battery (recommended).

    Returns
    -------
    ACEResult with per-feature mean importance, contrast bar, p-value, accept mask, and selected names.
    """
    if hasattr(X, "to_numpy"):
        names = list(X.columns) if feature_names is None and hasattr(X, "columns") else feature_names
        X_arr = X.to_numpy(dtype=np.float64, copy=True)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        names = feature_names
    y_arr = np.asarray(y)

    if X_arr.ndim != 2:
        raise ValueError(f"ACE expects 2-D X, got shape {X_arr.shape}.")
    n, p = X_arr.shape
    if names is None:
        names = [f"x{i}" for i in range(p)]
    names = list(names)
    if len(names) != p:
        raise ValueError(f"feature_names length {len(names)} != n_features {p}.")
    if n_replicates < 2:
        raise ValueError(f"ACE needs n_replicates >= 2 for the t-test, got {n_replicates}.")

    estimator = estimator if estimator is not None else _default_estimator(y_arr, n, random_state)

    imp_mean = np.zeros(p, dtype=np.float64)
    contrast_bar = np.full(p, np.nan, dtype=np.float64)
    p_values = np.ones(p, dtype=np.float64)
    accepted = np.zeros(p, dtype=bool)

    remaining = np.arange(p)  # column indices still competing (masking loop shrinks this)
    for _round in range(max(1, n_masking_rounds)):
        if remaining.size == 0:
            break
        idx = remaining
        real_imps, thr = _run_ace_round(
            estimator, X_arr[:, idx], y_arr, n_replicates=n_replicates, contrast_percentile=contrast_percentile,
            importance=importance, n_perm_repeats=n_perm_repeats, random_state=random_state + _round * 1000,
        )
        pvals_round = _ttest_greater(real_imps, thr)
        mean_round = real_imps.mean(axis=0)

        if fdr_control:
            acc_round = _benjamini_hochberg_reject(pvals_round, alpha)
        else:
            acc_round = pvals_round < alpha

        imp_mean[idx] = mean_round
        contrast_bar[idx] = thr
        p_values[idx] = pvals_round
        accepted[idx] |= acc_round

        newly = idx[acc_round]
        if newly.size == 0:
            break  # nothing new cleared the bar; masking cannot expose more
        remaining = idx[~acc_round]

    selected = [names[i] for i in range(p) if accepted[i]]
    return ACEResult(
        feature_names=names,
        importances_mean=imp_mean,
        contrast_threshold=contrast_bar,
        p_values=p_values,
        accepted=accepted,
        selected_features=selected,
    )


def _run_ace_round(
    estimator, X: np.ndarray, y: np.ndarray, *, n_replicates: int, contrast_percentile: float, importance: str,
    n_perm_repeats: int, random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One ACE pass over the given columns. Returns (real_imps [n_replicates x p], per-feature contrast bar).

    The per-feature bar is the ``contrast_percentile`` of the ENTIRE pooled contrast importances across all
    replicates and all contrast columns - a single scalar broadcast to every feature (Tuv 2009: the null is
    the distribution of importance for a variable known to be irrelevant, estimated from every contrast)."""
    p = X.shape[1]
    real_imps = np.empty((n_replicates, p), dtype=np.float64)
    contrast_pool: list[np.ndarray] = []
    for r in range(n_replicates):
        rng = np.random.default_rng(random_state + r)
        real_r, contrast_r = _one_replicate_importances(estimator, X, y, importance=importance, n_perm_repeats=n_perm_repeats, rng=rng, seed=random_state + r)
        real_imps[r] = real_r
        contrast_pool.append(contrast_r)
    pooled = np.concatenate(contrast_pool)
    bar = float(np.nanpercentile(pooled, contrast_percentile)) if pooled.size else 0.0
    return real_imps, np.full(p, bar, dtype=np.float64)


def _ttest_greater(real_imps: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """One-sample, one-sided t-test p-value that each feature's importance mean exceeds its contrast bar.

    ``real_imps`` is [n_replicates x p]; ``threshold`` is [p]. Returns [p] p-values for H1: mean > bar.
    Zero-variance columns (all replicates identical) map to p=0 when the mean beats the bar, else p=1."""
    from scipy.stats import t as _t

    x = real_imps - threshold[None, :]
    n = x.shape[0]
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    p = np.ones(x.shape[1], dtype=np.float64)
    # Zero-dispersion features: the t-statistic is degenerate; decide by the sign of the mean margin.
    zero_var = sd <= 1e-15
    p[zero_var & (mean > 0)] = 0.0
    p[zero_var & (mean <= 0)] = 1.0
    live = ~zero_var
    if live.any():
        se = sd[live] / np.sqrt(n)
        t_stat = mean[live] / se
        p[live] = _t.sf(t_stat, df=n - 1)  # P(T >= t) under H0: margin mean == 0
    return p


class ACESelector(BaseEstimator, TransformerMixin):
    """sklearn-compatible adapter over :func:`ace_select` for the training suite's pre-pipeline slot.

    ``ace_select`` is a FUNCTION returning ``ACEResult``; the suite drives selectors via the sklearn
    fit / get_support / transform contract (mirrors ShapProxiedFS). ``fit`` runs ACE once and materialises
    ``support_`` / ``selected_features_`` / ``feature_names_in_`` in INPUT-column order (``ACEResult`` arrays
    are already aligned to the input columns); ``transform`` narrows to the accepted columns positionally
    (no full-frame copy). ``classification`` is auto-derived inside ``ace_select`` from the target dtype, so
    no target_type threading is needed (unlike BorutaShap / ShapProxiedFS).
    """

    def __init__(
        self,
        estimator=None,
        *,
        n_replicates: int = 20,
        contrast_percentile: float = _DEFAULT_CONTRAST_PERCENTILE,
        alpha: float = 0.05,
        importance: str = "native",
        n_masking_rounds: int = 3,
        n_perm_repeats: int = 5,
        fdr_control: bool = True,
        random_state: int = 0,
    ):
        self.estimator = estimator
        self.n_replicates = n_replicates
        self.contrast_percentile = contrast_percentile
        self.alpha = alpha
        self.importance = importance
        self.n_masking_rounds = n_masking_rounds
        self.n_perm_repeats = n_perm_repeats
        self.fdr_control = fdr_control
        self.random_state = random_state

    def fit(self, X, y=None):
        """Run ``ace_select`` once and materialise the sklearn selector attributes (``support_``,
        ``selected_features_``, ``feature_names_in_``) in input-column order."""
        result = ace_select(
            X, y, estimator=self.estimator,
            n_replicates=self.n_replicates, contrast_percentile=self.contrast_percentile,
            alpha=self.alpha, importance=self.importance, n_masking_rounds=self.n_masking_rounds,
            n_perm_repeats=self.n_perm_repeats, fdr_control=self.fdr_control, random_state=self.random_state,
        )
        self.ace_result_ = result
        self.feature_names_in_ = np.asarray([str(c) for c in result.feature_names], dtype=object)
        self.n_features_in_ = len(result.feature_names)
        self.support_ = np.asarray(result.accepted, dtype=bool)
        self.selected_features_ = [str(c) for c in result.selected_features]
        return self

    def transform(self, X):
        """Narrow ``X`` to the accepted columns (positional slice, no full-frame copy)."""
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError("ACESelector.transform called before fit.")
        idx = np.where(self.support_)[0]
        if hasattr(X, "iloc"):
            return X.iloc[:, idx]
        return np.asarray(X)[:, self.support_]

    def fit_transform(self, X, y=None, **fit_params):
        """Convenience ``fit`` then ``transform`` in one call (fit_params accepted for sklearn API parity, unused)."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        """sklearn-style accepted-feature mask; returns positional indices instead when ``indices=True``."""
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError("ACESelector.get_support called before fit.")
        return np.where(self.support_)[0] if indices else self.support_

    def get_feature_names_out(self, input_features=None):
        """Names of the accepted features, in the order ``ace_select`` returned them (``input_features`` unused, kept for sklearn API parity)."""
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError("ACESelector.get_feature_names_out called before fit.")
        return np.asarray(self.selected_features_, dtype=object)


def _default_estimator(y: np.ndarray, n: int, random_state: int):
    """Task-appropriate RandomForest default: classifier for low-cardinality integer/label y, else regressor.

    Дьяконов recommends ensembles of trees for importance; a modest forest converges the importances
    (slide 19) without the runtime of a production fit."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    n_estimators = 120
    is_classification = False
    if y.dtype.kind in ("i", "u", "b", "O", "U", "S"):
        is_classification = np.unique(y).size <= max(20, int(np.sqrt(n)))
    if is_classification:
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state, max_features="sqrt")
    return RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state, max_features="sqrt")
