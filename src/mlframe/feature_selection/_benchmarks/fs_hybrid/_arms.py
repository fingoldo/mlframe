"""Phase-0 benchmark arms, each answering with an :class:`~._arm_result.ArmResult`.

Each arm is a small object with ``name``, a DECLARED ``score_kind`` and a ``run(X, y)`` method. The base
class times the fit (wall + process), builds the :class:`ArmResult` and lets its ``__post_init__`` enforce
the support/score contract -- so an arm that declares ``continuous`` and loses its score RAISES instead of
degrading to ``"none"``.

Support extraction reuses the public :func:`mlframe.feature_selection.extract_selected` /
:func:`mlframe.feature_selection.support_mask_from_selector` (promoted in commit ``5724effc1``), which
already handle every accessor shape in the package including MRMR's ``np.int64`` INDEX ``support_``. No
forked mask logic lives here.

The roster spans the paradigms Phase 0 needs a verdict on:

* ``all-features`` -- the NULL HYPOTHESIS every other arm is measured against;
* ``random-k`` / ``variance-sort`` -- controls (recall is uninterpretable without a same-cardinality
  random draw, and variance-sort is the varsortability tripwire);
* ``univariate-mi`` -- ``estimate_features_relevancy``, the univariate baseline the bench had none of;
* ``boruta`` / ``ace`` / ``knockoffs`` -- mlframe's all-relevant and FDR machinery;
* ``skb-f`` / ``skb-mi`` / ``select-fdr`` / ``sfm-lgbm`` / ``lars-order`` -- EXTERNAL sklearn anchors,
  without which no "mlframe wins" claim is falsifiable;
* ``mrmr`` / ``rfecv`` / ``boruta-shap`` / ``shap-proxied`` -- mlframe selectors driven BARE (not through
  ``feature_selection.registry``, whose factories wrap RFECV and BorutaShap in a cluster-medoid
  ``GroupAwareMRMR(expand=True)`` by default).
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from mlframe.feature_selection import extract_selected, support_mask_from_selector

from ._arm_result import ArmResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------- fit counting
# `n_model_fits` is the PRE-REGISTERED PRIMARY cost axis, chosen because it is deterministic while
# wall-clock on this contended host is not. The wrapper arms -- rfecv, boruta, boruta-shap, shap-proxied --
# are precisely the arms whose cost the axis exists to measure, and precisely the arms whose underlying
# selector publishes no fit count: each of them fans an estimator out internally and reports only its
# selection. Reporting `None` there and letting `cost_table` skip the row hides the expensive arms from the
# cost table entirely, which is worse than useless -- a reader compares the cheap arms and concludes
# wrappers are free.
#
# So the count is MEASURED rather than asked for: the estimator classes these selectors fit are patched for
# the duration of `_compute` and every `fit` call is tallied. Verified applicable here because all four
# wrapper selectors fan out over THREADS (`RFECV._fit_outer_loop` uses `Parallel(prefer="threads")`,
# BorutaShap's sub-fit loop uses `backend="threading"`, ShapProxiedFS's fold loop uses `prefer="threads"`),
# so every fit happens in this process and is visible to the patch. A selector that fanned out over
# PROCESSES would under-count, which is why the tally is published with its method in `provenance` instead
# of as a bare number.
_COUNTED_ESTIMATORS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("sklearn.ensemble", ("RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor", "IsolationForest")),
    ("sklearn.ensemble", ("GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier", "HistGradientBoostingRegressor")),
    ("sklearn.linear_model", ("Ridge", "Lasso", "LogisticRegression", "LinearRegression", "SGDClassifier")),
    ("sklearn.dummy", ("DummyClassifier", "DummyRegressor")),
    ("lightgbm", ("LGBMClassifier", "LGBMRegressor")),
    ("xgboost", ("XGBClassifier", "XGBRegressor")),
    ("catboost", ("CatBoostClassifier", "CatBoostRegressor")),
)

#: `provenance["n_model_fits_source"]` values. Each one is a different claim about the same empty-looking
#: number, and they are never collapsed: a MEASURED zero means "this arm fits no model", while
#: `not_measured` means nobody counted.
FITS_SOURCE_COUNTED = "counted_in_process_estimator_fits"
FITS_SOURCE_DECLARED = "declared_by_the_adapter"
FITS_SOURCE_NOT_MEASURED = "not_measured"


def _resolve_counted_classes() -> Tuple[List[Type[Any]], List[str]]:
    """Import the estimator classes to instrument, returning `(classes, names_that_could_not_be_imported)`."""
    classes: List[Type[Any]] = []
    missing: List[str] = []
    for module_name, class_names in _COUNTED_ESTIMATORS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.extend(f"{module_name}.{name}" for name in class_names)
            continue
        for name in class_names:
            obj = getattr(module, name, None)
            if isinstance(obj, type) and callable(getattr(obj, "fit", None)):
                classes.append(obj)
            else:
                missing.append(f"{module_name}.{name}")
    return classes, missing


class FitCounter:
    """Context manager tallying every in-process estimator `fit` call made inside its body.

    Attributes:
        by_class: `{qualified class name: call count}`, populated on exit.
        missing: Estimator classes that could not be imported, so a reader can see the counter's blind spots.
    """

    def __init__(self) -> None:
        self.by_class: Dict[str, int] = {}
        self.missing: List[str] = []
        self._lock = threading.Lock()
        self._restore: List[Tuple[Type[Any], bool, Any]] = []

    @property
    def total(self) -> int:
        """Total number of counted `fit` calls."""
        return int(sum(self.by_class.values()))

    @property
    def instrumented(self) -> bool:
        """True when at least one estimator class was successfully patched, i.e. the tally means something."""
        return bool(self._restore)

    def _wrap(self, cls: Type[Any], original: Any) -> Callable[..., Any]:
        """Build the counting replacement for one class's `fit`."""
        label = f"{cls.__module__.split('.')[0]}.{cls.__name__}"

        def counting_fit(estimator: Any, *args: Any, **kwargs: Any) -> Any:
            """Tally this call, then delegate to the estimator's real `fit`."""
            with self._lock:
                self.by_class[label] = self.by_class.get(label, 0) + 1
            return original(estimator, *args, **kwargs)

        return counting_fit

    def __enter__(self) -> "FitCounter":
        """Patch every resolvable estimator class's `fit` with a counting wrapper."""
        classes, self.missing = _resolve_counted_classes()
        for cls in classes:
            # Recorded as "was it in this class's OWN __dict__" so an inherited `fit` is removed rather than
            # pinned onto the subclass on exit, which would leave the class permanently shadowed.
            own = "fit" in cls.__dict__
            self._restore.append((cls, own, cls.__dict__.get("fit")))
            try:
                cls.fit = self._wrap(cls, cls.fit)
            except (AttributeError, TypeError):  # an immutable/extension type refuses the patch
                self._restore.pop()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        """Restore every patched `fit`, leaving no class permanently instrumented."""
        for cls, own, original in reversed(self._restore):
            if own:
                cls.fit = original
            else:
                try:
                    delattr(cls, "fit")
                except AttributeError:
                    pass


# ------------------------------------------------------------------------------------------------- helpers
def _feature_names(X: pd.DataFrame) -> List[str]:
    """Column names of ``X`` as plain ``str``, in input order."""
    return [str(c) for c in X.columns]


def _mask_from_names(names: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    """Boolean mask over ``names``, True for every entry of ``selected``; unknown names raise."""
    position = {str(n): i for i, n in enumerate(names)}
    mask = np.zeros(len(names), dtype=bool)
    missing = []
    for name in selected:
        idx = position.get(str(name))
        if idx is None:
            missing.append(str(name))
        else:
            mask[idx] = True
    if missing:
        raise ValueError(f"arm reported {len(missing)} selected name(s) absent from the input columns: {missing[:10]}")
    return mask


def _topk_mask(score: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask marking the ``k`` highest entries of ``score`` (ties broken by index, ascending)."""
    k = int(max(0, min(int(k), score.shape[0])))
    mask = np.zeros(score.shape[0], dtype=bool)
    if k:
        order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
        mask[order[:k]] = True
    return mask


def _integer_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Rank-based integer bin codes (0..n_bins-1) for one numeric column; constant columns yield all zeros."""
    v = np.asarray(values, dtype=np.float64)
    uniq = np.unique(v[np.isfinite(v)])
    if uniq.size <= 1:
        return np.zeros(v.shape[0], dtype=np.int64)
    if uniq.size <= n_bins:
        return np.searchsorted(uniq, v).astype(np.int64)
    ranks = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    return np.asarray(np.minimum((ranks * n_bins) // max(1, v.shape[0]), n_bins - 1), dtype=np.int64)


# ------------------------------------------------------------------------------------------------- base
class BaseArm:
    """Common timing/validation wrapper: subclasses implement ``_compute`` and declare ``score_kind``."""

    name = "base"
    score_kind = "none"

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the underlying selector and return the ``ArmResult`` payload fields (minus timings).

        Returns:
            A dict with keys ``support`` (required) and optionally ``score``, ``ranked_prefix``,
            ``selection_score``, ``n_model_fits``, ``provenance``.
        """
        raise NotImplementedError

    def run(self, X: pd.DataFrame, y: np.ndarray) -> ArmResult:
        """Fit the arm on ``(X, y)`` and return a validated :class:`ArmResult`."""
        t0, p0 = time.perf_counter(), time.process_time()
        with FitCounter() as counter:
            payload = self._compute(X, y)
        wall, proc = time.perf_counter() - t0, time.process_time() - p0
        support = np.asarray(payload["support"], dtype=bool)
        score = payload.get("score")
        if score is not None:
            score = np.asarray(score, dtype=np.float64)
        provenance = dict(payload.get("provenance") or {})
        provenance.setdefault("arm", self.name)
        provenance.setdefault("declared_score_kind", self.score_kind)
        ranked = payload.get("ranked_prefix")
        n_model_fits = self._resolve_fit_count(payload.get("n_model_fits"), counter, provenance)
        return ArmResult(
            support=support,
            score=score,
            score_kind=self.score_kind,  # type: ignore[arg-type]
            ranked_prefix=None if ranked is None else tuple(int(i) for i in ranked),
            n_features_selected=int(support.sum()),
            selection_score=payload.get("selection_score"),
            wall_time_s=float(wall),
            process_time_s=float(proc),
            n_model_fits=n_model_fits,
            provenance=provenance,
        )

    @staticmethod
    def _resolve_fit_count(declared: Optional[int], counter: "FitCounter", provenance: Dict[str, Any]) -> Optional[int]:
        """Pick the fit count to publish and record HOW it was obtained, so no reader has to guess.

        The measurement wins over the adapter's declaration whenever the counter was live: a declared
        number is a parameter (`n_trials`, `n_iterations`) and an arm that converges early pays less than
        it asked for. The declaration is kept in `provenance` alongside, and any gap between the two is
        itself informative. When nothing could be instrumented the count is `None` WITH a reason -- never a
        bare `None` a reader could mistake for "this arm is free".

        Args:
            declared: The count the arm adapter asserted, if any.
            counter: The (already exited) fit counter that wrapped `_compute`.
            provenance: Provenance dict, mutated in place with the source, the tally and the caveats.

        Returns:
            The fit count to publish, or `None` when it genuinely could not be measured or declared.
        """
        provenance["n_model_fits_declared"] = declared
        provenance["n_model_fits_counted"] = counter.total if counter.instrumented else None
        provenance["n_model_fits_by_class"] = dict(counter.by_class)
        if counter.missing:
            provenance["n_model_fits_uninstrumented_classes"] = list(counter.missing)
        if counter.instrumented:
            provenance["n_model_fits_source"] = FITS_SOURCE_COUNTED
            provenance["n_model_fits_caveat"] = (
                "in-process estimator fits only; a selector fanning out over PROCESSES would under-count. "
                "Every wrapper arm in this roster fans out over threads, so the tally is complete for them."
            )
            return counter.total
        if declared is not None:
            provenance["n_model_fits_source"] = FITS_SOURCE_DECLARED
            return int(declared)
        provenance["n_model_fits_source"] = FITS_SOURCE_NOT_MEASURED
        provenance["n_model_fits_reason"] = "no estimator class could be instrumented and the adapter declares no count: this cell's cost is UNMEASURED, not zero"
        return None


# ------------------------------------------------------------------------------------------------- P0 baselines / controls
class AllFeaturesArm(BaseArm):
    """The NULL HYPOTHESIS: keep every feature. No ranking, so ``score_kind='none'`` (not a flat score)."""

    name = "all-features"
    score_kind = "none"

    def _compute(self, X, y):
        """Return an all-True support of width ``X.shape[1]``."""
        return {"support": np.ones(X.shape[1], dtype=bool), "provenance": {"n_features_in": int(X.shape[1])}}


class RandomSelectionArm(BaseArm):
    """Uniform random subset of a caller-given cardinality: the control that makes recall interpretable."""

    score_kind = "none"

    def __init__(self, k: int, random_state: int = 0):
        self.k = int(k)
        self.random_state = int(random_state)
        self.name = f"random-{self.k}"

    def _compute(self, X, y):
        """Draw ``k`` distinct column positions without replacement from a seeded generator."""
        p = int(X.shape[1])
        k = min(self.k, p)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(p, size=k, replace=False)
        support = np.zeros(p, dtype=bool)
        support[idx] = True
        return {"support": support, "provenance": {"k": k, "random_state": self.random_state}}


class VarianceSortArm(BaseArm):
    """Rank by marginal variance -- the varsortability tripwire. Continuous, full coverage, zero model fits."""

    score_kind = "continuous"

    def __init__(self, k: Optional[int] = None, standardize: bool = False):
        self.k = k
        self.standardize = bool(standardize)
        self.name = "variance-sort" if k is None else f"variance-sort-{int(k)}"

    def _compute(self, X, y):
        """Score every column by its (optionally scale-normalised) variance and keep the top ``k``."""
        arr = np.asarray(X.to_numpy(), dtype=np.float64)
        score = np.nan_to_num(np.var(arr, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        if self.standardize:
            denom = np.abs(np.nan_to_num(np.mean(arr, axis=0), nan=0.0)) + 1.0
            score = score / denom
        k = arr.shape[1] if self.k is None else int(self.k)
        return {"support": _topk_mask(score, k), "score": score, "n_model_fits": 0, "provenance": {"k": k, "standardize": self.standardize}}


# ------------------------------------------------------------------------------------------------- P2 univariate
class UnivariateMIArm(BaseArm):
    """``estimate_features_relevancy`` (MI + Miller-Madow debias + G-test + BH) -- the univariate baseline.

    The function takes an INTEGER-BINS polars frame whose last column is the target, and returns
    ``(columns_to_drop, original_mi_results, all_permuted_mis, mi_algorithms_ranking)``; the raw per-feature
    MI row of ``original_mi_results`` is a genuine continuous score with full coverage.
    """

    name = "univariate-mi"
    score_kind = "continuous"

    def __init__(self, n_bins: int = 10, min_randomized_permutations: int = 1, max_runtime_mins: float = 1.0, random_state: int = 0):
        self.n_bins = int(n_bins)
        self.min_randomized_permutations = int(min_randomized_permutations)
        self.max_runtime_mins = float(max_runtime_mins)
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Bin features and target to integer codes, run the relevancy estimate, invert the drop list."""
        import polars as pl

        from mlframe.feature_selection.general import estimate_features_relevancy

        names = _feature_names(X)
        data = {name: _integer_bins(X.iloc[:, j].to_numpy(), self.n_bins) for j, name in enumerate(names)}
        target_name = "__target__"
        data[target_name] = np.asarray(pd.factorize(np.asarray(y))[0], dtype=np.int64)
        bins = pl.DataFrame(data)
        dropped, mi_results, _permuted, _ranking = estimate_features_relevancy(
            bins=bins,
            target_columns=[target_name],
            benchmark_mi_algorithms=False,
            min_randomized_permutations=self.min_randomized_permutations,
            max_runtime_mins=self.max_runtime_mins,
            random_state=self.random_state,
            verbose=0,
        )
        mi_row = np.asarray(np.asarray(mi_results)[0], dtype=np.float64)[: len(names)]
        score = np.nan_to_num(mi_row, nan=0.0, posinf=0.0, neginf=0.0)
        dropped_names = {str(c) for c in (dropped or [])}
        support = np.array([n not in dropped_names for n in names], dtype=bool)
        return {
            "support": support,
            "score": score,
            "n_model_fits": 0,
            "provenance": {"n_bins": self.n_bins, "n_dropped": len(dropped_names), "min_randomized_permutations": self.min_randomized_permutations},
        }


class SklearnScoreArm(BaseArm):
    """External sklearn univariate anchors: ``SelectKBest(f_classif|mutual_info_classif)`` and ``SelectFdr``.

    ``scores_`` is a full-length continuous statistic in every case, so the declared kind holds for all
    three variants.
    """

    score_kind = "continuous"

    def __init__(self, variant: str, k: int = 10, alpha: float = 0.05, random_state: int = 0):
        if variant not in ("kbest_f", "kbest_mi", "fdr_f"):
            raise ValueError(f"unknown SklearnScoreArm variant {variant!r}")
        self.variant = variant
        self.k = int(k)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.name = {"kbest_f": "skb-f", "kbest_mi": "skb-mi", "fdr_f": "select-fdr"}[variant]

    def _compute(self, X, y):
        """Fit the sklearn selector and read its ``scores_`` / ``get_support()`` pair."""
        from functools import partial

        from sklearn.feature_selection import SelectFdr, SelectKBest, f_classif, mutual_info_classif

        p = int(X.shape[1])
        if self.variant == "kbest_f":
            sel: Any = SelectKBest(f_classif, k=min(self.k, p))
        elif self.variant == "kbest_mi":
            sel = SelectKBest(partial(mutual_info_classif, random_state=self.random_state), k=min(self.k, p))
        else:
            sel = SelectFdr(f_classif, alpha=self.alpha)
        sel.fit(np.asarray(X.to_numpy(), dtype=np.float64), np.asarray(y))
        score = np.nan_to_num(np.asarray(sel.scores_, dtype=np.float64), nan=0.0, posinf=float(np.finfo(np.float64).max), neginf=0.0)
        return {"support": np.asarray(sel.get_support(), dtype=bool), "score": score, "n_model_fits": 0, "provenance": {"variant": self.variant, "k": self.k, "alpha": self.alpha}}


# ------------------------------------------------------------------------------------------------- P4 embedded
class SelectFromModelArm(BaseArm):
    """``SelectFromModel`` over a LightGBM classifier: external embedded-importance anchor."""

    name = "sfm-lgbm"
    score_kind = "continuous"

    def __init__(self, threshold: Optional[Any] = "mean", n_estimators: int = 100, random_state: int = 0):
        self.threshold = threshold
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Fit LGBM once and threshold its split-gain importances."""
        import lightgbm as lgb
        from sklearn.feature_selection import SelectFromModel

        est = lgb.LGBMClassifier(n_estimators=self.n_estimators, verbose=-1, random_state=self.random_state)
        sel = SelectFromModel(est, threshold=self.threshold)
        sel.fit(np.asarray(X.to_numpy(), dtype=np.float64), np.asarray(y))
        score = np.asarray(sel.estimator_.feature_importances_, dtype=np.float64)
        return {"support": np.asarray(sel.get_support(), dtype=bool), "score": score, "n_model_fits": 1, "provenance": {"threshold": str(self.threshold), "n_estimators": self.n_estimators}}


class LarsPathArm(BaseArm):
    """Regularization-path ENTRY ORDER (``lars_path`` / ``lasso_path``): the first embedded-path arm.

    Features that never enter the path have no score at all, so the honest kind is ``selection_order``:
    the order is published in ``ranked_prefix`` and no score vector is fabricated for the non-entrants.
    """

    score_kind = "selection_order"

    def __init__(self, method: str = "lasso", max_features: Optional[int] = None):
        if method not in ("lasso", "lar"):
            raise ValueError(f"LarsPathArm.method must be 'lasso' or 'lar'; got {method!r}")
        self.method = method
        self.max_features = max_features
        self.name = f"lars-order-{method}"

    def _compute(self, X, y):
        """Walk the path's coefficient matrix and record the column index order of first activation."""
        from sklearn.linear_model import lars_path

        arr = np.asarray(X.to_numpy(), dtype=np.float64)
        arr = arr - arr.mean(axis=0)
        std = arr.std(axis=0)
        arr = arr / np.where(std > 0, std, 1.0)
        target = np.asarray(y, dtype=np.float64)
        target = target - target.mean()
        _alphas, _active, coefs = lars_path(arr, target, method=self.method)
        entered: List[int] = []
        seen = set()
        for step in range(coefs.shape[1]):
            for j in np.flatnonzero(coefs[:, step] != 0.0):
                jj = int(j)
                if jj not in seen:
                    seen.add(jj)
                    entered.append(jj)
        k = len(entered) if self.max_features is None else min(int(self.max_features), len(entered))
        prefix = tuple(entered[:k])
        support = np.zeros(arr.shape[1], dtype=bool)
        if prefix:
            support[list(prefix)] = True
        return {"support": support, "ranked_prefix": prefix, "n_model_fits": 1, "provenance": {"method": self.method, "n_entered": len(entered), "n_path_steps": int(coefs.shape[1])}}


# ------------------------------------------------------------------------------------------------- P6 all-relevant
class BorutaArm(BaseArm):
    """``filters._boruta.boruta_select``: shadow-contrast all-relevant selection.

    ``win_rate`` is reported for EVERY input feature (not just the confirmed ones), so this arm ranks
    continuously; ``decision == "confirmed"`` defines the support.
    """

    name = "boruta"
    score_kind = "continuous"

    def __init__(self, n_iterations: int = 20, alpha: float = 0.05, n_estimators: int = 60, random_state: int = 0, include_tentative: bool = False):
        self.n_iterations = int(n_iterations)
        self.alpha = float(alpha)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.include_tentative = bool(include_tentative)

    def _importance_fn(self) -> Callable[[Any, np.ndarray], np.ndarray]:
        """Build the ``(X_with_shadows, y) -> importances`` callback boruta_select expects."""
        from sklearn.ensemble import RandomForestClassifier

        n_estimators, random_state = self.n_estimators, self.random_state

        def importance_fn(mat: Any, target: np.ndarray) -> np.ndarray:
            """Fit a small random forest on the real+shadow matrix and return impurity importances."""
            model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
            model.fit(np.asarray(mat, dtype=np.float64), np.asarray(target))
            return np.asarray(model.feature_importances_, dtype=np.float64)

        return importance_fn

    def _compute(self, X, y):
        """Run ``boruta_select`` and turn its decision list into a support and its win_rate into a score."""
        from mlframe.feature_selection.filters._boruta import boruta_select

        names = _feature_names(X)
        out = boruta_select(
            np.asarray(X.to_numpy(), dtype=np.float64),
            np.asarray(y),
            importance_fn=self._importance_fn(),
            feature_names=names,
            n_iterations=self.n_iterations,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        decisions = list(out["decision"])
        keep = {"confirmed", "tentative"} if self.include_tentative else {"confirmed"}
        reported = [str(c) for c in out["feature_names"]]
        selected = [reported[i] for i, d in enumerate(decisions) if d in keep]
        score_by_name = {reported[i]: float(v) for i, v in enumerate(np.asarray(out["win_rate"], dtype=np.float64))}
        score = np.array([score_by_name[n] for n in names], dtype=np.float64)
        return {
            "support": _mask_from_names(names, selected),
            "score": score,
            "n_model_fits": int(out.get("n_rounds_run", self.n_iterations)),
            "provenance": {"n_rounds_run": int(out.get("n_rounds_run", 0)), "n_tentative": sum(1 for d in decisions if d == "tentative")},
        }


class ACEArm(BaseArm):
    """``ace.ACESelector``: contrast-percentile all-relevant selection with a full-length score.

    The score is ``importances_mean - contrast_threshold`` (the margin over the feature's OWN contrast
    bar), NOT raw importance: raw impurity importance is confounded with cardinality bias, so ranking on
    it would measure a different statistic than the selection rule does.
    """

    name = "ace"
    score_kind = "continuous"

    def __init__(self, n_replicates: int = 5, n_masking_rounds: int = 2, n_perm_repeats: int = 3, alpha: float = 0.05, random_state: int = 0):
        self.n_replicates = int(n_replicates)
        self.n_masking_rounds = int(n_masking_rounds)
        self.n_perm_repeats = int(n_perm_repeats)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Fit ACESelector and read ``ace_result_`` for the contrast-margin score."""
        from mlframe.feature_selection.ace import ACESelector

        names = _feature_names(X)
        sel = ACESelector(
            n_replicates=self.n_replicates,
            n_masking_rounds=self.n_masking_rounds,
            n_perm_repeats=self.n_perm_repeats,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        sel.fit(X, np.asarray(y))
        res = sel.ace_result_
        margin = np.asarray(res.importances_mean, dtype=np.float64) - np.asarray(res.contrast_threshold, dtype=np.float64)
        score_by_name = {str(c): float(v) for c, v in zip(res.feature_names, margin)}
        score = np.array([score_by_name[n] for n in names], dtype=np.float64)
        return {
            "support": support_mask_from_selector(sel, names),
            "score": np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0),
            "provenance": {"n_replicates": self.n_replicates, "fdr_control": True},
        }


class KnockoffArm(BaseArm):
    """``wrappers._knockoffs``: ``knockoff_importance`` -> ``select_features_fdr`` at target FDR ``q``.

    ``knockoff_importance`` answers a ``{name: W_j}`` dict covering every feature, so ``W`` is the
    continuous score; the Barber-Candes threshold defines the support (legitimately empty on weak signal).
    """

    name = "knockoffs"
    score_kind = "continuous"

    def __init__(self, q: float = 0.2, n_estimators: int = 100, random_state: int = 0):
        self.q = float(q)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Build Gaussian knockoffs, difference the importances, and threshold at FDR ``q``."""
        from mlframe.feature_selection.wrappers import knockoff_importance, select_features_fdr

        names = _feature_names(X)
        n_estimators, random_state = self.n_estimators, self.random_state

        def model_factory():
            """Fresh small random forest for the 2p-column real+knockoff fit."""
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)

        w_stats = knockoff_importance(model_factory, X, np.asarray(y), random_state=self.random_state)
        missing = [n for n in names if n not in w_stats]
        if missing:
            raise ValueError(f"knockoff_importance returned no W statistic for {len(missing)} feature(s) {missing[:10]}; a 'continuous' arm may not be padded.")
        score = np.array([float(w_stats[n]) for n in names], dtype=np.float64)
        selected = select_features_fdr(w_stats, q=self.q)
        return {"support": _mask_from_names(names, selected), "score": score, "n_model_fits": 1, "provenance": {"q": self.q, "n_selected": len(selected)}}


# ------------------------------------------------------------------------------------------------- mlframe selectors, BARE
class MRMRArm(BaseArm):
    """MRMR driven BARE (not via ``registry``, which wraps it in nothing but still differs in defaults).

    ``support_`` holds ``np.int64`` POSITIONS in greedy-selection order and the non-selected features have
    no score at all, hence ``selection_order``. ``max_runtime_mins`` is passed EXPLICITLY: with
    ``fe_synergy_exhaustive="auto"`` and no runtime budget, the FE synergy stage escalates to an unbudgeted
    full ``C(p, 2)`` sweep.
    """

    def __init__(self, fe: bool = False, max_runtime_mins: float = 2.0, random_seed: int = 0):
        self.fe = bool(fe)
        self.max_runtime_mins = float(max_runtime_mins)
        self.random_seed = int(random_seed)
        self.name = "mrmr-fe" if fe else "mrmr"

    score_kind = "selection_order"

    def _compute(self, X, y):
        """Fit MRMR and read its selection-order ``support_`` through the shared extractor."""
        from mlframe.feature_selection.filters import MRMR

        names = _feature_names(X)
        model = MRMR(verbose=0, fe_max_steps=(1 if self.fe else 0), n_jobs=-1, random_seed=self.random_seed, max_runtime_mins=self.max_runtime_mins)
        model.fit(X, pd.Series(np.asarray(y)))
        raw_support = np.asarray(getattr(model, "support_", np.zeros(0)))
        if raw_support.dtype == np.bool_:
            prefix = tuple(int(i) for i in np.flatnonzero(raw_support))
        else:
            prefix = tuple(int(i) for i in raw_support if 0 <= int(i) < len(names))
        selected = [str(c) for c in extract_selected(model, names)]
        known = [c for c in selected if c in set(names)]
        support = _mask_from_names(names, known)
        if not prefix:
            prefix = tuple(int(i) for i in np.flatnonzero(support))
        return {
            "support": support,
            "ranked_prefix": prefix,
            "provenance": {"fe": self.fe, "max_runtime_mins": self.max_runtime_mins, "n_engineered_reported": len(selected) - len(known), "support_dtype": str(raw_support.dtype)},
        }


def _rfecv_rank_vector(model: Any, names: Sequence[str], selected: Sequence[str]) -> np.ndarray:
    """Rank vector for an mlframe RFECV, tolerating the three shapes its ``ranking_`` actually takes.

    The class docstring promises sklearn's contract -- an integer array where ``ranking_[i]`` is the i-th
    feature's rank and every survivor is 1 -- but the attribute is set from
    ``_rank_features_by_importance``, which returns an ordered list of feature NAMES, and on the ordinary
    (p < n) path it is not set at all. Reading it blindly as float raised
    ``could not convert string to float: 'f453'`` on every wide bed. All three shapes are handled here
    rather than in a guard at the call site, because the shape depends on which internal branch the fit
    took and is not knowable in advance.

    Args:
        model: A fitted RFECV.
        names: The input feature names, in column order.
        selected: The names the selector kept, already intersected with ``names``.

    Returns:
        A float rank vector aligned to ``names``, lower meaning better, never containing NaN.
    """
    raw = getattr(model, "ranking_", None)
    if raw is not None and len(np.asarray(raw, dtype=object).ravel()) > 0:
        arr = np.asarray(raw, dtype=object).ravel()
        if np.issubdtype(np.asarray(raw).dtype, np.number) and arr.shape[0] == len(names):
            return np.asarray(raw, dtype=np.float64)
        order = {str(nm): pos for pos, nm in enumerate(arr)}
        if set(order) & set(names):
            # An ordered NAME list: position in it IS the rank, and anything it omits ranks after all of it.
            return np.asarray([float(order.get(str(nm), len(order))) for nm in names], dtype=np.float64)
    # Attribute absent or unusable: fall back to the only ordering still available, kept-vs-dropped.
    keep = set(str(c) for c in selected)
    return np.asarray([0.0 if str(nm) in keep else 1.0 for nm in names], dtype=np.float64)


class RFECVArm(BaseArm):
    """mlframe ``wrappers.RFECV`` driven BARE, with an explicit refit/runtime budget.

    ``ranking_`` assigns rank 1 to EVERY survivor, so the honest kind is ``ordinal`` (ties included), NOT
    continuous: an AP computed on ``-ranking_`` without tie correction would score column order, not the
    selector. The score published here is ``-ranking_``, higher = better, ties intact.
    """

    name = "rfecv"
    score_kind = "ordinal"

    def __init__(self, max_refits: int = 12, max_runtime_mins: float = 2.0, cv: int = 3, random_state: int = 0):
        self.max_refits = int(max_refits)
        self.max_runtime_mins = float(max_runtime_mins)
        self.cv = int(cv)
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Fit RFECV over a small LightGBM and expose ``-ranking_`` as the ordinal score."""
        import lightgbm as lgb

        from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

        names = _feature_names(X)
        est = lgb.LGBMClassifier(n_estimators=80, verbose=-1, n_jobs=-1, random_state=self.random_state)
        model = RFECV(
            estimator=est,
            cv=self.cv,
            scoring=None,
            verbose=0,
            fi_config=FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min"),
            search_config=SearchConfig(max_refits=self.max_refits, max_runtime_mins=self.max_runtime_mins),
            random_state=self.random_state,
        )
        model.fit(X, pd.Series(np.asarray(y)))
        selected = [str(c) for c in extract_selected(model, names) if str(c) in set(names)]
        ranking = _rfecv_rank_vector(model, names, selected)
        best = getattr(model, "best_score_", None)
        return {
            "support": _mask_from_names(names, selected),
            "score": -ranking,
            "selection_score": None if best is None else float(best),
            "provenance": {"max_refits": self.max_refits, "max_runtime_mins": self.max_runtime_mins, "n_unique_ranks": int(np.unique(ranking).size)},
        }


class BorutaShapArm(BaseArm):
    """``boruta_shap.BorutaShap`` driven BARE (the registry factory wraps it in ``GroupAwareMRMR``).

    ``history_x`` accumulates one per-trial SHAP-importance row per column (NaN-padded once a column is
    dropped from the run), so a per-feature ``nanmean`` is a real continuous score with full coverage.
    """

    name = "boruta-shap"
    score_kind = "continuous"

    def __init__(self, n_trials: int = 30, n_estimators: int = 60, percentile: float = 95, random_state: int = 0):
        self.n_trials = int(n_trials)
        self.n_estimators = int(n_estimators)
        self.percentile = percentile
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Fit BorutaShap and average its per-trial importance history into a full-length score."""
        from sklearn.ensemble import RandomForestClassifier

        from mlframe.feature_selection.boruta_shap import BorutaShap

        names = _feature_names(X)
        model = BorutaShap(
            model=RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=-1, random_state=self.random_state),
            importance_measure="gini",
            classification=True,
            n_trials=self.n_trials,
            percentile=self.percentile,
            verbose=False,
            random_state=self.random_state,
        )
        model.fit(X, pd.Series(np.asarray(y)))
        history = getattr(model, "history_x", None)
        if not isinstance(history, pd.DataFrame) or not set(names).issubset(set(map(str, history.columns))):
            raise ValueError("BorutaShap.history_x is missing or does not cover every input feature; a 'continuous' arm may not fall back to a synthesised score.")
        hist = history.iloc[1:] if len(history) > 1 else history
        with np.errstate(invalid="ignore"):
            means = {str(c): float(np.nan_to_num(np.nanmean(np.asarray(hist[c], dtype=np.float64)), nan=0.0)) for c in names}
        score = np.array([means[n] for n in names], dtype=np.float64)
        selected = [str(c) for c in getattr(model, "selected_features_", []) if str(c) in set(names)]
        return {"support": _mask_from_names(names, selected), "score": score, "n_model_fits": self.n_trials, "provenance": {"n_trials": self.n_trials, "n_tentative": len(getattr(model, "tentative", []) or [])}}


class ShapProxiedArm(BaseArm):
    """``ShapProxiedFS`` driven BARE. Declares ``score_kind='none'`` -- verified, not assumed.

    The selector materialises only ``support_`` / ``selected_features_`` / ``shap_proxy_report_``; the
    ``mean_abs_shap`` key that ``registry._report_extract_shap_proxied_fs`` reads is NEVER written anywhere
    in the package, so no full-coverage per-feature score exists to publish. Declaring ``continuous`` here
    and letting the read fail would be the exact silent-degradation this harness forbids.
    """

    name = "shap-proxied"
    score_kind = "none"

    def __init__(self, top_n: int = 12, min_features: int = 3, n_splits: int = 3, random_state: int = 0):
        self.top_n = int(top_n)
        self.min_features = int(min_features)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def _compute(self, X, y):
        """Fit ShapProxiedFS and expose only its support (plus the report keys as provenance)."""
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

        names = _feature_names(X)
        p = int(X.shape[1])
        model = ShapProxiedFS(
            classification=True,
            n_splits=self.n_splits,
            top_n=self.top_n,
            min_features=min(self.min_features, p),
            prefilter_top=min(40, p),
            prefilter_n_estimators=40,
            oof_shap_n_estimators=40,
            revalidation_n_estimators=40,
            n_revalidation_models=2,
            random_state=self.random_state,
            verbose=False,
        )
        model.fit(X, pd.Series(np.asarray(y)))
        selected = [str(c) for c in getattr(model, "selected_features_", []) if str(c) in set(names)]
        report = getattr(model, "shap_proxy_report_", None)
        return {"support": _mask_from_names(names, selected), "provenance": {"report_keys": sorted(map(str, report)) if isinstance(report, dict) else None}}


# ------------------------------------------------------------------------------------------------- roster
def build_arm_roster(n_features: int, *, k: Optional[int] = None, random_state: int = 0) -> "Dict[str, Callable[[], BaseArm]]":
    """Factory map ``name -> zero-arg builder`` for the Phase-0 arms, sized to ``n_features``.

    Args:
        n_features: Width of the bench frame; sizes the random/variance controls when ``k`` is omitted.
        k: Cardinality for the fixed-K controls; defaults to ``max(1, n_features // 4)``.
        random_state: Seed threaded into every stochastic arm.

    Returns:
        Ordered dict of arm name to a builder returning a FRESH unfitted arm.
    """
    kk = int(k) if k is not None else max(1, int(n_features) // 4)
    roster: Dict[str, Callable[[], BaseArm]] = {}
    roster["all-features"] = lambda: AllFeaturesArm()
    roster[f"random-{kk}"] = lambda: RandomSelectionArm(k=kk, random_state=random_state)
    roster["variance-sort"] = lambda: VarianceSortArm(k=kk)
    roster["univariate-mi"] = lambda: UnivariateMIArm(random_state=random_state)
    roster["skb-f"] = lambda: SklearnScoreArm("kbest_f", k=kk, random_state=random_state)
    roster["skb-mi"] = lambda: SklearnScoreArm("kbest_mi", k=kk, random_state=random_state)
    roster["select-fdr"] = lambda: SklearnScoreArm("fdr_f", random_state=random_state)
    roster["sfm-lgbm"] = lambda: SelectFromModelArm(random_state=random_state)
    roster["lars-order"] = lambda: LarsPathArm(max_features=kk)
    roster["boruta"] = lambda: BorutaArm(random_state=random_state)
    roster["ace"] = lambda: ACEArm(random_state=random_state)
    roster["knockoffs"] = lambda: KnockoffArm(random_state=random_state)
    roster["mrmr"] = lambda: MRMRArm(random_seed=random_state)
    roster["rfecv"] = lambda: RFECVArm(random_state=random_state)
    roster["boruta-shap"] = lambda: BorutaShapArm(random_state=random_state)
    roster["shap-proxied"] = lambda: ShapProxiedArm(random_state=random_state)
    return roster


__all__ = [
    "ACEArm",
    "FITS_SOURCE_COUNTED",
    "FITS_SOURCE_DECLARED",
    "FITS_SOURCE_NOT_MEASURED",
    "FitCounter",
    "AllFeaturesArm",
    "BaseArm",
    "BorutaArm",
    "BorutaShapArm",
    "KnockoffArm",
    "LarsPathArm",
    "MRMRArm",
    "RFECVArm",
    "RandomSelectionArm",
    "SelectFromModelArm",
    "ShapProxiedArm",
    "SklearnScoreArm",
    "UnivariateMIArm",
    "VarianceSortArm",
    "build_arm_roster",
]
