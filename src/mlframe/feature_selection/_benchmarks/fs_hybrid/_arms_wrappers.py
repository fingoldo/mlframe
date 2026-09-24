"""Wrappers, stability selection, bandits and cascades: the arms that fit a model to decide.

Every selector here is in the repository and none was in the roster, including two whose ranking output
was upgraded specifically so the benchmark could read it and then never read. Each `score_kind` below was
set by running the selector and looking at what came back, and three of them differ from what the plan
assumed:

* **Greedy backward elimination is ordinal, not a selection order.** Its trace records the order columns
  were REMOVED; the survivors are a set with no order among them. So the survivors share the top rank and
  the removed columns rank by how late they went -- ties included, the same shape as RFECV.
* **Zero-importance pruning is a no-op with a linear model.** Its threshold is zero and a fitted
  coefficient is essentially never exactly zero, so with logistic regression the trace was empty and all
  twelve probe columns survived. A tree gives an unused column exactly zero importance, which is the case
  the method was written for, so the arm uses one.
* **The ridge prefilter cannot prune at all below sixteen columns** with its default size grid, which
  starts at sixteen and is capped at the column count -- leaving a single candidate, "everything". On a
  twelve-column probe it returned all twelve; given a grid it returned exactly the three informative
  columns. The arm passes a grid derived from `k`.

Wrappers score subsets by cross-validating a model on `predict`. For a classifier that is a hard label,
and an AUC over hard labels is one threshold -- nearly balanced accuracy -- which makes the elimination
signal coarse. The arms therefore hand these selectors an estimator whose `predict` returns the positive
class probability. That is a decision about the benchmark's inputs, not a change to the selectors, whose
`predict`-based scoring is left exactly as callers already rely on.

Wrapper arms are O(d^2) model fits and are registered only up to `WRAPPER_MAX_WIDTH` columns, as the plan's
tiering requires; past it they are absent from the roster rather than present and timing out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ._arms import BaseArm, _feature_names, _mask_from_names

__all__ = [
    "WRAPPER_MAX_WIDTH",
    "ProbabilityAsPrediction",
    "ForwardSelectArm",
    "GreedyBackwardArm",
    "ZeroImportanceArm",
    "NoiseFloorArm",
    "UnanimousPermutationArm",
    "BanditArm",
    "BanditEnsembleArm",
    "CascadeArm",
    "CascadeStableArm",
    "HeteroVoteArm",
    "NullImportanceArm",
    "RidgePrefilterArm",
    "RegistryWrappedArm",
]

#: Widest bed an O(d^2) wrapper arm is registered on. At fifty columns and a three-fold split a backward
#: elimination is already several thousand small fits.
WRAPPER_MAX_WIDTH = 50


def _small_tree(random_state: int) -> Any:
    """The wrapper arms' internal model: small, single-threaded and seeded, so a cell reproduces exactly."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=40, num_leaves=15, verbose=-1, n_jobs=1, random_state=random_state, deterministic=True, force_row_wise=True)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC, or 0.5 when a fold holds a single class and the statistic is undefined."""
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


class ProbabilityAsPrediction(ClassifierMixin, BaseEstimator):
    """A classifier whose `predict` returns the positive-class probability rather than a label.

    Selectors that score subsets through `predict` then compute a real AUC instead of a one-threshold one.
    It is a `BaseEstimator` so the selectors' own `clone` calls work on it unchanged.
    """

    def __init__(self, estimator: Any = None):
        self.estimator = estimator

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "ProbabilityAsPrediction":
        """Fit a fresh clone of the wrapped estimator."""
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the probability of the last class, which is the positive class for a binary target."""
        return np.asarray(self.estimator_.predict_proba(X))[:, -1]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Delegate unchanged."""
        return np.asarray(self.estimator_.predict_proba(X))

    @property
    def feature_importances_(self) -> np.ndarray:
        """The wrapped model's importances, for selectors that read them off the fitted estimator."""
        return np.asarray(self.estimator_.feature_importances_, dtype=np.float64)


def _probabilistic(random_state: int) -> ProbabilityAsPrediction:
    """The internal model every wrapper arm here fits."""
    return ProbabilityAsPrediction(_small_tree(random_state))


def _splitter(random_state: int, n_splits: int = 3) -> StratifiedKFold:
    """A stratified split -- the one three of these selectors could not accept until it was fixed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _ordinal_from_removal(names: Sequence[str], kept: Sequence[str], removed_in_order: Sequence[str]) -> np.ndarray:
    """Rank survivors jointly at the top and removed columns by how late they were removed.

    The survivors are a set; giving them distinct ranks would invent an order the selector never produced.
    """
    score = np.zeros(len(names), dtype=np.float64)
    position = {str(n): i for i, n in enumerate(names)}
    for order, name in enumerate(removed_in_order, start=1):
        if str(name) in position:
            score[position[str(name)]] = float(order)
    top = float(len(removed_in_order) + 1)
    for name in kept:
        if str(name) in position:
            score[position[str(name)]] = top
    return score


class ForwardSelectArm(BaseArm):
    """`forward_select`: grows the set one column at a time, publishing the order it grew in."""

    name = "forward-select"
    score_kind = "selection_order"

    def __init__(self, max_features: int, random_state: int = 0):
        self.max_features = int(max_features)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run forward selection with the probability-scoring model."""
        from mlframe.feature_selection.forward_select import forward_select

        names = _feature_names(X)
        chosen = [str(c) for c in forward_select(X, y, lambda: _probabilistic(self.random_state), scoring="roc_auc", cv=3, max_features=self.max_features)]
        order = tuple(names.index(c) for c in chosen if c in names)
        return {"support": _mask_from_names(names, chosen), "ranked_prefix": order, "provenance": {"max_features": self.max_features}}


class GreedyBackwardArm(BaseArm):
    """`greedy_backward_elimination`, read through its removal trace as an ordinal ranking."""

    name = "greedy-backward"
    score_kind = "ordinal"

    def __init__(self, min_features: int = 1, random_state: int = 0):
        self.min_features = int(min_features)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Eliminate by real CV score and rank by removal order."""
        from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination

        names = _feature_names(X)
        kept, trace = greedy_backward_elimination(
            _probabilistic(self.random_state), X, np.asarray(y), _auc, cv=_splitter(self.random_state), min_features=self.min_features, return_trace=True
        )
        kept_names = [str(c) for c in kept]
        removed = [str(step.dropped) for step in trace]
        best = max((float(step.score_after) for step in trace), default=None)
        return {
            "support": _mask_from_names(names, kept_names),
            "score": _ordinal_from_removal(names, kept_names, removed),
            "selection_score": best,
            "selection_metric": "roc_auc",
            "provenance": {"n_removed": len(removed), "n_kept": len(kept_names)},
        }


class ZeroImportanceArm(BaseArm):
    """`iterative_zero_importance_pruning` over a tree, ranked by the round in which a column was pruned."""

    name = "zero-importance"
    score_kind = "ordinal"

    def __init__(self, max_rounds: int = 10, random_state: int = 0):
        self.max_rounds = int(max_rounds)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Prune zero-importance columns round by round and rank by how late each went."""
        from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning

        names = _feature_names(X)
        kept, trace = iterative_zero_importance_pruning(
            _probabilistic(self.random_state),
            X,
            np.asarray(y),
            _auc,
            cv=_splitter(self.random_state),
            max_rounds=self.max_rounds,
            importance_fn=lambda model, frame, target: np.asarray(model.feature_importances_, dtype=np.float64),
            return_trace=True,
        )
        kept_names = [str(c) for c in kept]
        removed: List[str] = []
        for round_record in trace:
            removed.extend(str(c) for c in round_record.dropped)
        return {
            "support": _mask_from_names(names, kept_names),
            "score": _ordinal_from_removal(names, kept_names, removed),
            "provenance": {"n_rounds": len(trace), "n_pruned": len(removed)},
        }


class NoiseFloorArm(BaseArm):
    """`select_features_noise_floor`: cuts a tree-importance ranking where it stops beating permuted labels.

    The one mechanism in the repository with a measured real-data gain over doing nothing. Its order is the
    importance ranking it was handed, so the kind is a selection order over that ranking.
    """

    name = "noise-floor"
    score_kind = "selection_order"

    def __init__(self, n_perm: int = 50, random_state: int = 0):
        self.n_perm = int(n_perm)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Rank by gain importance, then let the noise floor choose how far down to cut."""
        from mlframe.feature_selection.wrappers._noise_floor import select_features_noise_floor

        names = _feature_names(X)
        importance = np.asarray(_small_tree(self.random_state).fit(X, np.asarray(y)).feature_importances_, dtype=np.float64)
        ranking = [names[i] for i in np.argsort(-importance, kind="stable")]
        result = select_features_noise_floor(lambda: _small_tree(self.random_state), X, np.asarray(y), ranking, n_perm=self.n_perm, random_state=self.random_state)
        selected = [str(c) for c in result["selected"]]
        return {
            "support": _mask_from_names(names, selected),
            "ranked_prefix": tuple(names.index(c) for c in ranking),
            "provenance": {"n_star": int(result["n_star"]), "n_perm": self.n_perm},
        }


class UnanimousPermutationArm(BaseArm):
    """`unanimous_permutation_prune`: drops a column only when every fold agrees it does not help."""

    name = "unanimous-permutation"
    score_kind = "none"

    def __init__(self, n_repeats: int = 5, max_iterations: int = 10, random_state: int = 0):
        self.n_repeats = int(n_repeats)
        self.max_iterations = int(max_iterations)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the unanimous-vote prune and read the survivors as a set."""
        from mlframe.feature_selection.unanimous_permutation_prune import unanimous_permutation_prune

        names = _feature_names(X)
        folds = list(_splitter(self.random_state).split(X, np.asarray(y)))
        kept = unanimous_permutation_prune(
            X, np.asarray(y), lambda: _small_tree(self.random_state), folds, n_repeats=self.n_repeats, max_iterations=self.max_iterations, random_state=self.random_state
        )
        return {"support": _mask_from_names(names, [str(c) for c in kept]), "provenance": {"n_repeats": self.n_repeats}}


class BanditArm(BaseArm):
    """`stochastic_bandit_selection`: one seed, one subset of fixed size, no ranking."""

    name = "bandit"
    score_kind = "none"

    def __init__(self, subset_size: int, n_epochs: int = 120, random_state: int = 0):
        self.subset_size = int(subset_size)
        self.n_epochs = int(n_epochs)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run one bandit and read its subset."""
        from mlframe.feature_selection.stochastic_bandit_selection import stochastic_bandit_selection

        names = _feature_names(X)
        size = min(self.subset_size, len(names))
        chosen = stochastic_bandit_selection(
            _probabilistic(self.random_state), X, np.asarray(y), _auc, subset_size=size, n_epochs=self.n_epochs, cv=_splitter(self.random_state), random_state=self.random_state
        )
        return {"support": _mask_from_names(names, [str(c) for c in chosen]), "provenance": {"subset_size": size, "n_epochs": self.n_epochs}}


class BanditEnsembleArm(BaseArm):
    """The bandit over several seeds, scored by the fraction of seeds whose subset held each column.

    A column no seed ever chose has a stability of exactly zero -- a measured value, not a padded one.
    """

    name = "bandit-ensemble"
    score_kind = "continuous"

    def __init__(self, subset_size: int, n_seeds: int = 5, n_epochs: int = 80, random_state: int = 0):
        self.subset_size = int(subset_size)
        self.n_seeds = int(n_seeds)
        self.n_epochs = int(n_epochs)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the seeded bandits and publish per-column stability."""
        from mlframe.feature_selection.stochastic_bandit_selection_ensemble import stochastic_bandit_selection_ensemble

        names = _feature_names(X)
        size = min(self.subset_size, len(names))
        seeds = [self.random_state * 1000 + s for s in range(self.n_seeds)]
        result = stochastic_bandit_selection_ensemble(
            _probabilistic(self.random_state), X, np.asarray(y), _auc, subset_size=size, seeds=seeds, n_epochs=self.n_epochs, cv=_splitter(self.random_state)
        )
        stability = {str(k): float(v) for k, v in dict(result.stability).items()}
        score = np.asarray([stability.get(n, 0.0) for n in names], dtype=np.float64)
        order = np.argsort(-score, kind="stable")
        return {
            "support": np.isin(np.arange(len(names)), order[:size]) & (score > 0),
            "score": score,
            "provenance": {"n_seeds": self.n_seeds, "subset_size": size, "union_size": len(result.union_top_feats)},
        }


class CascadeArm(BaseArm):
    """`cascade_select`: Boruta, then forward selection, then RFECV, keeping what survives all three."""

    name = "cascade"
    score_kind = "none"

    def __init__(self, forward_max_features: int, random_state: int = 0):
        self.forward_max_features = int(forward_max_features)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the cascade and read its final set."""
        from mlframe.feature_selection.cascade_select import cascade_select

        names = _feature_names(X)
        result = cascade_select(
            X, np.asarray(y), lambda: _small_tree(self.random_state), n_boruta_iterations=15, cv=3, forward_max_features=self.forward_max_features, random_state=self.random_state
        )
        final = [str(c) for c in result["final_selected"] if str(c) in set(names)]
        return {"support": _mask_from_names(names, final), "provenance": {"n_boruta": len(result.get("boruta_confirmed", [])), "n_forward": len(result.get("forward_selected", []))}}


class CascadeStableArm(BaseArm):
    """`cascade_select_stable`: the cascade over bootstraps, scored by selection frequency per column."""

    name = "cascade-stable"
    score_kind = "continuous"

    def __init__(self, forward_max_features: int, n_bootstrap: int = 8, random_state: int = 0):
        self.forward_max_features = int(forward_max_features)
        self.n_bootstrap = int(n_bootstrap)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the bootstrapped cascade and publish each column's selection frequency."""
        from mlframe.feature_selection.cascade_select_stability import cascade_select_stable

        names = _feature_names(X)
        result = cascade_select_stable(
            X,
            np.asarray(y),
            lambda: _small_tree(self.random_state),
            n_bootstrap=self.n_bootstrap,
            bootstrap_random_state=self.random_state,
            n_boruta_iterations=10,
            cv=3,
            forward_max_features=self.forward_max_features,
        )
        frequency = {str(k): float(v) for k, v in dict(result["selection_frequency"]).items()}
        stable = [str(c) for c in result["stable_selected"] if str(c) in set(names)]
        return {
            "support": _mask_from_names(names, stable),
            "score": np.asarray([frequency.get(n, 0.0) for n in names], dtype=np.float64),
            "provenance": {"n_bootstrap": self.n_bootstrap},
        }


class HeteroVoteArm(BaseArm):
    """`heterogeneous_relevance_vote`: several model families vote against shadows; the vote share is the score."""

    name = "hetero-vote"
    score_kind = "continuous"

    def __init__(self, n_shadow_trials: int = 3, random_state: int = 0):
        self.n_shadow_trials = int(n_shadow_trials)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Run the vote and publish each column's vote fraction."""
        from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote

        names = _feature_names(X)
        kept, report = heterogeneous_relevance_vote(X, np.asarray(y), n_shadow_trials=self.n_shadow_trials, random_state=self.random_state)
        votes = {str(k): float(v) for k, v in dict(report["vote_fraction"]).items()}
        return {
            "support": _mask_from_names(names, [str(c) for c in kept]),
            "score": np.asarray([votes.get(n, 0.0) for n in names], dtype=np.float64),
            "provenance": {"n_models": int(report.get("n_models", 0))},
        }


class NullImportanceArm(BaseArm):
    """`null_importance_filter`: real importance against a label-shuffled null; the margin is the score.

    It shuffles the TARGET, not the columns -- a different null from Boruta's shadow columns, which is why
    it is its own arm rather than a Boruta setting.
    """

    name = "null-importance"
    score_kind = "continuous"

    def __init__(self, n_shuffles: int = 30, random_state: int = 0):
        self.n_shuffles = int(n_shuffles)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Compare real gain importance with its label-permuted distribution."""
        from mlframe.feature_selection.filters._null_importance import null_importance_filter

        names = _feature_names(X)

        def importance(frame: Any, target: np.ndarray) -> np.ndarray:
            """Gain importance of one small tree fit."""
            return np.asarray(_small_tree(self.random_state).fit(frame, target).feature_importances_, dtype=np.float64)

        result = null_importance_filter(X, np.asarray(y), importance, n_shuffles=self.n_shuffles, random_state=self.random_state, return_margin_score=True)
        keep = np.asarray(result["keep_mask"], dtype=bool)
        if keep.shape[0] != len(names):
            raise ValueError(f"null_importance_filter returned a mask of {keep.shape[0]} for {len(names)} columns")
        return {
            "support": keep,
            "score": np.nan_to_num(np.asarray(result["margin_score"], dtype=np.float64), nan=0.0),
            "n_model_fits": self.n_shuffles + 1,
            "provenance": {"n_shuffles": self.n_shuffles},
        }


class RidgePrefilterArm(BaseArm):
    """`ridge_coefficient_prefilter` with a size grid built from `k`, since its default cannot prune below sixteen."""

    name = "ridge-prefilter"
    score_kind = "selection_order"

    def __init__(self, k: int, random_state: int = 0):
        self.k = int(k)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Rank by ridge coefficient and let the CV sweep choose the prefix."""
        from mlframe.feature_selection.ridge_forward_prefilter import ridge_coefficient_prefilter

        names = _feature_names(X)
        sizes = sorted({s for s in (max(1, self.k // 2), self.k, 2 * self.k, 4 * self.k) if s <= len(names)} | {len(names)})
        values = np.asarray(X.to_numpy(dtype=np.float64))
        # Standardised inside the arm: a ridge penalty is scale-dependent, and ranking raw coefficients would
        # rank columns partly by their units.
        values = (values - values.mean(axis=0)) / np.where(values.std(axis=0) > 0, values.std(axis=0), 1.0)
        chosen = [str(c) for c in ridge_coefficient_prefilter(values, np.asarray(y), names, candidate_sizes=sizes, is_classifier=True, random_state=self.random_state)]
        return {
            "support": _mask_from_names(names, chosen),
            "ranked_prefix": tuple(names.index(c) for c in chosen),
            "n_model_fits": 3 * len(sizes),
            "provenance": {"candidate_sizes": sizes},
        }


class RegistryWrappedArm(BaseArm):
    """A selector exactly as `feature_selection.registry` builds it, cluster-medoid wrapper included.

    The pre-registration requires RFECV and BorutaShap to be measured both bare and as the registry ships
    them, because the registry wraps both in a `GroupAwareMRMR` with `expand=True` that drags a selected
    medoid's whole cluster back in. The registry's own factory is called rather than the wrapper being
    rebuilt here, so what is measured is what runs. The wrapper returns its support sorted by column index,
    which is a set and not an order, so the kind is `none`.
    """

    score_kind = "none"

    def __init__(self, which: str, random_state: int = 0, max_runtime_mins: float = 2.0):
        if which not in ("rfecv", "boruta_shap"):
            raise ValueError(f"no registry arm for {which!r}")
        self.which = which
        self.random_state = int(random_state)
        self.max_runtime_mins = float(max_runtime_mins)
        self.name = "rfecv-registry" if which == "rfecv" else "boruta-shap-registry"

    def _build(self) -> Any:
        """Call the registry's factory with the same inner settings the bare arm uses."""
        from mlframe.feature_selection import registry

        if self.which == "rfecv":
            import lightgbm as lgb

            from mlframe.feature_selection.wrappers import FIConfig, SearchConfig

            return registry._instantiate_rfecv(
                estimator=lgb.LGBMClassifier(n_estimators=80, verbose=-1, n_jobs=-1, random_state=self.random_state),
                cv=3,
                scoring=None,
                verbose=0,
                fi_config=FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min"),
                search_config=SearchConfig(max_refits=12, max_runtime_mins=self.max_runtime_mins),
                random_state=self.random_state,
            )
        from sklearn.ensemble import RandomForestClassifier

        from ._arms import BorutaShapArm

        # The bare arm's own settings, read off it rather than restated, so the two arms cannot drift apart
        # and the only difference between them stays the registry's wrapper.
        bare = BorutaShapArm(random_state=self.random_state)
        return registry._instantiate_boruta_shap(
            model=RandomForestClassifier(n_estimators=bare.n_estimators, n_jobs=-1, random_state=self.random_state),
            importance_measure="gini",
            classification=True,
            n_trials=bare.n_trials,
            percentile=bare.percentile,
            verbose=False,
            random_state=self.random_state,
        )

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Fit the registry-built selector and read its final support as a set."""
        names = _feature_names(X)
        model = self._build()
        model.fit(X, pd.Series(np.asarray(y)))
        selected = [str(c) for c in model.get_feature_names_out() if str(c) in set(names)]
        return {
            "support": _mask_from_names(names, selected),
            "provenance": {"registry_factory": f"_instantiate_{self.which}", "wrapped": type(model).__name__, "reduction": float(getattr(model, "reduction_", float("nan")))},
        }


def registered_wrapper_names(n_features: int) -> List[str]:
    """Return the wrapper arm names that are registered at this width, for the roster and its tests."""
    base = ["noise-floor", "unanimous-permutation", "bandit", "bandit-ensemble", "cascade", "cascade-stable", "hetero-vote", "null-importance", "ridge-prefilter"]
    if n_features <= WRAPPER_MAX_WIDTH:
        base += ["forward-select", "greedy-backward", "zero-importance"]
    return base
