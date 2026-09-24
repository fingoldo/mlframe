"""Reference lines that bound the leaderboard, and the univariate filters it was missing.

**The oracle pair.** `all-features` is the null hypothesis, which bounds a selector from one side: it
answers "is selecting worth anything". Nothing bounded it from the other. `oracle-informative` hands the
panel exactly the bed's declared answer key, and `all-except-informative` hands it everything BUT the key.
Together they turn every arm's number into a position between two measured endpoints instead of a raw
score: an arm that matches the oracle has nothing left to gain on this bed, and one that scores no better
than the complement has found nothing at all. Both need the truth, so they exist only on beds that declare
one and are simply absent from a real bed rather than present and meaningless.

Two P0 entries from the plan are deliberately NOT arms. `true-prob` and `shuffled-prob` replace the model's
PREDICTION, not its inputs, so they cannot be expressed as a feature selection at all; the ceiling they
stand for is already computed exactly by the dataset oracle, and the floor is the base rate every cell
records.

**Univariate filters.** Permutation importance with the tail cut; the unsupervised prescreen (constant and
all-null columns), which on most synthetic beds removes nothing and is kept because on the beds with a
point mass it is the whole method; the near-noise univariate-AUC drop; a KSG mutual information, the one
continuous-estimator member of the family; and the Mann-Whitney / Kruskal-Wallis / Kendall relevance table
with Benjamini-Yekutieli control, scored by `-log10 p`.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ._arms import BaseArm, _feature_names, _mask_from_names

__all__ = [
    "OracleInformativeArm",
    "AllExceptInformativeArm",
    "PermutationTopKArm",
    "UnsupervisedPrescreenArm",
    "NearNoiseAucArm",
    "KSGArm",
    "RelevanceTableArm",
]


class OracleInformativeArm(BaseArm):
    """Hands the panel exactly the bed's declared answer key: the ceiling a selector can reach on this bed."""

    name = "oracle-informative"
    score_kind = "none"

    def __init__(self, relevant: Sequence[str]):
        self.relevant = [str(c) for c in relevant]

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Select the declared relevant columns that are present in this frame."""
        names = _feature_names(X)
        present = [c for c in self.relevant if c in set(names)]
        return {"support": _mask_from_names(names, present), "n_model_fits": 0, "provenance": {"n_declared": len(self.relevant), "n_present": len(present)}}


class AllExceptInformativeArm(BaseArm):
    """Hands the panel every column EXCEPT the answer key: the floor that says a selector found nothing."""

    name = "all-except-informative"
    score_kind = "none"

    def __init__(self, relevant: Sequence[str]):
        self.relevant = [str(c) for c in relevant]

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Select the complement of the declared answer key."""
        names = _feature_names(X)
        excluded = set(self.relevant)
        return {"support": np.asarray([n not in excluded for n in names], dtype=bool), "n_model_fits": 0, "provenance": {"n_excluded": len(excluded & set(names))}}


class PermutationTopKArm(BaseArm):
    """Permutation importance on a held-out slice of TRAIN, with the tail below `k` cut.

    The slice comes out of the training rows, never the benchmark's honest holdout, so the arm cannot see
    the rows it will be scored on.
    """

    name = "permutation-topk"
    score_kind = "continuous"

    def __init__(self, k: int, n_repeats: int = 5, random_state: int = 0):
        self.k = int(k)
        self.n_repeats = int(n_repeats)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Fit on part of train, permute on the rest, and keep the top `k`."""
        import lightgbm as lgb
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        names = _feature_names(X)
        x_fit, x_val, y_fit, y_val = train_test_split(X, np.asarray(y), test_size=0.3, random_state=self.random_state, stratify=np.asarray(y))
        model = lgb.LGBMClassifier(n_estimators=80, verbose=-1, n_jobs=1, random_state=self.random_state, deterministic=True, force_row_wise=True)
        model.fit(x_fit, y_fit)
        result = permutation_importance(model, x_val, y_val, scoring="roc_auc", n_repeats=self.n_repeats, random_state=self.random_state, n_jobs=1)
        score = np.nan_to_num(np.asarray(result.importances_mean, dtype=np.float64), nan=0.0)
        support = np.zeros(len(names), dtype=bool)
        support[np.argsort(-score, kind="stable")[: min(self.k, len(names))]] = True
        return {"support": support, "score": score, "n_model_fits": 1, "provenance": {"n_repeats": self.n_repeats}}


class UnsupervisedPrescreenArm(BaseArm):
    """`compute_unsupervised_drops`: removes constant and nearly-all-null columns without looking at the target."""

    name = "unsupervised-prescreen"
    score_kind = "none"

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Drop what the target-free screen drops."""
        from mlframe.feature_selection.pre_screen import compute_unsupervised_drops

        names = _feature_names(X)
        dropped = {str(c) for c in compute_unsupervised_drops(X)}
        return {"support": np.asarray([n not in dropped for n in names], dtype=bool), "n_model_fits": 0, "provenance": {"n_dropped": len(dropped)}}


class NearNoiseAucArm(BaseArm):
    """`drop_near_noise_univariate_auc`: drops a column whose univariate AUC sits within tolerance of 0.5."""

    name = "near-noise-auc"
    score_kind = "none"

    def __init__(self, tolerance: float = 0.02, random_state: int = 0):
        self.tolerance = float(tolerance)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Keep the complement of the near-noise drop list."""
        from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc

        names = _feature_names(X)
        dropped = {str(c) for c in drop_near_noise_univariate_auc(X, np.asarray(y), tolerance=self.tolerance, random_state=self.random_state)}
        return {"support": np.asarray([n not in dropped for n in names], dtype=bool), "n_model_fits": 0, "provenance": {"n_dropped": len(dropped), "tolerance": self.tolerance}}


class KSGArm(BaseArm):
    """KSG mutual information with the target: the continuous-estimator member of the MI family."""

    name = "ksg-mi"
    score_kind = "continuous"

    def __init__(self, k: int, n_neighbors: int = 3, random_state: int = 0):
        self.k = int(k)
        self.n_neighbors = int(n_neighbors)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Score every column by KSG MI and keep the top `k`."""
        from mlframe.feature_selection.filters.estimators import ksg_mi_with_target

        names = _feature_names(X)
        values = np.asarray(X.to_numpy(dtype=np.float64))
        score = np.nan_to_num(
            np.asarray(ksg_mi_with_target(values, np.asarray(y), list(range(len(names))), n_neighbors=self.n_neighbors, random_state=self.random_state), dtype=np.float64),
            nan=0.0,
        )
        support = np.zeros(len(names), dtype=bool)
        support[np.argsort(-score, kind="stable")[: min(self.k, len(names))]] = True
        return {"support": support, "score": score, "n_model_fits": 0, "provenance": {"n_neighbors": self.n_neighbors}}


class RelevanceTableArm(BaseArm):
    """`calculate_relevance_table`: per-column hypothesis tests with Benjamini-Yekutieli control, scored `-log10 p`."""

    name = "relevance-table"
    score_kind = "continuous"

    def __init__(self, fdr_level: float = 0.05, random_state: int = 0):
        self.fdr_level = float(fdr_level)
        self.random_state = int(random_state)

    def _compute(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Test every column against the target and publish the evidence against each null."""
        from mlframe.feature_selection.wrappers._univariate_ht import calculate_relevance_table

        names = _feature_names(X)
        table = calculate_relevance_table(X, pd.Series(np.asarray(y)), ml_task="classification", fdr_level=self.fdr_level, random_state=self.random_state)
        by_name = table.set_index(table["feature"].astype(str))
        p_values = np.asarray([float(by_name.at[n, "p_value"]) if n in by_name.index else 1.0 for n in names], dtype=np.float64)
        relevant = [n for n in names if n in by_name.index and bool(by_name.at[n, "relevant"])]
        # A p-value of exactly zero is a float underflow, not infinite evidence; the floor keeps it the largest
        # finite score instead of an infinity that would break every rank statistic downstream.
        score = -np.log10(np.clip(p_values, np.finfo(np.float64).tiny, 1.0))
        return {"support": _mask_from_names(names, relevant), "score": score, "n_model_fits": 0, "provenance": {"fdr_level": self.fdr_level}}
