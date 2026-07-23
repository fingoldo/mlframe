"""Social-choice ranking/election methods mixed into :class:`~mlframe.votenrank.Leaderboard`: mean, plurality, threshold, Borda, Dowdall, Condorcet, Baldwin, Copeland, minimax, and optimality-gap."""
from __future__ import annotations

import pandas as pd
import numpy as np

from scipy.stats import gmean

from ..utils import ranking2top


def mean_ranking(self, mean_type: str = "arithmetic"):
    """Rank models by weighted arithmetic or geometric mean score across tasks; NaN cells filled with the task median when the table is partial.

    ``mean_type="geometric"`` requires every score to be strictly positive (``scipy.stats.gmean``'s own
    domain constraint) -- a task column with a zero or negative score (plausible for a raw margin/log-loss
    metric) otherwise silently drives that model's geometric-mean aggregate to ``0``/``NaN``.
    """
    table = self.table.copy()
    if self.is_partial:
        table = table.fillna(table.median())

    if mean_type == "arithmetic":
        return (table * self.weights / self.weights.sum()).sum(axis=1).sort_values(ascending=False)
    elif mean_type == "geometric":
        if not (table > 0).all().all():
            raise ValueError(
                "mean_ranking(mean_type='geometric') requires every score to be strictly positive "
                "(scipy.stats.gmean's domain constraint); found a zero/negative score in the table."
            )
        scores = pd.Series(index=table.index, data=gmean(table, axis=1, weights=self.weights))
        return scores.sort_values(ascending=False)
    else:
        raise ValueError(f"Only arithmetic and geometric mean is supported, got {mean_type}")


def mean_election(self, mean_type: str = "arithmetic"):
    """Winning model(s) under :func:`mean_ranking`."""
    return ranking2top(self.mean_ranking(mean_type=mean_type))


def _partial_table_guard(self, method_name: str) -> None:
    """F6: raise on a partial table for methods with no fill/skip strategy of their own.

    ``pandas.sum(skipna=True)`` silently drops each model's NaN tasks, so different models would be
    compared on effectively different numbers of tasks with no warning -- these methods are only ever
    routed away from partial tables via ``elect_all``/``rank_all``'s ``PARTIAL_METHODS`` filter; called
    directly they had no protection of their own.
    """
    if self.is_partial:
        raise ValueError(f"{method_name}: table is partial (contains NaN); this method has no NaN-fill strategy. Use a complete table or a method in PARTIAL_METHODS.")


def _approval_ranking(self, acceptance_threshold: int, rank_type: str = "max"):
    """Approval-style ranking: weighted count of tasks where a model's rank is at or above ``acceptance_threshold``."""
    _partial_table_guard(self, "plurality_ranking/_approval_ranking")
    if rank_type == "min":
        return ((self.ranks <= acceptance_threshold) * self.weights).sum(axis=1).sort_values(ascending=False)
    elif rank_type == "max":
        return ((self.max_ranks <= acceptance_threshold) * self.weights).sum(axis=1).sort_values(ascending=False)
    else:
        raise ValueError("Rank type should be min or max")


def plurality_ranking(self):
    """Rank models by how often each is ranked 1st across tasks."""
    return self._approval_ranking(1)


def plurality_election(self):
    """Winning model(s) under :func:`plurality_ranking`."""
    return ranking2top(self.plurality_ranking())


def threshold_election(self):
    """Iteratively eliminate models ranked last-among-remaining, task-weighted, until only the joint winner(s) remain."""
    _partial_table_guard(self, "threshold_election")
    candidate_models = self.models
    for step in range(self.n_models, 1, -1):
        current_ranking = ((self.max_ranks.loc[candidate_models] != step) * self.weights).sum(axis=1)
        candidate_models = ranking2top(current_ranking)

    return candidate_models


def borda_ranking(self):
    """Rank models by Borda count: weighted sum of ``n_models - rank`` (points) across tasks."""
    _partial_table_guard(self, "borda_ranking")
    return ((self.n_models - self.max_ranks) * self.weights).sum(axis=1).sort_values(ascending=False)


def borda_election(self):
    """Winning model(s) under :func:`borda_ranking`."""
    return ranking2top(self.borda_ranking())


def dowdall_ranking(self):
    """Rank models by Dowdall (reciprocal-rank) score: weighted sum of ``1/rank`` across tasks."""
    _partial_table_guard(self, "dowdall_ranking")
    return ((1 / self.ranks) * self.weights).sum(axis=1).sort_values(ascending=False)


def dowdall_election(self):
    """Winning model(s) under :func:`dowdall_ranking`."""
    return ranking2top(self.dowdall_ranking())


def condorcet_election(self):
    """Model(s) that win every pairwise weighted-majority comparison against all others, if any exist."""
    self._ensure_majority_graph()
    return self.majority_graph.index[(self.majority_graph == 1).all(axis=1)].tolist()


def baldwin_election(self):
    """Iteratively eliminate the lowest-Borda-score model(s) and recompute Borda on the remainder until a unique winner (or tied set) remains."""
    current_borda = self.borda_ranking()
    while current_borda.min() != current_borda.max():
        candidates = current_borda.index[current_borda != current_borda.min()]
        current_max_ranks = self.table.loc[candidates].rank(method="max", ascending=False).astype(int)
        weighted_ranks = (len(candidates) - current_max_ranks) * self.weights
        current_borda = weighted_ranks.sum(axis=1).sort_values(ascending=False)

    return current_borda.index.tolist()


def copeland_ranking(self, slice_type: str = "lower_with_ties"):
    """Rank models by Copeland score (wins minus losses in the pairwise majority graph), with several tie/slice conventions."""
    self._ensure_majority_graph()
    if slice_type == "lower_with_ties":
        return (self.majority_graph.sum(axis=1) - 1).sort_values(ascending=False)
    elif slice_type == "difference":
        lower_ranking = (self.majority_graph == 1).sum(axis=1) - 1
        upper_ranking = (self.majority_graph == 0).sum(axis=1)
        return (lower_ranking - upper_ranking).sort_values(ascending=False)
    elif slice_type == "lower":
        return ((self.majority_graph == 1).sum(axis=1) - 1).sort_values(ascending=False)
    elif slice_type == "upper":
        return (-(self.majority_graph == 0).sum(axis=1)).sort_values(ascending=False)
    else:
        raise ValueError("Slice type should be lower_with_ties, difference, lower or upper")


def copeland_election(self, slice_type: str = "lower_with_ties"):
    """Winning model(s) under :func:`copeland_ranking`."""
    return ranking2top(self.copeland_ranking(slice_type))


def minimax_ranking(self, score_type: str = "winning_votes"):
    """Rank models by minimax (Simpson-Kramer): each model's worst pairwise opposition score, negated so the least-bad model ranks first."""
    ranks = []
    for model in self.models:
        if score_type == "winning_votes":
            # The weighted "less-than" sum is identical for models_scores and the
            # does_win LHS; compute it once instead of twice per model.
            models_scores = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(axis=1)
            does_win = models_scores > ((self.ranks > self.ranks.loc[model]) * self.weights).sum(axis=1)
            models_scores *= does_win
        elif score_type == "margins":
            models_scores = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(axis=1) - ((self.ranks.loc[model] < self.ranks) * self.weights).sum(
                axis=1
            )
        elif score_type == "pairwise_opposition":
            models_scores = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(axis=1)
        else:
            raise ValueError("Score type should be winning_votes, margins or pairwise_opposition")

        score = models_scores.drop(model).max()
        ranks.append(score)

    return (-pd.Series(data=ranks, index=pd.Series(self.models, name="Name"))).sort_values(ascending=False)


def minimax_election(self, score_type: str = "winning_votes"):
    """Winning model(s) under :func:`minimax_ranking`."""
    return ranking2top(self.minimax_ranking(score_type))


def optimality_gap_ranking(self, gamma: int):
    """Rank models by weighted mean gap-to-``gamma`` score (``min(score, gamma) - gamma``, capping credit for scores above ``gamma``); NaN cells filled with the task median when partial."""
    table = self.table.copy()
    if self.is_partial:
        table = table.fillna(table.median())

    gap_scores_np = np.minimum(table, gamma) - gamma
    gap_scores = pd.DataFrame(index=table.index, columns=table.columns, data=gap_scores_np)
    return (gap_scores * self.weights / self.weights.sum()).sum(axis=1).sort_values(ascending=False)


def optimality_gap_election(self, gamma: int):
    """Winning model(s) under :func:`optimality_gap_ranking`."""
    return ranking2top(self.optimality_gap_ranking(gamma))
