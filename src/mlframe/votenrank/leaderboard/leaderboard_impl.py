"""``Leaderboard``: aggregates a model x task score table into a consensus ranking via social-choice methods (mean, Borda, Copeland, minimax, etc.), with support for partial tables and grouped two-step rankings."""
from __future__ import annotations

import itertools

from pyutilz.system import tqdmu as tqdm
import pandas as pd
import numpy as np

from typing import Any, Dict, List, Optional

from sklearn.model_selection import ParameterGrid

from .settings import (
    RANKING_METHODS,
    ELECTION_METHODS,
    PARTIAL_METHODS,
    METHODS_SETTINGS,
    PRETTY_NAMES,
)

# Weighted win margin a Condorcet winner must reach over every rival; 0 would accept a weighted tie, which ``condorcet_election`` does not count as a win.
_CONDORCET_STRICT_MARGIN = 1e-6

_METHODS_REQUIRING_GAMMA = frozenset({"optimality_gap"})


def _check_gamma_given(method: str, params: dict) -> None:
    """Raise a clear error when a method that needs ``gamma`` runs without one (42-F6), instead of a bare TypeError."""
    if method in _METHODS_REQUIRING_GAMMA and "gamma" not in params:
        raise ValueError(f"{method} requires 'gamma'; pass it explicitly, e.g. use_methods={{'{method}': {{'gamma': <value>}}}}.")


class Leaderboard:
    """Model x task score table plus per-task weights; ranks/elects models via the methods mixed in from ``_rules``/``_cw``.

    Store the score table and per-task weights (default 1.0), then build ranks.

    ``allow_partial``: bypass ``_partial_table_guard`` for methods with no NaN-fill strategy
    of their own (see its docstring) -- for deliberate back-compat callers ONLY (e.g. an
    explicit "skip" imputation policy reproducing the pre-fix pandas skipna=True bias for
    A/B comparison). Everyone else should go through ``elect_all``/``rank_all``'s
    ``PARTIAL_METHODS`` filter instead of setting this.
    """

    from ._rules import (  # type: ignore[misc]  # class-scoped import binds these functions as unbound methods; mypy doesn't model this pattern
        mean_ranking,
        mean_election,
        plurality_ranking,
        plurality_election,
        threshold_election,
        borda_ranking,
        borda_election,
        dowdall_ranking,
        dowdall_election,
        condorcet_election,
        baldwin_election,
        copeland_ranking,
        copeland_election,
        minimax_ranking,
        minimax_election,
        optimality_gap_ranking,
        optimality_gap_election,
        _approval_ranking,
    )
    from ._cw import _get_tasks_onehot, _find_weights_for_majority_graph  # type: ignore[misc]  # same class-scoped-import pattern as above

    def __init__(
        self, table: pd.DataFrame, weights: Optional[Dict[str, float]] = None, allow_partial: bool = False,
        partial_fill: str = "min",
    ):
        self.allow_partial = allow_partial
        # How the score-based rules (mean / optimality-gap) fill a model's missing task. "min" (default) gives it the
        # task's WORST observed score; "median" is the historical behaviour, which credited a model that crashed or
        # timed out on a task with a median result there - so a model missing the two hardest tasks could beat one
        # that ran everywhere and scored below median on them. The filled cells are listed in ``imputed_cells_``.
        if partial_fill not in ("min", "median"):
            raise ValueError(f"Leaderboard: partial_fill must be 'min' or 'median', got {partial_fill!r}")
        self.partial_fill = partial_fill
        self.imputed_cells_ = [(str(table.index[int(r)]), str(table.columns[int(c)])) for r, c in zip(*np.where(table.isna().to_numpy()))]
        # Duplicate task (column) names make list.index and self.weights address only the first occurrence (42-F4).
        if not table.columns.is_unique:
            dupes = table.columns[table.columns.duplicated()].unique().tolist()
            raise ValueError(f"Leaderboard: duplicate task column name(s) {dupes}; every task must be a unique column.")
        self.table = table
        self.tasks = list(self.table.columns)
        self.n_tasks = len(self.tasks)

        self.weights = pd.Series(index=self.tasks, data=1.0)
        weight_dict = weights or {}
        # A typo'd/nonexistent task key would otherwise silently enlarge self.weights via pandas
        # .loc, inflating weights.sum() (the mean_ranking/optimality_gap_ranking denominator) with a
        # bogus entry that never multiplies anything in the numerator -- a silent score-magnitude bug.
        unknown_tasks = set(weight_dict) - set(self.tasks)
        if unknown_tasks:
            raise ValueError(f"Leaderboard: weights key(s) {sorted(unknown_tasks)} do not match any table column (tasks: {self.tasks}).")
        for task, weight in weight_dict.items():
            self.weights.loc[task] = weight
        # mean_ranking(mean_type="arithmetic") / optimality_gap_ranking both divide by
        # self.weights.sum() -- an all-zero-weights Leaderboard (a legal Dict[str, float]) would
        # otherwise silently produce inf/NaN scores for every model instead of raising.
        if self.weights.sum() <= 0:
            raise ValueError(f"Leaderboard: weights must sum to a positive value; got {self.weights.sum()!r} from {weight_dict!r}.")

        self.models = self.table.index.tolist()
        self.n_models = len(self.models)
        # Every _rules.py/_cw.py method does self.ranks.loc[model] expecting a 1-D Series; a
        # duplicated model name returns a 2-D DataFrame instead and silently corrupts every pairwise
        # comparison. data_processing.py's preprocess_glue/preprocess_sglue already dedupe for exactly
        # this reason -- a caller who skips that step (or merges two tables) hits this directly.
        if not self.table.index.is_unique:
            dupes = self.table.index[self.table.index.duplicated()].unique().tolist()
            raise ValueError(
                f"Leaderboard: table.index has duplicate model name(s) {dupes}; every model name must be "
                "unique. See data_processing.preprocess_glue/preprocess_sglue for the expected dedup step."
            )

        self.ranks: pd.DataFrame = None  # set for real by build_ranks() below before any use
        self.max_ranks: pd.DataFrame = None
        self.majority_graph: Optional[pd.DataFrame] = None

        self.build_ranks()

    def build_ranks(self):
        """(Re)compute ``self.ranks``/``self.max_ranks`` from ``self.table`` and invalidate any cached majority graph."""
        # Perf split: the prior single-step build_ranks ALWAYS
        # materialised the n_models x n_models majority_graph, which is the
        # dominant cost on wide tables (10k features = 100M entries / ~800MB
        # at float64, recomputed per Leaderboard rebuild). The majority_graph
        # is only consumed by copeland_ranking() / minimax_ranking() / their
        # election variants. The default RFECV path uses borda_ranking() /
        # mean_ranking() / dowdall_ranking() which need only self.ranks.
        # Lazy-build the graph in _ensure_majority_graph() called from the
        # graph-using methods.

        self.ranks = self.table.rank(method="min", ascending=False)
        self.max_ranks = self.table.rank(method="max", ascending=False)

        self.is_partial = self.table.isna().to_numpy().sum() > 0

        if not self.is_partial:
            self.ranks = self.ranks.astype(int)
            self.max_ranks = self.max_ranks.astype(int)

        # Invalidate any previously materialised majority graph: it was built from a stale
        # ranks state and reusing it would yield wrong condorcet/copeland/minimax results.
        self.majority_graph = None

    def _ensure_majority_graph(self):
        """Lazy-construct the n_models x n_models majority graph. Idempotent
        once self.majority_graph is non-None for the current ranks state.
        Reset by build_ranks() (which sets self.majority_graph = None)."""
        if self.majority_graph is not None:
            return
        # One pass per model: the weighted-below and weighted-above sums are computed once and reused for the win and tie tests (42-F9).
        wins, ties = {}, {}
        for model in self.models:
            below = ((self.ranks.loc[model] < self.ranks) * self.weights).sum(axis=1)
            above = ((self.ranks.loc[model] > self.ranks) * self.weights).sum(axis=1)
            wins[model] = below > above
            ties[model] = below == above

        self.majority_graph = pd.DataFrame(wins).transpose() + 0.5 * pd.DataFrame(ties).transpose() + np.eye(self.n_models) * 0.5

    def elect_all(self, use_methods: Optional[Dict[str, Any]] = None, drop_mean=False):
        """Run every applicable election method (or ``use_methods``' subset, each with its parameter grid) and return a DataFrame of (method, winners, n_winners)."""
        result = []

        if use_methods is not None:
            methods_to_choose = list(use_methods.keys())
            settings_dict = use_methods.copy()
        else:
            methods_to_choose = list(ELECTION_METHODS)
            settings_dict = METHODS_SETTINGS

        # Order-preserving filters (no ``set``): set iteration order is PYTHONHASHSEED-dependent, which would make the assembled DataFrame's row order non-reproducible.
        if self.is_partial:
            _partial = set(PARTIAL_METHODS)
            methods_to_choose = [m for m in methods_to_choose if m in _partial]

        if drop_mean:
            methods_to_choose = [m for m in methods_to_choose if m != "mean"]

        for method in methods_to_choose:
            func = getattr(self, f"{method}_election")
            if method in settings_dict:
                params = ParameterGrid(settings_dict[method])
                for some_params in params:
                    _check_gamma_given(method, some_params)
                    election_result = func(**some_params)
                    result.append((f"Method: {method}, Params: {some_params}", election_result, len(election_result)))
            else:
                _check_gamma_given(method, {})
                election_result = func()
                result.append((f"Method: {method}, Params: {{}}", election_result, len(election_result)))
        final = pd.DataFrame(result, columns=["method", "winners", "n_winners"])
        final["method"] = final["method"].replace(PRETTY_NAMES)
        return final

    def rank_all(self, task_groups=None, group_weights=None, insert_nan=True, use_methods: Optional[Dict[str, Any]] = None, drop_mean=False, return_tie_numbers=False):
        """Run every applicable ranking method (optionally via a two-step grouped ranking) and return a "score: model" table, one column per method/param combo."""
        result = []

        if use_methods is not None:
            methods_to_choose = list(use_methods.keys())
            settings_dict = use_methods.copy()
        else:
            methods_to_choose = list(RANKING_METHODS)
            settings_dict = METHODS_SETTINGS

        # Order-preserving filters (no ``set``): set iteration order is PYTHONHASHSEED-dependent, which would make the assembled ranking DataFrame's column order non-reproducible.
        if self.is_partial:
            _partial = set(PARTIAL_METHODS)
            methods_to_choose = [m for m in methods_to_choose if m in _partial]

        if drop_mean:
            methods_to_choose = [m for m in methods_to_choose if m != "mean"]

        tie_numbers = {}

        result = pd.DataFrame()
        for method in methods_to_choose:
            func = getattr(self, f"{method}_ranking")
            if method in settings_dict:
                params = ParameterGrid(settings_dict[method])
                for some_params in params:
                    if task_groups is None:
                        _check_gamma_given(method, some_params)
                        ranking = func(**some_params)
                    else:
                        ranking = self.two_step_ranking(
                            method,
                            task_groups,
                            group_weights=group_weights,
                            ranking_params=some_params,
                            insert_nan=insert_nan,
                        )
                    to_print = (ranking.apply(lambda x: f"{x:.2f}: ") + ranking.index).to_numpy()
                    result[f"Method: {method}, Params: {some_params}"] = to_print

                    tie_numbers[f"Method: {method}, Params: {some_params}"] = ranking.shape[0] - ranking.nunique(dropna=False)  # NaN ranks count as one shared value, not as absent (42-F13)
            else:
                if task_groups is None:
                    _check_gamma_given(method, {})
                    ranking = func()
                else:
                    ranking = self.two_step_ranking(
                        method,
                        task_groups,
                        group_weights=group_weights,
                        insert_nan=insert_nan,
                    )
                to_print = (ranking.apply(lambda x: f"{x:.2f}: ") + ranking.index).to_numpy()
                result[f"Method: {method}, Params: {{}}"] = to_print
                tie_numbers[f"Method: {method}, Params: {{}}"] = ranking.shape[0] - ranking.nunique(dropna=False)  # NaN ranks count as one shared value, not as absent (42-F13)

        result_df = pd.DataFrame(result).rename(columns=PRETTY_NAMES)
        result_df.index = pd.Series(result_df.index + 1, name="Ranking position")
        # same tolerant lookup as the result frame's rename (audit 42-F1)
        tie_numbers = {PRETTY_NAMES.get(key, key): value for key, value in tie_numbers.items()}

        if not return_tie_numbers:
            return result_df
        else:
            return result_df, tie_numbers

    def get_meta_leaderboard(
        self,
        ranking_method,
        task_groups,
        group_weights=None,
        ranking_params=None,
        insert_nan=True,
    ):
        """Rank models within each ``task_groups`` group (a partition of ``self.tasks``) via ``ranking_method``, then wrap the per-group scores as a new :class:`Leaderboard`."""
        group_merged: list = list(itertools.chain.from_iterable(task_groups.values()))
        group_set, group_counts = np.unique(group_merged, return_counts=True)
        # assert -> ValueError: an assert-based check silently disappears under -O, letting
        # partition violations produce wrong meta-tables with no error at all.
        if set(group_set) != set(self.tasks):
            raise ValueError(f"get_meta_leaderboard: group partition tasks {set(group_set)} " f"do not match self.tasks {set(self.tasks)}.")
        if group_counts.max() != 1:
            raise ValueError(
                f"get_meta_leaderboard: task assignment is not a partition "
                f"(max group_counts={group_counts.max()}); each task must "
                f"belong to exactly one group."
            )

        meta_table = pd.DataFrame(np.nan, index=self.table.index, columns=list(task_groups.keys()), dtype=np.float64)
        for key, tasks in task_groups.items():
            # The parent's partial-table settings apply to every group; dropping them made two_step_ranking silently use the defaults (audit 42-F2).
            sub_lb = Leaderboard(self.table[tasks], weights=self.weights[tasks].to_dict(), allow_partial=self.allow_partial, partial_fill=self.partial_fill)

            func = getattr(sub_lb, f"{ranking_method}_ranking")
            ranking_params = ranking_params or {}
            group_scores = func(**ranking_params)
            if isinstance(group_scores, pd.Series):
                # A ranking that omits a model (e.g. one with no data in the group) leaves it NaN rather than shifting the column.
                group_scores = group_scores.reindex(meta_table.index)
            meta_table[key] = np.asarray(group_scores, dtype=np.float64)

            if insert_nan:
                meta_table.loc[self.table[tasks].isna().all(axis=1), key] = np.nan

        return Leaderboard(meta_table, weights=group_weights)

    def two_step_ranking(
        self,
        ranking_method,
        task_groups,
        group_weights=None,
        ranking_params=None,
        insert_nan=True,
    ):
        """Two-step ranking: rank within task groups via :func:`get_meta_leaderboard`, then rank the resulting group leaderboard by ``ranking_method`` again."""
        meta_lb = self.get_meta_leaderboard(
            ranking_method,
            task_groups,
            group_weights,
            ranking_params,
            insert_nan=insert_nan,
        )
        func = getattr(meta_lb, f"{ranking_method}_ranking")
        ranking_params = ranking_params or {}
        return func(**ranking_params)

    def find_weights_for_condorcet(self, winner_model: str, restrictions: Optional[Dict[str, List]] = None):
        """Find task weights (if any) under which ``winner_model`` beats every other model pairwise (i.e. is the Condorcet winner)."""
        edge_list = []
        for model in self.models:
            if model == winner_model:
                continue
            else:
                edge_list.append((model, winner_model))

        return self._find_weights_for_majority_graph(edge_list, restrictions=restrictions, margin=_CONDORCET_STRICT_MARGIN)

    def split_models_by_feasibility(self, restrictions: Optional[Dict[str, List]] = None):
        """Partition all models into "feasible" (some task-weighting makes them the Condorcet winner) vs "infeasible", per :func:`find_weights_for_condorcet`."""
        result: Dict[str, list] = {"feasible": [], "infeasible": []}
        for model in tqdm(self.models):
            if self.find_weights_for_condorcet(model, restrictions) is None:
                result["infeasible"].append(model)
            else:
                result["feasible"].append(model)

        return result
