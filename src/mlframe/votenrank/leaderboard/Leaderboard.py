
from __future__ import annotations

from pyutilz.system import tqdmu as tqdm
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple

from sklearn.model_selection import ParameterGrid
from scipy.optimize import linprog

from .settings import (
    RANKING_METHODS,
    ELECTION_METHODS,
    PARTIAL_METHODS,
    METHODS_SETTINGS,
    PRETTY_NAMES,
)


class Leaderboard:
    from ._rules import (
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
    from ._cw import _get_tasks_onehot, _find_weights_for_majority_graph

    def __init__(self, table: pd.DataFrame = None, weights: Dict[str, float] = None):

        self.table = table
        self.tasks = list(self.table.columns) if self.table is not None else []
        self.n_tasks = len(self.tasks)

        self.weights = pd.Series(index=self.tasks, data=1.0)
        weight_dict = weights or {}
        for task, weight in weight_dict.items():
            self.weights.loc[task] = weight

        self.models = self.table.index.tolist() if self.table is not None else []
        self.n_models = len(self.models)

        self.ranks, self.max_ranks, self.majority_graph = None, None, None

        self.build_ranks()

    def build_ranks(self):
        # Perf split (2026-05-10): the prior single-step build_ranks ALWAYS
        # materialised the n_models x n_models majority_graph, which is the
        # dominant cost on wide tables (10k features = 100M entries / ~800MB
        # at float64, recomputed per Leaderboard rebuild). The majority_graph
        # is only consumed by copeland_ranking() / minimax_ranking() / their
        # election variants. The default RFECV path uses borda_ranking() /
        # mean_ranking() / dowdall_ranking() which need only self.ranks.
        # Lazy-build the graph in _ensure_majority_graph() called from the
        # graph-using methods.
        self.table = self.table.reindex(self.table.index)

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
        wins = {
            model: ((self.ranks.loc[model] < self.ranks) * self.weights).sum(axis=1)
            > ((self.ranks.loc[model] > self.ranks) * self.weights).sum(axis=1)
            for model in self.models
        }
        ties = {
            model: ((self.ranks.loc[model] < self.ranks) * self.weights).sum(axis=1)
            == ((self.ranks.loc[model] > self.ranks) * self.weights).sum(axis=1)
            for model in self.models
        }

        self.majority_graph = (
            pd.DataFrame(wins).transpose()
            + 0.5 * pd.DataFrame(ties).transpose()
            + np.eye(self.n_models) * 0.5
        )

    def elect_all(self, use_methods: List[str] = None, drop_mean=False):
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
                params = ParameterGrid(METHODS_SETTINGS[method])
                for some_params in params:
                    election_result = func(**some_params)
                    result.append(
                        (
                            f"Method: {method}, Params: {some_params}",
                            election_result,
                            len(election_result)
                        )
                    )
            else:
                election_result = func()
                result.append((f"Method: {method}, Params: {{}}", election_result, len(election_result)))
        final = pd.DataFrame(result, columns=["method", "winners", "n_winners"])
        final["method"].replace(PRETTY_NAMES, inplace=True)
        return final

    def rank_all(
        self,
        task_groups=None,
        group_weights=None,
        insert_nan=True,
        use_methods: List[str] = None,
        drop_mean=False,
        return_tie_numbers=False
    ):
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
                        ranking = func(**some_params)
                    else:
                        ranking = self.two_step_ranking(
                            method,
                            task_groups,
                            group_weights=group_weights,
                            ranking_params=some_params,
                            insert_nan=insert_nan,
                        )
                    to_print = (
                        ranking.apply(lambda x: f"{x:.2f}: ") + ranking.index
                    ).to_numpy()
                    result[f"Method: {method}, Params: {some_params}"] = to_print

                    tie_numbers[f"Method: {method}, Params: {some_params}"] = ranking.shape[0] - ranking.nunique()
            else:
                if task_groups is None:
                    ranking = func()
                else:
                    ranking = self.two_step_ranking(
                        method,
                        task_groups,
                        group_weights=group_weights,
                        insert_nan=insert_nan,
                    )
                to_print = (
                    ranking.apply(lambda x: f"{x:.2f}: ") + ranking.index
                ).to_numpy()
                result[f"Method: {method}, Params: {{}}"] = to_print
                tie_numbers[f"Method: {method}, Params: {{}}"] = ranking.shape[0] - ranking.nunique()

        result_df = pd.DataFrame(result).rename(columns=PRETTY_NAMES)
        result_df.index = pd.Series(result_df.index + 1, name="Ranking position")
        tie_numbers = {PRETTY_NAMES[key]: value for key, value in tie_numbers.items()}

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
        group_merged = sum(list(task_groups.values()), start=[])
        group_set, group_counts = np.unique(group_merged, return_counts=True)
        # Wave 31 (2026-05-20): assert -> ValueError. SILENT-CORRECTNESS
        # bug under -O: partition violations produced wrong meta-tables.
        if set(group_set) != set(self.tasks):
            raise ValueError(
                f"get_meta_leaderboard: group partition tasks {set(group_set)} "
                f"do not match self.tasks {set(self.tasks)}."
            )
        if group_counts.max() != 1:
            raise ValueError(
                f"get_meta_leaderboard: task assignment is not a partition "
                f"(max group_counts={group_counts.max()}); each task must "
                f"belong to exactly one group."
            )

        meta_table = pd.DataFrame(index=self.table.index, columns=task_groups.keys())
        for key, tasks in task_groups.items():
            sub_lb = Leaderboard(
                self.table[tasks], weights=self.weights[tasks].to_dict()
            )

            func = getattr(sub_lb, f"{ranking_method}_ranking")
            ranking_params = ranking_params or {}
            meta_table[key] = func(**ranking_params)

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

    def find_weights_for_condorcet(
        self, winner_model: str, restrictions: Dict[str, List] = None
    ):
        edge_list = []
        for model in self.models:
            if model == winner_model:
                continue
            else:
                edge_list.append((model, winner_model))

        return self._find_weights_for_majority_graph(
            edge_list, restrictions=restrictions
        )

    def split_models_by_feasibility(self, restrictions: Dict[str, List] = None):
        result = {"feasible": [], "infeasible": []}
        for model in tqdm(self.models):
            if self.find_weights_for_condorcet(model, restrictions) == "infeasible":
                result["infeasible"].append(model)
            else:
                result["feasible"].append(model)

        return result
