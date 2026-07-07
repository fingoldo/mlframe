
from __future__ import annotations

import numpy as np

from pyutilz.system import tqdmu as tqdm

from . import Leaderboard


def fine_sorted_ranking(ranking):
    big_list = [(rank, model) for model, rank in ranking.items()]
    big_list.sort(reverse=True)
    return [model for rank, model in big_list]


def compute_iia_for_fixed_models(method, table, models_order, weights):
    result = 0

    ranking_kwargs = {"gamma": 95} if method == "optimality_gap" else {}

    base_lb = Leaderboard(table.loc[models_order[:2]], weights)
    last_ranking = fine_sorted_ranking(getattr(base_lb, f"{method}_ranking")(**ranking_kwargs))

    for current_models_order in range(3, len(models_order) + 1):
        current_lb = Leaderboard(table.loc[models_order[:current_models_order]], weights)
        current_ranking = fine_sorted_ranking(getattr(current_lb, f"{method}_ranking")(**ranking_kwargs))
        current_ranking_without_new_model = current_ranking.copy()
        current_ranking_without_new_model.remove(models_order[current_models_order - 1])
        result += last_ranking != current_ranking_without_new_model
        last_ranking = current_ranking
    
    return result


def compute_iia(method, table, weights, num_repetitions):
    results = []
    for i in tqdm(range(num_repetitions), leave=False):
        models_order = table.index.tolist()
        # Wave 49 (2026-05-20): use a local Generator instead of mutating the
        # process-global np.random RNG.
        rng = np.random.default_rng(i)
        rng.shuffle(models_order)
        results.append(compute_iia_for_fixed_models(method, table, models_order, weights))
    return np.mean(results), np.std(results), results
