"""Independence-of-Irrelevant-Alternatives (IIA) violation rate for a leaderboard ranking method.

IIA asks: does adding a new model to the leaderboard change the RELATIVE order of the models
already there? A social-choice-theoretic ranking method that violates IIA can flip which of two
existing models looks "better" just because a third, unrelated model joined the comparison --
undesirable for a leaderboard, where new submissions arrive continuously. ``compute_iia`` measures
the violation rate empirically by repeatedly adding models one at a time in a random order and
counting how often the ranking of the already-present models changes.
"""
from __future__ import annotations

import numpy as np

from pyutilz.system import tqdmu as tqdm

from . import Leaderboard


def fine_sorted_ranking(ranking):
    """Sort a ``{model: rank}`` mapping into a model-name list ordered by descending rank (ties broken by name)."""
    big_list = [(rank, model) for model, rank in ranking.items()]
    big_list.sort(reverse=True)
    return [model for rank, model in big_list]


def compute_iia_for_fixed_models(method, table, models_order, weights):
    """Count IIA violations for one fixed model-arrival order: how many times adding the next model changes the relative ranking of the models already present."""
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
    """Monte-Carlo IIA violation rate for ``method`` over ``num_repetitions`` random model-arrival orders. Returns ``(mean, std, per_repetition_counts)``."""
    results = []
    for i in tqdm(range(num_repetitions), leave=False):
        models_order = table.index.tolist()
        # Use a local Generator instead of mutating the process-global np.random RNG.
        rng = np.random.default_rng(i)
        rng.shuffle(models_order)
        results.append(compute_iia_for_fixed_models(method, table, models_order, weights))
    return np.mean(results), np.std(results), results
