"""Deterministic synthetic dataset generators for the MRMR benchmark suite.

Each generator returns ``(X, y, scenario_id)`` where ``scenario_id`` is a stable
string identifier used as a key in the benchmark output JSON. Generators are pure
functions of their inputs (random_state included) so two runs on the same machine
produce identical outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    p: int
    informative: int
    task: str  # "classification" | "regression"
    n_classes: int = 5
    use_gpu: bool = False
    fe_max_steps: int = 0


# 6 CPU + 2 GPU-paired (paired with #2 and #6 respectively).
if __name__ == "__main__":
    SCENARIOS: dict[str, Scenario] = {
        "n10k_p100_clf": Scenario("n10k_p100_clf", 10_000, 100, 10, "classification", n_classes=5),
        "n100k_p100_clf": Scenario("n100k_p100_clf", 100_000, 100, 20, "classification", n_classes=5),
        "n10k_p1000_clf": Scenario("n10k_p1000_clf", 10_000, 1_000, 30, "classification", n_classes=5),
        "n100k_p1000_clf": Scenario("n100k_p1000_clf", 100_000, 1_000, 50, "classification", n_classes=5),
        "n10k_p100_reg": Scenario("n10k_p100_reg", 10_000, 100, 10, "regression"),
        "n50k_p200_fe": Scenario("n50k_p200_fe", 50_000, 200, 20, "classification", n_classes=5, fe_max_steps=1),
        "n100k_p100_clf_gpu": Scenario("n100k_p100_clf_gpu", 100_000, 100, 20, "classification", n_classes=5, use_gpu=True),
        "n50k_p200_fe_gpu": Scenario("n50k_p200_fe_gpu", 50_000, 200, 20, "classification", n_classes=5, fe_max_steps=1, use_gpu=True),
    }

    CPU_SCENARIOS = [s for s in SCENARIOS.values() if not s.use_gpu]
    GPU_SCENARIOS = [s for s in SCENARIOS.values() if s.use_gpu]


    def make_scenario_data(scenario: Scenario, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate (X, y) for the given scenario. Deterministic for fixed random_state."""
        if scenario.task == "classification":
            X, y = make_classification(
                n_samples=scenario.n,
                n_features=scenario.p,
                n_informative=scenario.informative,
                n_redundant=0,
                n_classes=scenario.n_classes,
                n_clusters_per_class=1,
                random_state=random_state,
            )
        elif scenario.task == "regression":
            X, y = make_regression(
                n_samples=scenario.n,
                n_features=scenario.p,
                n_informative=scenario.informative,
                noise=0.1,
                random_state=random_state,
            )
        else:
            raise ValueError(f"unknown task: {scenario.task}")
        cols = [f"f_{i}" for i in range(scenario.p)]
        X_df = pd.DataFrame(X, columns=cols)
        return X_df, y


    def list_scenarios(include_gpu: bool = False) -> list[Scenario]:
        """Return scenarios in stable order. GPU scenarios appended last when included."""
        if include_gpu:
            return CPU_SCENARIOS + GPU_SCENARIOS
        return list(CPU_SCENARIOS)
