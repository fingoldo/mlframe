"""cProfile benchmark for RoundedNumericCategoricalInteraction.

COMPETITION / EXPLORATORY ONLY — see mlframe.competition.rounded_categorical_interaction.

Run: python -m mlframe.competition._benchmarks.bench_rounded_categorical_interaction
"""

from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.competition.rounded_categorical_interaction import RoundedNumericCategoricalInteraction


def run(n: int = 2_000_000) -> pd.Series:
    rng = np.random.default_rng(0)
    numeric = pd.Series(rng.uniform(0.0, 1000.0, size=n), name="numeric_col")
    categorical = pd.Series(rng.choice([f"cat_{i}" for i in range(500)], size=n), name="cat_col")

    interaction = RoundedNumericCategoricalInteraction(decimals=2, sep="|")
    return interaction.transform(numeric, categorical)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    result = run()
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())
    print(f"rows produced: {len(result)}")


if __name__ == "__main__":
    main()
