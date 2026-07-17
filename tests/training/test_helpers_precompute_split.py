"""Wave 95 (2026-05-21): split helpers.py (1232 lines)
into helpers.py (now 987 lines) + new _precompute.py (290 lines).

Moved to the sibling file:
  - get_trainset_features_stats / _polars (train-set min/max/cat-vals)
  - TrainMlframeSuitePrecomputed dataclass
  - precompute_composite_target_specs (NotImplementedError stub)
  - precompute_dummy_baselines (NotImplementedError stub)
  - precompute_trainset_features_stats (backend dispatcher)
  - precompute_all (one-shot helper)

Original re-exports the 7 names so existing
``from mlframe.training.helpers import precompute_all`` imports still
resolve.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_precompute_symbols_still_importable_from_facade() -> None:
    from mlframe.training.helpers import (
        get_trainset_features_stats,
        get_trainset_features_stats_polars,
        TrainMlframeSuitePrecomputed,
        precompute_composite_target_specs,
        precompute_dummy_baselines,
        precompute_trainset_features_stats,
        precompute_all,
    )

    assert callable(get_trainset_features_stats)
    assert callable(get_trainset_features_stats_polars)
    assert TrainMlframeSuitePrecomputed is not None
    assert callable(precompute_composite_target_specs)
    assert callable(precompute_dummy_baselines)
    assert callable(precompute_trainset_features_stats)
    assert callable(precompute_all)


def test_other_helpers_symbols_still_importable() -> None:
    from mlframe.training.helpers import (
        parse_catboost_devices,
        get_training_configs,
        compute_cb_text_processing,
    )

    assert callable(parse_catboost_devices)
    assert callable(get_training_configs)
    assert callable(compute_cb_text_processing)


def test_facade_below_1k_line_threshold() -> None:
    root = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training"
    facade = root / "helpers.py"
    n = len(facade.read_text(encoding="utf-8").splitlines())
    assert n < 1000, f"helpers.py is {n} lines, still over the 1k threshold"


def test_precompute_module_owns_the_moved_symbols() -> None:
    """Identity: facade and sibling module expose the SAME object."""
    from mlframe.training import helpers, _precompute

    for name in (
        "get_trainset_features_stats",
        "get_trainset_features_stats_polars",
        "TrainMlframeSuitePrecomputed",
        "precompute_composite_target_specs",
        "precompute_dummy_baselines",
        "precompute_trainset_features_stats",
        "precompute_all",
    ):
        assert getattr(helpers, name) is getattr(_precompute, name), name


def test_precompute_trainset_features_stats_round_trip() -> None:
    """Functional smoke: pandas frame -> stats dict has min/max keys."""
    from mlframe.training.helpers import precompute_trainset_features_stats

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    stats = precompute_trainset_features_stats(df)
    assert "min" in stats
    assert "max" in stats
    assert float(stats["min"]["x"]) == 1.0
    assert float(stats["max"]["y"]) == 30.0


def test_precompute_dummy_and_composite_stubs_raise() -> None:
    """The two stub helpers are documented as NotImplementedError."""
    from mlframe.training.helpers import (
        precompute_composite_target_specs,
        precompute_dummy_baselines,
    )

    with pytest.raises(NotImplementedError):
        precompute_composite_target_specs(train_df=None, target_by_type={}, config=None)
    with pytest.raises(NotImplementedError):
        precompute_dummy_baselines(train_df=None, target_by_type={}, config=None)


def test_precompute_all_fills_only_stats_slot() -> None:
    """precompute_all should populate trainset_features_stats and leave the
    other slots at None so the suite's in-line compute still runs."""
    from mlframe.training.helpers import precompute_all

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    bundle = precompute_all(df, target_by_type=None)
    assert bundle.trainset_features_stats is not None
    assert bundle.dummy_baselines is None
    assert bundle.composite_target_specs is None
    assert bundle.train_df_fingerprint is None
