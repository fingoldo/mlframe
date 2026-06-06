"""A4-06: the verbose-only strategy resolution in ``_phase_pandas_conversion_and_cat_prep``
reuses the precomputed ``strategy_by_model`` map instead of re-walking ``get_strategy``."""
from __future__ import annotations

import pandas as pd

from mlframe.training.core import _phase_helpers as ph
from mlframe.training.strategies import get_strategy


def test_a4_06_verbose_branch_reuses_strategy_map(monkeypatch) -> None:
    """With a strategy_by_model map threaded in, the verbose conversion-phase log path must
    not call module-level get_strategy (the re-walk the precomputed map exists to avoid)."""
    models = ["lgb", "linear"]
    strategy_map = {id(m): get_strategy(m) for m in models}

    calls: list = []
    real_get_strategy = ph.get_strategy

    def _spy(m):
        calls.append(m)
        return real_get_strategy(m)

    monkeypatch.setattr(ph, "get_strategy", _spy)

    # pandas inputs -> not deferred -> hits the verbose "conversion needed" branch that resolves strategies.
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    val = pd.DataFrame({"a": [1.0, 2.0]})
    test = pd.DataFrame({"a": [1.0]})

    ph._phase_pandas_conversion_and_cat_prep(
        train_df=train, val_df=val, test_df=test,
        train_df_polars_pre=None, val_df_polars_pre=None, test_df_polars_pre=None,
        cat_features=[], was_polars_input=False, all_models_polars_native=False,
        needs_polars_pre_clone=False, mlframe_models=models,
        recurrent_models=[], rfecv_models=[],
        baseline_rss_mb=0.0, df_size_mb=0.0, verbose=True,
        strategy_by_model=strategy_map,
    )
    assert calls == [], f"get_strategy was re-walked despite a threaded map: {calls}"


def test_a4_06_falls_back_to_get_strategy_without_map(monkeypatch) -> None:
    """Without a map, the verbose branch must still resolve strategies via get_strategy (correctness)."""
    models = ["lgb", "linear"]
    calls: list = []
    real_get_strategy = ph.get_strategy

    def _spy(m):
        calls.append(m)
        return real_get_strategy(m)

    monkeypatch.setattr(ph, "get_strategy", _spy)

    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    ph._phase_pandas_conversion_and_cat_prep(
        train_df=train, val_df=None, test_df=None,
        train_df_polars_pre=None, val_df_polars_pre=None, test_df_polars_pre=None,
        cat_features=[], was_polars_input=False, all_models_polars_native=False,
        needs_polars_pre_clone=False, mlframe_models=models,
        recurrent_models=[], rfecv_models=[],
        baseline_rss_mb=0.0, df_size_mb=0.0, verbose=True,
        strategy_by_model=None,
    )
    assert calls, "get_strategy should be used as fallback when no map is threaded in"
