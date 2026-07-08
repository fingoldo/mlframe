"""Wave-31 sensor: ``assert`` -> ``raise ValueError`` for production input validation.

The wave-31 audit found 47 production sites using ``assert`` as the
default validation syntax. Under ``python -O`` (production-perf deploys,
embedded inference), ALL asserts are stripped -- public-API entries lost
their input validation silently.

Two of the most-impactful sites were silent-correctness bugs (-O lets
wrong reductions reach the metric):
- ``metrics/core.py:4176`` (``bins.index.is_unique``)
- ``votenrank/Leaderboard.py:230-231`` (group_set / group_counts partition)

This sensor confirms that the top-priority sites now raise ValueError
(not assert) so the check survives -O. Behavioural verification: invalid
input -> ValueError; valid input -> normal return.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---- core/stats.py -----------------------------------------------------


def test_get_tukey_fences_quantile_zero_raises_value_error():
    from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile
    with pytest.raises(ValueError, match="quantile must be in"):
        get_tukey_fences_multiplier_for_quantile(quantile=0.0)


def test_get_tukey_fences_quantile_one_raises_value_error():
    from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile
    with pytest.raises(ValueError, match="quantile must be in"):
        get_tukey_fences_multiplier_for_quantile(quantile=1.0)


def test_get_tukey_fences_valid_quantile_returns_float():
    from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile
    out = get_tukey_fences_multiplier_for_quantile(quantile=0.25, sd_sigma=2.7)
    assert isinstance(out, float)


# ---- estimators/custom.py ---------------------------------------------


def test_soft_winsorize_bad_distribution_raises_value_error():
    from mlframe.estimators.custom import soft_winsorize
    with pytest.raises(ValueError, match="distribution must be"):
        soft_winsorize(
            np.array([1.0, 2.0, 3.0, 100.0]),
            rel_lower_limit=0.1, rel_upper_limit=0.1,
            abs_lower_threshold=1.0, abs_upper_threshold=10.0,
            distribution="banana",
        )


def test_clip_to_quantiles_bad_method_raises_value_error():
    from mlframe.estimators.custom import clip_to_quantiles
    with pytest.raises(ValueError, match="method must be"):
        clip_to_quantiles(np.array([1.0, 2.0, 3.0]), method="banana")


def test_clip_to_quantiles_bad_quantile_raises_value_error():
    from mlframe.estimators.custom import clip_to_quantiles
    with pytest.raises(ValueError, match="quantile must be in"):
        clip_to_quantiles(np.array([1.0, 2.0]), quantile=1.5)


def test_identity_estimator_no_features_raises_value_error():
    from mlframe.estimators.custom import IdentityEstimator
    import pandas as pd
    est = IdentityEstimator()  # neither feature_names nor feature_indices set
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="feature"):
        est.predict(df)


# ---- preprocessing/cleaning.py ----------------------------------------


def test_is_variable_truly_continuous_none_inputs_raises():
    from mlframe.preprocessing.cleaning import is_variable_truly_continuous
    with pytest.raises(ValueError, match="must be provided"):
        is_variable_truly_continuous(values=None, df=None, variable_name=None)


# ---- metrics/core.py --------------------------------------------------


def test_metrics_fairness_bad_min_pop_cat_thresh_raises():
    """Invalid min_pop_cat_thresh must raise ValueError (survives -O), not assert."""
    import pandas as pd
    from mlframe.metrics._fairness_metrics import create_fairness_subgroups
    df = pd.DataFrame({"a": list(range(1, 11)), "b": list("aabbccddee")})
    with pytest.raises(ValueError, match=r"min_pop_cat_thresh.*must be in"):
        create_fairness_subgroups(df=df, features=["a", "b"], min_pop_cat_thresh=0.0)
    with pytest.raises(ValueError, match=r"min_pop_cat_thresh"):
        create_fairness_subgroups(df=df, features=["a", "b"], min_pop_cat_thresh=-1)


def test_metrics_bins_index_unique_raises_value_error():
    """SILENT-CORRECTNESS under -O: a non-unique bins.index made bins.loc[arr]
    return multiple rows per key. The guard must raise ValueError, not assert."""
    import numpy as np
    import pandas as pd
    from mlframe.metrics._fairness_metrics import create_fairness_subgroups_indices
    subgroups = {"grp": {"bins": pd.Series(["a", "b", "a"], index=[0, 0, 1])}}
    with pytest.raises(ValueError, match="unique index"):
        create_fairness_subgroups_indices(
            subgroups=subgroups,
            train_idx=np.array([0, 1]),
            val_idx=np.array([0]),
            test_idx=np.array([1]),
        )


# ---- models/optimization.py -------------------------------------------


def test_mbh_optimizer_params_checks_raise_value_error():
    """MBHOptimizer.__init__ param checks must raise ValueError (survives -O), not assert."""
    from mlframe.models.optimization import MBHOptimizer
    with pytest.raises(ValueError, match=r"quantile must be in"):
        MBHOptimizer(search_space={"x": [1, 2, 3]}, ground_truth=None, quantile=0.6)
    with pytest.raises(ValueError, match="search_space must be non-empty"):
        MBHOptimizer(search_space={}, ground_truth=None)
    with pytest.raises(ValueError, match="acquisition_method"):
        MBHOptimizer(search_space={"x": [1, 2, 3]}, ground_truth=None, acquisition_method="banana")


# ---- models/ensembling.py ---------------------------------------------


def test_ensembling_unknown_method_raises_value_error():
    from mlframe.models.ensembling import ensemble_probabilistic_predictions
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=(10, 2))
    p /= p.sum(axis=1, keepdims=True)
    # Function signature is variadic: *preds, then ensemble_method as kwarg.
    with pytest.raises(ValueError, match="unknown ensemble_method"):
        ensemble_probabilistic_predictions(p, p, ensemble_method="banana")


# ---- feature_selection/filters/screen.py -----------------------------


def test_mrmr_select_features_input_checks_raise_value_error():
    """screen_predictors input checks must raise ValueError (survives -O), not assert."""
    import numpy as np
    from mlframe.feature_selection.filters._screen_predictors import screen_predictors
    fd = np.random.default_rng(0).random((50, 3))
    td = np.random.default_rng(1).integers(0, 2, (50, 1))
    with pytest.raises(ValueError, match="mrmr_relevance_algo must be"):
        screen_predictors(factors_data=fd, factors_nbins=[3, 3, 3], targets_data=td,
                          targets_nbins=[2], mrmr_relevance_algo="banana")
    with pytest.raises(ValueError, match="at least 10 rows"):
        screen_predictors(factors_data=fd[:5], factors_nbins=[3, 3, 3],
                          targets_data=td[:5], targets_nbins=[2])
    with pytest.raises(ValueError, match=r"must equal len\(targets_nbins\)"):
        screen_predictors(factors_data=fd, factors_nbins=[3, 3, 3], targets_data=td,
                          targets_nbins=[2, 2])


def test_mrmr_invariant_no_self_target_raises_runtime_error():
    """Source guard for the silent-correctness 'target subset of factors' invariant.

    Kept source-level: the guard fires deep in the index-derivation branch that is not
    cheaply reachable from the public entry without reverse-engineering internal x/y sets.
    """
    import pathlib
    import mlframe as _mlframe
    _dir = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection" / "filters"
    src = (
        (_dir / "screen.py").read_text(encoding="utf-8")
        + "\n"
        + (_dir / "_screen_predictors.py").read_text(encoding="utf-8")
    )
    assert "assert not set(y).issubset(set(x))" not in src
    assert "target index set is a subset" in src


# ---- votenrank/Leaderboard.py -----------------------------------------


def test_leaderboard_partition_check_raises_value_error():
    """SILENT-CORRECTNESS under -O: a bad task->group partition produced a wrong
    meta-table. get_meta_leaderboard must raise ValueError, not assert."""
    import pandas as pd
    from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard
    table = pd.DataFrame({"t1": [1.0, 2.0], "t2": [3.0, 4.0], "t3": [5.0, 6.0]}, index=["m1", "m2"])
    lb = Leaderboard(table, weights={"t1": 1.0, "t2": 1.0, "t3": 1.0})
    with pytest.raises(ValueError, match="do not match"):
        lb.get_meta_leaderboard("mean", {"g1": ["t1"], "g2": ["t2"]})
    with pytest.raises(ValueError, match="not a partition"):
        lb.get_meta_leaderboard("mean", {"g1": ["t1", "t2", "t3"], "g2": ["t1"]})


# ---- cross-cutting source guard ---------------------------------------


def test_input_validation_raises_in_targeted_modules():
    """Behavioural cross-cut: invalid input to the migrated entrypoints must raise
    ValueError (survives -O), not assert. Covers the cheaply-reachable audit sites."""
    import numpy as np
    import pandas as pd
    from mlframe.estimators.custom import (
        create_dummy_lagged_predictions, soft_winsorize, clip_to_quantiles,
    )
    from mlframe.preprocessing.cleaning import is_variable_truly_continuous

    with pytest.raises(ValueError, match="strategy must be"):
        create_dummy_lagged_predictions(np.arange(20.0), strategy="banana", lag=1)
    with pytest.raises(ValueError, match="distribution must be"):
        soft_winsorize(np.array([1.0, 2.0, 3.0, 100.0]), rel_lower_limit=0.1,
                       rel_upper_limit=0.1, abs_lower_threshold=1.0,
                       abs_upper_threshold=10.0, distribution="banana")
    with pytest.raises(ValueError, match="method must be"):
        clip_to_quantiles(np.array([1.0, 2.0, 3.0]), method="banana")
    with pytest.raises(ValueError, match="use_quantile"):
        is_variable_truly_continuous(
            df=pd.DataFrame({"a": np.arange(20.0)}), variable_name="a", use_quantile=1.5,
        )


def test_no_input_validation_asserts_in_residual_modules():
    """Source-level residual guard for the few audit sites whose validation fires deep in
    plotting / tuning / permutation paths that are not cheaply reachable behaviourally;
    confirm the pre-fix ``assert <param-check>`` shapes have not reappeared (-O strips them)."""
    import pathlib
    import mlframe as _mlframe
    root = pathlib.Path(_mlframe.__file__).resolve().parent
    banned = [
        ("metrics/core.py", 'assert backend in ("plotly", "matplotlib")'),
        ("metrics/core.py", 'assert method in ("multicrit", "brier_score", "precision")'),
        ("models/tuning.py", 'assert sampler in ("random", "ml")'),
        ("feature_selection/general.py", "assert min_randomized_permutations >= 1"),
    ]
    for rel, shape in banned:
        text = (root / rel).read_text(encoding="utf-8")
        assert shape not in text, (
            f"Wave 31 regression at {rel}: pre-fix assert shape `{shape}` reappeared; -O would strip it."
        )
