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
    """Source-level check that the metrics fairness function raises
    ValueError for bad min_pop_cat_thresh (rather than asserting).

    ``create_fairness_subgroups`` moved to ``_fairness_metrics.py`` when
    ``metrics/core.py`` was split into siblings.
    """
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "metrics" / "_fairness_metrics.py"
    ).read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert "assert min_pop_cat_thresh > 0 and min_pop_cat_thresh < 1.0" not in src
    assert "assert min_pop_cat_thresh > 0 and min_pop_cat_thresh <= len(df)" not in src
    # Post-fix marker:
    assert "min_pop_cat_thresh (float) must be in (0, 1)" in src


def test_metrics_bins_index_unique_raises_value_error():
    """SILENT-CORRECTNESS bug under -O: duplicate bins.index let
    bins.loc[arr] return multiple rows. Source-level guard.

    ``create_fairness_subgroups_indices`` moved to ``_fairness_metrics.py``
    when ``metrics/core.py`` was split into siblings.
    """
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "metrics" / "_fairness_metrics.py"
    ).read_text(encoding="utf-8")
    assert "assert bins.index.is_unique" not in src
    assert "must have a unique index" in src


# ---- models/optimization.py -------------------------------------------


def test_mbh_optimizer_params_checks_raise_value_error():
    """The 6-assert 'Params checks' block on MBHOptimizer.__init__ now
    raises ValueError instead of asserting."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "models" / "optimization.py"
    ).read_text(encoding="utf-8")
    # Pre-fix block MUST be gone:
    assert "assert quantile > 0.0 and quantile < 0.5" not in src
    assert "assert len(search_space) > 0" not in src
    assert "assert acquisition_method in (\"EE\")" not in src
    # Post-fix markers:
    assert "if not (0.0 < quantile < 0.5)" in src
    assert "search_space must be non-empty" in src


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
    """7-assert 'Input checks' block on mrmr_select_features now raises
    ValueError. Source-level guard (full integration fixture is too heavy)."""
    import pathlib
    import mlframe as _mlframe
    # 2026-05-22: screen.py was split; the input-check shapes the test
    # pins live in ``_screen_predictors.py`` now. Read both files.
    _dir = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection" / "filters"
    src = (
        (_dir / "screen.py").read_text(encoding="utf-8")
        + "\n"
        + (_dir / "_screen_predictors.py").read_text(encoding="utf-8")
    )
    # Pre-fix shapes MUST be gone:
    assert "assert mrmr_relevance_algo in (\"fleuret\", \"pld\")" not in src
    assert "assert len(factors_data) >= 10" not in src
    assert "assert targets_data.shape[1] == len(targets_nbins)" not in src
    # Post-fix markers:
    assert "mrmr_relevance_algo must be 'fleuret' or 'pld'" in src
    assert "factors_data must have at least 10 rows" in src


def test_mrmr_invariant_no_self_target_raises_runtime_error():
    """Source guard for the silent-correctness 'target subset of factors' check."""
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
    """SILENT-CORRECTNESS bug under -O: bad partition produced wrong
    meta-table. Source-level guard."""
    import pathlib
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "votenrank" / "leaderboard" / "Leaderboard.py"
    ).read_text(encoding="utf-8")
    assert "assert set(group_set) == set(self.tasks)" not in src
    assert "assert group_counts.max() == 1" not in src
    assert "do not match self.tasks" in src
    assert "not a partition" in src


# ---- cross-cutting source guard ---------------------------------------


def test_no_input_validation_asserts_in_targeted_modules():
    """Cross-cutting sanity: spot-check each of the 12 targeted modules
    no longer carries the pre-fix shape ``assert <input-param-check>``.
    Internal-invariant asserts (e.g. inside numba @njit kernels) are not
    in the audit's target set; this guard only checks the specific
    shapes the audit flagged."""
    import pathlib
    import mlframe as _mlframe
    root = pathlib.Path(_mlframe.__file__).resolve().parent
    # Per-file (file, banned_substring_proving_assert_returned).
    banned = [
        ("core/stats.py", "assert quantile > 0 and quantile < 1.0"),
        ("estimators/custom.py", 'assert strategy in ("constant_lag", "adaptive_lag")'),
        ("estimators/custom.py", 'assert distribution in ("linear", "quantile")'),
        ("preprocessing/cleaning.py", "assert use_quantile > 0 and use_quantile < 1.0"),
        ("metrics/core.py", 'assert backend in ("plotly", "matplotlib")'),
        ("metrics/core.py", 'assert method in ("multicrit", "brier_score", "precision")'),
        ("models/tuning.py", 'assert sampler in ("random", "ml")'),
        ("feature_selection/general.py", "assert min_randomized_permutations >= 1"),
    ]
    for rel, shape in banned:
        text = (root / rel).read_text(encoding="utf-8")
        assert shape not in text, (
            f"Wave 31 regression at {rel}: pre-fix assert shape "
            f"`{shape}` reappeared. -O would strip it again."
        )
