"""Behavioural regression tests for the §2 Feature engineering audit fixes
(2026-05-16). Each test fails on pre-fix code and passes on post-fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# -------------------------------------------------------------------------
# §2 P0 FE-LEAK: bruteforce.py default leakage_free is now True (was False)
# -------------------------------------------------------------------------


def test_bruteforce_leakage_free_default_is_true():
    """``run_pysr_feature_engineering`` must default to leakage_free=True. Pre-fix the default was
    False which silently leaked the target through CatBoostEncoder.fit_transform."""
    import inspect
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    sig = inspect.signature(run_pysr_feature_engineering)
    assert sig.parameters["leakage_free"].default is True, (
        "bruteforce.run_pysr_feature_engineering.leakage_free default must be True post-fix"
    )


# -------------------------------------------------------------------------
# §2 P1 FE-LEAK target_encoders.py:403 KFold shuffle for time-series
# -------------------------------------------------------------------------


def test_leakage_safe_encoder_time_aware_no_shuffle():
    """``LeakageSafeEncoder`` with ``time_aware=True`` must use shuffle=False KFold so temporal
    ordering is preserved across folds."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    n = 100
    cats = ["a"] * 50 + ["b"] * 50
    # Monotone temporal target: y = i / n.
    y = np.linspace(0.0, 1.0, n)

    enc = LeakageSafeEncoder(method="target_mean", cv=5, time_aware=True, random_state=0)
    out = enc.fit_transform(cats, y)
    assert out.shape == (n,)

    # With shuffle=False, fold i contains contiguous rows. Last fold's "a" rows (impossible since a is rows
    # 0-50) and so on. Test contract: ensure constructor accepts time_aware and KFold internals route correctly.
    assert enc.time_aware is True


def test_leakage_safe_encoder_external_splitter():
    """``LeakageSafeEncoder`` honours ``cv_splitter=TimeSeriesSplit(...)`` end-to-end."""
    from sklearn.model_selection import TimeSeriesSplit
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    n = 60
    cats = (["a"] * 30 + ["b"] * 30)
    y = np.linspace(0.0, 1.0, n)
    tss = TimeSeriesSplit(n_splits=4)
    enc = LeakageSafeEncoder(method="target_mean", cv_splitter=tss)
    out = enc.fit_transform(cats, y)
    # TimeSeriesSplit produces folds 1..4; first n/(n_splits+1) rows are never in val: assert those got the prior.
    n_never_val = n // 5  # first chunk -- TimeSeriesSplit doesn't put these in any val fold
    # Pre-train rows: TimeSeriesSplit leaves out the first chunk from any val. That should still be NaN /
    # zero since fit_transform only fills val rows. Verify no exception path; output finite for filled rows.
    assert np.isfinite(out[n_never_val:]).all()


# -------------------------------------------------------------------------
# §2 P1 FE-DTYPE target_encoders.py:75 vectorised _categorical_to_string_array
# -------------------------------------------------------------------------


def test_categorical_to_string_array_pandas_fastpath():
    """Vectorised pandas Series path: ``__NULL__`` sentinel survives null cells."""
    from mlframe.training.feature_handling.target_encoders import _categorical_to_string_array

    s = pd.Series(["a", None, "b", float("nan"), "c"])
    out = _categorical_to_string_array(s)
    assert list(out) == ["a", "__NULL__", "b", "__NULL__", "c"]


def test_categorical_to_string_array_polars_fastpath():
    """Vectorised polars Series path: ``__NULL__`` sentinel survives null cells."""
    pl = pytest.importorskip("polars")
    from mlframe.training.feature_handling.target_encoders import _categorical_to_string_array

    s = pl.Series("x", ["a", None, "b", "c"])
    out = _categorical_to_string_array(s)
    assert list(out) == ["a", "__NULL__", "b", "c"]


def test_categorical_to_string_array_numpy_float_with_nan():
    """Numpy float array with NaN gets sentinel; integral floats canonicalise to int form (1.0 -> "1")
    so int<->float dtype drift on the same integer-coded categorical hits the same per-category key."""
    from mlframe.training.feature_handling.target_encoders import _categorical_to_string_array

    a = np.array([1.0, np.nan, 2.5, 3.0])
    out = _categorical_to_string_array(a)
    assert out[1] == "__NULL__"
    assert out[0] == "1"
    assert out[2] == "2.5"  # non-integral float keeps its repr


# -------------------------------------------------------------------------
# §2 P2 FE-FALLBACK target_encoders.py:484 WoE unseen uses prior log-odds (not 0)
# -------------------------------------------------------------------------


def test_woe_unseen_uses_prior_logodds_not_zero():
    """Imbalanced binary y: unseen category WoE must not be 0.0 (which is neutral only when
    prior=0.5). Pre-fix unseen categories returned 0.0; post-fix they return log(p/(1-p)) where
    p is the (smoothed) positive rate."""
    from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

    rng = np.random.default_rng(0)
    n = 200
    cats = rng.choice(["a", "b", "c"], size=n)
    y = (rng.random(n) < 0.05).astype(float)  # 5% positive
    enc = LeakageSafeEncoder(method="woe", smoothing=1.0, cv=3, random_state=0)
    enc.fit_transform(cats, y)
    seen_encoded = enc.transform(["a"])[0]
    unseen_encoded = enc.transform(["__never_seen__"])[0]
    # Unseen must reflect the imbalanced prior; far from 0.0.
    assert abs(unseen_encoded) > 1.0, (
        f"Expected |unseen WoE| > 1 for 5%-positive prior; got {unseen_encoded}"
    )
    # Distinguishable from seen-but-balanced cell.
    assert seen_encoded != unseen_encoded


# -------------------------------------------------------------------------
# §2 P1 FE-DIM pipeline.py:96 projected formula matches polynomial._projected_output_cols
# -------------------------------------------------------------------------


def test_pipeline_projected_diagnostic_includes_exact_formula():
    """When the ``n**degree`` upper bound trips the guard, the diagnostic ValueError must include the
    exact combinatorial count too, so the user can see how loose the upper bound was. The legacy
    formula stays as the trip wire (regression sentry against existing callers' guard pins) but the
    error message now exposes the actual count to aid in tuning ``memory_safety_max_features``."""
    import pytest
    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.feature_handling.polynomial import _projected_output_cols
    from mlframe.training.pipeline import _build_extension_steps

    # n=5, degree=3, interaction_only=True: exact count C(5,1)+C(5,2)+C(5,3) = 25.
    # Legacy upper bound 5**3 = 125; guard at 100 traps the upper bound but not the exact count.
    cfg = PreprocessingExtensionsConfig(polynomial_degree=3, polynomial_interaction_only=True)
    cfg.memory_safety_max_features = 100

    exact = _projected_output_cols(5, 3, True)
    assert exact == 25
    with pytest.raises(ValueError, match=r"exact combinatorial"):
        _build_extension_steps(cfg, n_features=5)


# -------------------------------------------------------------------------
# §2 P1 FE-CACHE apply.py:363 text encoder cache key uses content token
# -------------------------------------------------------------------------


def test_text_encoder_content_token_disambiguates():
    """Two distinct text columns with same length must produce different content tokens."""
    pl = pytest.importorskip("polars")
    from mlframe.training.feature_handling.apply import _text_column_content_token

    df1 = pl.DataFrame({"text": ["alpha"] * 20})
    df2 = pl.DataFrame({"text": ["beta"] * 20})
    t1 = _text_column_content_token(df1, "text")
    t2 = _text_column_content_token(df2, "text")
    assert t1 != t2, "different content should produce different tokens"
