"""Regression sensor for S67 (Wave 9c).

W7D shipped a WARN when the bare ``pl.Categorical`` fallback fires; W9c
completes the persistence story:

1. ``apply_polars_categorical_fixes`` returns the train+val Enum domains
   it built so the suite can stash them into model metadata.
2. The metadata-resident ``enum_domains`` dict survives any reasonable
   round-trip (dict copy / dill / etc.) and matches the domains used at
   train.
3. At predict time, ``_coerce_cat_dtype_for_lgb_xgb`` reads
   ``metadata["enum_domains"]`` and casts polars XGB inputs to pl.Enum
   against the persisted domain -- out-of-domain test-only categories
   land as null (matches training's ``strict=False`` semantics), not a
   crash.
4. Legacy bundles without ``enum_domains`` fall back to bare
   ``pl.Categorical`` cast with a WARN (backwards-compat preserved).
"""

from __future__ import annotations

import logging
import pytest


def test_s67_polars_fixes_returns_enum_domains():
    """``apply_polars_categorical_fixes`` exposes the per-column train+val union."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes

    train_df = pl.DataFrame({"cat_col": ["a", "b", "a"], "n": [1.0, 2.0, 3.0]})
    val_df = pl.DataFrame({"cat_col": ["b", "c"], "n": [4.0, 5.0]})
    test_df = pl.DataFrame({"cat_col": ["a", "c"], "n": [6.0, 7.0]})

    result = apply_polars_categorical_fixes(
        train_df_polars=train_df,
        val_df_polars=val_df,
        test_df_polars=test_df,
        train_df_pd=train_df,
        val_df_pd=val_df,
        test_df_pd=test_df,
        filtered_train_df=train_df,
        filtered_val_df=val_df,
        cat_features=["cat_col"],
        align_polars_categorical_dicts=True,
        defer_pandas_conv=True,
        was_polars_input=True,
        verbose=False,
    )
    assert result.enum_domains is not None
    assert "cat_col" in result.enum_domains
    domain = sorted(result.enum_domains["cat_col"])
    # Train+val union = {a, b, c}; test ("c") was already in val so it stays in domain too here
    assert "a" in domain and "b" in domain and "c" in domain


def test_s67_predict_cat_cast_uses_persisted_enum_domain():
    """When metadata has enum_domains, predict-time XGB-polars cat-cast lands on pl.Enum (not pl.Categorical)."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _FakeXGBModel:
        """Duck-types as an XGBClassifier (module/class name only) so the cat-cast dispatch treats it as XGBoost."""

        __module__ = "xgboost.sklearn"
        __class__ = type("XGBClassifier", (), {})

    fake_model = _FakeXGBModel()
    fake_model.__class__.__name__ = "XGBClassifier"
    fake_model.__class__.__module__ = "xgboost.sklearn"

    inp = pl.DataFrame({"cat_col": ["a", "b", "z"]})
    out = _coerce_cat_dtype_for_lgb_xgb(
        inp,
        model=fake_model,
        cat_features=["cat_col"],
        enum_domains={"cat_col": ["a", "b", "c"]},
    )
    assert isinstance(out.schema["cat_col"], pl.Enum)
    # Unseen category "z" -> null (strict=False)
    assert out["cat_col"].to_list() == ["a", "b", None]


def test_s67_predict_cat_cast_legacy_bundle_falls_back_with_warn(caplog):
    """Legacy bundle (no enum_domains): cast still works but emits WARN."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core.predict import _coerce_cat_dtype_for_lgb_xgb

    class _FakeXGBModel:
        """Bare stand-in later renamed to XGBClassifier, simulating a legacy bundle with no enum_domains."""

        pass

    fake_model = _FakeXGBModel()
    fake_model.__class__.__name__ = "XGBClassifier"
    fake_model.__class__.__module__ = "xgboost.sklearn"

    inp = pl.DataFrame({"cat_col": ["a", "b"]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        out = _coerce_cat_dtype_for_lgb_xgb(
            inp,
            model=fake_model,
            cat_features=["cat_col"],
            enum_domains=None,
        )
    assert out.schema["cat_col"] == pl.Categorical
    msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("enum_domains" in m and "cat_col" in m for m in msgs), f"Expected legacy-bundle WARN; got: {msgs}"


def test_s67_enum_domains_roundtrips_via_dill():
    """``enum_domains`` is a plain dict[str, list[str]] -- dill-safe for sidecar saves."""
    pl = pytest.importorskip("polars")
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
    from mlframe.training.core._phase_polars_fixes import apply_polars_categorical_fixes

    train_df = pl.DataFrame({"cat_col": ["a", "b"], "n": [1.0, 2.0]})
    result = apply_polars_categorical_fixes(
        train_df_polars=train_df,
        val_df_polars=None,
        test_df_polars=None,
        train_df_pd=train_df,
        val_df_pd=None,
        test_df_pd=None,
        filtered_train_df=train_df,
        filtered_val_df=None,
        cat_features=["cat_col"],
        align_polars_categorical_dicts=True,
        defer_pandas_conv=True,
        was_polars_input=True,
        verbose=False,
    )
    blob = dill.dumps(result.enum_domains)
    restored = dill.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
    assert restored == result.enum_domains
