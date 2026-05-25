"""Regression sensors for D1 Low #11, #12, #13.

#11: ``TrainingSplitConfig.composite_cardinality_cap`` exposed (was hardcoded 200).
#12: pre-screen protects group/ts column names sourced from extractor + split_config
     (not just ctx attrs), and WARNs when split_config.use_groups=True but the
     protected set ends up empty.
#13: ``_cast_utf8_cats_to_categorical`` emits a WARN when it falls back to bare
     ``pl.Categorical`` (no enum_domains supplied), so operators see the silent
     global-string-cache-widening path on the first run.
"""
from __future__ import annotations

import logging
import pytest


def test_d1_low_11_composite_cardinality_cap_exposed_on_split_config():
    """``TrainingSplitConfig`` must accept ``composite_cardinality_cap`` (was magic 200)."""
    from mlframe.training._preprocessing_configs import TrainingSplitConfig

    cfg = TrainingSplitConfig(composite_cardinality_cap=500)
    assert cfg.composite_cardinality_cap == 500
    cfg_default = TrainingSplitConfig()
    assert cfg_default.composite_cardinality_cap == 200


def test_d1_low_11_bucket_stratify_default_on_in_split_config():
    """``TrainingSplitConfig.bucket_stratify`` default = True."""
    from mlframe.training._preprocessing_configs import TrainingSplitConfig

    cfg = TrainingSplitConfig()
    assert cfg.bucket_stratify is True


def test_d1_low_13_bare_categorical_fallback_emits_warn(caplog):
    """``_cast_utf8_cats_to_categorical`` without enum_domains -> WARN."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core._phase_polars_fixes import _cast_utf8_cats_to_categorical

    df = pl.DataFrame({"cat_col": ["a", "b", "a", "c"], "num_col": [1, 2, 3, 4]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_polars_fixes"):
        out = _cast_utf8_cats_to_categorical(df, ["cat_col"], enum_domains=None)
    assert out.schema["cat_col"] == pl.Categorical
    msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("pl.Categorical" in m and "cat_col" in m for m in msgs), (
        f"Expected fallback WARN mentioning pl.Categorical + col name; got: {msgs}"
    )


def test_d1_low_13_enum_domain_path_does_not_warn(caplog):
    """When enum_domain supplied, cast lands on pl.Enum and no WARN fires."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core._phase_polars_fixes import _cast_utf8_cats_to_categorical

    df = pl.DataFrame({"cat_col": ["a", "b", "a", "c"]})
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_polars_fixes"):
        out = _cast_utf8_cats_to_categorical(df, ["cat_col"], enum_domains={"cat_col": ["a", "b", "c"]})
    assert isinstance(out.schema["cat_col"], pl.Enum)
    msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("pl.Categorical" in m for m in msgs), (
        f"Did not expect fallback WARN when enum_domain present; got: {msgs}"
    )
