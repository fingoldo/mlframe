"""Tests for `apply_preprocessing_extensions` + `PreprocessingExtensionsConfig`.

These tests use pandas inputs directly (bypassing the suite) so they don't
transitively import torch/pytorch_lightning during collection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions

from tests.conftest import fast_subset


@pytest.fixture
def small_df():
    """Small df."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((60, 5)), columns=[f"x{i}" for i in range(5)])


def test_none_config_is_noop(small_df):
    """None config is noop."""
    val = small_df.iloc[:10].copy()
    a, b, c, p = apply_preprocessing_extensions(small_df, val, None, None)
    assert p is None
    assert a is small_df
    assert b is val
    assert c is None


def test_empty_config_is_noop(small_df):
    """Empty config is noop."""
    # ``row_wise_summary_stats_enabled`` / ``row_wise_extreme_columns_enabled`` default to True on
    # PreprocessingExtensionsConfig itself (documented, intentional -- generically-safe additive
    # row-wise FE steps enabled by default; see the class docstring). The bare constructor is NOT an
    # all-off config, so a genuine "empty config" test must disable them explicitly.
    cfg = PreprocessingExtensionsConfig(row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False)
    a, _, _, p = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert p is None
    assert a is small_df


# Full PreprocessingExtensionsConfig.scaler Literal: every value routes through
# the same apply_preprocessing_extensions code path. Fast mode trims to one
# representative; full mode covers every variant the validator accepts.
@pytest.mark.parametrize(
    "scaler",
    fast_subset(
        [
            "StandardScaler",
            "StandardScaler_nomean",
            "RobustScaler",
            "MinMaxScaler",
            "MaxAbsScaler",
            "PowerTransformer_yj",
            "PowerTransformer_yj_nostd",
            "QuantileTransformer_uniform",
            "QuantileTransformer_normal",
        ],
        representative="StandardScaler",
    ),
)
def test_scaler_variants_produce_expected_shape(small_df, scaler):
    """Scaler variants produce expected shape."""
    # Isolate the scaler's own shape effect: row_wise_summary_stats_enabled / row_wise_extreme_columns_enabled
    # default to True on PreprocessingExtensionsConfig (documented, intentional additive row-wise FE) and
    # would otherwise add columns unrelated to what this test checks.
    cfg = PreprocessingExtensionsConfig(scaler=scaler, row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False)
    out, _, _, pipe = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert out.shape == small_df.shape
    assert pipe is not None


def test_normalizer_l2_rejected_at_validation():
    """Regression: row-wise normalization mislabeled as a scaler silently broke
    GBDT models. Removed 2026-05-15 - see README.md Roadmap."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PreprocessingExtensionsConfig(scaler="Normalizer_l2")


def test_pca_reduces_dimension(small_df):
    """Pca reduces dimension."""
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", dim_reducer="PCA", dim_n_components=3)
    out, _, _, _ = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert out.shape == (60, 3)


def test_polynomial_features_guard_triggers(small_df):
    """Polynomial features guard triggers."""
    cfg = PreprocessingExtensionsConfig(polynomial_degree=3, memory_safety_max_features=50)
    with pytest.raises(ValueError, match="memory_safety_max_features"):
        apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)


def test_binarization_and_kbins_mutually_exclusive():
    """Binarization and kbins mutually exclusive."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PreprocessingExtensionsConfig(binarization_threshold=0.5, kbins=5)


def test_kbins_min_bins():
    """Kbins min bins."""
    with pytest.raises(ValueError, match="kbins"):
        PreprocessingExtensionsConfig(kbins=1)


def test_polynomial_min_degree():
    """Polynomial min degree."""
    with pytest.raises(ValueError, match="polynomial_degree"):
        PreprocessingExtensionsConfig(polynomial_degree=1)


def test_umap_missing_raises_importerror(monkeypatch, small_df):
    """Umap missing raises importerror."""
    import importlib.util as ilu

    orig = ilu.find_spec
    monkeypatch.setattr(ilu, "find_spec", lambda name: None if name == "umap" else orig(name))
    cfg = PreprocessingExtensionsConfig(dim_reducer="UMAP", dim_n_components=2)
    with pytest.raises(ImportError, match="umap-learn"):
        apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)


def test_binarizer_produces_binary(small_df):
    """Binarizer produces binary."""
    cfg = PreprocessingExtensionsConfig(binarization_threshold=0.0)
    out, _, _, _ = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert set(np.unique(out.values)) <= {0.0, 1.0}


def test_kbins_produces_integers(small_df):
    """Kbins produces integers."""
    cfg = PreprocessingExtensionsConfig(kbins=4)
    out, _, _, _ = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert out.values.max() < 4
    assert out.values.min() >= 0


def test_val_and_test_follow_train(small_df):
    """Val and test follow train."""
    val = small_df.iloc[:20].copy()
    test = small_df.iloc[20:30].copy()
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", dim_reducer="PCA", dim_n_components=2)
    tr, va, te, _pipe = apply_preprocessing_extensions(small_df, val, test, cfg, verbose=0)
    assert tr.shape == (60, 2)
    assert va.shape == (20, 2)
    assert te.shape == (10, 2)


def _polars_frame_with_wide_categoricals(n_rows: int = 300, n_cat: int = 8):
    """Polars frame mixing numeric/bool/duration columns with many Categorical ones."""
    pl = pytest.importorskip("polars")
    import datetime

    rng = np.random.default_rng(0)
    data: dict = {f"num{i}": rng.standard_normal(n_rows) for i in range(4)}
    data["flag"] = rng.integers(0, 2, n_rows).astype(bool)
    data["dur"] = [datetime.timedelta(days=int(v)) for v in rng.integers(0, 5, n_rows)]
    for c in range(n_cat):
        data[f"cat{c}"] = pl.Series([f"v{v}" for v in rng.integers(0, 20, n_rows)], dtype=pl.Categorical)
    return pl.DataFrame(data)


def test_polars_categoricals_are_not_converted_before_being_dropped():
    """Categorical/Enum columns no extension stage reads must never reach the polars->pandas bridge.

    ``to_pandas`` materialises them as pandas ``object`` -- one Python ``str`` per cell -- and the numeric
    gate then drops every one of them. On a production 2.7M-row frame that round trip cost 3.9 minutes for
    23 columns whose values were never read. Pinning the behaviour by asserting the bridge is handed only
    the relevant subset: a pure output-shape assertion would pass just as well on the wasteful path.
    """
    pl = pytest.importorskip("polars")
    df = _polars_frame_with_wide_categoricals()
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False)

    seen_widths = []
    _orig = pl.DataFrame.to_pandas

    def _spy(self, *a, **kw):
        """Record the frame width handed to the bridge, then delegate to the real to_pandas."""
        seen_widths.append(self.width)
        return _orig(self, *a, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl.DataFrame, "to_pandas", _spy)
        tr, _va, _te, _pipe = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)

    assert seen_widths, "expected the polars->pandas bridge to be exercised"
    # Only the 4 numeric columns + the bool cross the bridge. The 8 Categoricals and the Duration are
    # preselected away; without the fix all 14 columns were converted and then dropped.
    assert max(seen_widths) == 5, f"irrelevant columns still crossed the bridge: widths={seen_widths}"
    assert not [c for c in tr.columns if c.startswith("cat")]


def test_timedelta_column_is_dropped_instead_of_crashing_the_sklearn_bridge():
    """A timedelta column must take the documented "non-numeric -> drop with a WARN" path.

    ``select_dtypes(include="number")`` selects timedelta64, but every sklearn step downstream then calls
    ``np.result_type`` across the frame and timedelta64 has no common dtype with float64 -- raising a
    ``DTypePromotionError`` from inside ``sklearn.utils.validation`` that names neither the offending column
    nor this pipeline. Passing the numeric gate never made a timedelta column usable, it only swapped a
    clean drop for an opaque crash.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"num{i}": rng.standard_normal(100) for i in range(3)})
    df["dur"] = pd.to_timedelta(rng.integers(0, 5, 100), unit="D")
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False)

    tr, _va, _te, _pipe = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)

    assert "dur" not in tr.columns
    assert tr.shape[1] == 3


def test_polars_duration_column_is_dropped_not_crashed():
    """Polars-input twin of the timedelta drop above (polars Duration -> pandas timedelta64)."""
    pytest.importorskip("polars")
    df = _polars_frame_with_wide_categoricals()
    cfg = PreprocessingExtensionsConfig(row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False, scaler="StandardScaler")
    tr, _va, _te, _pipe = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    # 4 numeric + bool reach the scaler; the Duration and all 8 Categoricals are dropped.
    assert "dur" not in tr.columns
    assert tr.shape[1] == 5, f"expected bool+4 numeric to survive, got {list(tr.columns)}"


def test_tfidf_column_survives_the_preselect():
    """A declared ``tfidf_columns`` text column is consumed AFTER the bridge, so it must not be
    preselected away as "non-numeric" -- otherwise TF-IDF silently finds nothing to vectorise."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "num0": rng.standard_normal(200),
            "txt": [f"word{v} common" for v in rng.integers(0, 10, 200)],
            "cat0": pl.Series([f"v{v}" for v in rng.integers(0, 5, 200)], dtype=pl.Categorical),
        }
    )
    cfg = PreprocessingExtensionsConfig(
        tfidf_columns=["txt"], tfidf_max_features=8, row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False
    )
    tr, _va, _te, _pipe = apply_preprocessing_extensions(df, None, None, cfg, verbose=0)
    assert [c for c in tr.columns if c.startswith("txt__tfidf_")], f"TF-IDF produced no features: {list(tr.columns)}"
    assert "cat0" not in tr.columns
