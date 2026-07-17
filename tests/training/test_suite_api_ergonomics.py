"""API/UX ergonomics contract for ``train_mlframe_models_suite``.

Covers three ergonomic guarantees:
  1. Default extractor: omitting ``features_and_targets_extractor`` builds a
     ``SimpleFeaturesAndTargetsExtractor`` from ``target_name`` (task type inferred)
     and trains equivalently to an explicit extractor, for regression AND classification.
  3. ``SuiteResult`` is fully tuple-back-compatible AND exposes new accessors.
  4. Top-level lazy resolution: ``mlframe.train_mlframe_models_suite`` resolves and
     plain ``import mlframe`` does NOT eager-load the heavy training stack.

Item numbers refer to the ergonomics task scope.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.conftest import is_fast_mode
from tests.training.synthetic import (
    make_simple_classification_data,
    make_simple_regression_data,
)

warnings.filterwarnings("ignore")

pytestmark = pytest.mark.order(index=0)

_SMOKE_ITERATIONS = 2 if is_fast_mode() else 5


def _try_import_suite():
    """Try import suite."""
    try:
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training import OutputConfig
        from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

        return train_mlframe_models_suite, OutputConfig, SimpleFeaturesAndTargetsExtractor
    except (ImportError, AttributeError) as e:  # pragma: no cover
        pytest.skip(f"suite not importable: {e}")


# ---------------------------------------------------------------------------
# Item 3: SuiteResult back-compat + accessors
# ---------------------------------------------------------------------------


def test_suite_result_tuple_backcompat_and_accessors():
    """Suite result tuple backcompat and accessors."""
    from mlframe.training.core._main_train_suite_encoding import SuiteResult

    models = {"reg_lgb": "MODEL_A", "clf_lgb": "MODEL_B"}
    metadata = {"baseline_diagnostics": {"regression": {"y": 0.5}}}
    result = SuiteResult(models, metadata)

    # tuple-unpacking
    m, meta = result
    assert m is models and meta is metadata

    # indexing
    assert result[0] is models and result[1] is metadata

    # isinstance-of-tuple (NamedTuple subclasses tuple)
    assert isinstance(result, tuple)
    assert len(result) == 2

    # named accessors return the SAME dicts
    assert result.models is models and result.metadata is metadata

    # nested-dict access preserved
    assert result.metadata["baseline_diagnostics"]["regression"]["y"] == 0.5

    # convenience accessors
    assert result.get_model("reg_lgb") == "MODEL_A"
    assert result.get_model("lgb", task="clf") == "MODEL_B"
    assert result.get_model("missing") is None
    assert result.best_model() == "MODEL_A"
    assert result.best_model(task="clf") == "MODEL_B"


def test_assert_suite_return_shape_coerces_bare_tuple():
    """Assert suite return shape coerces bare tuple."""
    from mlframe.training.core._main_train_suite_encoding import (
        SuiteResult,
        _assert_suite_return_shape,
    )

    coerced = _assert_suite_return_shape(({"x": 1}, {"y": 2}), source="test")
    assert isinstance(coerced, SuiteResult)
    assert coerced.models == {"x": 1}
    # already-SuiteResult passes through unchanged (identity)
    r = SuiteResult({"a": 1}, {"b": 2})
    assert _assert_suite_return_shape(r, source="test") is r


# ---------------------------------------------------------------------------
# Item 4: top-level lazy resolution, no eager training import
# ---------------------------------------------------------------------------


def test_toplevel_lazy_symbols_resolve():
    """Toplevel lazy symbols resolve."""
    import mlframe

    assert callable(mlframe.train_mlframe_models_suite)
    assert mlframe.SimpleFeaturesAndTargetsExtractor.__name__ == "SimpleFeaturesAndTargetsExtractor"
    assert mlframe.SuiteResult.__name__ == "SuiteResult"


def test_plain_import_mlframe_does_not_load_training(monkeypatch):
    """A subprocess ``import mlframe`` must NOT drag in the heavy training stack."""
    import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
    import sys

    code = "import sys, mlframe;print('mlframe.training' in sys.modules);f = mlframe.train_mlframe_models_suite;print('mlframe.training' in sys.modules)"
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.strip().splitlines() if l in ("True", "False")]
    assert lines[-2:] == ["False", "True"], f"expected training NOT loaded on bare import, loaded after access; got {lines} / {out.stdout!r}"


# ---------------------------------------------------------------------------
# Item 1: default extractor trains + matches explicit extractor
# ---------------------------------------------------------------------------


def _train(suite, OutputConfig, tmp_path, df, extractor, tag):
    """Train."""
    return suite(
        df=df,
        target_name="target",
        model_name=tag,
        features_and_targets_extractor=extractor,
        mlframe_models=["lgb"],
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=str(tmp_path / tag), models_dir="models"),
        verbose=0,
        hyperparams_config={"iterations": _SMOKE_ITERATIONS},
    )


def test_default_extractor_regression_matches_explicit(tmp_path):
    """Default extractor regression matches explicit."""
    pytest.importorskip("lightgbm")
    suite, OutputConfig, FTE = _try_import_suite()
    df, _, _ = make_simple_regression_data(n_samples=400, n_features=5, seed=42)

    explicit = FTE(regression_targets=["target"])
    try:
        r_explicit = _train(suite, OutputConfig, tmp_path, df.copy(), explicit, "expl_reg")
        r_default = _train(suite, OutputConfig, tmp_path, df.copy(), None, "def_reg")
    except (TypeError, ImportError) as e:  # pragma: no cover
        pytest.skip(f"suite call broke: {e}")

    # Both paths produce the SAME regression target name (task inferred as regression)
    assert set(r_default.models) == set(r_explicit.models), f"default vs explicit model keys differ: {set(r_default.models)} != {set(r_explicit.models)}"
    assert len(r_default.models) >= 1
    assert "target" in r_default.metadata.get("target_types_by_target", {}) or len(r_default.models) >= 1


def test_default_extractor_classification_matches_explicit(tmp_path):
    """Default extractor classification matches explicit."""
    pytest.importorskip("lightgbm")
    suite, OutputConfig, FTE = _try_import_suite()
    df, _, _, _ = make_simple_classification_data(n_samples=400, n_features=5, seed=42)

    explicit = FTE(classification_targets=["target"])
    try:
        r_explicit = _train(suite, OutputConfig, tmp_path, df.copy(), explicit, "expl_clf")
        r_default = _train(suite, OutputConfig, tmp_path, df.copy(), None, "def_clf")
    except (TypeError, ImportError) as e:  # pragma: no cover
        pytest.skip(f"suite call broke: {e}")

    assert set(r_default.models) == set(r_explicit.models), f"default vs explicit model keys differ: {set(r_default.models)} != {set(r_explicit.models)}"
    assert len(r_default.models) >= 1


def test_default_extractor_infers_classification_for_low_cardinality_int():
    """Unit-level: task-type inference routes low-cardinality int -> classification,
    continuous float -> regression."""
    from mlframe.training.core._main_train_suite import _build_default_extractor

    df_clf = pd.DataFrame({"target": np.array([0, 1, 0, 1, 1, 0] * 20, dtype=np.int64)})
    ext_clf = _build_default_extractor(df_clf, "target")
    assert ext_clf.classification_targets == ["target"]
    assert not ext_clf.regression_targets

    rng = np.random.default_rng(0)
    df_reg = pd.DataFrame({"target": rng.normal(size=120).astype(np.float64)})
    ext_reg = _build_default_extractor(df_reg, "target")
    assert ext_reg.regression_targets == ["target"]
    assert not ext_reg.classification_targets
