"""Wave 37 (2026-05-20): wrong exception type at validation/dispatch boundaries.

Audit class: production code raised the wrong exception class for the failure mode,
which breaks sklearn pipeline machinery (catches NotFittedError, not RuntimeError),
duck-type dispatch contracts (TypeError vs ValueError), and -O optimization
(AssertionError gets stripped).

Sites covered:
  P1 (not-fitted: RuntimeError -> NotFittedError):
    - feature_selection/wrappers/_rfecv.py (2 sites)
    - training/feature_handling/polynomial.py (3 sites)
    - training/feature_handling/text_encoder.py (1 site)
    - training/feature_handling/custom_handler.py (1 site)
    - training/pu_learning.py:417
    - training/neural/base.py:560
    - training/neural/recurrent.py (2 sites)
    - training/neural/keras_compat.py:146
  P1 (type-vs-value: ValueError -> TypeError for isinstance failures):
    - training/core/predict.py:687 (models_path type check)
    - feature_engineering/bruteforce.py:158 (df type check)
    - training/neural/base.py:723 (mixin type check)
    - training/neural/base.py:849 (period type check)
    - training/neural/flat.py:116-128 (4 type checks split out)
    - training/neural/flat.py:433 (batch format type check)
  P2 (AssertionError -> ValueError/RuntimeError):
    - feature_engineering/categorical.py:83
    - training/ranking.py:261
    - training/ranking.py:487
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    2026-05-21 monolith split compat: when the requested file is
    ``training/core/predict.py``, append the main + pp siblings so
    source-pattern sensors for the relocated raise sites still match.
    """
    _path = MLFRAME_ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read the package
        # __init__ and append every submodule so source-pattern sensors for
        # relocated raise sites still match regardless of which submodule owns them.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            primary = _init.read_text(encoding="utf-8")
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    primary = primary + "\n" + _sub.read_text(encoding="utf-8")
            return primary
    primary = _path.read_text(encoding="utf-8")
    if rel == "training/core/predict.py":
        _core = MLFRAME_ROOT / "training" / "core"
        # Concat every ``_predict*.py`` sibling so the source-grep sensor
        # picks up the relocated raise sites regardless of which sibling
        # owns them after the predict monolith-split waves.
        for _sib_path in sorted(_core.glob("_predict*.py")):
            primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    elif rel == "feature_selection/wrappers/rfecv/__init__.py":
        _wraps = MLFRAME_ROOT / "feature_selection" / "wrappers" / "rfecv"
        for _sib_path in sorted(_wraps.glob("*.py")):
            if _sib_path.name != "__init__.py":
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    return primary


# ---------------------------------------------------------------------------
# P1 / cluster #1: not-fitted now raises sklearn.exceptions.NotFittedError
# ---------------------------------------------------------------------------

def _not_fitted_calls() -> dict:
    """Zero-argument callables, one per not-fitted guard, each invoking the guarded method on an unfitted object."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    x2 = np.zeros((3, 2))

    def rfecv(method):
        """Return a callable running ``method`` on an unfitted RFECV."""

        def call():
            """Invoke the method on a fresh unfitted RFECV."""
            from mlframe.feature_selection.wrappers.rfecv import RFECV

            return getattr(RFECV(estimator=LogisticRegression()), method)(*([x2] if method == "transform" else []))

        return call

    def rfecv_stability():
        """Run the RFECV stability diagnostic on an unfitted selector."""
        from mlframe.feature_selection.wrappers.rfecv import RFECV
        from mlframe.feature_selection.wrappers.rfecv import _diagnostics

        return _diagnostics.selection_stability_(RFECV(estimator=LogisticRegression()))

    def polynomial(attr):
        """Return a callable touching ``attr`` (or calling it) on an unfitted expander."""

        def call():
            """Touch the attribute on a fresh unfitted PolynomialFeatureExpander."""
            from mlframe.training.feature_handling.polynomial import PolynomialFeatureExpander

            obj = PolynomialFeatureExpander(degree=2)
            return getattr(obj, attr)(x2) if attr == "transform" else getattr(obj, attr)

        return call

    def text_encoder():
        """Transform with an unfitted text encoder."""
        from mlframe.training.feature_handling.text_encoder import TextColumnEncoder, TfidfParams

        return TextColumnEncoder("c", TfidfParams()).transform(pd.DataFrame({"c": ["a"]}))

    def custom_handler():
        """Transform with an unfitted custom handler."""
        from sklearn.preprocessing import StandardScaler

        from mlframe.training.feature_handling.custom_handler import CustomHandler
        from mlframe.training.feature_handling.handlers import CustomParams

        return CustomHandler("c", CustomParams(transformer=StandardScaler())).transform(pd.DataFrame({"c": [1.0]}))

    def pu_learning():
        """predict_proba on an unfitted PU wrapper."""
        from mlframe.training.pu_learning import PULearningWrapper

        return PULearningWrapper(LogisticRegression()).predict_proba(x2)

    def neural_base():
        """predict on an unfitted Lightning estimator."""
        from mlframe.training.neural.base import PytorchLightningRegressor

        est = PytorchLightningRegressor(
            model_class=object, model_params={}, network_params={}, datamodule_class=object, datamodule_params={}, trainer_params={}
        )
        return est.predict(x2)

    def recurrent(cls_name, method):
        """Return a callable running ``method`` on an unfitted recurrent wrapper."""

        def call():
            """Invoke the method on a fresh unfitted wrapper."""
            from mlframe.training.neural import recurrent as rec

            return getattr(getattr(rec, cls_name)(), method)(x2)

        return call

    def keras_compat():
        """predict on an unfitted Keras-compatible MLP."""
        from mlframe.training.neural.keras_compat import KerasCompatibleMLP

        return KerasCompatibleMLP().predict(x2)

    return {
        "rfecv.get_feature_names_out": rfecv("get_feature_names_out"),
        "rfecv.get_support": rfecv("get_support"),
        "rfecv.transform": rfecv("transform"),
        "rfecv.selection_stability": rfecv_stability,
        "polynomial.transform": polynomial("transform"),
        "polynomial.n_features_in": polynomial("n_features_in"),
        "polynomial.feature_names_out": polynomial("feature_names_out"),
        "text_encoder.transform": text_encoder,
        "custom_handler.transform": custom_handler,
        "pu_learning.predict_proba": pu_learning,
        "neural_base.predict": neural_base,
        "recurrent_classifier.predict_proba": recurrent("RecurrentClassifierWrapper", "predict_proba"),
        "recurrent_classifier.predict": recurrent("RecurrentClassifierWrapper", "predict"),
        "recurrent_regressor.predict": recurrent("RecurrentRegressorWrapper", "predict"),
        "keras_compat.predict": keras_compat,
    }


NOT_FITTED_TARGETS = sorted(_not_fitted_calls())


@pytest.mark.parametrize("target", NOT_FITTED_TARGETS)
def test_not_fitted_uses_notfittederror(target: str) -> None:
    """Every not-fitted code path raises sklearn's NotFittedError so pipelines catch it, never a bare RuntimeError."""
    from sklearn.exceptions import NotFittedError

    if target.startswith("rfecv"):
        pytest.importorskip("mlframe.feature_selection.wrappers.rfecv")
    if target.startswith(("neural_base", "recurrent")):
        pytest.importorskip("lightning")
    with pytest.raises(NotFittedError) as info:
        _not_fitted_calls()[target]()
    assert type(info.value) is NotFittedError
    assert not isinstance(info.value, RuntimeError)
    assert str(info.value).strip()


def test_notfittederror_is_importable_in_each_file() -> None:
    """The NotFittedError every guard raises is sklearn's own class, so ``except sklearn.exceptions.NotFittedError`` catches it."""
    from sklearn.exceptions import NotFittedError

    available = [t for t in NOT_FITTED_TARGETS if not t.startswith(("neural_base", "recurrent"))]
    assert len(available) >= 10
    caught = []
    for target in available:
        try:
            _not_fitted_calls()[target]()
        except NotFittedError as exc:
            caught.append((target, type(exc)))
    assert [t for t, _ in caught] == available
    assert {cls for _, cls in caught} == {NotFittedError}


# ---------------------------------------------------------------------------
# P1 / cluster #2: isinstance failures now raise TypeError (not ValueError)
# ---------------------------------------------------------------------------


def test_bruteforce_df_type_is_typeerror() -> None:
    """A non-DataFrame input raises TypeError (not ValueError) naming the supported frame types."""
    pytest.importorskip("pysr")
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    with pytest.raises(TypeError, match="pandas or polars DataFrame"):
        run_pysr_feature_engineering(df=[1, 2, 3], target_col="y")  # type: ignore[arg-type]


def test_neural_base_mixin_type_is_typeerror() -> None:
    """An estimator that is neither a regressor nor a classifier is refused by ``score`` with a TypeError naming its class."""
    pytest.importorskip("lightning")
    from mlframe.training.neural.base._base_predict import _PredictMixin

    class _NeitherKind(_PredictMixin):
        """Mixin host that is not a RegressorMixin or ClassifierMixin."""

        def predict(self, X, **kwargs):
            """Return fixed predictions so ``score`` reaches its type dispatch."""
            return np.zeros(len(X))

    with pytest.raises(TypeError, match=r"Estimator must be a RegressorMixin or ClassifierMixin, got _NeitherKind"):
        _NeitherKind().score(np.zeros((3, 2)), np.zeros(3))


def test_neural_base_period_type_is_typeerror() -> None:
    """``PeriodicLearningRateFinder`` rejects a non-int or bool period with TypeError."""
    pytest.importorskip("pytorch_lightning")
    from mlframe.training.neural._base_callbacks import PeriodicLearningRateFinder

    for bad in ("3", 2.5, True, None):
        with pytest.raises(TypeError, match="period must be an int"):
            PeriodicLearningRateFinder(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs,exc,match",
    [
        # isinstance failures -> TypeError
        (dict(nlayers=2.0), TypeError, "nlayers must be an int"),
        (dict(nlayers=True), TypeError, "nlayers must be an int"),
        (dict(min_layer_neurons=1.5), TypeError, "min_layer_neurons must be an int"),
        (dict(num_classes=2.5), TypeError, "num_classes must be None or an int"),
        (dict(num_classes=True), TypeError, "num_classes must be None or an int"),
        (dict(first_layer_num_neurons=5.0), TypeError, "first_layer_num_neurons must be an int"),
        # range failures -> ValueError
        (dict(nlayers=0), ValueError, "nlayers must be >= 1"),
        (dict(min_layer_neurons=0), ValueError, "min_layer_neurons must be >= 1"),
        (dict(num_classes=-1), ValueError, "num_classes must be >= 0"),
        (dict(first_layer_num_neurons=2, min_layer_neurons=3), ValueError, "first_layer_num_neurons must be >= min_layer_neurons"),
    ],
)
def test_neural_flat_validation_uses_typeerror_and_valueerror(kwargs, exc, match) -> None:
    """generate_mlp rejects a wrong-typed argument with TypeError and an out-of-range one with ValueError."""
    pytest.importorskip("torch")
    from mlframe.training.neural.flat import generate_mlp

    call = dict(num_features=4, num_classes=1)
    call.update(kwargs)
    with pytest.raises(exc, match=match) as info:
        generate_mlp(**call)
    assert type(info.value) is exc, f"{kwargs}: expected exactly {exc.__name__}, got {type(info.value).__name__}"


def test_neural_flat_validation_accepts_valid_arguments() -> None:
    """Control: the same validators let a well-formed call through."""
    pytest.importorskip("torch")
    from mlframe.training.neural.flat import generate_mlp

    assert generate_mlp(num_features=4, num_classes=2, nlayers=2, first_layer_num_neurons=8, min_layer_neurons=2) is not None


def test_neural_flat_batch_format_is_typeerror() -> None:
    """``MLPTorchModel`` unpacks tuple / list / dict batches and refuses anything else with a TypeError naming the type."""
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from mlframe.training.neural._flat_torch_module._flat_torch_loss import _LossMixin

    loss = _LossMixin()
    assert loss._unpack_batch((1, 2)) == (1, 2, None)
    assert loss._unpack_batch([1, 2, 3]) == (1, 2, 3)
    assert loss._unpack_batch({"features": 1, "labels": 2}) == (1, 2, None)
    with pytest.raises(TypeError, match=r"Unexpected batch format: int"):
        loss._unpack_batch(5)
    with pytest.raises(TypeError, match=r"Unexpected batch format: tuple"):
        loss._unpack_batch((1,))


# ---------------------------------------------------------------------------
# P2: AssertionError -> ValueError/RuntimeError (would survive -O optimization)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel,forbidden_assertion_substring",
    [
        ("feature_engineering/categorical.py", "compute_numaggs(directional_only=True) returned"),
        ("training/ranking.py", '"unreachable"'),
        ("training/ranking.py", 'AssertionError(f"unknown flavor'),
    ],
)
def test_assertion_error_not_used_at_validation_boundary(rel: str, forbidden_assertion_substring: str) -> None:
    """Assertion error not used at validation boundary."""
    src = _read(rel)
    # The forbidden string must NOT co-occur with raise AssertionError on the same line.
    assert src.strip(), f"{rel}: the source to scan is empty"
    for line in src.splitlines():
        if forbidden_assertion_substring in line and "AssertionError" in line:
            pytest.fail(f"{rel}: still raises AssertionError at validation boundary; would be stripped by python -O\n  line: {line.strip()!r}")


def test_categorical_numaggs_count_mismatch_is_runtimeerror(monkeypatch) -> None:
    """A directional-numaggs width that disagrees with the registered names raises RuntimeError (not an assert stripped by -O)."""
    import pandas as pd

    from mlframe.feature_engineering import categorical

    monkeypatch.setattr(categorical, "compute_numaggs", lambda **kwargs: [1.0])
    series = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    with pytest.raises(RuntimeError, match=r"compute_numaggs\(directional_only=True\) returned 1 values but \d+ names are registered"):
        categorical.compute_countaggs(series, counts_compute_numaggs=False, counts_compute_values_numaggs=True)


def test_ranking_unreachable_is_runtimeerror(monkeypatch) -> None:
    """A ranker family the dispatcher does not recognise raises RuntimeError rather than falling off the end."""
    from types import SimpleNamespace

    from mlframe.training.ranking import ranking as r

    strategy = SimpleNamespace(supports_native_ranking=True, get_ranker_objective_kwargs=lambda **kwargs: {})
    monkeypatch.setattr(r, "_strategy_flavor", lambda s: "banana")
    with pytest.raises(RuntimeError, match=r"unreachable: unhandled ranker family after backend dispatch"):
        r.fit_ranker(strategy, np.zeros((4, 2)), np.zeros(4), np.zeros(4, dtype=int))


# ---------------------------------------------------------------------------
# Behavioural smokes: the wrong-exception fixes should be reachable.
# ---------------------------------------------------------------------------


def test_predict_models_path_typeerror_behavioural() -> None:
    """Predict models path typeerror behavioural."""
    import polars as pl
    from mlframe.training.core.predict import predict_mlframe_models_suite

    df = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError, match="models_path must be a str"):
        predict_mlframe_models_suite(df, 12345)  # type: ignore[arg-type]


def test_polynomial_not_fitted_is_notfittederror() -> None:
    """Polynomial not fitted is notfittederror."""
    from sklearn.exceptions import NotFittedError
    import numpy as np
    from mlframe.training.feature_handling.polynomial import PolynomialFeatureExpander

    tr = PolynomialFeatureExpander(degree=2)
    with pytest.raises(NotFittedError):
        tr.transform(np.zeros((4, 3)))


def test_ranking_unknown_flavor_valueerror_behavioural() -> None:
    """Behavioural: unknown ranker ``flavor`` must raise ValueError with
    a useful message naming the unknown flavor. Previously skipped via
    ``hasattr(r, "_predict_ranker_scores")`` after the helper was
    renamed from underscored-private to public ``predict_ranker_scores``;
    the skip silently masked the raise-site contract on every CI run.
    """
    import numpy as np
    from mlframe.training import ranking as r

    # 2026-05-24: the helper signature is now ``predict_ranker_scores
    # (fitted: dict, X, group_ids=None)`` where ``fitted`` must carry
    # ``{"model": ..., "flavor": str}``; the unknown-flavor raise
    # lives at ranking.py:510. Build a minimal ``fitted`` dict so the
    # flavor check is the only thing the function reaches.
    with pytest.raises(ValueError, match="unknown ranker flavor"):
        r.predict_ranker_scores(
            fitted={"model": object(), "flavor": "banana"},
            X=np.zeros((2, 2)),
        )
