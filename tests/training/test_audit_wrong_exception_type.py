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

NOT_FITTED_TARGETS = [
    "feature_selection/wrappers/rfecv/__init__.py",
    "training/feature_handling/polynomial.py",
    "training/feature_handling/text_encoder.py",
    "training/feature_handling/custom_handler.py",
    "training/pu_learning.py",
    "training/neural/base.py",
    "training/neural/recurrent.py",
    "training/neural/keras_compat.py",
]


@pytest.mark.parametrize("rel", NOT_FITTED_TARGETS)
def test_not_fitted_uses_notfittederror(rel: str) -> None:
    """Every not-fitted code path must use sklearn's NotFittedError so pipelines catch it."""
    src = _read(rel)
    # No bare RuntimeError("...not been fitted...") or "...is not fitted..." patterns.
    forbidden_phrases = ["not been fitted", "is not fitted", "not fitted; call fit"]
    for phrase in forbidden_phrases:
        # Allow the phrase in the NEW NotFittedError messages; forbid only with RuntimeError class.
        # Lightweight check: if phrase appears, ensure no RuntimeError on the same logical raise line.
        assert src.strip(), f"{rel}: the source to scan is empty"
        for line_idx, line in enumerate(src.splitlines()):
            if phrase in line and "RuntimeError" in line:
                pytest.fail(f"{rel}: line {line_idx + 1} still raises RuntimeError for a not-fitted state: {line.strip()!r}")


def test_notfittederror_is_importable_in_each_file() -> None:
    """If a file mentions NotFittedError, sklearn must be the source."""
    for rel in NOT_FITTED_TARGETS:
        src = _read(rel)
        if "NotFittedError" not in src:
            continue
        assert "from sklearn.exceptions import NotFittedError" in src, f"{rel}: NotFittedError referenced but not imported from sklearn.exceptions"


# ---------------------------------------------------------------------------
# P1 / cluster #2: isinstance failures now raise TypeError (not ValueError)
# ---------------------------------------------------------------------------


def _raise_type_for_isinstance_fail(src: str, sentinel_token: str) -> str:
    """Return raise-class on the line following the isinstance check naming sentinel_token."""
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if "isinstance" in ln and sentinel_token in ln:
            # The raise should be in the same suite within 3 lines.
            for j in range(i, min(i + 5, len(lines))):
                if "raise " in lines[j]:
                    return lines[j].strip()
    return ""


def test_bruteforce_df_type_is_typeerror() -> None:
    """A non-DataFrame input raises TypeError (not ValueError) naming the supported frame types."""
    pytest.importorskip("pysr")
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    with pytest.raises(TypeError, match="pandas or polars DataFrame"):
        run_pysr_feature_engineering(df=[1, 2, 3], target_col="y")  # type: ignore[arg-type]


def test_neural_base_mixin_type_is_typeerror() -> None:
    """Neural base mixin type is typeerror."""
    src = _read("training/neural/base.py")
    assert 'raise TypeError(f"Estimator must be a RegressorMixin or ClassifierMixin' in src, "neural/base.py: mixin dispatch failure should raise TypeError"


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
    """``MLPTorchModel`` (where the batch-format dispatch lives) moved to
    sibling _flat_torch_module.py after the flat-module monolith split;
    concat so the source sensor still matches."""
    src = _read("training/neural/flat.py") + "\n" + _read("training/neural/_flat_torch_module.py")
    assert 'raise TypeError(f"Unexpected batch format' in src, "neural/flat.py: batch format dispatch failure should raise TypeError"


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


def test_categorical_numaggs_count_mismatch_is_runtimeerror() -> None:
    """Categorical numaggs count mismatch is runtimeerror."""
    src = _read("feature_engineering/categorical.py")
    assert (
        "raise RuntimeError(" in src and "compute_numaggs(directional_only=True) returned" in src
    ), "categorical.py: numaggs count mismatch should raise RuntimeError"


def test_ranking_unreachable_is_runtimeerror() -> None:
    """Ranking unreachable is runtimeerror."""
    src = _read("training/ranking.py")
    assert 'raise RuntimeError("unreachable' in src, "ranking.py: unreachable sentinel should be RuntimeError"


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
