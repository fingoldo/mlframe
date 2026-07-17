"""Wave 88 (2026-05-21): _try_predict closure extracted from predict.py:1372
mega-try body to module-level _try_predict_with_pp_fallback.

The 90-line nested closure was previously redefined on every per-model
iteration inside the 525-line outer try-block. Wave 88 lifts it to module
level with an explicit keyword-arg surface (model, expected_list,
pandas_view_cache, model_name, verbose) so:
  1. The function is unit-testable in isolation (no for-loop closure).
  2. Per-iteration def overhead is gone.
  3. The mega-try body shrinks by 90 lines (visual readability +
     per the user's "до хера уродливых блоков try" pushback).
  4. The original nested def is kept as a 9-line thin wrapper inside the
     loop, preserving the call-site convention so the 4 internal call sites
     stay readable.

This refactor is behaviour-preserving: same fallback paths, same logging,
same exception classes caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_try_predict_with_pp_fallback_importable() -> None:
    """The extracted helper is a module-level symbol."""
    from mlframe.training.core.predict import _try_predict_with_pp_fallback

    assert callable(_try_predict_with_pp_fallback)


def test_try_predict_with_pp_fallback_passthrough_on_success() -> None:
    """When the primary fn succeeds, the helper returns its output verbatim."""
    from mlframe.training.core.predict import _try_predict_with_pp_fallback

    class _Model:
        def predict(self, X):
            return np.full(len(X), 0.5)

    # Use a function that does NOT match "predict"/"predict_proba" by name so
    # we bypass the _predict_with_fallback path and exercise the direct fn(primary) branch.
    def plain_fn(X):
        return np.full(len(X), 0.5)

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = _try_predict_with_pp_fallback(
        plain_fn,
        df,
        None,
        model=_Model(),
        expected_list=["x"],
        pandas_view_cache={},
        model_name="test",
        verbose=0,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, [0.5, 0.5, 0.5])


def test_try_predict_with_pp_fallback_encoder_mismatch_retries_on_fallback() -> None:
    """TypeError matching the encoder-mismatch signature retries on fallback frame."""
    from mlframe.training.core.predict import _try_predict_with_pp_fallback

    primary = pd.DataFrame({"x": [1.0, 2.0]})
    fallback = pd.DataFrame({"x": ["a", "b"]})

    calls: list[str] = []

    def maybe_fail(X):
        if X is primary:
            calls.append("primary")
            raise TypeError("ufunc 'isnan' not supported for input types")
        calls.append("fallback")
        return np.array([0.0, 1.0])

    out = _try_predict_with_pp_fallback(
        maybe_fail,
        primary,
        fallback,
        model=object(),
        expected_list=["x"],
        pandas_view_cache={},
        model_name="test_encoder_mismatch",
        verbose=0,
    )
    assert calls == ["primary", "fallback"]
    np.testing.assert_array_equal(out, [0.0, 1.0])


def test_try_predict_with_pp_fallback_unrelated_typeerror_raises() -> None:
    """A TypeError that doesn't match the encoder-mismatch signature re-raises."""
    from mlframe.training.core.predict import _try_predict_with_pp_fallback

    primary = pd.DataFrame({"x": [1.0, 2.0]})

    def always_fail(X):
        raise TypeError("some unrelated dtype problem")

    with pytest.raises(TypeError, match="some unrelated dtype problem"):
        _try_predict_with_pp_fallback(
            always_fail,
            primary,
            None,
            model=object(),
            expected_list=None,
            pandas_view_cache={},
            model_name="test_unrelated",
            verbose=0,
        )


def test_outer_try_body_shrunk() -> None:
    """Verify the mega-try body is shorter now -- nested def is gone."""
    from pathlib import Path

    # ``predict.py`` was carved up into ``_predict_main.py`` /
    # ``_predict_pre_pipeline.py`` siblings during the monolith-split
    # wave; the lifted ``_try_predict_with_pp_fallback`` helper now
    # lives in ``_predict_pre_pipeline.py``. Concat all three so the
    # source-grep boundary check still matches the relocated code.
    _core = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe" / "training" / "core"
    src_parts = []
    for _p in sorted(_core.glob("predict.py")) + sorted(_core.glob("_predict*.py")):
        src_parts.append(_p.read_text(encoding="utf-8"))
    src = "\n".join(src_parts)
    # The lifted helper is now at module level.
    assert "\ndef _try_predict_with_pp_fallback(" in src
    # The thin-wrapper closure call inside the for-loop body now delegates.
    assert "return _try_predict_with_pp_fallback(" in src
    # The original 90-line nested closure body is gone -- the encoder-mismatch
    # variable now lives only inside the module-level helper, NOT inside the
    # per-iteration nested def. Confirm via a structural marker: the
    # "fallback is not None" check appears exactly once.
    assert src.count("_is_encoder_mismatch and fallback is not None") == 1
