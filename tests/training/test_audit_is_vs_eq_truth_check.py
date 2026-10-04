"""Wave-28 sensors: ``is True``/``is False`` confusion fixes (5 sites).

Three P0 + 2 P1 sites where the pre-fix shape used identity-comparison
(``is True``/``is False``/``is not True``/``is not False``) on a value
that can legitimately be ``np.bool_(True)``/``np.bool_(False)`` from
config dicts. The numpy bool fails identity checks against Python's
``True``/``False`` singletons -> code paths got mis-routed silently
or raised cryptic TypeError / ValueError.

P0 sites:

#1 feature_engineering/transformer/random_features.py:_resolve_use_gpu
   ``use_gpu is False``/``is True`` raised
   ``ValueError("must be True, False, or 'auto'")`` when called with
   ``np.bool_(...)`` from a config dict. Fix: handle the string
   ``"auto"`` first via isinstance, then ``bool(use_gpu)``.

#2 feature_engineering/transformer/row_attention.py:_select_stage4_backend
   Same shape as #1 with ``gpu_stage4``. Fix: same.

#3 feature_engineering/transformer/random_features.py:compute_rff_features
   ``if standardize is not True and standardize is not False:`` strict
   type-guard rejected ``np.bool_`` and ``int(1)``. Fix:
   ``isinstance(standardize, (bool, np.bool_))``.

P1 sites:

#4 training/train_eval.py:_select  -- ``idx is False`` matched only
   Python False; ``numpy.False_`` slipped past the guard. Fix: explicit
   isinstance check.

#5 feature_selection/filters/screen.py:449 -- ``use_simple_mode is False``
   rejected ``np.bool_(False)`` from config, silently forcing the
   ``len < n`` branch. Fix: ``not use_simple_mode``.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---- #1 _resolve_use_gpu ------------------------------------------------


def test_resolve_use_gpu_accepts_numpy_bool_false():
    """Pre-fix np.bool_(False) raised ValueError; post-fix it's accepted."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    out = _resolve_use_gpu(use_gpu=np.bool_(False), work=100, threshold=None)
    assert out is False or out == False  # noqa: E712


def test_resolve_use_gpu_accepts_python_bool_false():
    """Resolve use gpu accepts python bool false."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    assert _resolve_use_gpu(use_gpu=False, work=100, threshold=None) == False  # noqa: E712


def test_resolve_use_gpu_string_auto_still_works():
    """Resolve use gpu string auto still works."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    # Without GPU available, "auto" returns False.
    out = _resolve_use_gpu(use_gpu="auto", work=0, threshold=None)
    assert isinstance(out, bool)


def test_resolve_use_gpu_invalid_string_raises():
    """Resolve use gpu invalid string raises."""
    from mlframe.feature_engineering.transformer.random_features import _resolve_use_gpu

    with pytest.raises(ValueError, match="must be True, False, or 'auto'"):
        _resolve_use_gpu(use_gpu="invalid", work=0, threshold=None)


# ---- #2 _select_stage4_backend source guard ------------------------------


def test_row_attention_stage4_backend_handles_numpy_bool():
    """np.bool_ flags select the CPU njit backend exactly like Python bools, and an unknown string is rejected."""
    from mlframe.feature_engineering.transformer._kernels_njit import row_attention_stage4_njit
    from mlframe.feature_engineering.transformer.row_attention import _select_stage4_backend

    assert _select_stage4_backend(gpu_stage4=np.bool_(False), n_queries_hint=1000) is row_attention_stage4_njit
    assert _select_stage4_backend(gpu_stage4=False, n_queries_hint=1000) is row_attention_stage4_njit
    with pytest.raises(ValueError, match="gpu_stage4 must be True, False, or 'auto'"):
        _select_stage4_backend(gpu_stage4="sometimes", n_queries_hint=1000)


# ---- #3 standardize isinstance guard -------------------------------------


def test_compute_rff_features_accepts_numpy_bool_standardize():
    """np.bool_ standardize behaves like the Python bool; a non-bool (int) is still rejected with TypeError."""
    from mlframe.feature_engineering.transformer.random_features import compute_rff_features

    X = np.random.default_rng(0).normal(size=(40, 3)).astype(np.float32)
    kwargs = dict(seed=1, n_features=8, use_gpu=False)
    plain = compute_rff_features(X, standardize=True, **kwargs).to_numpy()
    numpy_flag = compute_rff_features(X, standardize=np.bool_(True), **kwargs).to_numpy()
    np.testing.assert_array_equal(plain, numpy_flag)
    unstandardized = compute_rff_features(X, standardize=np.bool_(False), **kwargs).to_numpy()
    assert not np.allclose(plain, unstandardized)
    with pytest.raises(TypeError, match="standardize must be bool"):
        compute_rff_features(X, standardize=1, **kwargs)  # type: ignore[arg-type]


# ---- #4 _select np.False_ guard -----------------------------------------


def test_train_eval_select_handles_numpy_bool_false():
    """Pre-fix ``idx is False`` matched only Python False; np.False_
    slipped past."""
    import pathlib
    import mlframe as _mlframe

    # ``select_target`` was carved out of ``train_eval.py`` into
    # ``_train_eval_select_target.py``; the ``idx is False`` -> isinstance
    # fix lives in the sibling. Concat parent + sibling so the source-grep
    # guard survives the split.
    _dir = pathlib.Path(_mlframe.__file__).resolve().parent / "training"
    src = (_dir / "train_eval.py").read_text(encoding="utf-8")
    sib = _dir / "targets" / "_train_eval_select_target.py"
    if sib.exists():
        src += "\n" + sib.read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert 'if idx is None or idx is False or (hasattr(idx, "__len__") and len(idx) == 0):' not in src
    # Post-fix isinstance marker:
    assert "isinstance(idx, (bool, np.bool_))" in src


# ---- #5 screen.py use_simple_mode ---------------------------------------


def test_screen_use_simple_mode_uses_not_not_is_false():
    """Pre-fix ``use_simple_mode is False`` rejected np.bool_; post-fix
    ``not use_simple_mode`` works uniformly."""
    import pathlib
    import mlframe as _mlframe

    # 2026-05-22 split: screen_predictors moved to _screen_predictors.py.
    # Second split (2026-05-2x): confirm-step body moved to _confirm_predictor.py
    # where this literal now lives.
    _dir = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_selection" / "filters"
    src = (_dir / "screen.py").read_text(encoding="utf-8") + "\n" + (_dir / "_screen_predictors.py").read_text(encoding="utf-8")
    _confirm = _dir / "_confirm_predictor.py"
    if _confirm.exists():
        src += "\n" + _confirm.read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert "and (use_simple_mode is False or len(cached_MIs)" not in src
    # Post-fix marker:
    assert "and (not use_simple_mode or len(cached_MIs)" in src
