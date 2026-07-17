"""Regression tests for fixes in
``mlframe.feature_selection.wrappers.rfecv._stability_select``.

Finding [17][Low]: ``_fit_stability_selection`` had its triple-quoted docstring
placed AFTER the ``if X.shape[0] < 20: raise`` guard, so Python did not bind it
to ``__doc__`` (a docstring must be the first statement in the function body).
The literal was evaluated-and-discarded on every call and introspection showed
no docstring. The fix moves the docstring to be the first statement.
"""

import types

import pytest


def _load_func():
    mod = pytest.importorskip("mlframe.feature_selection.wrappers.rfecv._stability_select")
    return mod._fit_stability_selection


def test_fit_stability_selection_has_docstring():
    """The Stability-Selection literal is now bound to __doc__, not a dead string."""
    func = _load_func()
    assert func.__doc__ is not None, (
        "_fit_stability_selection.__doc__ is None: the docstring is still placed below the n_samples guard and is being evaluated-and-discarded."
    )
    # The exact wording that was intended as the docstring must be reachable via
    # standard introspection (help()/IDE) now.
    assert "Stability Selection (Meinshausen & Buhlmann 2010, JRSS-B)." in func.__doc__
    assert "Bootstrap-based feature selection." in func.__doc__


def test_fit_stability_selection_guard_still_raises_on_small_n():
    """Moving the docstring must not break the n<20 guard that follows it."""
    func = _load_func()
    np = pytest.importorskip("numpy")

    # Minimal stand-in for the bound `self`: the guard fires before any of the
    # self.* attributes are touched, so a bare namespace is sufficient and keeps
    # this a focused unit test (no RFECV construction / training run).
    fake_self = types.SimpleNamespace()
    X = np.zeros((5, 3))  # n=5 < 20 -> must raise
    y = np.zeros(5)

    with pytest.raises(ValueError, match=r"stability_selection requires n_samples >= 20"):
        func(fake_self, X, y, signature="sig")
