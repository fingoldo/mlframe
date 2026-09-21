"""The screening feature matrix must be gathered from the frame once per fit, not once by auto-base and again by fit.

Auto-base draws the same screening sample and gathers the same columns before ``fit`` does, so every auto-base fit
paid two full gathers over the frame. ``fit`` now reuses auto-base's matrix when the frame, the columns and the rows
match (permuting for a time key), and the resulting specs are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.composite.discovery import CompositeTargetDiscovery as _Disc
from mlframe.training.composite.discovery._fit_helpers import take_screen_matrix
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_FEATS = ["b1", "b2", "x1", "x2"]


def _frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A frame with two base candidates, so auto-base has something to rank."""
    rng = np.random.default_rng(seed)
    b1 = rng.uniform(10.0, 50.0, n)
    b2 = 0.5 * b1 + rng.normal(size=n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"b1": b1, "b2": b2, "x1": x1, "x2": x2, "y": 1.5 * b1 + 2.0 * x1 + rng.normal(size=n)})


def _fit(df: pd.DataFrame, **kw) -> CompositeTargetDiscovery:
    """An auto-base discovery fit."""
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(
        enabled=True, random_state=0, base_candidates="auto", transforms=["diff", "linear_residual"], screening="mi",
    ))
    disc.fit(df, "y", _FEATS, np.arange(len(df)), **kw)
    return disc


def _count_builds(monkeypatch) -> dict:
    """Count full screening-matrix gathers (the all-columns build on a screening sample)."""
    seen = {"n": 0}
    real = _Disc._build_feature_matrix

    def counting(self, df, cols, rows):
        """Count gathers of every usable feature column, then build as usual."""
        if len(cols) >= len(_FEATS):
            seen["n"] += 1
        return real(self, df, cols, rows)

    monkeypatch.setattr(_Disc, "_build_feature_matrix", counting)
    return seen


def test_an_auto_base_fit_gathers_the_screen_matrix_once(monkeypatch):
    """Against a fit that re-gathers the screen, exactly one full-frame gather is saved (the gates' own are untouched)."""
    import mlframe.training.composite.discovery._fit as fit_mod

    df = _frame()
    seen = _count_builds(monkeypatch)
    disc = _fit(df)
    assert disc.specs_, "the fixture must produce specs"
    with_reuse = seen["n"]
    seen["n"] = 0
    monkeypatch.setattr(fit_mod, "take_screen_matrix", lambda self, df_, cols, rows: self._build_feature_matrix(df_, cols, rows))
    _fit(df)
    assert with_reuse == seen["n"] - 1, f"expected one gather saved; with reuse {with_reuse}, without {seen['n']}"


def test_the_specs_match_a_fit_that_gathers_twice(monkeypatch):
    """Reuse is a speed change only: the discovered specs equal those of a fit that re-gathers the matrix."""
    df = _frame()
    reused = [(s.name, round(s.mi_gain, 12)) for s in _fit(df).specs_]
    import mlframe.training.composite.discovery._fit as fit_mod

    monkeypatch.setattr(fit_mod, "take_screen_matrix", lambda self, df_, cols, rows: self._build_feature_matrix(df_, cols, rows))
    regathered = [(s.name, round(s.mi_gain, 12)) for s in _fit(df).specs_]
    assert reused == regathered


def test_a_time_ordered_screen_reuses_the_permuted_rows():
    """With a time key fit reorders the sample; the stash is permuted into that order, not reused blindly."""
    x = np.arange(20, dtype=np.float64).reshape(10, 2)

    class _Stub:
        """The attribute surface ``take_screen_matrix`` reads."""

        def _build_feature_matrix(self, df, cols, rows):
            """Gather rows directly (the reference the reuse must equal)."""
            return x[np.asarray(rows)]

    stub, frame = _Stub(), object()
    rows = np.array([7, 2, 9, 4, 0])
    stub._screen_matrix_stash = (frame, ("a", "b"), rows, x[rows])
    reordered = np.array([0, 2, 4, 7, 9])
    got = take_screen_matrix(stub, frame, ["a", "b"], reordered)
    np.testing.assert_array_equal(got, x[reordered])
    assert stub._screen_matrix_stash is None, "the stash must be released after use"


def test_a_different_frame_or_column_list_gathers_afresh():
    """Any mismatch falls back to a plain gather."""
    x = np.arange(20, dtype=np.float64).reshape(10, 2)

    class _Stub:
        """Records whether a fresh gather happened."""

        built = 0

        def _build_feature_matrix(self, df, cols, rows):
            """Count and gather."""
            type(self).built += 1
            return x[np.asarray(rows)]

    rows = np.array([1, 3, 5])
    for frame, cols in ((object(), ("a", "b")), (None, ("a",))):
        stub = _Stub()
        stash_frame = object()
        stub._screen_matrix_stash = (stash_frame, ("a", "b"), rows, x[rows])
        take_screen_matrix(stub, frame if frame is not None else stash_frame, list(cols), rows)
    assert _Stub.built == 2
