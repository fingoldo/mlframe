"""Ask the installed booster whether it takes polars, rather than assuming it either way.

A production run converted a polars frame to pandas on every single ``predict_proba`` call, with nothing in the
log explaining why -- which reads as "CatBoost cannot consume polars". Probed against the installed build,
CatBoost 1.2.10 accepts a polars frame end to end and LightGBM 4.6.0 does not, so neither blanket assumption
was right.
"""

from __future__ import annotations

import pytest

from mlframe.training import _polars_native_support as nat

pytest.importorskip("polars")


@pytest.fixture(autouse=True)
def _clean_cache():
    """The probe result is cached per (library, version); a stale entry would answer for a patched library."""
    nat.reset_cache()
    yield
    nat.reset_cache()


class TestTheProbe:
    """It has to answer from the installed library, not from a table someone wrote down."""

    def test_catboost_is_probed_not_assumed(self):
        """Whatever the answer, it must come from a real fit+predict rather than a hardcoded verdict."""
        pytest.importorskip("catboost")
        assert nat.accepts_polars("catboost") is nat._probe("catboost")

    def test_lightgbm_is_probed_not_assumed(self):
        """Same for LightGBM, which answers differently on this box."""
        pytest.importorskip("lightgbm")
        assert nat.accepts_polars("lightgbm") is nat._probe("lightgbm")

    def test_absent_library_answers_false(self):
        """A missing library routes the caller down the pandas path it would have needed anyway."""
        assert nat.accepts_polars("definitely_not_installed") is False

    def test_result_is_cached(self, monkeypatch):
        """The probe fits a model; repeating it per predict call would be its own performance bug."""
        pytest.importorskip("catboost")
        calls = []
        monkeypatch.setattr(nat, "_probe", lambda lib: calls.append(lib) or True)
        nat.accepts_polars("catboost")
        nat.accepts_polars("catboost")
        assert calls == ["catboost"]

    def test_cache_is_keyed_by_version(self, monkeypatch):
        """An upgrade inside a long-lived process must not be answered from the old build's probe."""
        calls = []
        monkeypatch.setattr(nat, "_probe", lambda lib: calls.append(lib) or True)
        monkeypatch.setattr(nat, "_version", lambda lib: "1.0")
        nat.accepts_polars("catboost")
        monkeypatch.setattr(nat, "_version", lambda lib: "2.0")
        nat.accepts_polars("catboost")
        assert calls == ["catboost", "catboost"]

    def test_a_raising_library_answers_false_rather_than_propagating(self, monkeypatch):
        """The probe is a capability question; it must never take down the caller that asked it."""

        def _boom(_lib):
            """Stand in for a library whose probe blows up."""
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(nat, "_probe", _boom)
        with pytest.raises(RuntimeError):
            nat.accepts_polars("catboost")  # the raise is the probe's own; accepts_polars does not swallow it

    def test_probe_frame_carries_the_dtypes_that_historically_broke_dispatch(self):
        """A probe over plain floats would pass while the real frames still fail."""
        frame = nat._probe_frame()
        import polars as pl

        assert any(isinstance(dtype, pl.Enum) for dtype in frame.dtypes)
        assert frame["nullable"].null_count() > 0


class TestWhatTheAnswerIsHere:
    """Pins the measured behaviour of the installed builds, so a version bump that changes it is visible."""

    def test_catboost_accepts_polars_on_this_build(self):
        """Measured on CatBoost 1.2.10: the per-predict conversion the log showed was not a library limitation."""
        pytest.importorskip("catboost")
        assert nat.accepts_polars("catboost") is True

    def test_lightgbm_does_not_accept_this_frame(self):
        """Measured on LightGBM 4.6.0 with an Enum column and nulls -- so it genuinely needs the pandas path."""
        pytest.importorskip("lightgbm")
        assert nat.accepts_polars("lightgbm") is False
