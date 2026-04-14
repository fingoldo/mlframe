"""Tests for mlframe.training.logging_transformers."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from mlframe.training.logging_transformers import (
    log_resources,
    log_methods,
    wrap_with_logging,
)

LOGGER_NAME = "mlframe.training.logging_transformers"


class _Dummy:
    """Trivial sklearn-like class used to exercise the decorators."""

    def __init__(self, factor: float = 1.0):
        self.factor = factor
        self.fitted_ = False

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.asarray(X) * self.factor

    def get_params(self, deep: bool = True):
        return {"factor": self.factor}


def test_log_resources_emits_structured_record(caplog):
    class C:
        @log_resources()
        def fit(self, X):
            return len(X)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        out = C().fit([1, 2, 3])
    assert out == 3

    records = [r for r in caplog.records if r.name == LOGGER_NAME]
    assert len(records) == 1
    rec = records[0]
    for key in ("stage", "cls", "dt_s", "rss_mb", "d_rss_mb"):
        assert hasattr(rec, key), f"missing extra key: {key}"
    assert rec.cls == "C"
    assert rec.dt_s >= 0.0


def test_log_resources_custom_stage_and_extra_factory(caplog):
    def factory(self, X, y=None):
        return {"n_samples": len(X), "custom": "yes"}

    class C:
        @log_resources(stage="custom_stage", extra_factory=factory)
        def fit(self, X, y=None):
            return self

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        C().fit([0, 1, 2, 3])

    rec = [r for r in caplog.records if r.name == LOGGER_NAME][-1]
    assert rec.stage == "custom_stage"
    assert rec.n_samples == 4
    assert rec.custom == "yes"


def test_log_methods_stamps_fit_and_transform(caplog):
    @log_methods("fit", "transform", stage_prefix="dummy")
    class Logged(_Dummy):
        pass

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        obj = Logged(factor=2.0)
        obj.fit([1, 2, 3])
        out = obj.transform([1, 2, 3])

    assert list(out) == [2, 4, 6]
    stages = [r.stage for r in caplog.records if r.name == LOGGER_NAME]
    assert "dummy.fit" in stages
    assert "dummy.transform" in stages


def test_log_methods_skips_missing_methods():
    # Should not raise even though _Dummy lacks 'fit_transform'.
    @log_methods("fit", "transform", "fit_transform")
    class Logged(_Dummy):
        pass

    obj = Logged()
    assert not hasattr(obj, "fit_transform") or callable(getattr(obj, "fit_transform"))


def test_wrap_with_logging_on_standard_scaler(caplog):
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    raw = StandardScaler()
    wrapped = wrap_with_logging(raw, stage="scaler")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        out = wrapped.fit_transform(X)

    expected = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(out, expected)

    stages = [r.stage for r in caplog.records if r.name == LOGGER_NAME]
    assert "scaler.fit_transform" in stages
    # Exactly one log record for the single fit_transform call.
    assert stages.count("scaler.fit_transform") == 1


def test_wrap_with_logging_proxy_delegates_get_params():
    raw = StandardScaler(with_mean=False)
    wrapped = wrap_with_logging(raw)
    params = wrapped.get_params()
    assert params["with_mean"] is False


def test_wrap_with_logging_forwards_arbitrary_attributes():
    obj = _Dummy(factor=7.0)
    wrapped = wrap_with_logging(obj, methods=("fit", "transform"))
    assert wrapped.factor == 7.0
    wrapped.fit([1, 2])
    assert wrapped.fitted_ is True


def test_wrap_with_logging_skips_missing_methods():
    # _Dummy has no fit_transform, should silently skip.
    wrapped = wrap_with_logging(_Dummy(), methods=("fit", "transform", "fit_transform"))
    # fit_transform shouldn't exist as an instrumented method on the proxy class,
    # but attribute access falls through __getattr__ to the inner object, which
    # also lacks it — so AttributeError is the expected outcome.
    with pytest.raises(AttributeError):
        wrapped.fit_transform([1, 2, 3])
