"""Regression benchmark: `preprocessing_extensions=None` must remain a
byte-identical no-op that returns before any extension work is reached.

Rationale (from Audit #02 plan, Phase 3.7): every sklearn-based extension
is opt-in; when the user omits `preprocessing_extensions`, the Polars-native
fastpath must be preserved. We assert identity and that execution never gets past the short-circuit.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


def _make_df(n_rows: int = 10_000, n_cols: int = 20) -> pd.DataFrame:
    """Make df."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])


def test_fastpath_returns_inputs_unchanged():
    """Fastpath returns inputs unchanged."""
    train, val, test = _make_df(), _make_df(500), _make_df(500)
    out_train, out_val, out_test, pipe = apply_preprocessing_extensions(train, val, test, config=None, verbose=0)
    assert out_train is train
    assert out_val is val
    assert out_test is test
    assert pipe is None


def test_fastpath_polars_inputs_unchanged():
    """Fastpath polars inputs unchanged."""
    train = pl.DataFrame({"a": np.arange(1000), "b": np.arange(1000).astype(float)})
    val = pl.DataFrame({"a": np.arange(100), "b": np.arange(100).astype(float)})
    test = pl.DataFrame({"a": np.arange(100), "b": np.arange(100).astype(float)})
    out_train, out_val, out_test, pipe = apply_preprocessing_extensions(train, val, test, config=None, verbose=0)
    assert out_train is train
    assert out_val is val
    assert out_test is test
    assert pipe is None


def test_a_none_config_returns_before_the_short_circuit(monkeypatch):
    """A ``config=None`` call must return before any of the work behind the short-circuit runs.

    Asked of the first function past that ``return`` rather than of the clock. 50ms for 1000 calls is 50us
    each, which the nightly coverage job (tracing every line) or a contended worker multiplies severalfold
    on entirely healthy code -- and an aggregate loop of trivial work is unlikely to catch anything short
    of an order-of-magnitude regression anyway. A call count catches the first added statement.
    """
    from mlframe.training.pipeline import _pipeline_extensions as pe

    train, val, test = _make_df(), _make_df(500), _make_df(500)

    reached = []
    real_gate = pe._has_active_extension_stage

    def _spy(*args, **kwargs):
        """Record that execution got past the ``config is None`` return."""
        reached.append(1)
        return real_gate(*args, **kwargs)

    monkeypatch.setattr(pe, "_has_active_extension_stage", _spy)

    out = apply_preprocessing_extensions(train, val, test, config=None, verbose=0)
    assert not reached, "a config=None call ran past the `if config is None: return ...` short-circuit"
    assert out[0] is train and out[1] is val and out[2] is test and out[3] is None, "the fastpath must return the inputs untouched"

    # The spy has to be reachable at all, or the assertion above holds for the wrong reason.
    apply_preprocessing_extensions(train, val, test, config=PreprocessingExtensionsConfig(), verbose=0)
    assert reached, "the gate past the short-circuit was never reached even with a config supplied; this test has lost its subject"
