"""Opt-in switches for wavedec_dispatch and the curated FE pipelines.

Each switch is listed in the gate's _OPT_IN_SLOW. The default path must keep its current output shape, and the explicit
switch must reach the other branch, so each test checks the shape that only one branch produces.
"""

import numpy as np
import pytest

from mlframe.feature_engineering import curated_fe
from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter
from mlframe.feature_engineering.wavelet_dwt import wavedec_dispatch


@pytest.fixture
def signals():
    """Four short random signals for the dispatcher branches."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 64))


def test_wavedec_dispatch_loop_path_returns_list_of_coefficient_lists(signals):
    """path='loop' yields one coefficient list per signal."""
    out = wavedec_dispatch(signals, "db2", 2, path="loop")
    assert isinstance(out, list) and len(out) == signals.shape[0]
    assert all(isinstance(level, np.ndarray) for level in out[0])


def test_wavedec_dispatch_batched_path_returns_flat_tuple(signals):
    """path='batched' yields the flat (approx, details, lengths) tuple, not the per-signal list."""
    out = wavedec_dispatch(signals, "db2", 2, path="batched")
    assert isinstance(out, tuple) and len(out) == 3


def test_wavedec_dispatch_default_path_keeps_small_batch_loop_shape(signals):
    """The default path below the threshold still takes the loop branch."""
    out = wavedec_dispatch(signals, "db2", 2)
    assert isinstance(out, list) and len(out) == signals.shape[0]


def test_wavedec_dispatch_rejects_unknown_path(signals):
    """An unknown path name fails loudly instead of silently taking one branch."""
    with pytest.raises(ValueError):
        wavedec_dispatch(signals, "db2", 2, path="gpu")


def test_curated_pipelines_opt_in_builds_shortlist_adapters():
    """The recommended curated names build ShortlistTransformerAdapter pipelines for custom_pre_pipelines."""
    names = curated_fe.recommended_curated_fe_names("regression")
    assert names
    pipes = curated_fe.curated_fe_pipelines(task="regression", names=list(names), seed=0, passthrough=False)
    assert set(pipes) == set(names)
    assert all(isinstance(step, ShortlistTransformerAdapter) for p in pipes.values() for _, step in p.steps)
