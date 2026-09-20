"""The honest-holdout re-score must not rebuild the same feature matrix per spec, nor read an uncapped holdout.

The re-score pulls every usable feature on the holdout rows for each final spec, concurrently. The holdout is a
fraction of train with no row cap of its own, and specs that share a base column were each building the identical
matrix: on a multi-million-row frame that is a full feature block per spec, held at once across threads. The matrix is
now built once per base set and the rows are capped by ``mi_sample_n``, the same cap the in-screen MI uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeSpec
from mlframe.training.composite.discovery import _honest_holdout
from mlframe.training.composite.discovery._honest_holdout import rescore_specs_on_holdout
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class _Disc:
    """The attribute surface the re-score reads off the discovery instance."""

    def __init__(self, **cfg_kwargs):
        self.config = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, **cfg_kwargs)


_FEATS = ["base", "f1", "f2", "f3"]


def _frame(n: int = 1200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame whose target is an additive residual on ``base``."""
    rng = np.random.default_rng(seed)
    base = rng.normal(50.0, 5.0, n)
    f1, f2, f3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = base + 2.0 * f1 - f2 + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"base": base, "f1": f1, "f2": f2, "f3": f3, "y": y}), y


def _spec(name: str, transform_name: str = "diff") -> CompositeSpec:
    """A spec on the shared ``base`` column."""
    return CompositeSpec(
        name=name, target_col="y", transform_name=transform_name, base_column="base",
        fitted_params={}, mi_gain=1.0, mi_y=0.0, mi_t=1.0, valid_domain_frac=1.0, n_train_rows=100,
    )


def _count_builds(monkeypatch) -> dict:
    """Count calls to the holdout matrix builder."""
    seen = {"n": 0}
    original = _honest_holdout._build_x_remaining_holdout

    def counting(*args, **kwargs):
        """Count the build, then defer to the real one."""
        seen["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_honest_holdout, "_build_x_remaining_holdout", counting)
    return seen


def test_specs_sharing_a_base_build_the_matrix_once(monkeypatch):
    """Four specs on one base need one matrix, not four."""
    df, y = _frame()
    holdout = np.arange(800, 1200)
    seen = _count_builds(monkeypatch)
    specs = [_spec(f"y-diff-{i}") for i in range(4)]
    rescore_specs_on_holdout(_Disc(), df, "y", specs, _FEATS, holdout, y)
    assert seen["n"] == 1, f"the identical holdout matrix was built {seen['n']} times"
    assert all(s.honest_holdout_gain is not None for s in specs), "every spec must still be scored"


def test_the_stamped_gains_are_unchanged_by_the_cache(monkeypatch):
    """Sharing the matrix is a memory and time change only: the numbers must match a per-spec build."""
    df, y = _frame()
    holdout = np.arange(800, 1200)
    shared = [_spec("y-diff-a"), _spec("y-diff-b", "linear_residual")]
    rescore_specs_on_holdout(_Disc(), df, "y", shared, _FEATS, holdout, y)

    monkeypatch.setattr(_honest_holdout, "_build_x_remaining_holdout", _honest_holdout._build_x_remaining_holdout)
    per_spec = [_spec("y-diff-a"), _spec("y-diff-b", "linear_residual")]
    for one in per_spec:  # one spec at a time: the cache can hold nothing across calls
        rescore_specs_on_holdout(_Disc(), df, "y", [one], _FEATS, holdout, y)
    for a, b in zip(shared, per_spec):
        assert a.honest_holdout_gain == pytest.approx(b.honest_holdout_gain)
        assert a.honest_holdout_n_rows == b.honest_holdout_n_rows


def test_a_cap_above_the_holdout_size_changes_nothing():
    """Below the cap every holdout row is read, so the stamped numbers are the uncapped ones."""
    df, y = _frame()
    holdout = np.arange(800, 1200)
    capped = [_spec("y-diff-a")]
    uncapped = [_spec("y-diff-a")]
    rescore_specs_on_holdout(_Disc(mi_sample_n=100_000), df, "y", capped, _FEATS, holdout, y)
    rescore_specs_on_holdout(_Disc(mi_sample_n=None), df, "y", uncapped, _FEATS, holdout, y)
    assert capped[0].honest_holdout_gain == pytest.approx(uncapped[0].honest_holdout_gain)
    assert capped[0].honest_holdout_n_rows == uncapped[0].honest_holdout_n_rows == 400


def test_a_cap_below_the_holdout_size_limits_the_rows_read():
    """Above the cap the re-score reads a seeded draw, so it never scales with the holdout."""
    df, y = _frame()
    holdout = np.arange(800, 1200)
    specs = [_spec("y-diff-a")]
    rescore_specs_on_holdout(_Disc(mi_sample_n=150), df, "y", specs, _FEATS, holdout, y)
    assert specs[0].honest_holdout_n_rows == 150, "the cap must bound the rows the honest gain is measured on"


def test_the_capped_draw_is_reproducible():
    """Two runs at the same seed must stamp the same honest gain, or the diagnostic would drift run to run."""
    df, y = _frame()
    holdout = np.arange(800, 1200)
    first, second = [_spec("y-diff-a")], [_spec("y-diff-a")]
    rescore_specs_on_holdout(_Disc(mi_sample_n=200), df, "y", first, _FEATS, holdout, y)
    rescore_specs_on_holdout(_Disc(mi_sample_n=200), df, "y", second, _FEATS, holdout, y)
    assert first[0].honest_holdout_gain == pytest.approx(second[0].honest_holdout_gain)
