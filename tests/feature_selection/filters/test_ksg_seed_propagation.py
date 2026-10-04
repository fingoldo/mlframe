"""The caller's seed must reach the KSG mutual-information jitter in the hermite pair search and the trivial baselines."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _hermite_fe_optimise as hfo
from mlframe.feature_selection.filters import _hermite_fe_optimise_pair as hfp
from mlframe.feature_selection.filters.fe_baselines import _mi_1d, best_trivial_pair, ksg_random_state


def _tied_xy(n: int = 400):
    """Integer-valued x with a noisy continuous y: ties make the KSG jitter (and so the seed) visible in the MI value."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, size=n).astype(np.float64)
    return x, rng.normal(size=n) + x


def test_mi_1d_ksg_depends_on_random_state_and_is_reproducible() -> None:
    """Different random_state gives a different KSG MI on tied data; equal state is bit-reproducible; default equals 42."""
    x, y = _tied_xy()
    kw = dict(discrete_target=False, mi_estimator="ksg")
    a = _mi_1d(x, y, random_state=42, **kw)
    assert a == _mi_1d(x, y, random_state=42, **kw)
    assert a == _mi_1d(x, y, **kw)
    assert a != _mi_1d(x, y, random_state=7, **kw)


def test_ksg_random_state_accepts_any_integer_seed() -> None:
    """Negative and oversized seeds are mapped into sklearn's valid range instead of raising."""
    assert 0 <= ksg_random_state(-5) < 2**32
    assert ksg_random_state(2**40 + 3) == 3
    x, y = _tied_xy()
    assert np.isfinite(_mi_1d(x, y, discrete_target=False, mi_estimator="ksg", random_state=-5))


def test_best_trivial_pair_ksg_seed_changes_mi() -> None:
    """best_trivial_pair's KSG path follows the supplied seed."""
    x, y = _tied_xy()
    x_b = np.roll(x, 1)
    kw = dict(discrete_target=False, mi_estimator="ksg")
    m42 = best_trivial_pair(x, x_b, y, random_state=42, **kw)
    m7 = best_trivial_pair(x, x_b, y, random_state=7, **kw)
    assert m42 is not None and m7 is not None
    assert m42[2] != m7[2]


def test_baseline_mi_pair_ksg_seed_changes_mi() -> None:
    """The identity baseline of the pair search follows the supplied seed."""
    x, y = _tied_xy()
    x_b = np.roll(x, 1)
    kw = dict(discrete_target=False, mi_estimator="ksg")
    assert hfo._baseline_mi_pair(x, x_b, y, random_state=42, **kw) != hfo._baseline_mi_pair(x, x_b, y, random_state=7, **kw)


@pytest.mark.parametrize("seed", [7, 123])
def test_optimise_hermite_pair_threads_seed_into_every_ksg_call(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    """Every sklearn KSG call made by the pair search receives the caller's seed rather than a hardcoded 42."""
    seen: list[int] = []
    real = {"c": hfo.mutual_info_classif, "r": hfo.mutual_info_regression}

    def _spy_r(*args, **kwargs):
        """Record random_state then delegate to sklearn regression MI."""
        seen.append(kwargs["random_state"])
        return real["r"](*args, **kwargs)

    monkeypatch.setattr(hfo, "mutual_info_regression", _spy_r)
    monkeypatch.setattr(hfp, "mutual_info_regression", _spy_r)
    x, y = _tied_xy(300)
    x_b = np.random.default_rng(1).normal(size=300)
    hfp.optimise_hermite_pair(
        x, x_b, y, discrete_target=False, mi_estimator="ksg", n_trials=4, max_degree=2, min_degree=2, seed=seed,
    )
    assert seen, "KSG path was not exercised"
    assert set(seen) == {ksg_random_state(seed)}
