"""Behavioural checks for the module-level caches that gained locks: concurrent writers must not lose or corrupt entries."""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest


def _run_threads(target, n_threads: int = 8) -> None:
    """Start n_threads copies of target(i), release them together, join, re-raise the first error."""
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def _wrap(i: int) -> None:
        """Wait at the barrier then run the target, collecting any exception."""
        try:
            barrier.wait()
            target(i)
        except BaseException as exc:  # the harness collects every failure, including assertion errors
            errors.append(exc)

    threads = [threading.Thread(target=_wrap, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def test_pipeline_roundtrip_cache_keeps_every_concurrent_write_and_snapshots_safely(tmp_path, monkeypatch):
    """Writers store distinct keys while another thread persists snapshots; nothing is lost and no snapshot iteration fails."""
    from mlframe.training.core import _setup_helpers_pipeline_cache as pc

    monkeypatch.setattr(pc, "_PIPELINE_JSON_DISK_CACHE_PATH", str(tmp_path / "rt.json"))
    pc._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()

    def work(i: int) -> None:
        """Even threads write 300 keys each; odd threads persist snapshots meanwhile."""
        if i % 2 == 0:
            for k in range(300):
                pc.pipeline_roundtrip_store(f"k{i}_{k}", k % 2 == 0)
        else:
            for _ in range(20):
                pc._persist_pipeline_disk_cache()

    _run_threads(work)
    assert len(pc._PIPELINE_JSON_ROUNDTRIP_CACHE) == 4 * 300
    assert pc.pipeline_roundtrip_lookup("k0_0") is True and pc.pipeline_roundtrip_lookup("k0_1") is False
    assert pc.pipeline_roundtrip_lookup("never_stored") is None
    pc._PIPELINE_JSON_ROUNDTRIP_CACHE.clear()


def test_tweedie_warning_counter_does_not_lose_increments():
    """N concurrent identical events must count to exactly N, so recurrence numbers are never repeated."""
    from mlframe.metrics.regression import _regression_deviance as rd

    rd._TWEEDIE_WARN_COUNTS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run_threads(lambda i: [rd._maybe_warn_tweedie("tweedie", 3, 100) for _ in range(200)])
    assert rd._TWEEDIE_WARN_COUNTS[("tweedie", 3, 100)] == 8 * 200
    rd._TWEEDIE_WARN_COUNTS.clear()


def test_metric_registry_survives_concurrent_register_unregister_and_iteration():
    """Registering, unregistering and iterating from different threads never raises 'dict changed size'."""
    from mlframe.training import metrics_registry as mr
    from mlframe.training.configs import TargetTypes

    tt = TargetTypes.REGRESSION
    stop = threading.Event()

    def churn(i: int) -> None:
        """Thread 0 iterates while the rest register and unregister their own names."""
        if i == 0:
            while not stop.is_set():
                list(mr.iter_extra_metrics(tt, np.zeros(3), None, np.zeros(3)))
                mr.list_registered(tt)
        else:
            for k in range(300):
                name = f"_lock_test_{i}_{k % 5}"
                mr.register_metric(tt, name, lambda y, p, q: 0.0)
                mr.unregister_metric(tt, name)
            if i == 7:
                stop.set()

    try:
        _run_threads(churn)
    finally:
        stop.set()
        for name in list(mr.list_registered(tt)):
            if name.startswith("_lock_test_"):
                mr.unregister_metric(tt, name)
    assert not [n for n in mr.list_registered(tt) if n.startswith("_lock_test_")]


def test_axis_registry_conflict_is_still_detected_under_races():
    """Racing registrations of two different classes for one axis: exactly one wins, the loser gets ValueError."""
    from mlframe.training.feature_handling import axis as ax

    axis = next(iter(ax.Axis))
    saved = dict(ax._AXIS_REGISTRY)
    ax._AXIS_REGISTRY.pop(axis, None)

    class _A(ax.HandlerSpec):
        """Spec A."""

        @classmethod
        def axis(cls):
            """Return the contested axis."""
            return axis

    class _B(ax.HandlerSpec):
        """Spec B."""

        @classmethod
        def axis(cls):
            """Return the contested axis."""
            return axis

    results: list[str] = []

    def work(i: int) -> None:
        """Half the threads register A, half B."""
        try:
            ax.register_handler_spec(_A if i % 2 == 0 else _B)
            results.append("ok")
        except ValueError:
            results.append("conflict")

    try:
        _run_threads(work)
        winner = ax._AXIS_REGISTRY[axis]
        assert winner in (_A, _B)
        losers = _B if winner is _A else _A
        assert results.count("conflict") == 4 and results.count("ok") == 4
        with pytest.raises(ValueError):
            ax.register_handler_spec(losers)
    finally:
        ax._AXIS_REGISTRY.clear()
        ax._AXIS_REGISTRY.update(saved)


def test_permutation_block_cache_bytes_stay_consistent_with_its_entries():
    """After concurrent identical draws the byte counter equals the sum of the cached blocks and the stream is reproducible."""
    from mlframe.feature_selection.filters import _family_maxt_gate as g

    g._perm_block_cache.clear()
    g._perm_block_cache_bytes = 0
    y = np.arange(400) % 3

    def work(i: int) -> None:
        """Every thread draws the same seeded blocks."""
        rng = np.random.default_rng(5)
        for _ in range(5):
            g._permutation_block(rng, y, 16)

    _run_threads(work)
    assert g._perm_block_cache_bytes == sum(v[0].nbytes for v in g._perm_block_cache.values())
    a = g._permutation_block(np.random.default_rng(5), y, 16)
    b = g._permutation_block(np.random.default_rng(5), y, 16)
    np.testing.assert_array_equal(a, b)
    g._perm_block_cache.clear()
    g._perm_block_cache_bytes = 0
