"""Five handlers substituted a value that made the check they guard pass, or latched a transient fault forever.

The shape is the same each time: something raises, the handler substitutes a placeholder, and the placeholder
happens to be exactly the value that means "everything is fine" to the next line.

  * MRMR's rescue loop substituted `p_value = 0.0` -- maximally significant -- so a broken permutation probe
    admitted every scanned candidate through the gate that exists because plug-in MI is upward-biased.
  * The same loop substituted `pair_mi = 0.0`, the one value that makes the redundancy test fail, so a failed
    pair-MI silently admitted an algebraic near-duplicate.
  * The CatBoost GPU probe and the numba.cuda compile probe latched `False` for the whole process on ANY
    exception, so one moment of GPU contention at startup cost the entire run its GPU path, logged at debug.
  * `_ensure_cb_mtr_loss` substituted `params = {}` when `get_param()` raised, concluded the user had set no
    loss, and overwrote a deliberately-chosen objective with `MultiRMSE`.

Every one of them now fails CLOSED or leaves the decision unresolved, and says so above debug level.
"""

from __future__ import annotations

import logging

import numpy as np


class _FakeMRMR:
    """The handful of attributes `_finalise_empty_support_fallback` reads, with three above-floor raw candidates.

    `min_features_fallback=1` matters: the count-floor top-up further down re-adds candidates until the floor is
    met, so with a floor of 3 it would refill everything the gates dropped and the gates would look inert.
    """

    def __init__(self):
        """Three above-floor raw candidates with descending cached MI."""
        self.n_features_in_ = 3
        self.feature_names_in_ = ["a", "b", "c"]
        self.min_features_fallback = 1
        self.min_relevance_gain = 0.0
        self.min_relevance_gain_relative_to_first = 0.0
        self.cached_MIs = {(0,): 0.9, (1,): 0.8, (2,): 0.7}
        self.quantization_dtype = np.int32
        self._engineered_recipes_ = {}
        self.support_ = None
        self.n_features_ = 0


def _rescue(monkeypatch, *, sig_raises=False, pair_raises=False, p_value=0.001, pair_mi=0.0):
    """Run the empty-support rescue with the two probes stubbed, and return the rescued support."""
    import mlframe.feature_selection.filters.info_theory as it
    import mlframe.feature_selection.filters.permutation as perm
    from mlframe.feature_selection.filters._mrmr_fit_impl._finalise import _finalise_empty_support_fallback

    def _boom(*a, **k):
        """Stand in for a dtype mismatch, a numba typing failure, or a device fault inside the probe."""
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(perm, "mi_direct", _boom if sig_raises else (lambda *a, **k: (0, 0, 0, p_value)))
    monkeypatch.setattr(it, "mi", _boom if pair_raises else (lambda *a, **k: pair_mi))

    m = _FakeMRMR()
    data = np.random.default_rng(0).integers(0, 4, (200, 4)).astype(np.int32)
    _finalise_empty_support_fallback(m, 0, ["a", "b", "c", "y"], data, np.array([4, 4, 4, 4], dtype=np.int32), np.array([3], dtype=np.int64))
    return sorted(int(i) for i in m.support_)


class TestTheMrmrRescueGatesFailClosed:
    """A probe that cannot answer must not be read as an affirmative answer."""

    def test_working_probes_rescue_every_significant_candidate(self, monkeypatch):
        """The control: with p well under alpha and no redundancy, all three are admitted."""
        assert _rescue(monkeypatch) == [0, 1, 2]

    def test_a_failed_significance_probe_drops_the_candidate(self, monkeypatch, caplog):
        """`p_value = 0.0` is maximally significant, so the gate one line later passed for everything."""
        with caplog.at_level(logging.WARNING):
            rescued = _rescue(monkeypatch, sig_raises=True)
        assert rescued == [0], "a broken significance probe still admitted candidates on magnitude alone"
        assert any("dropping the candidate" in r.message for r in caplog.records)

    def test_that_matches_a_genuinely_insignificant_verdict(self, monkeypatch):
        """Fail-closed means "as if the probe had said no", stated as an equality."""
        assert _rescue(monkeypatch, sig_raises=True) == _rescue(monkeypatch, p_value=0.9)

    def test_a_failed_pair_mi_treats_the_pair_as_redundant(self, monkeypatch, caplog):
        """`0.0` was exactly the value that makes the redundancy comparison fail."""
        with caplog.at_level(logging.WARNING):
            rescued = _rescue(monkeypatch, pair_raises=True)
        assert rescued == [0], "a broken pair-MI probe still admitted an unchecked near-duplicate"
        assert any("treating the pair as redundant" in r.message for r in caplog.records)

    def test_a_high_pair_mi_still_dedupes_normally(self, monkeypatch):
        """The redundancy gate itself must be unchanged for a working probe."""
        assert _rescue(monkeypatch, pair_mi=10.0) == [0]


class TestTheGpuProbesDoNotLatchOnATransientFault:
    """A device that is busy right now is not a device that cannot work."""

    def test_the_catboost_probe_latches_only_on_the_absence_signature(self, monkeypatch, caplog):
        """A GPU OOM must leave the cache unresolved so the next caller re-probes."""
        from mlframe.training.cb import _cb_pool

        monkeypatch.setattr(_cb_pool, "_CB_GPU_USABLE_CACHE", None)
        monkeypatch.setattr(_cb_pool, "_cached_gpu_info", lambda: True)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        class _Boom:
            """A CatBoostRegressor stand-in whose probe fit hits a device fault."""

            def __init__(self, **kw):
                """Accept and ignore the probe's constructor arguments."""

            def fit(self, *a, **k):
                """Stand in for a device fault during the probe fit."""
                raise RuntimeError("out of memory")

        import catboost

        monkeypatch.setattr(catboost, "CatBoostRegressor", _Boom)
        with caplog.at_level(logging.WARNING):
            assert _cb_pool._cb_gpu_usable() is False
        assert _cb_pool._CB_GPU_USABLE_CACHE is None, "a transient fault latched the process onto CPU CatBoost"
        assert any("transient" in r.message for r in caplog.records)

    def test_the_catboost_probe_still_latches_on_a_cpu_only_wheel(self, monkeypatch):
        """The one failure that IS a permanent property of the host must still be cached."""
        from mlframe.training.cb import _cb_pool

        monkeypatch.setattr(_cb_pool, "_CB_GPU_USABLE_CACHE", None)
        monkeypatch.setattr(_cb_pool, "_cached_gpu_info", lambda: True)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

        class _NoGpu:
            """A CatBoostRegressor stand-in behaving like a CPU-only wheel."""

            def __init__(self, **kw):
                """Accept and ignore the probe's constructor arguments."""

            def fit(self, *a, **k):
                """The exact error a CPU-only catboost wheel raises."""
                raise RuntimeError("catboost/libs/train_lib: Environment for task type [GPU] not found")

        import catboost

        monkeypatch.setattr(catboost, "CatBoostRegressor", _NoGpu)
        assert _cb_pool._cb_gpu_usable() is False
        assert _cb_pool._CB_GPU_USABLE_CACHE is False

    def test_the_numba_probe_names_its_permanent_faults(self):
        """Latching is correct for these three and for nothing else."""
        from mlframe.feature_selection.filters._internals import _PERMANENT_CUDA_FAULTS

        assert {t.__name__ for t in _PERMANENT_CUDA_FAULTS} >= {"ImportError", "NvvmSupportError", "CudaSupportError"}

    def test_a_transient_numba_fault_leaves_the_cache_unset(self, monkeypatch, caplog):
        """The failure mode: one contended startup routed every filter to CPU for the process."""
        import mlframe.feature_selection.filters._internals as internals

        monkeypatch.setattr(internals, "_NUMBA_CUDA_CAN_COMPILE", None)
        from numba import cuda as _cuda

        monkeypatch.setattr(_cuda, "is_available", lambda: True)
        monkeypatch.setattr(_cuda, "to_device", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")))
        with caplog.at_level(logging.WARNING):
            assert internals.numba_cuda_can_compile() is False
        assert internals._NUMBA_CUDA_CAN_COMPILE is None
        assert any("transient" in r.message for r in caplog.records)


class TestUnknownParamsAreNotEmptyParams:
    """The objective overwrite."""

    def test_a_failing_get_param_leaves_the_objective_alone(self, monkeypatch, caplog):
        """`params = {}` made the next line conclude the caller set no loss."""
        from mlframe.training import _training_loop as tl

        class CatBoostRegressor:
            """A model whose parameter accessor is broken, with a deliberately-chosen objective."""

            def __init__(self):
                """Start with no recorded set_params calls."""
                self.set_calls = []

            def get_param(self, *a, **k):
                """Stand in for a CatBoost version whose accessor signature drifted."""
                raise TypeError("get_param() missing 1 required positional argument")

            def set_params(self, **kw):
                """Record any attempt to overwrite the objective."""
                self.set_calls.append(kw)

        m = CatBoostRegressor()
        with caplog.at_level(logging.WARNING):
            tl._ensure_cb_mtr_loss(m, None)
        assert not m.set_calls, f"a user-supplied objective was overwritten with {m.set_calls}"
        assert any("untouched" in r.message for r in caplog.records)
