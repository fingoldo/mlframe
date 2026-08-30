"""The JIT prewarm imported the whole neural stack for runs that never touch it.

A production fit with ``mlframe_models=['cb']`` spent 382.73s in ``_warmup_numba_kernels``. The stack trace was
unambiguous: ``_core_numba_warmup.py`` imports ``lightning.fabric``, which pulls lightning.pytorch.callbacks ->
torchmetrics -> torchmetrics.functional.text.bert -> ``from transformers import AutoModel``, and transformers'
``import_utils`` then probes for tensorflow. None of it is reachable from a CatBoost-only run.

The cost was also invisible: the per-group line printed ``dummy_baselines=0.0s, feature_selection=0.0s`` while
382s elapsed, because the import cascade was timed by nothing.
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace

import pytest

from mlframe.metrics._core_numba_warmup import _prewarm_numba_cache_body, prewarm_numba_cache
from mlframe.training.core._phase_config_setup import _heavy_libs_needed


class TestWhoNeedsTheNeuralStack:
    """The predicate that decides whether the import is work or waste."""

    def test_a_catboost_only_run_does_not(self):
        """The production case, stated directly."""
        assert _heavy_libs_needed(["cb"], None, SimpleNamespace(use_shap=False)) is False

    @pytest.mark.parametrize("model", ["mlp", "nn", "lstm", "gru", "rnn", "transformer"])
    def test_every_neural_tag_does(self, model):
        """Whatever ``is_neural_model`` recognises has to keep its prewarm."""
        assert _heavy_libs_needed(["cb", model], None, SimpleNamespace(use_shap=False)) is True

    def test_a_recurrent_model_list_does(self):
        """``recurrent_models`` is a separate argument and was not even reaching this decision before."""
        assert _heavy_libs_needed(["cb"], ["lstm"], SimpleNamespace(use_shap=False)) is True

    def test_shap_does(self):
        """The same block prewarms shap, which the trainer imports when use_shap is on."""
        assert _heavy_libs_needed(["cb"], None, SimpleNamespace(use_shap=True)) is True

    def test_an_unreadable_config_warms_rather_than_skips(self):
        """Skipping wrongly costs a slow first fit and a confusing profile; warming wrongly only costs time."""
        class _Explodes:
            """A config whose attribute access raises, standing in for any shape this cannot read."""

            @property
            def use_shap(self):
                """Always raises."""
                raise RuntimeError("no")

        assert _heavy_libs_needed(["cb"], None, _Explodes()) is True

    def test_no_models_and_no_shap_skips(self):
        """Nothing declared means nothing to warm for."""
        assert _heavy_libs_needed(None, None, SimpleNamespace(use_shap=False)) is False


class TestTheGateReachesTheWarmup:
    """A predicate nobody threads through changes nothing."""

    @pytest.mark.parametrize("fn", [prewarm_numba_cache, _prewarm_numba_cache_body])
    def test_the_entry_points_accept_the_flag(self, fn):
        """A gate the entry points cannot receive is a gate nobody can use."""
        assert "include_heavy_libs" in inspect.signature(fn).parameters

    def test_the_baselines_warmup_forwards_it(self):
        """The suite calls this one, so it is where the flag has to arrive."""
        from mlframe.training.baselines.dummy import _warmup_numba_kernels

        assert "include_heavy_libs" in inspect.signature(_warmup_numba_kernels).parameters

    def test_setup_configuration_accepts_recurrent_models(self):
        """It could not decide correctly before, because the argument never reached it."""
        from mlframe.training.core._phase_config_setup import setup_configuration

        assert "recurrent_models" in inspect.signature(setup_configuration).parameters


class TestTheCostIsAttributed:
    """A 382-second group that reports nothing is how this stayed hidden."""

    def test_the_per_group_line_names_the_import_group(self, caplog):
        """Skipping still reports the group, so the line accounts for every part of the step."""
        with caplog.at_level(logging.INFO, logger="mlframe.metrics._core_numba_warmup"):
            _prewarm_numba_cache_body(include_feature_selection=False, include_heavy_libs=False)
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "heavy_lib_imports" in text

    def test_skipping_reports_zero_for_that_group(self, caplog):
        """A skipped group still appears, so the line accounts for the whole step."""
        with caplog.at_level(logging.INFO, logger="mlframe.metrics._core_numba_warmup"):
            _prewarm_numba_cache_body(include_feature_selection=False, include_heavy_libs=False)
        line = next(r.getMessage() for r in caplog.records if "heavy_lib_imports" in r.getMessage())
        assert "heavy_lib_imports=0.0s" in line

    def test_lightning_is_never_probed_when_the_gate_is_off(self, monkeypatch):
        """The behavioural end: the block resolves lightning through find_spec, so a gated-off run must not ask."""
        import importlib.util as _ilu

        asked = []
        _real = _ilu.find_spec

        def _spy(name, *a, **k):
            """Record every module the prewarm looks up."""
            asked.append(name)
            return _real(name, *a, **k)

        monkeypatch.setattr(_ilu, "find_spec", _spy)
        _prewarm_numba_cache_body(include_feature_selection=False, include_heavy_libs=False)
        assert not [
            n for n in asked if n in {"lightning", "pytorch_lightning", "shap", "torch"}
        ], f"gated-off prewarm still probed the heavy stack: {sorted(set(asked))}"

    def test_the_gate_on_does_probe(self, monkeypatch):
        """Guards the test above: with the gate ON the lookups really do happen, so the assertion has meaning."""
        import importlib.util as _ilu

        asked = []
        _real = _ilu.find_spec

        def _spy(name, *a, **k):
            """Record every module the prewarm looks up."""
            asked.append(name)
            return _real(name, *a, **k)

        monkeypatch.setattr(_ilu, "find_spec", _spy)
        _prewarm_numba_cache_body(include_feature_selection=False, include_heavy_libs=True)
        assert {"lightning", "pytorch_lightning", "shap"} & set(asked)
