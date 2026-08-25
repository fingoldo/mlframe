"""Coverage for three previously-untested KTC-backed backend dispatchers that share the same
env-var-override -> KTC-measured-region -> caller-fallback contract:
``inference._ktc_dispatch.choose_logical_constraints_backend``,
``votenrank._confidence_gated_blend_ktc_dispatch.choose_confidence_blend_backend``, and
``calibration._ktc_dispatch.choose_odds_combine_backend`` (the first two route through the
shared ``_ktc_dispatch_shared`` helper; the third has its own local ``_get_cache``, same
contract). Their ``_make_tuner`` closures do real backend timing (njit/cupy) and are
exercised indirectly via the cache's ``get_or_tune``, not called directly here -- this test
targets the DISPATCH decision logic (env override / cache-absent / cache-exception
fallback), which is what every caller actually depends on."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlframe.calibration._ktc_dispatch import choose_odds_combine_backend
from mlframe.inference._ktc_dispatch import choose_logical_constraints_backend
from mlframe.votenrank._confidence_gated_blend_ktc_dispatch import choose_confidence_blend_backend

pytestmark = pytest.mark.fast


class TestChooseLogicalConstraintsBackend:
    """choose_logical_constraints_backend (inference._ktc_dispatch)."""

    def test_env_override_wins_over_everything(self, monkeypatch):
        """A valid env-var value short-circuits before ever touching the KTC cache."""
        monkeypatch.setenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", "cupy")
        out = choose_logical_constraints_backend(1000, 10, 3, fallback="njit_single")
        assert out == "cupy"

    def test_invalid_env_value_is_ignored(self, monkeypatch):
        """An env value outside the valid backend set is treated as unset, not forced verbatim."""
        monkeypatch.setenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", "not_a_real_backend")
        monkeypatch.setattr("mlframe.inference._ktc_dispatch.get_ktc_cache", lambda: None)
        out = choose_logical_constraints_backend(1000, 10, 3, fallback="njit_parallel")
        assert out == "njit_parallel"

    def test_cache_unavailable_returns_fallback(self, monkeypatch):
        """get_ktc_cache() returning None (pyutilz/FS unavailable) falls straight to the caller's fallback."""
        monkeypatch.delenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", raising=False)
        monkeypatch.setattr("mlframe.inference._ktc_dispatch.get_ktc_cache", lambda: None)
        out = choose_logical_constraints_backend(500, 5, 2, fallback="njit_single")
        assert out == "njit_single"

    def test_cache_returns_valid_backend_string(self, monkeypatch):
        """A cache hit returning a plain valid backend string (not a dict) is used as-is."""
        monkeypatch.delenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.return_value = "njit_parallel"
        monkeypatch.setattr("mlframe.inference._ktc_dispatch.get_ktc_cache", lambda: cache)
        out = choose_logical_constraints_backend(100_000, 10, 3, fallback="njit_single")
        assert out == "njit_parallel"

    def test_cache_returns_dict_with_backend_choice(self, monkeypatch):
        """A cache hit returning a {'backend_choice': ...} dict (the tuner's own region shape) is unpacked."""
        monkeypatch.delenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.return_value = {"backend_choice": "cupy", "wall_ms_cupy": 1.2}
        monkeypatch.setattr("mlframe.inference._ktc_dispatch.get_ktc_cache", lambda: cache)
        out = choose_logical_constraints_backend(1_000_000, 20, 5, fallback="njit_parallel")
        assert out == "cupy"

    def test_cache_returns_unrecognized_backend_falls_back(self, monkeypatch):
        """A cache result naming a backend outside the valid set is discarded, not trusted verbatim."""
        monkeypatch.delenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.return_value = {"backend_choice": "gibberish"}
        monkeypatch.setattr("mlframe.inference._ktc_dispatch.get_ktc_cache", lambda: cache)
        out = choose_logical_constraints_backend(1000, 10, 3, fallback="njit_single")
        assert out == "njit_single"

    def test_cache_lookup_exception_falls_back_silently(self, monkeypatch):
        """A cache.get_or_tune exception (any measurement/cache hiccup) never propagates -- falls back."""
        monkeypatch.delenv("MLFRAME_LOGICAL_CONSTRAINTS_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.side_effect = RuntimeError("boom")
        monkeypatch.setattr("mlframe.inference._ktc_dispatch.get_ktc_cache", lambda: cache)
        out = choose_logical_constraints_backend(1000, 10, 3, fallback="njit_single")
        assert out == "njit_single"


class TestChooseConfidenceBlendBackend:
    """choose_confidence_blend_backend (votenrank._confidence_gated_blend_ktc_dispatch)."""

    def test_env_override_wins(self, monkeypatch):
        """A valid env-var value short-circuits before ever touching the KTC cache."""
        monkeypatch.setenv("MLFRAME_CONFIDENCE_BLEND_BACKEND", "njit")
        out = choose_confidence_blend_backend(1000, fallback="numpy")
        assert out == "njit"

    def test_invalid_env_value_is_ignored(self, monkeypatch):
        """An env value outside the valid backend set is treated as unset."""
        monkeypatch.setenv("MLFRAME_CONFIDENCE_BLEND_BACKEND", "bogus")
        monkeypatch.setattr("mlframe.votenrank._confidence_gated_blend_ktc_dispatch.get_ktc_cache", lambda: None)
        out = choose_confidence_blend_backend(1000, fallback="njit_parallel")
        assert out == "njit_parallel"

    def test_cache_unavailable_returns_fallback(self, monkeypatch):
        """get_ktc_cache() returning None falls straight to the caller's fallback."""
        monkeypatch.delenv("MLFRAME_CONFIDENCE_BLEND_BACKEND", raising=False)
        monkeypatch.setattr("mlframe.votenrank._confidence_gated_blend_ktc_dispatch.get_ktc_cache", lambda: None)
        out = choose_confidence_blend_backend(500, fallback="numpy")
        assert out == "numpy"

    def test_cache_returns_dict_with_backend_choice(self, monkeypatch):
        """A cache hit returning a {'backend_choice': ...} dict is unpacked."""
        monkeypatch.delenv("MLFRAME_CONFIDENCE_BLEND_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.return_value = {"backend_choice": "cupy"}
        monkeypatch.setattr("mlframe.votenrank._confidence_gated_blend_ktc_dispatch.get_ktc_cache", lambda: cache)
        out = choose_confidence_blend_backend(1_000_000, fallback="numpy")
        assert out == "cupy"

    def test_cache_lookup_exception_falls_back_silently(self, monkeypatch):
        """A cache.get_or_tune exception never propagates -- falls back to the caller's default."""
        monkeypatch.delenv("MLFRAME_CONFIDENCE_BLEND_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.side_effect = RuntimeError("boom")
        monkeypatch.setattr("mlframe.votenrank._confidence_gated_blend_ktc_dispatch.get_ktc_cache", lambda: cache)
        out = choose_confidence_blend_backend(1000, fallback="numpy")
        assert out == "numpy"


class TestChooseOddsCombineBackend:
    """choose_odds_combine_backend (calibration._ktc_dispatch) -- has its own local _get_cache, same contract."""

    def test_env_override_wins(self, monkeypatch):
        """A valid env-var value short-circuits before ever touching the KTC cache."""
        monkeypatch.setenv("MLFRAME_ODDS_COMBINE_BACKEND", "njit_parallel")
        out = choose_odds_combine_backend(1000, 3, fallback="njit_single")
        assert out == "njit_parallel"

    def test_invalid_env_value_is_ignored(self, monkeypatch):
        """An env value outside the valid backend set is treated as unset."""
        monkeypatch.setenv("MLFRAME_ODDS_COMBINE_BACKEND", "bogus")
        monkeypatch.setattr("mlframe.calibration._ktc_dispatch._get_cache", lambda: None)
        out = choose_odds_combine_backend(1000, 3, fallback="njit_single")
        assert out == "njit_single"

    def test_cache_unavailable_returns_fallback(self, monkeypatch):
        """_get_cache() returning None falls straight to the caller's fallback."""
        monkeypatch.delenv("MLFRAME_ODDS_COMBINE_BACKEND", raising=False)
        monkeypatch.setattr("mlframe.calibration._ktc_dispatch._get_cache", lambda: None)
        out = choose_odds_combine_backend(500, 2, fallback="njit_parallel")
        assert out == "njit_parallel"

    def test_cache_returns_dict_with_backend_choice(self, monkeypatch):
        """A cache hit returning a {'backend_choice': ...} dict is unpacked."""
        monkeypatch.delenv("MLFRAME_ODDS_COMBINE_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.return_value = {"backend_choice": "cupy"}
        monkeypatch.setattr("mlframe.calibration._ktc_dispatch._get_cache", lambda: cache)
        out = choose_odds_combine_backend(1_000_000, 5, fallback="njit_single")
        assert out == "cupy"

    def test_cache_returns_unrecognized_backend_falls_back(self, monkeypatch):
        """A cache result naming a backend outside the valid set is discarded, not trusted verbatim."""
        monkeypatch.delenv("MLFRAME_ODDS_COMBINE_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.return_value = {"backend_choice": "gibberish"}
        monkeypatch.setattr("mlframe.calibration._ktc_dispatch._get_cache", lambda: cache)
        out = choose_odds_combine_backend(1000, 3, fallback="njit_single")
        assert out == "njit_single"

    def test_cache_lookup_exception_falls_back_silently(self, monkeypatch):
        """A cache.get_or_tune exception never propagates -- falls back to the caller's default."""
        monkeypatch.delenv("MLFRAME_ODDS_COMBINE_BACKEND", raising=False)
        cache = MagicMock()
        cache.get_or_tune.side_effect = RuntimeError("boom")
        monkeypatch.setattr("mlframe.calibration._ktc_dispatch._get_cache", lambda: cache)
        out = choose_odds_combine_backend(1000, 3, fallback="njit_single")
        assert out == "njit_single"
