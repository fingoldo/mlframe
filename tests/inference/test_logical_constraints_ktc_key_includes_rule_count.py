"""The kernel-tuning lookup for the logical-constraint backend is keyed on the rule count too, not only on rows and labels."""

from __future__ import annotations

from mlframe.inference import _ktc_dispatch as mod


class _SpyCache:
    """Records the dims / axes of the lookup and answers with a fixed backend."""

    def __init__(self):
        self.dims = None
        self.axes = None

    def get_or_tune(self, kernel, *, dims, tuner, axes, fallback, async_sweep):
        """Capture the call and return a valid backend."""
        self.dims, self.axes = dims, axes
        return "njit_single"


def test_rule_count_is_part_of_the_tuning_key(monkeypatch):
    """A region measured at 1 rule must not be reused for 50 rules at the same (n, n_labels)."""
    spy = _SpyCache()
    monkeypatch.setattr(mod, "get_ktc_cache", lambda: spy)
    monkeypatch.setattr(mod, "forced_backend", lambda *a, **k: None)
    mod.choose_logical_constraints_backend(1000, 8, 50, fallback="njit_single")
    assert spy.dims["n_rules"] == 50 and "n_rules" in spy.axes
