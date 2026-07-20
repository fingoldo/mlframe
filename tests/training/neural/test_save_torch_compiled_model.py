"""Regression: save_mlframe_model used to fail with "cannot pickle
'ConfigModuleInstance' object" when the payload contained a torch.compile'd
nn.Module (PyTorch 2.x + Lightning leaves a non-picklable closure inside
the OptimizedModule wrapper).

Pre-fix repro from a real run (90-minute MLP fit lost on save):
    ERROR mlframe.training.io: Could not save model to file ...mlp__sch_*.dump:
    cannot pickle 'ConfigModuleInstance' object

Fix walks the payload before dill.dump, temp-swaps any ``_orig_mod``-bearing
attribute (the marker torch.compile leaves on its wrapper) with the underlying
un-compiled module, dumps, then restores so the caller keeps the optimized
graph after save returns.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


@pytest.mark.fast
def test_save_unwraps_torch_compile_wrapper_for_pickle(tmp_path, monkeypatch):
    """A SimpleNamespace whose .model.network is torch.compile'd must save
    successfully (was raising TypeError: cannot pickle ConfigModuleInstance)."""
    from mlframe.training.io import save_mlframe_model

    class _Net(nn.Module):
        """Groups tests covering net."""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 1)

        def forward(self, x):
            """Forward."""
            return self.fc(x)

    inner = _Net()

    class _FakeOptimizedModule:
        """Mimics torch.compile's OptimizedModule wrapper (the ``_orig_mod`` attribute is all
        save_mlframe_model's unwrap logic duck-types on -- see _io_save.py's ``getattr(_v, "_orig_mod", None)``)
        without paying torch.compile's real Dynamo/Inductor JIT-warmup cost (~100s+ on a cold cache),
        which this test has no need to trigger since it never calls the compiled module."""

        def __init__(self, orig_mod):
            self._orig_mod = orig_mod

    compiled = _FakeOptimizedModule(inner)

    # Sanity: the fake exposes _orig_mod the same way a real torch.compile wrapper would.
    assert hasattr(compiled, "_orig_mod"), "fake wrapper should expose _orig_mod"

    class _ModelHolder:
        """Groups tests covering model holder."""
        def __init__(self, net):
            self.network = net

    payload = SimpleNamespace(model=_ModelHolder(compiled), other_metadata={"foo": "bar"})

    out = tmp_path / "compiled_model.dump"
    ok = save_mlframe_model(payload, str(out), verbose=0)

    assert ok is True, "save_mlframe_model should succeed on a torch.compile'd payload"
    assert out.exists(), "save should produce a non-empty file"
    assert out.stat().st_size > 0

    # Original payload's compile wrapper restored after save (so subsequent
    # predict() keeps the optimized graph).
    assert payload.model.network is compiled, "After save, the caller's payload must keep the compiled wrapper (not be left swapped with _orig_mod)."


@pytest.mark.fast
def test_save_payload_without_torch_compile_unaffected(tmp_path):
    """Sanity: regular (non-compiled) payloads must still save identically."""
    from mlframe.training.io import save_mlframe_model

    payload = SimpleNamespace(plain={"a": 1, "b": [1, 2, 3]})
    out = tmp_path / "plain.dump"
    assert save_mlframe_model(payload, str(out), verbose=0) is True
    assert out.exists()
    assert out.stat().st_size > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
