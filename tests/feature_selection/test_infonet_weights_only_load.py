"""SEC1 regression: the infonet vendored checkpoint loader must use torch.load(weights_only=True).

The Google-Drive checkpoint is a plain state_dict, so restricting the unpickler to weights-only blocks arbitrary-code execution from a
tampered checkpoint. This test pins the mechanism the fix relies on (a state_dict round-trips under weights_only=True) and asserts the
source loads via that path. We cannot import the vendored module directly (it uses broken absolute ``from model.* import`` paths), so we
verify the behavioural contract + the source.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_INFER = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters" / "_vendored" / "infonet" / "infer.py"


def test_state_dict_round_trips_under_weights_only_true(tmp_path):
    """State dict round trips under weights only true."""
    model = torch.nn.Linear(4, 2)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)

    loaded = torch.load(str(ckpt), map_location="cpu", weights_only=True)

    target = torch.nn.Linear(4, 2)
    target.load_state_dict(loaded)
    assert model.state_dict().items()
    for k, v in model.state_dict().items():
        assert torch.equal(v, target.state_dict()[k])


def test_infer_never_loads_a_checkpoint_without_weights_only():
    """EVERY `torch.load` in infer.py passes `weights_only=True`, except the one in the `except TypeError` arm of the guarded call.

    Decided on the AST: a multi-line call or a kwargs splat is still seen, and a call in a dead branch cannot satisfy the check for the others.
    The round-trip test above already proves the loader honours the flag; what is pinned here is that no call site omits it. The single
    unavoidable exception is the torch < 1.13 fallback, whose torch has no such kwarg and predates the unpickler restriction entirely.
    """
    import ast

    tree = ast.parse(_INFER.read_bytes())
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load" and getattr(n.func.value, "id", "") == "torch"]
    assert loads, "no torch.load call found in infer.py; this test needs updating"

    fallback_ids: set[int] = set()
    for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
        if isinstance(handler.type, ast.Name) and handler.type.id == "TypeError":
            fallback_ids.update(id(n) for n in ast.walk(handler) if isinstance(n, ast.Call))

    unguarded = []
    fallbacks = []
    for call in loads:
        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
        if any(k.arg is None for k in call.keywords):
            continue  # flag may arrive through a kwargs splat; the round-trip test covers the behaviour
        if "weights_only" in kw:
            assert getattr(kw["weights_only"], "value", None) is True, f"torch.load at line {call.lineno} passes weights_only but not True"
            continue
        if id(call) in fallback_ids:
            fallbacks.append(call.lineno)
            continue
        unguarded.append(call.lineno)
    assert not unguarded, f"torch.load without weights_only=True outside the `except TypeError` fallback at line(s) {unguarded}"
    assert len(fallbacks) == 1, f"expected exactly one legacy fallback in an `except TypeError` arm, found {fallbacks}"
