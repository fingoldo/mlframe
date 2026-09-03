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
    for k, v in model.state_dict().items():
        assert torch.equal(v, target.state_dict()[k])


def test_infer_never_loads_a_checkpoint_without_weights_only():
    """EVERY `torch.load` in infer.py must pass `weights_only=True`, checked on the AST rather than by regex.

    The regex this replaces failed on a multi-line call and on one passing the flag through a kwargs dict --
    both correct -- and passed whenever any such call existed anywhere in the file, including in a dead branch,
    while saying nothing about the other call sites. The round-trip test above already proves the loader honours
    the flag; what is worth pinning here is that no call site omits it.
    """
    import ast

    tree = ast.parse(_INFER.read_text(encoding="utf-8"))
    loads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "load" and getattr(n.func.value, "id", "") == "torch"
    ]
    assert loads, "no torch.load call found in infer.py; this test needs updating"

    lines = _INFER.read_text(encoding="utf-8").splitlines()
    unguarded = []
    for call in loads:
        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
        if any(k.arg is None for k in call.keywords):
            continue  # flag may arrive through a kwargs splat; the round-trip test covers the behaviour
        if "weights_only" in kw:
            assert getattr(kw["weights_only"], "value", None) is True, f"torch.load at line {call.lineno} passes weights_only but not True"
            continue
        # ONE documented exception is expected and is genuinely unavoidable: torch < 1.13 has no such kwarg, and
        # those versions predate the unpickler restriction entirely, so there is nothing safer to call. It lives
        # in the `except TypeError` arm of the guarded call and says so on its own line. Anything else is a real
        # gap -- and is exactly what the regex this replaced could not see, since it only needed ONE matching
        # call anywhere in the file to pass.
        if "torch<1.13 fallback" in lines[call.lineno - 1]:
            continue
        unguarded.append(call.lineno)
    assert not unguarded, f"torch.load without weights_only=True and without the documented torch<1.13 note at line(s) {unguarded}"

    legacy = [c.lineno for c in loads if "torch<1.13 fallback" in lines[c.lineno - 1]]
    assert len(legacy) == 1, f"expected exactly one documented legacy fallback, found {legacy}"
