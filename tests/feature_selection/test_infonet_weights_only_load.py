"""SEC1 regression: the infonet vendored checkpoint loader must use torch.load(weights_only=True).

The Google-Drive checkpoint is a plain state_dict, so restricting the unpickler to weights-only blocks arbitrary-code execution from a
tampered checkpoint. This test pins the mechanism the fix relies on (a state_dict round-trips under weights_only=True) and asserts the
source loads via that path. We cannot import the vendored module directly (it uses broken absolute ``from model.* import`` paths), so we
verify the behavioural contract + the source.
"""

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_INFER = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters" / "_vendored" / "infonet" / "infer.py"


def test_state_dict_round_trips_under_weights_only_true(tmp_path):
    model = torch.nn.Linear(4, 2)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ckpt)

    loaded = torch.load(str(ckpt), map_location="cpu", weights_only=True)

    target = torch.nn.Linear(4, 2)
    target.load_state_dict(loaded)
    for k, v in model.state_dict().items():
        assert torch.equal(v, target.state_dict()[k])


def test_infer_source_calls_torch_load_with_weights_only():
    src = _INFER.read_text(encoding="utf-8")
    assert re.search(r"torch\.load\([^)]*weights_only\s*=\s*True", src), "infer.py must call torch.load(..., weights_only=True)"
