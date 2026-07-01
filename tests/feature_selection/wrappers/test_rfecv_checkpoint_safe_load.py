"""SEC4 regression: RFECV resume-checkpoint load must go through the sidecar-gated safe loader.

Previously _load_checkpoint did a bare pickle.load(fh) on the caller-supplied checkpoint path -- arbitrary pickle from a tampered file.
The fix saves a .sha256 sidecar in _save_checkpoint and loads via safe_pickle.safe_load, so a tampered checkpoint (sidecar digest
mismatch) is refused (start fresh) while a legit one round-trips.

We invoke the two methods as unbound functions against a minimal stub carrying only the attributes they read (checkpoint_path,
_CHECKPOINT_VERSION), avoiding the heavy RFECV constructor while exercising the exact save/load code path.
"""
from mlframe.feature_selection.wrappers.rfecv import RFECV


class _Stub:
    _CHECKPOINT_VERSION = RFECV._CHECKPOINT_VERSION

    def __init__(self, path):
        self.checkpoint_path = path


def _legit_state():
    return {"version": RFECV._CHECKPOINT_VERSION, "payload": [1, 2, 3]}


def test_legit_checkpoint_round_trips(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    ckpt = str(tmp_path / "rfecv.ckpt")
    stub = _Stub(ckpt)

    RFECV._save_checkpoint(stub, _legit_state())
    loaded = RFECV._load_checkpoint(stub)

    assert loaded is not None
    assert loaded["payload"] == [1, 2, 3]


def test_tampered_checkpoint_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    ckpt = str(tmp_path / "rfecv.ckpt")
    stub = _Stub(ckpt)
    RFECV._save_checkpoint(stub, _legit_state())

    # Tamper with the payload bytes WITHOUT updating the sidecar -> digest mismatch.
    with open(ckpt, "ab") as fh:
        fh.write(b"\x00malicious")

    # Refused -> start fresh, no unpickling of the divergent bytes.
    assert RFECV._load_checkpoint(stub) is None


def test_missing_sidecar_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    import pickle

    ckpt = str(tmp_path / "rfecv.ckpt")
    with open(ckpt, "wb") as fh:
        pickle.dump(_legit_state(), fh)  # planted payload, no .sha256 sidecar
    stub = _Stub(ckpt)

    assert RFECV._load_checkpoint(stub) is None
