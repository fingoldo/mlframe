"""Security-focused tests for RCE-prone code paths in mlframe."""

import io
import os
import pickle
import tempfile

import pytest


# ---------------------------------------------------------------------------
# 1) torch.load weights_only=True wrapper test
# ---------------------------------------------------------------------------
def test_torch_load_rejects_malicious_pickle(tmp_path):
    torch = pytest.importorskip("torch")

    class _Exploit:
        def __reduce__(self):
            # os.system is a classic RCE marker — weights_only should refuse it.
            return (os.system, ("echo pwned",))

    bad_path = tmp_path / "bad.pt"
    with open(bad_path, "wb") as f:
        pickle.dump({"state_dict": _Exploit()}, f)

    with pytest.raises(Exception):
        torch.load(str(bad_path), map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------
# 2) _SafeUnpickler allowlist enforcement
# ---------------------------------------------------------------------------
def test_safe_unpickler_rejects_os_system_accepts_numpy_ndarray():
    pytest.importorskip("dill")
    np = pytest.importorskip("numpy")
    from mlframe.training.io import _SafeUnpickler

    # numpy.ndarray class should be resolvable.
    class _AllowedRef:
        def __reduce__(self):
            # Just reference numpy.core.multiarray.array constructor-like callable.
            # We use np.array via __reduce__ returning (np.asarray, ([1,2,3],))
            import numpy as _np
            return (_np.asarray, ([1, 2, 3],))

    buf = io.BytesIO()
    import dill
    dill.dump(_AllowedRef(), buf)
    buf.seek(0)
    arr = _SafeUnpickler(buf).load()
    assert list(arr) == [1, 2, 3]

    # os.system should be blocked.
    class _Bad:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    buf2 = io.BytesIO()
    dill.dump(_Bad(), buf2)
    buf2.seek(0)
    with pytest.raises(Exception):
        _SafeUnpickler(buf2).load()


# ---------------------------------------------------------------------------
# 3) inference.read_trained_models rejects paths outside trusted_root
# ---------------------------------------------------------------------------
def test_read_trained_models_rejects_untrusted_path(tmp_path):
    pytest.importorskip("joblib")
    pd = pytest.importorskip("pandas")
    from mlframe import inference

    trusted = tmp_path / "trusted"
    trusted.mkdir()
    # inference_folder resolves outside the trusted root.
    outside = tmp_path / "outside"
    outside.mkdir()

    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        inference.read_trained_models(
            featureset="fs",
            X=X,
            inference_folder=str(outside),
            trusted_root=str(trusted),
        )


# ---------------------------------------------------------------------------
# 4) experiments field whitelist
# ---------------------------------------------------------------------------
def test_experiments_field_whitelist():
    pytest.importorskip("pyutilz")
    from mlframe.experiments import _validate_and_join_fields, _ALLOWED_EXPERIMENT_FIELDS

    # SQL injection attempt must be rejected.
    with pytest.raises(ValueError):
        _validate_and_join_fields("x; DROP TABLE y", _ALLOWED_EXPERIMENT_FIELDS)

    # Whitelisted fields pass through.
    out = _validate_and_join_fields("id,name", _ALLOWED_EXPERIMENT_FIELDS)
    assert out == "id,name"

    # Sequence input also works.
    out2 = _validate_and_join_fields(["id", "started_at"], _ALLOWED_EXPERIMENT_FIELDS)
    assert out2 == "id,started_at"

    # Unknown field rejected.
    with pytest.raises(ValueError):
        _validate_and_join_fields(["id", "evil_col"], _ALLOWED_EXPERIMENT_FIELDS)
