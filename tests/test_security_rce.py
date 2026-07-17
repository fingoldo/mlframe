"""Security-focused tests for RCE-prone code paths in mlframe."""

import io
import os
import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import pytest


# ---------------------------------------------------------------------------
# 1) torch.load weights_only=True wrapper test
# ---------------------------------------------------------------------------
def test_torch_load_rejects_malicious_pickle(tmp_path):
    """Torch load rejects malicious pickle."""
    torch = pytest.importorskip("torch")

    class _Exploit:
        """Groups tests covering Exploit."""
        def __reduce__(self):
            # os.system is a classic RCE marker — weights_only should refuse it.
            return (os.system, ("echo pwned",))

    bad_path = tmp_path / "bad.pt"
    with open(bad_path, "wb") as f:
        pickle.dump({"state_dict": _Exploit()}, f)

    # weights_only=True refuses any non-tensor global; torch raises pickle.UnpicklingError
    # (subclass) whose message names the blocked global / the weights_only restriction.
    with pytest.raises(pickle.UnpicklingError) as exc:
        torch.load(str(bad_path), map_location="cpu", weights_only=True)
    msg = str(exc.value).lower()
    assert "weights_only" in msg or "system" in msg or "global" in msg


# ---------------------------------------------------------------------------
# 2) _SafeUnpickler allowlist enforcement
# ---------------------------------------------------------------------------
def test_safe_unpickler_rejects_os_system_accepts_numpy_ndarray():
    """Safe unpickler rejects os system accepts numpy ndarray."""
    pytest.importorskip("dill")
    pytest.importorskip("numpy")
    from mlframe.training.io import _SafeUnpickler

    # numpy.ndarray class should be resolvable.
    class _AllowedRef:
        """Groups tests covering AllowedRef."""
        def __reduce__(self):
            # Just reference numpy.core.multiarray.array constructor-like callable.
            # We use np.array via __reduce__ returning (np.asarray, ([1,2,3],))
            import numpy as _np

            return (_np.asarray, ([1, 2, 3],))

    buf = io.BytesIO()
    import dill  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

    dill.dump(_AllowedRef(), buf)
    buf.seek(0)
    arr = _SafeUnpickler(buf).load()
    assert list(arr) == [1, 2, 3]

    # os.system should be blocked.
    class _Bad:
        """Groups tests covering Bad."""
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    buf2 = io.BytesIO()
    dill.dump(_Bad(), buf2)
    buf2.seek(0)
    # The allowlist refuses os.system via the module-prefix gate -> dill.UnpicklingError
    # whose message names the blocked global.
    with pytest.raises(dill.UnpicklingError) as exc:
        _SafeUnpickler(buf2).load()
    msg = str(exc.value)
    assert "blocked by _SafeUnpickler" in msg and "system" in msg


# ---------------------------------------------------------------------------
# 3) inference.read_trained_models rejects paths outside trusted_root
# ---------------------------------------------------------------------------
def test_read_trained_models_rejects_untrusted_path(tmp_path):
    """Read trained models rejects untrusted path."""
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
    """Experiments field whitelist."""
    pytest.importorskip("pyutilz")
    from mlframe.utils.experiments import _validate_and_join_fields, _ALLOWED_EXPERIMENT_FIELDS

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
