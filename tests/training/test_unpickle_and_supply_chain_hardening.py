"""Regression tests for the restricted unpickler policy, metadata loading, decompression caps, HF revision pins and dispatch import guard."""

from __future__ import annotations

import io
import os
import pickle  # nosec B403 - test-only round-trips of locally built payloads
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import dill  # nosec B403 - test-only round-trips of locally built payloads
import joblib
import numpy as np
import pytest
import zstandard as zstd

from mlframe.training._bounded_zstd import BoundedReader, DecompressedSizeError, decompress_bounded, max_decompressed_bytes
from mlframe.training.io import _SafeUnpickler, load_mlframe_model, safe_joblib_load, save_mlframe_model


def _marker_exploit_by_value(marker: Path):
    """Build an object whose unpickling would run a by-value function (code object) that writes ``marker``."""

    def _write_marker() -> int:
        """Write the marker file (executed only if the unpickler runs attacker code)."""
        with open(str(marker), "w") as fh:
            fh.write("pwned")
        return 1

    class _Exploit:
        """Pickles to a call of the by-value function."""

        def __reduce__(self):
            """Reduce to ``_write_marker()`` so loading calls it."""
            return (_write_marker, ())

    return _Exploit()


def _global_ref_pickle(module: str, name: str) -> bytes:
    """Protocol-4 pickle that merely resolves the global ``module.name`` (STACK_GLOBAL), no call."""
    mod = module.encode()
    nm = name.encode()
    return b"\x80\x04" + b"\x8c" + bytes([len(mod)]) + mod + b"\x8c" + bytes([len(nm)]) + nm + b"\x93."


_FORBIDDEN_REFS = [
    ("types", "CodeType"),
    ("types", "FunctionType"),
    ("types", "ModuleType"),
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "type"),
    ("builtins", "memoryview"),
    ("functools", "reduce"),
    ("operator", "methodcaller"),
    ("dill._dill", "_create_function"),
    ("dill._dill", "_create_code"),
    ("dill", "loads"),
    ("os", "system"),
    ("numpy", "load"),
    ("numpy.testing._private.utils", "runstring"),
    ("pandas", "read_pickle"),
    ("pandas", "eval"),
    ("torch", "load"),
    ("torch.hub", "load"),
    ("torch.serialization", "load"),
    ("mlframe.training.io", "load_mlframe_model"),
    ("mlframe.training.io", "save_mlframe_model"),
    ("joblib", "load"),
]


@pytest.mark.parametrize("module,name", _FORBIDDEN_REFS)
def test_safe_unpickler_refuses_gadget_references(module, name):
    """Every code-execution / IO / nested-loader gadget reference is refused by the allowlist unpickler, not only top-level eval."""
    with pytest.raises(dill.UnpicklingError):
        _SafeUnpickler(io.BytesIO(_global_ref_pickle(module, name))).load()


def test_safe_unpickler_blocks_by_value_function_payload(tmp_path):
    """A dill by-value function (code object + FunctionType) in a bundle must not execute under the safe default; the marker stays absent."""
    marker = tmp_path / "marker.txt"
    payload = dill.dumps(_marker_exploit_by_value(marker), byref=False, recurse=True)
    with pytest.raises(dill.UnpicklingError):
        _SafeUnpickler(io.BytesIO(payload)).load()
    assert not marker.exists()


def test_load_mlframe_model_safe_default_rejects_function_payload(tmp_path):
    """The public loader returns None for a tampered bundle carrying a by-value function and never runs it."""
    marker = tmp_path / "marker.txt"
    bundle = tmp_path / "evil.bundle"
    raw = dill.dumps(_marker_exploit_by_value(marker), byref=False, recurse=True)
    bundle.write_bytes(zstd.ZstdCompressor().compress(raw))
    assert load_mlframe_model(str(bundle)) is None
    assert not marker.exists()


def test_safe_unpickler_still_loads_legitimate_bundle(tmp_path):
    """A realistic bundle (sklearn estimator, numpy array, functools.partial of a builtin, torch module and tensor) round-trips through the safe loader."""
    torch = pytest.importorskip("torch")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    lin = torch.nn.Linear(3, 2)
    bundle = SimpleNamespace(
        model=LogisticRegression().fit(X, y),
        arr=np.arange(6, dtype=np.float32).reshape(2, 3),
        part=__import__("functools").partial(int, base=2),
        torch_module=lin,
        tensor=torch.arange(4, dtype=torch.float32),
        nested={"a": [1, 2, (3, 4)], "s": {1, 2}, "o": None},
    )
    path = tmp_path / "ok.bundle"
    assert save_mlframe_model(bundle, str(path), verbose=0)
    loaded = load_mlframe_model(str(path))
    assert loaded is not None
    np.testing.assert_allclose(loaded.model.coef_, bundle.model.coef_)
    np.testing.assert_array_equal(loaded.arr, bundle.arr)
    assert loaded.part("101") == 5
    assert torch.equal(loaded.tensor, bundle.tensor)
    assert torch.equal(loaded.torch_module.weight, lin.weight)
    assert loaded.nested == bundle.nested


def test_safe_joblib_load_blocks_types_function_constructor(tmp_path):
    """The joblib denylist loader refuses types.FunctionType/CodeType, io.open, operator.methodcaller and numpy.load references."""
    for module, name in [("types", "FunctionType"), ("types", "CodeType"), ("io", "open"), ("operator", "methodcaller"), ("numpy", "load"), ("builtins", "type")]:
        path = tmp_path / f"{module}_{name}.pkl"
        path.write_bytes(_global_ref_pickle(module, name))
        with pytest.raises(dill.UnpicklingError):
            safe_joblib_load(str(path))


def test_safe_joblib_load_still_loads_custom_estimator(tmp_path):
    """The denylist loader keeps accepting ordinary estimators and arrays (arbitrary custom classes stay supported)."""
    from sklearn.linear_model import Ridge

    model = Ridge().fit(np.eye(4), np.arange(4.0))
    path = tmp_path / "ridge.pkl"
    joblib.dump({"m": model, "ns": SimpleNamespace(a=1)}, str(path))
    loaded = safe_joblib_load(str(path))
    np.testing.assert_allclose(loaded["m"].coef_, model.coef_)
    assert loaded["ns"].a == 1


# metadata loading


def _write_meta(path: Path, obj: object, sidecar: bool) -> None:
    """Write ``obj`` as a zstd pickle at ``path``, optionally with its sha256 sidecar."""
    path.write_bytes(zstd.ZstdCompressor().compress(pickle.dumps(obj, protocol=5)))
    if sidecar:
        from mlframe.utils.safe_pickle import write_sidecar

        write_sidecar(str(path))


def test_metadata_without_sidecar_is_refused_by_default(tmp_path, monkeypatch):
    """A metadata.pkl.zst with no sha256 sidecar is refused unless the explicit env opt-in is set."""
    from mlframe.training.core._metadata_loader import load_metadata_file

    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    meta = tmp_path / "metadata.pkl.zst"
    _write_meta(meta, {"schema_version": 1}, sidecar=False)
    with pytest.raises(RuntimeError, match="sidecar"):
        load_metadata_file(str(meta), "pkl.zst", "test")
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    assert load_metadata_file(str(meta), "pkl.zst", "test") == {"schema_version": 1}


def test_metadata_with_valid_sidecar_loads_and_tampered_digest_is_refused(tmp_path, monkeypatch):
    """A matching sidecar loads the metadata; a corrupted payload (digest mismatch) is refused."""
    from mlframe.training.core._metadata_loader import load_metadata_file

    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    meta = tmp_path / "metadata.pkl.zst"
    _write_meta(meta, {"a": np.arange(3), "b": ["x"]}, sidecar=True)
    loaded = load_metadata_file(str(meta), "pkl.zst", "test")
    np.testing.assert_array_equal(loaded["a"], np.arange(3))
    meta.write_bytes(meta.read_bytes() + b"\x00")
    with pytest.raises(RuntimeError, match="sidecar"):
        load_metadata_file(str(meta), "pkl.zst", "test")


def test_metadata_with_attacker_resealed_sidecar_still_cannot_run_code(tmp_path, monkeypatch):
    """Even when the attacker recomputes the sidecar, a metadata payload carrying an eval gadget is blocked by the restricted unpickler."""
    from mlframe.training.core._metadata_loader import load_metadata_file

    class _Evil:
        """Reduces to builtins.eval."""

        def __reduce__(self):
            """Reduce to an eval call."""
            return (eval, ("1+1",))

    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    meta = tmp_path / "metadata.pkl.zst"
    _write_meta(meta, _Evil(), sidecar=True)
    with pytest.raises(dill.UnpicklingError):
        load_metadata_file(str(meta), "pkl.zst", "test")


def test_legacy_joblib_metadata_goes_through_restricted_loader(tmp_path):
    """Legacy metadata.joblib with an eval gadget is blocked, a benign dict still loads."""
    from mlframe.training.core._metadata_loader import load_metadata_file

    class _Evil:
        """Reduces to builtins.eval."""

        def __reduce__(self):
            """Reduce to an eval call."""
            return (eval, ("1+1",))

    bad = tmp_path / "bad.joblib"
    joblib.dump(_Evil(), str(bad))
    with pytest.raises(dill.UnpicklingError):
        load_metadata_file(str(bad), "joblib", "test")
    good = tmp_path / "metadata.joblib"
    joblib.dump({"k": 1}, str(good))
    assert load_metadata_file(str(good), "joblib", "test") == {"k": 1}


# decompression caps


def test_decompress_bounded_rejects_bomb(monkeypatch):
    """A tiny zstd frame that expands past the configured ceiling is refused instead of allocated."""
    monkeypatch.setenv("MLFRAME_MAX_DECOMPRESSED_BYTES", "100000")
    bomb = zstd.ZstdCompressor().compress(b"\x00" * 5_000_000)
    assert len(bomb) < 1000
    with pytest.raises(DecompressedSizeError):
        decompress_bounded(bomb)
    ok = zstd.ZstdCompressor().compress(b"a" * 1000)
    assert decompress_bounded(ok) == b"a" * 1000


def test_bounded_reader_stops_streaming_bomb(monkeypatch):
    """The streaming reader used by load_mlframe_model raises once the produced bytes pass the ceiling."""
    bomb = zstd.ZstdCompressor().compress(b"\x00" * 5_000_000)
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(bomb)) as raw:
        reader = BoundedReader(raw, 100_000)
        with pytest.raises(DecompressedSizeError):
            reader.read()


def test_load_mlframe_model_returns_none_on_oversized_stream(tmp_path, monkeypatch):
    """An over-ceiling bundle makes load_mlframe_model return None rather than expand it."""
    monkeypatch.setenv("MLFRAME_MAX_DECOMPRESSED_BYTES", "10000")
    path = tmp_path / "big.bundle"
    path.write_bytes(zstd.ZstdCompressor().compress(pickle.dumps(np.zeros(100_000))))
    assert load_mlframe_model(str(path)) is None


def test_max_decompressed_bytes_env_parsing(monkeypatch):
    """Positive values are honoured, 0 disables, garbage falls back to the default ceiling."""
    monkeypatch.setenv("MLFRAME_MAX_DECOMPRESSED_BYTES", "123")
    assert max_decompressed_bytes() == 123
    monkeypatch.setenv("MLFRAME_MAX_DECOMPRESSED_BYTES", "0")
    assert max_decompressed_bytes() == 0
    monkeypatch.setenv("MLFRAME_MAX_DECOMPRESSED_BYTES", "junk")
    assert max_decompressed_bytes() == 64 * 1024**3


# Hugging Face revision pins


def test_resolve_revision_pin_flag_and_explicit_override():
    """pinned_revision=True resolves the shipped SHA for a default model, an explicit revision wins, and False leaves hub main."""
    from mlframe.training.feature_handling.hf_provider import PINNED_REVISIONS, resolve_revision

    model = "intfloat/multilingual-e5-small"
    sha = PINNED_REVISIONS[model]
    assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")
    assert resolve_revision(model, None, True) == sha
    assert resolve_revision(model, "deadbeef", True) == "deadbeef"
    assert resolve_revision(model, None, False) is None


def test_resolve_revision_warns_for_unpinned_model():
    """An unlisted model without an explicit revision still loads (None) but warns that it is unpinned."""
    from mlframe.training.feature_handling.hf_provider import resolve_revision

    with pytest.warns(UserWarning, match="no pinned revision"):
        assert resolve_revision("some-org/unlisted-model", None, True) is None


def _install_fake_transformers(monkeypatch, calls: list):
    """Install a stub ``transformers`` module whose from_pretrained calls are recorded in ``calls``."""
    torch = pytest.importorskip("torch")

    class _Cfg:
        """Model config stub."""

        hidden_size = 4

    class _Model:
        """Model stub supporting the .to().eval() chain."""

        config = _Cfg()

        def to(self, device):
            """Return self."""
            return self

        def eval(self):
            """Return self."""
            return self

    class _Tok:
        """Tokenizer stub."""

        pad_token = "x"
        eos_token = "y"

    class _AutoModel:
        """AutoModel stub."""

        @staticmethod
        def from_pretrained(name, **kw):
            """Record the call."""
            calls.append(("model", name, kw))
            return _Model()

    class _AutoTokenizer:
        """AutoTokenizer stub."""

        @staticmethod
        def from_pretrained(name, **kw):
            """Record the call."""
            calls.append(("tok", name, kw))
            return _Tok()

    mod = types.ModuleType("transformers")
    mod.AutoModel = _AutoModel
    mod.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return torch


def test_hf_provider_acquire_passes_pinned_revision_for_default_model(monkeypatch, tmp_path):
    """The default provider config loads tokenizer and model at the pinned commit; pinned_revision=False restores hub main."""
    from mlframe.training.feature_handling.hf_provider import PINNED_REVISIONS, HuggingFaceProvider
    from mlframe.training.feature_handling.providers import EmbeddingProvider

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    calls: list = []
    _install_fake_transformers(monkeypatch, calls)
    model = "intfloat/multilingual-e5-small"
    cfg = EmbeddingProvider(kind="huggingface", model=model, params={"device": "cpu", "dtype": "fp32"})
    HuggingFaceProvider(cfg).acquire()
    assert [c[2]["revision"] for c in calls] == [PINNED_REVISIONS[model]] * 2
    calls.clear()
    HuggingFaceProvider(cfg, pinned_revision=False).acquire()
    assert [c[2]["revision"] for c in calls] == [None, None]


def test_mist_model_loaded_at_pinned_revision(monkeypatch):
    """The MIST checkpoints are fetched at a pinned commit SHA, per loss variant."""
    from mlframe.feature_selection.filters import _neural_mi as nm

    seen: list = []

    class _FakeModel:
        """MISTForHF stub."""

        def eval(self):
            """Return self."""
            return self

        def to(self, dev):
            """Return self."""
            return self

    class _FakeMIST:
        """Stub exposing from_pretrained."""

        @staticmethod
        def from_pretrained(repo, **kw):
            """Record repo and revision."""
            seen.append((repo, kw.get("revision")))
            return _FakeModel()

    fake = types.ModuleType("mist_statinf")
    fake.MISTForHF = _FakeMIST
    monkeypatch.setitem(sys.modules, "mist_statinf", fake)
    monkeypatch.setattr(nm, "_MIST_MODEL_CACHE", {})
    monkeypatch.setattr(nm, "_resolve_device", lambda device: "cpu")
    nm._get_mist_hf_model("mse", "cpu")
    nm._get_mist_hf_model("qr", "cpu")
    assert seen == [("grgera/MIST", nm.MIST_PINNED_REVISIONS["grgera/MIST"]), ("grgera/MIST-QR", nm.MIST_PINNED_REVISIONS["grgera/MIST-QR"])]
    assert all(rev and len(rev) == 40 for _, rev in seen)


# dispatch import guard


def test_instantiate_recommended_estimator_refuses_non_mlframe_module():
    """A recommendation naming a module outside mlframe is refused before anything is imported; an mlframe one still instantiates."""
    from mlframe.training.composite._estimator_dispatch import instantiate_recommended_estimator

    with pytest.raises(ValueError, match="mlframe module"):
        instantiate_recommended_estimator({"module": "os", "estimator": "getcwd"})
    with pytest.raises(ValueError, match="mlframe module"):
        instantiate_recommended_estimator({"module": "mlframeevil.pkg", "estimator": "X"})
    obj = instantiate_recommended_estimator({"module": "mlframe.training.io", "estimator": "SimpleNamespace"}, a=1)
    assert obj.a == 1
    assert instantiate_recommended_estimator(None) is None


# dependency floor


def test_pyproject_lightning_floor_excludes_vulnerable_release():
    """The lightning requirement for Python >= 3.10 excludes 2.6.5 (PYSEC-2026-3624) while the 3.9 requirement stays satisfiable."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    from packaging.requirements import Requirement
    from packaging.version import Version

    root = Path(__file__).resolve().parents[2]
    deps = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    all_deps = list(deps.get("dependencies", []))
    for extra in deps.get("optional-dependencies", {}).values():
        all_deps.extend(extra)
    lightning = [Requirement(d) for d in all_deps if Requirement(d).name == "lightning"]
    modern = [r for r in lightning if r.marker is None or r.marker.evaluate({"python_version": "3.12"})]
    legacy = [r for r in lightning if r.marker is not None and r.marker.evaluate({"python_version": "3.9"})]
    assert modern and all(Version("2.6.5") not in r.specifier and Version("2.6.6") in r.specifier for r in modern)
    assert legacy and all(Version("2.4.0") in r.specifier for r in legacy)
    assert os.path.exists(root / "uv.lock")
    lock_text = (root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "lightning"\nversion = "2.6.6"' in lock_text
