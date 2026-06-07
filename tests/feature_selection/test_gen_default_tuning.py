"""Tests for the anonymized default kernel-tuning generator + its runtime
registration.

All tests are HERMETIC + DISK-LIGHT: they use a tmp ``PYUTILZ_KERNEL_CACHE_DIR``
with a PRE-SEEDED ``.hw_fingerprint.json`` (so ``KernelTuningCache()`` never
imports cupy to probe the GPU -- that import stalls for minutes when the box's
disk is saturated), inject fake ``discover_fn`` / ``tune_fn`` into
``generate_defaults`` so NO real discovery walk or sweep (and no heavy mlframe
fit) ever runs, and write only under ``tmp_path``. They cover:

* the generator emits valid, SORTED JSON for the current kernels + code_versions;
* ``register_default_cache`` loads that JSON;
* a LOCAL-miss ``get_or_tune`` returns the DEFAULT region (not the hand fallback);
* ``skip_existing`` skips kernels already present at the live code_version;
* ``--force`` re-sweeps even a present kernel;
* the anonymization maps backend_choice -> abstract cpu/gpu and drops wall_ms.
"""
from __future__ import annotations

import os

import orjson
import pytest

from pyutilz.performance.kernel_tuning.cache import (
    KernelTuningCache,
    register_default_cache,
)
from pyutilz.performance.kernel_tuning.registry import TunerSpec

from mlframe.feature_selection._benchmarks import gen_default_tuning as gdt

_SEED_FP = "cpu_test_no-gpu"


def _assert_canonical_on_disk(raw: str, doc: dict) -> None:
    """``write_defaults`` writes the document sorted (keys) + 2-space indented +
    trailing newline. Verify the on-disk text round-trips to ``doc``, ends in a
    newline, and has every (nested) object's keys in sorted order."""
    assert orjson.loads(raw) == doc
    assert raw.endswith("\n")

    def _keys_sorted(obj) -> bool:
        if isinstance(obj, dict):
            keys = list(obj.keys())
            return keys == sorted(keys) and all(_keys_sorted(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_keys_sorted(v) for v in obj)
        return True

    assert _keys_sorted(orjson.loads(raw))


def _seed_hw_fingerprint(cache_dir: str) -> None:
    """Write a fresh ``.hw_fingerprint.json`` into ``cache_dir`` so
    ``hw_fingerprint()`` resolves from disk and never imports cupy (which stalls
    under disk saturation)."""
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, ".hw_fingerprint.json"), "wb") as f:
        f.write(orjson.dumps({"schema_version": 1, "fingerprint": _SEED_FP, "ts_utc": "2026-06-07T00:00:00"}))


# --------------------------------------------------------------------------- #
# Fixtures: a fake registry + a tmp per-host cache so no real sweep runs.
# --------------------------------------------------------------------------- #

def _fake_spec(kernel_name="fake_kernel", *, gpu=True, salt=3):
    """A minimal real TunerSpec with a deterministic code_version. ``variant_fns``
    is a tiny local function so ``code_version()`` is stable + non-None."""
    def _ref(x):  # body hashed into code_version
        return x + 1

    return TunerSpec(
        kernel_name=kernel_name,
        variant_fns=(_ref,),
        tuner=lambda: [],  # never called in these tests (we seed the cache directly)
        axes={"n_samples": [100, 1000], "n_pairs": [10, 100]},
        fallback={"backend_choice": "njit_serial"},
        gpu_capable=gpu,
        salt=salt,
    )


def _seed_regions():
    """Two bands + a catch-all, in the shape ``sweep_backend_grid`` emits
    (with a ``wall_ms`` the anonymizer must drop)."""
    return [
        {"n_samples_max": 100, "n_pairs_max": 100, "backend_choice": "njit_serial", "wall_ms": 0.5, "max_abs_diff": 0.0},
        {"n_samples_max": 1000, "n_pairs_max": 100, "backend_choice": "cupy", "wall_ms": 0.2, "max_abs_diff": 1e-7},
        {"n_samples_max": None, "n_pairs_max": None, "backend_choice": "cupy", "wall_ms": 0.1},
    ]


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point the v3 per-host cache at a tmp dir (with a seeded hw fingerprint) so
    seeding + reads are isolated and cupy is never probed."""
    d = str(tmp_path / "kcache")
    _seed_hw_fingerprint(d)
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", d)
    return tmp_path


def _use_empty_host(tmp_path, monkeypatch, name):
    """Switch the per-host cache to a fresh EMPTY dir (with seeded hw fingerprint)
    so the next lookup is a genuine LOCAL miss without a cupy probe."""
    d = str(tmp_path / name)
    _seed_hw_fingerprint(d)
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", d)
    return d


class _Reg:
    """A fake registry + sweep counter for the ``generate_defaults`` injection
    seams. ``kwargs`` is spread into ``generate_defaults(...)`` so the generator
    sees exactly ``spec`` and a no-op (counted) ``tune_fn`` -- NO real discovery
    walk, NO sweep. ``cache_cls`` is left as the real ``KernelTuningCache`` so the
    seeded per-host regions are read back."""

    def __init__(self, spec):
        self.spec = spec
        self.tune_calls = 0

        def _disc(package="mlframe"):
            return {spec.kernel_name: spec}

        def _tune(s, force=False, skip_existing=True):
            self.tune_calls += 1
            return 0

        self.kwargs = {"discover_fn": _disc, "tune_fn": _tune}


@pytest.fixture
def patched_registry():
    """One fake spec wired through the ``generate_defaults`` injection seams."""
    return _Reg(_fake_spec())


# --------------------------------------------------------------------------- #
# classify_device / anonymize_regions  (pure -- no cache, no pyutilz heavy path)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("token,expected", [
    ("cuda", "gpu"), ("cupy", "gpu"), ("gpu", "gpu"),
    ("njit_serial", "cpu"), ("njit_parallel", "cpu"), ("numpy", "cpu"),
    ("serial", "cpu"), ("parallel", "cpu"), ("sklearn", "cpu"), ("hnsw", "cpu"),
    ("", "cpu"), (None, "cpu"), ("unknown_backend", "cpu"),
])
def test_classify_device(token, expected):
    assert gdt.classify_device(token) == expected


def test_anonymize_drops_wall_ms_adds_device_keeps_caps():
    anon = gdt.anonymize_regions(_seed_regions())
    assert len(anon) == 3
    for r in anon:
        assert "wall_ms" not in r, "per-host latency must be stripped"
        assert "device" in r, "abstract device profile must be added"
    gpu_band = anon[1]
    assert gpu_band["backend_choice"] == "cupy"
    assert gpu_band["device"] == "gpu"
    assert gpu_band["n_samples_max"] == 1000
    assert "max_abs_diff" in gpu_band  # correctness property -> KEPT
    assert anon[0]["device"] == "cpu"


def test_anonymize_does_not_mutate_input():
    src = _seed_regions()
    before = orjson.dumps(src, option=orjson.OPT_SORT_KEYS)
    gdt.anonymize_regions(src)
    assert orjson.dumps(src, option=orjson.OPT_SORT_KEYS) == before


# --------------------------------------------------------------------------- #
# generate_defaults: valid sorted JSON for current kernels + code_versions
# --------------------------------------------------------------------------- #

def test_generate_emits_valid_sorted_json(tmp_cache, patched_registry):
    reg = patched_registry
    spec = reg.spec
    out = str(tmp_cache / "default_kernel_tuning.json")

    # Seed the per-host cache with measured regions at the live code_version so
    # get_regions returns them (tune_fn is a no-op).
    KernelTuningCache().update(spec.kernel_name, axes=list(spec.axes.keys()), regions=_seed_regions(),
                               code_version=spec.code_version(), salt=spec.salt)

    doc = gdt.generate_defaults(output_path=out, skip_existing=True, **reg.kwargs)

    assert doc["schema_version"] == gdt.DEFAULTS_SCHEMA_VERSION
    assert "hw_fingerprint" not in doc, "the defaults file must be hw-agnostic"
    # The kernel is NOT yet in the committed defaults file, so the generator calls
    # the ensure-tune seam once (real tune_spec is a no-op when already cached).
    assert reg.tune_calls == 1, "a kernel absent from the committed file must be ensure-tuned"
    entry = doc["kernels"][spec.kernel_name]
    assert entry["code_version"] == spec.code_version()
    assert entry["axes"] == list(spec.axes.keys())
    assert entry["salt"] == spec.salt
    assert len(entry["regions"]) == 3

    gdt.write_defaults(doc, out)
    with open(out, "r", encoding="utf-8") as f:
        raw = f.read()
    reparsed = orjson.loads(raw)
    assert reparsed == doc
    _assert_canonical_on_disk(raw, doc)


def test_generate_kernels_sorted_by_name(tmp_cache):
    a = _fake_spec("zzz_kernel")
    b = _fake_spec("aaa_kernel")
    for s in (a, b):
        KernelTuningCache().update(s.kernel_name, axes=list(s.axes.keys()), regions=_seed_regions(),
                                   code_version=s.code_version(), salt=s.salt)
    doc = gdt.generate_defaults(
        output_path=str(tmp_cache / "d.json"),
        discover_fn=lambda package="mlframe": {a.kernel_name: a, b.kernel_name: b},
        tune_fn=lambda s, force=False, skip_existing=True: 0,
    )
    assert list(doc["kernels"].keys()) == ["aaa_kernel", "zzz_kernel"]


# --------------------------------------------------------------------------- #
# register_default_cache loads it; a LOCAL miss returns the DEFAULT region.
# --------------------------------------------------------------------------- #

def test_register_default_cache_loads_and_local_miss_returns_default(tmp_path, tmp_cache, patched_registry, monkeypatch):
    reg = patched_registry
    spec = reg.spec
    out = str(tmp_cache / "default_kernel_tuning.json")

    KernelTuningCache().update(spec.kernel_name, axes=list(spec.axes.keys()), regions=_seed_regions(),
                               code_version=spec.code_version(), salt=spec.salt)
    doc = gdt.generate_defaults(output_path=out, **reg.kwargs)
    gdt.write_defaults(doc, out)

    # Fresh EMPTY per-host cache dir => a genuine LOCAL miss.
    _use_empty_host(tmp_path, monkeypatch, "empty_host")
    assert register_default_cache(out) is True

    local = KernelTuningCache()
    assert not local.has(spec.kernel_name), "the fresh host must be a local miss"

    monkeypatch.setenv("PYUTILZ_KERNEL_DISABLE_SWEEP", "1")
    result = local.get_or_tune(
        spec.kernel_name, dims={"n_samples": 500, "n_pairs": 50},
        tuner=spec.tuner, axes=list(spec.axes.keys()), fallback=spec.fallback,
        code_version=spec.code_version(), once_per_process=False,
    )
    bc = result if isinstance(result, str) else result.get("backend_choice")
    assert bc == "cupy", f"local miss should serve the DEFAULT (cupy), not the hand fallback; got {result!r}"


def test_default_ignored_when_code_version_stale(tmp_path, tmp_cache, patched_registry, monkeypatch):
    """If the live code_version differs from the default file's, the default is
    treated as stale and the hand fallback is used instead."""
    reg = patched_registry
    spec = reg.spec
    out = str(tmp_cache / "default_kernel_tuning.json")
    KernelTuningCache().update(spec.kernel_name, axes=list(spec.axes.keys()), regions=_seed_regions(),
                               code_version=spec.code_version(), salt=spec.salt)
    gdt.write_defaults(gdt.generate_defaults(output_path=out, **reg.kwargs), out)

    _use_empty_host(tmp_path, monkeypatch, "empty_host2")
    register_default_cache(out)
    local = KernelTuningCache()
    monkeypatch.setenv("PYUTILZ_KERNEL_DISABLE_SWEEP", "1")
    result = local.get_or_tune(
        spec.kernel_name, dims={"n_samples": 500, "n_pairs": 50},
        tuner=spec.tuner, axes=list(spec.axes.keys()), fallback=spec.fallback,
        code_version="a-different-code-version", once_per_process=False,
    )
    bc = result if isinstance(result, str) else result.get("backend_choice")
    assert bc == "njit_serial", f"stale default must fall through to hand fallback; got {result!r}"


# --------------------------------------------------------------------------- #
# skip_existing skips already-present kernels (no sweep, entry preserved).
# --------------------------------------------------------------------------- #

def test_skip_existing_skips_present_kernel(tmp_cache, patched_registry):
    reg = patched_registry
    spec = reg.spec
    out = str(tmp_cache / "default_kernel_tuning.json")

    # Pre-write a defaults file that ALREADY has the kernel at the live cv with a
    # sentinel region we can detect was carried over (not re-derived).
    sentinel = {"schema_version": gdt.DEFAULTS_SCHEMA_VERSION, "kernels": {
        spec.kernel_name: {
            "axes": list(spec.axes.keys()),
            "code_version": spec.code_version(),
            "regions": [{"n_samples_max": None, "n_pairs_max": None,
                         "backend_choice": "SENTINEL", "device": "cpu"}],
            "salt": spec.salt,
        }
    }}
    gdt.write_defaults(sentinel, out)

    doc = gdt.generate_defaults(output_path=out, skip_existing=True, **reg.kwargs)
    assert reg.tune_calls == 0, "skip_existing must not sweep an already-current kernel"
    assert doc["kernels"][spec.kernel_name]["regions"][0]["backend_choice"] == "SENTINEL", \
        "the existing entry must be carried over verbatim"


def test_force_resweeps_even_if_present(tmp_cache, patched_registry):
    reg = patched_registry
    spec = reg.spec
    out = str(tmp_cache / "default_kernel_tuning.json")
    sentinel = {"schema_version": gdt.DEFAULTS_SCHEMA_VERSION, "kernels": {
        spec.kernel_name: {"axes": list(spec.axes.keys()), "code_version": spec.code_version(),
                           "regions": [{"backend_choice": "SENTINEL"}], "salt": spec.salt}
    }}
    gdt.write_defaults(sentinel, out)

    # Seed the per-host cache so the re-derived entry has real regions.
    KernelTuningCache().update(spec.kernel_name, axes=list(spec.axes.keys()), regions=_seed_regions(),
                               code_version=spec.code_version(), salt=spec.salt)

    doc = gdt.generate_defaults(output_path=out, force=True, **reg.kwargs)
    assert reg.tune_calls == 1, "force must re-sweep even a present kernel"
    assert doc["kernels"][spec.kernel_name]["regions"][0]["backend_choice"] != "SENTINEL"


# --------------------------------------------------------------------------- #
# --check drift detection (ignores generated_utc).
# --------------------------------------------------------------------------- #

def test_check_ignores_timestamp(tmp_cache, patched_registry):
    reg = patched_registry
    spec = reg.spec
    out = str(tmp_cache / "default_kernel_tuning.json")
    KernelTuningCache().update(spec.kernel_name, axes=list(spec.axes.keys()), regions=_seed_regions(),
                               code_version=spec.code_version(), salt=spec.salt)
    doc = gdt.generate_defaults(output_path=out, **reg.kwargs)
    gdt.write_defaults(doc, out)

    # A freshly-regenerated document differs only in generated_utc; --check must
    # consider it in-sync.
    doc2 = gdt.generate_defaults(output_path=out, **reg.kwargs)
    assert gdt._documents_equivalent(doc, doc2)


def test_committed_defaults_file_is_valid():
    """The repo-committed defaults file must be valid, sorted JSON with the
    expected schema (an empty-kernels seed is valid)."""
    path = gdt.DEFAULT_OUTPUT_PATH
    if not os.path.isfile(path):
        pytest.skip("default_kernel_tuning.json not present in this checkout")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    doc = orjson.loads(raw)
    assert doc["schema_version"] == gdt.DEFAULTS_SCHEMA_VERSION
    assert isinstance(doc["kernels"], dict)
    assert "hw_fingerprint" not in doc
    _assert_canonical_on_disk(raw, doc)
    for name, entry in doc["kernels"].items():
        assert "axes" in entry and "regions" in entry and "code_version" in entry, name
        for r in entry["regions"]:
            assert "device" in r, f"{name} region missing abstract device profile: {r}"
