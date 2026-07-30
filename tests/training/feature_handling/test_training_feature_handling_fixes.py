"""Regression tests for the training_feature_handling.md audit fix wave (2026-07-21).

One test per finding (F1-F17, including F6/F8/F9 which were fixed in an earlier pass but not yet pinned
here) plus the proposals implemented alongside the findings (PR1 already covered by
``tests/training/test_biz_val_leakage_safe_encoder_woe_smoothing.py``, PR2 the LOC carve, PR3 Generator
reproducibility, PR4 the WoE imbalanced-fold regression, PR6 the presets resolution test). See
``audits/full_audit_2026-07-21/training_feature_handling.md``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.feature_handling import (
    CacheConfig,
    CatHandlerSpec,
    ContentFingerprint,
    DiskKey,
    FeatureCache,
    FeatureHandlingConfig,
    InMemoryKey,
    TargetEncodeParams,
    TextHandlerSpec,
    cb_native_only,
    fingerprint_df,
)
from mlframe.training.feature_handling.handlers import NoParams
from mlframe.training.feature_handling.ordered_target_encoder import (
    ordered_target_encode_batch,
)
from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder

# =====================================================================
# F1 / PR4 -- WoE k-fold unseen-category clip on an imbalanced fold
# =====================================================================


def test_f1_woe_kfold_finite_on_highly_imbalanced_fold():
    """A tiny, highly-imbalanced target (cv=5 on 20 rows, all-but-the-last positive) can leave a
    fold's training rows with zero negatives -- pre-fix, ``log(0)`` poisoned the OOF column with
    -inf/nan for that fold's unseen categories. Post-fix, ``_kfold_encode``'s woe branch clips the
    same way ``_encode_per_row`` already did."""
    cats = np.array([f"c{i % 8}" for i in range(20)])
    y = np.zeros(20)
    y[-1] = 1.0  # only the last row is positive
    enc = LeakageSafeEncoder(method="woe", cv=5, random_state=0)
    out = enc.fit_transform(cats, y)
    assert np.isfinite(out).all(), f"non-finite WoE OOF values: {out}"


def test_f1_woe_kfold_finite_all_negative_target():
    """Degenerate all-negative target (every fold's prior_t == 0.0) must still yield finite WoE."""
    cats = np.array([f"c{i % 5}" for i in range(30)])
    y = np.zeros(30)
    enc = LeakageSafeEncoder(method="woe", cv=5, random_state=0)
    out = enc.fit_transform(cats, y)
    assert np.isfinite(out).all()


# =====================================================================
# F2 / PR3 -- ordered_target_encode_batch honors a Generator random_state
# =====================================================================


def test_f2_ordered_target_encode_batch_generator_random_state_reproducible():
    """Passing an ``np.random.Generator`` (not a bare int) for ``random_state`` must still be
    reproducible: two independently-constructed Generators seeded identically must produce
    identical noisy output. Pre-fix, a Generator instance was discarded (``base_seed = None``)
    and ``SeedSequence(None)`` drew OS entropy on every call -- no two runs matched."""
    rng_state = np.random.default_rng(123)
    n = 200
    y = rng_state.integers(0, 2, size=n).astype(np.float64)
    order = np.arange(n)
    columns = {
        "a": rng_state.integers(0, 5, size=n).astype(str),
        "b": rng_state.integers(0, 5, size=n).astype(str),
    }

    batch_a = ordered_target_encode_batch(
        columns, y, order=order, smoothing=1.0, noise_std=0.4,
        random_state=np.random.default_rng(7),
    )
    batch_b = ordered_target_encode_batch(
        columns, y, order=order, smoothing=1.0, noise_std=0.4,
        random_state=np.random.default_rng(7),
    )
    for name in columns:
        np.testing.assert_allclose(batch_a[name], batch_b[name])


def test_f2_ordered_target_encode_batch_generator_differs_from_default_entropy():
    """Sanity check the test above isn't vacuously true because noise_std==0: two DIFFERENT seeded
    Generators must diverge, proving the seed is actually driving the draw."""
    n = 200
    y = np.random.default_rng(1).integers(0, 2, size=n).astype(np.float64)
    order = np.arange(n)
    columns = {"a": np.random.default_rng(2).integers(0, 5, size=n).astype(str)}

    batch_a = ordered_target_encode_batch(
        columns, y, order=order, smoothing=1.0, noise_std=0.4,
        random_state=np.random.default_rng(7),
    )
    batch_b = ordered_target_encode_batch(
        columns, y, order=order, smoothing=1.0, noise_std=0.4,
        random_state=np.random.default_rng(999),
    )
    assert not np.allclose(batch_a["a"], batch_b["a"])


# =====================================================================
# F3 -- disk-tier LRU-by-mtime eviction
# =====================================================================


def _disk_key(fp: ContentFingerprint, column: str) -> DiskKey:
    """Disk key."""
    return DiskKey(content=fp, column=column, params_canonical_hash="0" * 32, provider_signature="x")


def test_f3_disk_tier_evicts_when_free_space_below_threshold(tmp_path, monkeypatch):
    """Pre-fix, ``FeatureCache``'s disk tier never referenced ``disk_evict_when_free_below_gb`` /
    ``disk_min_free_gb`` at all -- the directory grew unbounded. Force the "free space below
    threshold" branch via a monkeypatched ``shutil.disk_usage`` and confirm the oldest entries get
    removed."""
    import mlframe.training.feature_handling.cache as cache_mod

    cache_dir = tmp_path / "cache"
    cfg = CacheConfig(
        persistence="auto", dir=str(cache_dir),
        disk_evict_when_free_below_gb=100.0,
        disk_min_free_gb=100.0,  # unreachable target -> evicts everything it can
        # allow_pickle=True: pytest's tmp_path under Windows long-path (``\\?\``) prefixes
        # occasionally trips np.load's npz reader; the pickle fallback recovers transparently
        # (same convention test_feature_cache.py's TestDiskPersistence already documents).
        allow_pickle=True,
    )
    cache = FeatureCache(cfg)

    df = pd.DataFrame({"txt": [f"row{i}" for i in range(20)]})
    fp = fingerprint_df(df)
    for i in range(5):
        in_mem_key = InMemoryKey(
            session_id="s", df_token=id(df), train_idx_token=0,
            column=f"col{i}", params_canonical_hash="0" * 32, provider_signature="x",
        )
        cache.get_or_compute(
            in_mem_key, lambda i=i: np.full(4, float(i)),
            disk_key=_disk_key(fp, f"col{i}"),
        )

    files_before = list(cache_dir.glob("*.bin"))
    assert len(files_before) == 5, "sanity: all 5 entries should have been written to disk first"

    class _FakeUsage:
        """Fake Usage."""
        free = 1.0 * 1e9  # 1 GB free, below the 100 GB evict_below_gb threshold

    monkeypatch.setattr(cache_mod.shutil, "disk_usage", lambda _d: _FakeUsage())

    # A fresh write triggers _maybe_evict_disk again; disk_min_free_gb=100GB is unreachable from a
    # fake 1GB-free state, so it evicts every removable .bin entry (oldest mtime first).
    in_mem_key_new = InMemoryKey(
        session_id="s", df_token=id(df), train_idx_token=0,
        column="trigger", params_canonical_hash="0" * 32, provider_signature="x",
    )
    cache.get_or_compute(in_mem_key_new, lambda: np.zeros(4), disk_key=_disk_key(fp, "trigger"))

    files_after = list(cache_dir.glob("*.bin"))
    assert len(files_after) < len(files_before) + 1, "disk eviction should have removed at least one stale entry"
    assert cache.stats()["evictions"] > 0


def test_f3_disk_tier_no_eviction_when_threshold_unset():
    """Default ``CacheConfig`` (no explicit thresholds) must not evict -- eviction is opt-in."""
    from mlframe.training.feature_handling.cache import FeatureCache as _FC

    cfg = CacheConfig(persistence="off")
    cache = _FC(cfg)
    cache._maybe_evict_disk()  # no dir configured; must be a no-op, not raise
    assert cache.stats()["evictions"] == 0


# =====================================================================
# F4 -- group_columns ABC symmetry + no-op warning
# =====================================================================


def test_f4_text_handler_spec_declares_group_columns():
    """TextHandlerSpec must declare ``group_columns`` for genuine ABC symmetry with CatHandlerSpec
    (axis.py's HandlerSpec contract explicitly claims this symmetry)."""
    spec = TextHandlerSpec(method="drop", params=NoParams(kind="drop"), group_columns=["region"])
    assert spec.group_columns == ["region"]


def test_f4_cat_group_columns_warns_when_set(caplog):
    """A user setting ``CatHandlerSpec.group_columns`` (group-aware encoding, not yet implemented)
    must see a WARNING rather than silently getting ordinary global-fold encoding."""
    from mlframe.training.feature_handling.apply import feature_handling_apply

    train_df = pd.DataFrame({"cat": [f"c{i % 4}" for i in range(60)], "region": [f"r{i % 3}" for i in range(60)]})
    target = np.random.RandomState(0).randint(0, 2, size=60).astype(np.int32)
    fhc = FeatureHandlingConfig(
        default_cat=[
            CatHandlerSpec(
                method="target_mean",
                params=TargetEncodeParams(kind="target_mean", smoothing=10.0, cv=3),
                group_columns=["region"],
            )
        ],
        default_text=[],
    )
    caplog.set_level(logging.WARNING)
    feature_handling_apply(
        train_df=train_df, fhc=fhc, model_kind="xgb",
        train_target=target, candidate_cat_columns=["cat"],
    )
    assert any("group_columns" in rec.message and "not" in rec.message for rec in caplog.records)


# =====================================================================
# F5 -- target-encoder cache key disambiguated by column content
# =====================================================================


def test_f5_target_encoder_cache_key_differs_by_column_content_not_just_target():
    """Pre-fix, ``_apply_target_encoder``'s InMemoryKey hashed only ``train_target`` content -- two
    calls sharing an identical target but a DIFFERENT categorical column would collide and replay
    each other's cached (encoder, oof_train). Folding in a column-content token (mirroring the
    text-encoder cache key's existing fix) must make the combined tokens differ."""
    from mlframe.training.feature_handling.apply import (
        _combine_content_tokens,
        _target_content_token,
        _text_column_content_token,
    )

    df1 = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "a"] * 10})
    df2 = pd.DataFrame({"cat": ["x", "y", "z", "w", "x", "y"] * 10})
    target = pd.Series([1, 0, 1, 0, 1, 0] * 10, dtype=float)

    tok1 = _combine_content_tokens(_target_content_token(target), _text_column_content_token(df1, "cat"))
    tok2 = _combine_content_tokens(_target_content_token(target), _text_column_content_token(df2, "cat"))
    assert tok1 != tok2, "cache key must distinguish different columns even under an identical target"


def test_f5_combine_content_tokens_not_xor_cancelling():
    """``_combine_content_tokens`` must not reduce to plain XOR, which silently cancels to 0 (a
    degenerate, collision-prone key) whenever two input tokens happen to be equal."""
    from mlframe.training.feature_handling.apply import _combine_content_tokens

    t = 123456789
    combined_equal_inputs = _combine_content_tokens(t, t)
    assert combined_equal_inputs != 0
    assert combined_equal_inputs != (t ^ t)


# =====================================================================
# F7 / PR6 -- cb_native_only gives CatBoost genuine native cat handling
# =====================================================================


def test_f7_cb_native_only_resolves_native_cat_for_cb():
    """``cb_native_only()``'s entire stated purpose is native handling for CatBoost -- pre-fix,
    every model (including cb) got ``method='ordinal'`` for cat columns."""
    cfg = cb_native_only()
    cb_cat_specs = cfg._effective_cat_specs("cb")
    assert len(cb_cat_specs) == 1
    assert cb_cat_specs[0].method == "native"


def test_f7_cb_native_only_other_models_still_ordinal():
    """Non-cb models keep the ordinal cat default (no native-cat override for them)."""
    cfg = cb_native_only()
    xgb_cat_specs = cfg._effective_cat_specs("xgb")
    assert len(xgb_cat_specs) == 1
    assert xgb_cat_specs[0].method == "ordinal"


def test_f7_cb_native_only_text_also_native_for_cb():
    """The pre-existing text override must still hold (this was never broken by F7)."""
    cfg = cb_native_only()
    cb_text_specs = cfg._effective_text_specs("cb")
    assert len(cb_text_specs) == 1
    assert cb_text_specs[0].method == "native"


# =====================================================================
# F6 -- polars_capability.py's docstring now honestly documents the hasattr-only probe
# =====================================================================


def test_f6_polars_capability_docstring_documents_hasattr_not_try_invoke():
    """F6: the module must explicitly document that capability detection is a ``hasattr`` presence
    check, not a try-invoke probe on a synthetic frame -- pre-fix the docstring implied the stronger
    guarantee. Callers must still handle a runtime failure even when ``has(op)`` is True."""
    import inspect

    import mlframe.training.feature_handling.polars_capability as mod

    src = inspect.getsource(mod)
    assert "hasattr" in src
    assert "try-invoke" in src


def test_f6_detect_polars_ds_capabilities_still_returns_frozenset():
    """The detection function's actual behaviour is unaffected by the F6 docstring fix."""
    from mlframe.training.feature_handling.polars_capability import detect_polars_ds_capabilities

    caps = detect_polars_ds_capabilities()
    assert isinstance(caps, frozenset)


# =====================================================================
# F8 -- bench_woe_laplace_alpha.py now sweeps the parameter WoE actually reads
# =====================================================================


def test_f8_woe_bench_sweeps_woe_smoothing_not_dead_smoothing_kwarg():
    """F8: pre-fix the bench swept ``smoothing=alpha`` on ``LeakageSafeEncoder(method="woe", ...)``, a
    kwarg the WoE code path never reads (only ``woe_smoothing`` controls WoE's Laplace cushion) -- every
    swept alpha produced bit-identical output and the bench measured no real signal."""
    import inspect

    import mlframe.training.feature_handling._benchmarks.bench_woe_laplace_alpha as mod

    src = inspect.getsource(mod)
    assert "woe_smoothing=alpha" in src
    assert 'method="woe", smoothing=alpha' not in src


def test_f8_woe_smoothing_kwarg_actually_changes_transform_output():
    """Sweeping the real ``woe_smoothing`` kwarg (unlike the dead ``smoothing`` kwarg it replaced) must
    produce different transform output across alpha values -- confirms the bench's fixed parameter now
    has a real, measurable effect."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"cat": rng.choice(["a", "b", "c", "d"], size=n)})
    y = (rng.random(n) < 0.15).astype(int)  # imbalanced, so the Laplace cushion actually matters

    outputs = []
    for alpha in (0.5, 20.0):
        enc = LeakageSafeEncoder(method="woe", woe_smoothing=alpha, random_state=0)
        outputs.append(enc.fit(df["cat"], y).transform(df["cat"]))
    assert not np.allclose(outputs[0], outputs[1]), "woe_smoothing=0.5 vs 20.0 produced identical output"


# =====================================================================
# F9 -- LocalDiskBackend/CacheBackend documented as an intentionally-unwired future seam
# =====================================================================


def test_f9_feature_cache_does_not_import_cache_backend():
    """F9: FeatureCache's disk tier is still the hand-rolled implementation, not routed through
    CacheBackend/LocalDiskBackend -- confirmed not silently half-wired since the earlier audit pass."""
    import inspect

    import mlframe.training.feature_handling.cache as cache_mod

    src = inspect.getsource(cache_mod)
    assert "CacheBackend" not in src
    assert "LocalDiskBackend" not in src


def test_f9_cache_backend_module_documents_why_it_is_not_wired_in():
    """F9: the non-use must be an explicit, reasoned disposition (bytes-oriented Protocol vs.
    FeatureCache's mmap-based large-array reads), not a silent unexplained dead seam."""
    import mlframe.training.feature_handling.cache_backend as mod

    assert mod.__doc__ is not None
    assert "mmap_mode" in mod.__doc__
    assert "Not yet wired into" in mod.__doc__


def test_f9_local_disk_backend_still_functions_standalone(tmp_path):
    """The LocalDiskBackend implementation itself (atomic writes, LRU eviction) stays correct and
    tested even while unused by FeatureCache -- it's a ready seam, not bit-rotted dead code."""
    from mlframe.training.feature_handling.cache_backend import LocalDiskBackend

    backend = LocalDiskBackend(str(tmp_path / "cache_root"), max_entries=2)
    backend.write("k1", b"v1")
    backend.write("k2", b"v2")
    assert backend.read("k1") == b"v1"
    assert backend.exists("k2")
    backend.write("k3", b"v3")  # exceeds max_entries=2, should evict the LRU entry
    remaining = set(backend.list_keys())
    assert len(remaining) == 2
    assert "k3" in remaining


# =====================================================================
# F10 -- PolynomialFeatureExpander no longer builds a dead dispatcher
# =====================================================================


def test_f10_polynomial_expander_has_no_dispatcher_attribute():
    """Pre-fix, the constructor built ``self._dispatcher = PolarsNativeDispatcher(...)`` but never
    referenced it again -- a real but entirely dead capability probe. ``prefer_polarsds`` is now
    stored directly, honestly documented as a currently-inert no-op."""
    from mlframe.training.feature_handling.polynomial import PolynomialFeatureExpander

    expander = PolynomialFeatureExpander(degree=2, prefer_polarsds=True)
    assert not hasattr(expander, "_dispatcher")
    assert expander.prefer_polarsds is True


def test_f10_polynomial_expander_fit_transform_unaffected_by_prefer_polarsds():
    """``prefer_polarsds`` must remain a genuine no-op: True vs False produce identical output
    (sklearn PolynomialFeatures is always used)."""
    from mlframe.training.feature_handling.polynomial import PolynomialFeatureExpander

    X = np.random.RandomState(0).randn(30, 3).astype(np.float32)
    e_true = PolynomialFeatureExpander(degree=2, prefer_polarsds=True)
    e_false = PolynomialFeatureExpander(degree=2, prefer_polarsds=False)
    out_true = e_true.fit_transform(X, feature_names=["a", "b", "c"])
    out_false = e_false.fit_transform(X, feature_names=["a", "b", "c"])
    np.testing.assert_array_equal(out_true, out_false)


# =====================================================================
# F11 -- fingerprint_df memo keyed on n_sample too
# =====================================================================


def test_f11_fingerprint_cache_key_includes_n_sample():
    """Pre-fix, ``_fp_cache_key`` ignored ``n_sample`` -- two calls on the identical df object with
    different ``n_sample`` would silently reuse the first call's memoised fingerprint."""
    from mlframe.training.feature_handling.fingerprint import _fp_cache_key

    df = pd.DataFrame({"a": range(50)})
    key_small = _fp_cache_key(df, 8)
    key_large = _fp_cache_key(df, 4096)
    assert key_small != key_large
    assert key_small[-1] == 8
    assert key_large[-1] == 4096


def test_f11_fingerprint_df_distinguishes_n_sample_on_repeat_call():
    """End-to-end: two ``fingerprint_df`` calls on the SAME df object with different ``n_sample``
    must not spuriously hit each other's memo (content differs because sampled rows differ)."""
    n = 20000
    df = pd.DataFrame({"a": np.arange(n, dtype=np.float64)})
    fp_small = fingerprint_df(df, n_sample=8)
    fp_large = fingerprint_df(df, n_sample=4096)
    assert fp_small.sampled_rows_hash != fp_large.sampled_rows_hash


# =====================================================================
# F12 -- CUDA-context-lost CPU fallback uses a real torch.dtype
# =====================================================================


def test_f12_cuda_context_lost_cpu_fallback_uses_real_torch_dtype(monkeypatch):
    """The CUDA-context-lost recovery path must pass ``torch.float32`` (a real dtype object,
    matching the primary load path), not the bare string ``"float32"`` whose acceptance by
    ``from_pretrained`` is transformers-version-dependent."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from mlframe.training.feature_handling.hf_provider import HuggingFaceProvider
    from mlframe.training.feature_handling.providers import EmbeddingProvider

    cfg = EmbeddingProvider(kind="huggingface", model="stub-model", params={})
    provider = HuggingFaceProvider(cfg)
    provider._device = "cuda"
    provider._load_model_name = "stub-model"
    provider._load_revision = None
    provider._load_trust_remote = False
    provider._embedding_dim = 4
    provider._is_loaded = True

    class _FailingModel:
        """Failing Model."""
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA error: unknown error")

    class _StubOutput:
        """Stub Output."""
        last_hidden_state = torch.zeros(1, 3, 4)

    class _CpuModel:
        """Stub model whose .to()/.eval() calls are recorded instead of moving a real device."""
        def __call__(self, **kwargs):
            return _StubOutput()

        def to(self, device):
            """No-op stub matching torch.nn.Module.to()'s signature."""
            return self

        def eval(self):
            """No-op stub matching torch.nn.Module.eval()'s signature."""
            return self

    def _fake_tokenizer(batch, **kwargs):
        """Fake tokenizer."""
        n = len(batch)
        return {
            "input_ids": torch.ones(n, 3, dtype=torch.long),
            "attention_mask": torch.ones(n, 3, dtype=torch.long),
        }

    provider._tokenizer = _fake_tokenizer
    provider._model = _FailingModel()

    captured = {}

    def _fake_from_pretrained(model_name, revision=None, torch_dtype=None, trust_remote_code=False):
        """Fake from pretrained."""
        captured["torch_dtype"] = torch_dtype
        return _CpuModel()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(_fake_from_pretrained))

    out = provider._batched_inference(["hello"], batch_size=1, max_length=8, pool="mean")

    assert isinstance(captured["torch_dtype"], torch.dtype)
    assert not isinstance(captured["torch_dtype"], str)
    assert captured["torch_dtype"] is torch.float32
    assert out.shape == (1, 4)


# =====================================================================
# F13 -- dead _smoothed_mean removed; inline formula still correct
# =====================================================================


def test_f13_smoothed_mean_helper_no_longer_exists():
    """``_smoothed_mean`` was dead code (defined, never called); its 3 duplicated inline call sites
    remain, so removing it must not change behaviour."""
    import mlframe.training.feature_handling.target_encoders as te_mod

    assert not hasattr(te_mod, "_smoothed_mean")


def test_f13_target_mean_smoothed_formula_still_correct():
    """Regression sensor for the inline duplicates the dead helper used to (partially) shadow:
    smoothed encoding must match ``(n*mean + smoothing*prior) / (n+smoothing)`` by construction."""
    cats = np.array(["a"] * 8 + ["b"] * 2)
    y = np.array([1.0] * 8 + [0.0] * 2)
    enc = LeakageSafeEncoder(method="target_mean", smoothing=2.0, cv=2, random_state=0)
    enc.fit(cats, y)
    transformed = enc.transform(np.array(["a", "b"]))
    expected_a = (8 * 1.0 + 2.0 * enc._global_prior) / (8 + 2.0)
    expected_b = (2 * 0.0 + 2.0 * enc._global_prior) / (2 + 2.0)
    np.testing.assert_allclose(transformed, [expected_a, expected_b])


# =====================================================================
# F14 -- assembler routing no longer has a redundant tuple check
# =====================================================================


def test_f14_sparse_routing_unaffected_for_cb_xgb_lgb():
    """Simplifying ``if model_kind in ("cb","xgb","lgb") or accepts_sparse(...)`` to just
    ``accepts_sparse(...)`` (a strict superset) must not change routing for the named models."""
    from mlframe.training.feature_handling.routing import accepts_sparse

    for model_kind in ("cb", "xgb", "lgb"):
        assert accepts_sparse(model_kind) is True


def test_f14_sparse_routing_also_covers_linear_models():
    """The redundant tuple didn't include linear/ridge/sgd; SPARSE_AWARE_MODELS already did --
    confirming the simplification lost no coverage."""
    from mlframe.training.feature_handling.routing import SPARSE_AWARE_MODELS, accepts_sparse

    assert {"cb", "xgb", "lgb", "linear", "ridge", "sgd"} <= SPARSE_AWARE_MODELS
    for model_kind in ("linear", "ridge", "sgd"):
        assert accepts_sparse(model_kind) is True


# =====================================================================
# F15 -- string-shorthand for a real (non-no-params) method raises a clear error
# =====================================================================


def test_f15_string_shorthand_valid_noparams_method_still_works():
    """The documented string-shorthand form must keep working for genuine no-params methods."""
    fhc = FeatureHandlingConfig(default_cat="ordinal", default_text=[])
    specs = fhc._effective_cat_specs("xgb")
    assert len(specs) == 1
    assert specs[0].method == "ordinal"


def test_f15_string_shorthand_real_method_raises_clear_value_error():
    """Pre-fix, ``default_text="tfidf"`` raised a raw, opaque pydantic ValidationError from
    ``NoParams(kind="tfidf")``. Post-fix it's a clear ValueError naming the actual problem."""
    fhc = FeatureHandlingConfig(default_text="tfidf", default_cat=[])
    with pytest.raises(ValueError, match="not a no-params method"):
        fhc._effective_text_specs("xgb")


# =====================================================================
# F16 -- unused `import pandas as pd` removed from the bench file
# =====================================================================


def test_f16_bench_encode_full_train_stat_transform_imports_cleanly():
    """The bench module must still import (and run its top-level setup) without the dead
    ``import pandas as pd``."""
    import importlib

    mod = importlib.import_module("mlframe.training.feature_handling._benchmarks.bench_encode_full_train_stat_transform")
    assert not hasattr(mod, "pd")


# =====================================================================
# F17 -- no audit/phase-marker junk left in the 4 named files
# =====================================================================


def test_f17_no_audit_phase_markers_in_named_files():
    """``cache.py``, ``target_encoders.py``, ``locking.py``, ``cache_backend.py`` must be free of
    "Wave N (date)" / "Round-3 X" process-metadata markers per this repo's own commenting
    convention (process metadata belongs in git history / PR text, not source)."""
    import re

    import mlframe.training.feature_handling as fh_pkg

    pkg_dir = fh_pkg.__path__[0]
    pattern = re.compile(r"Wave \d+|Round-3")
    for fname in ("cache.py", "target_encoders.py", "locking.py", "cache_backend.py"):
        path = f"{pkg_dir}/{fname}"
        with open(path, encoding="utf-8") as f:
            text = f.read()
        matches = pattern.findall(text)
        assert not matches, f"{fname} still has phase markers: {matches}"


# =====================================================================
# PR2 -- target_encoders.py LOC carve (helpers moved to a sibling module)
# =====================================================================


def test_pr2_target_encoders_under_carve_threshold():
    """``target_encoders.py`` was at 903 LOC, right at this repo's ~800-900 LOC carve guidance
    (CLAUDE.md 'New code goes in focused submodules from the start'). The canonicalisation helpers
    now live in a sibling ``_target_encoders_canon.py``, re-exported for backward compatibility."""
    import mlframe.training.feature_handling.target_encoders as te_mod

    path = te_mod.__file__
    with open(path, encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    assert n_lines < 700, f"target_encoders.py grew back to {n_lines} LOC; re-carve needed"


def test_pr2_canon_helpers_reexported_from_target_encoders():
    """Callers importing the canonicalisation helpers from ``target_encoders`` (the pre-carve
    location) must keep working unchanged."""
    from mlframe.training.feature_handling.target_encoders import (
        _canonical_cat_token,
        _categorical_to_string_array,
        _coerce_y_to_float64,
        _compute_prior,
    )

    assert _canonical_cat_token(1.0) == "1"
    assert _compute_prior(np.array([1.0, 0.0, 1.0, 0.0]), "mean") == 0.5
    arr = _categorical_to_string_array(np.array(["a", "b", None], dtype=object))
    assert arr[2] == "__NULL__"
    assert _coerce_y_to_float64(np.array([1, 0, 1])).dtype == np.float64
