"""Regression tests for CRITICAL findings of the 2026-05-17 audit.

Each test pins a specific bug surfaced by the audit. Tests must fail
on pre-fix code and pass on post-fix; see
``mlframe/audit/CODE_REVIEW_2026-05-17.md`` for the full disposition
table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")


# ---------------------------------------------------------------------------
# C1: composite_estimator.py:563 -- mask-reindex bug
# ---------------------------------------------------------------------------


def test_c1_auto_variance_stabilise_with_dropped_rows() -> None:
    """When ``auto_variance_stabilise=True``, ``transform_name`` is
    ``"ratio"``/``"logratio"``, no caller-supplied ``sample_weight``, and
    some rows fail ``domain_check`` (so ``drop_invalid_rows=True`` filters
    them), the pre-fix branch did ``base_train[valid]`` where
    ``base_train`` was already filtered (length ``valid.sum()``) and
    ``valid`` retained the original length -- producing an ``IndexError``
    on numpy bool indexing.
    """
    from mlframe.training.composite import CompositeTargetEstimator

    rng = np.random.default_rng(0)
    n = 300
    base = rng.lognormal(mean=2.0, sigma=0.5, size=n)
    y = base * np.exp(rng.normal(0, 0.3, size=n))

    # Inject 30 rows of negative base -> filtered by logratio domain_check.
    base[:30] = -1.0
    df = pd.DataFrame({"base": base, "x1": rng.normal(size=n), "x2": rng.normal(size=n)})

    wrapper = CompositeTargetEstimator(
        base_estimator=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
        transform_name="logratio",
        base_column="base",
        drop_invalid_rows=True,
        auto_variance_stabilise=True,
    )

    # Pre-fix: IndexError("boolean index did not match indexed array...")
    # Post-fix: succeeds; sample_weight has length matching the filtered
    # train rows.
    wrapper.fit(df, y)

    assert wrapper.fitted_params_["n_train_invalid"] >= 30
    n_valid = wrapper.fitted_params_["n_train_valid"]
    assert n_valid == n - wrapper.fitted_params_["n_train_invalid"]

    # Predict should succeed and produce finite outputs on the same frame.
    yhat = wrapper.predict(df.iloc[30:])  # only the valid-domain rows
    assert yhat.shape == (n - 30,)
    assert np.all(np.isfinite(yhat))


# ---------------------------------------------------------------------------
# C2: feature_handling/cache.py -- xref write outside lock could strand entry
# ---------------------------------------------------------------------------


def test_c2_xref_invariant_under_eviction(tmp_path) -> None:
    """``_key_xref`` must never contain a key absent from ``_mem``.

    Pre-fix: the xref assignment lived OUTSIDE the lock that owned
    ``_mem`` insertion + eviction (line 202 / 221 in the disk-tier and
    compute paths). If ``_evict_if_needed_locked`` ran during
    ``_insert_in_memory`` and picked the just-inserted key as victim,
    the eviction popped ``_mem`` + ``_key_xref`` atomically; but THEN
    the caller (still outside the lock) wrote ``_key_xref[in_mem_key]
    = disk_key``, stranding the xref. A future ``get_or_compute``
    hitting the disk tier would then replay the stale ``DiskKey``.

    Post-fix the xref write is folded into ``_insert_in_memory`` under
    the same lock, so the eviction sweep at the end of insertion
    cleans both atomically.

    Deterministic reproduction: tiny ``ram_max_gb`` + ``size_weighted``
    eviction picks the largest entry each time. We insert several
    oversized values via ``get_or_compute``; each insertion triggers
    eviction. Post-condition: every ``_key_xref`` key must still be in
    ``_mem``.
    """
    pytest.importorskip("polars")
    from mlframe.training.feature_handling.cache import FeatureCache
    from mlframe.training.feature_handling.config import CacheConfig
    from mlframe.training.feature_handling.fingerprint import (
        ContentFingerprint,
        DiskKey,
        InMemoryKey,
    )

    cfg = CacheConfig(
        persistence="off",  # in-memory only; xref still maintained for unit testing
        ram_max_gb=0.000001,  # 1 KB -> any non-trivial entry triggers eviction
        ram_reserve_gb=0.0,
        eviction_strategy="lru",
    )
    cache = FeatureCache(cache_cfg=cfg)

    # Build a stable ContentFingerprint to derive DiskKeys (even though
    # persistence is off here, we still exercise the xref code path).
    cf = ContentFingerprint(
        n_rows=100,
        n_cols=4,
        column_dtypes_hash="a" * 32,
        sampled_rows_hash="b" * 32,
    )

    # Trigger eviction with explicit cross-key writes: directly call
    # the lower-level ``_insert_in_memory`` and verify the invariant
    # holds after each insert under tight RAM cap.
    big_payload = np.zeros(10_000, dtype=np.float64)  # 80 KB >> 1 KB cap
    for i in range(5):
        mem_key = InMemoryKey(
            session_id="s",
            df_token=i,
            train_idx_token=i,
            column=f"col_{i}",
            params_canonical_hash=f"h{i}",
            provider_signature="prov",
        )
        disk_key = DiskKey(
            content=cf,
            column=f"col_{i}",
            params_canonical_hash=f"h{i}" * 8,
            provider_signature="prov",
        )
        # Direct write through the API that the disk-tier-read /
        # compute-fresh paths now use: xref assignment under lock.
        cache._insert_in_memory(
            key=mem_key,
            value=big_payload,
            recompute_time_s=0.1,
            size_estimator=lambda v: v.nbytes,
            disk_key=disk_key,
        )
        # FH-XREF-NO-EVICT invariant.
        orphan = set(cache._key_xref) - set(cache._mem)
        assert not orphan, f"After insert #{i}: xref has {len(orphan)} orphan keys: {orphan}"


# ---------------------------------------------------------------------------
# C3: feature_handling/cache.py -- pickle RCE vector, default-deny
# ---------------------------------------------------------------------------


def test_c3_pickle_refused_by_default_on_write(tmp_path) -> None:
    """Writing a non-ndarray / non-sparse value with default
    ``allow_pickle=False`` must raise rather than silently fall through
    to ``pickle.dump``. Pre-fix the cache would unconditionally pickle
    arbitrary values, making the disk-tier readable file an RCE vector
    for any principal with write access to the cache directory."""
    import io
    from mlframe.training.feature_handling.cache import (
        CachePickleRefusedError,
        _serialize,
    )

    # Default allow_pickle=False refuses opaque payload.
    with pytest.raises(CachePickleRefusedError, match="allow_pickle is False"):
        _serialize({"opaque": object()}, io.BytesIO(), allow_pickle=False)

    # Opt-in succeeds (still writes pickle, but only when the caller
    # explicitly authorised it).
    buf = io.BytesIO()
    _serialize({"opaque": "hello"}, buf, allow_pickle=True)
    assert len(buf.getvalue()) > 0


def test_c3_pickle_refused_by_default_on_read(tmp_path) -> None:
    """Reading a payload that requires pickle deserialisation must
    raise with ``allow_pickle=False``. Pre-fix ``np.load(..., allow_pickle=True)``
    was unconditional and the pickle-fallback path was always reached
    on a non-numpy file."""
    import pickle
    from mlframe.training.feature_handling.cache import (
        CachePickleRefusedError,
        _deserialize,
    )

    p = tmp_path / "evil.bin"
    with open(p, "wb") as f:
        pickle.dump({"payload": [1, 2, 3]}, f, protocol=5)

    with pytest.raises(CachePickleRefusedError, match="allow_pickle is False"):
        _deserialize(str(p), allow_pickle=False)

    # Opt-in returns the value.
    val = _deserialize(str(p), allow_pickle=True)
    assert val == {"payload": [1, 2, 3]}


# ---------------------------------------------------------------------------
# C4/C5/C6: neural/recurrent.py weighted loss math (regression, multilabel, CE)
# ---------------------------------------------------------------------------


def test_c4_c5_c6_weighted_loss_normalises_by_weight_sum() -> None:
    """``_compute_weighted_loss`` must divide by ``sum(sample_weights)``,
    not by ``N``. Pre-fix ``(losses * weights).mean()`` divided by N --
    with ``weights=[10, 0, 0]`` and per-sample losses=[1, 1, 1] the
    weighted mean is 1.0 (only the first sample contributes), but the
    pre-fix code returned 10/3 ≈ 3.33. The bug distorted every
    weighted training -- both fit metric and gradient signal.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")

    # Build a minimal module exposing _compute_weighted_loss by routing
    # through the same code path. We construct a lightweight stand-in
    # for the LightningModule with the minimum attributes touched.
    from mlframe.training.neural.recurrent import RecurrentTorchModel

    class _Stub:
        is_regression = True
        task_type = "regression"
        _loss_fn_unreduced = staticmethod(lambda preds, targets: (preds - targets) ** 2)
        _loss_fn_mean = staticmethod(lambda preds, targets: ((preds - targets) ** 2).mean())

    stub = _Stub()
    compute = RecurrentTorchModel._compute_weighted_loss.__get__(stub)

    # Regression branch: logits (N, 1), labels (N,).
    logits = torch.tensor([[2.0], [3.0], [4.0]])  # preds after squeeze: [2, 3, 4]
    labels = torch.tensor([1.0, 2.0, 3.0])  # losses: (1)^2=[1, 1, 1]
    weights = torch.tensor([10.0, 0.0, 0.0])
    out = compute(logits, labels, weights).item()

    # True weighted mean = sum(w*L) / sum(w) = (10*1 + 0 + 0) / 10 = 1.0
    # Pre-fix returned (10*1 + 0 + 0) / 3 ≈ 3.333.
    assert abs(out - 1.0) < 1e-6, f"expected 1.0, got {out}"

    # Multilabel branch.
    class _StubML:
        is_regression = False
        task_type = "multilabel"
        _loss_fn_unreduced = staticmethod(lambda logits, labels_f: (logits - labels_f) ** 2)
        _loss_fn_mean = staticmethod(lambda logits, labels_f: ((logits - labels_f) ** 2).mean())

    stub_ml = _StubML()
    compute_ml = RecurrentTorchModel._compute_weighted_loss.__get__(stub_ml)
    logits_ml = torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    labels_ml = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).long()
    out_ml = compute_ml(logits_ml, labels_ml, weights).item()
    # Per-sample per-label loss = 1.0 everywhere; per-sample mean over
    # K = 1.0; weighted mean across N = sum([10*1, 0, 0])/10 = 1.0.
    assert abs(out_ml - 1.0) < 1e-6, f"expected 1.0, got {out_ml}"

    # Multiclass CE branch.
    class _StubCE:
        is_regression = False
        task_type = "classification"
        _loss_fn_unreduced = staticmethod(lambda logits, labels: torch.ones(logits.shape[0]))
        _loss_fn_mean = staticmethod(lambda logits, labels: torch.tensor(1.0))

    stub_ce = _StubCE()
    compute_ce = RecurrentTorchModel._compute_weighted_loss.__get__(stub_ce)
    logits_ce = torch.zeros(3, 5)
    labels_ce = torch.tensor([0, 1, 2])
    out_ce = compute_ce(logits_ce, labels_ce, weights).item()
    assert abs(out_ce - 1.0) < 1e-6, f"expected 1.0, got {out_ce}"


# ---------------------------------------------------------------------------
# C7: neural/base.py:476 UnboundLocalError on prediction_datamodule branch
# ---------------------------------------------------------------------------


def test_c7_predict_uses_prediction_datamodule_when_set() -> None:
    """Pre-fix the ``hasattr(...) or ... is None`` check had no
    ``else`` branch, so ``datamodule`` was left unbound; the subsequent
    ``datamodule.setup_predict(...)`` then raised ``UnboundLocalError``
    whenever a training-time datamodule was retained on the estimator.

    Behavioural test: construct a minimal stand-in with the exact
    attribute shape that the ``_predict_raw`` else-branch reads, drive
    the method to the point of the bug, and assert ``setup_predict``
    on the pre-attached datamodule was actually called.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    import numpy as np

    from mlframe.training.neural.base import PytorchLightningEstimator

    class _RecordingDM:
        def __init__(self):
            self.setup_predict_calls = []
            self.predict_dataloader_calls = 0
            self.batch_size = 64

        def setup_predict(self, X, batch_size=None):
            self.setup_predict_calls.append({"X_shape": getattr(X, "shape", None), "batch_size": batch_size})

        def predict_dataloader(self):  # pragma: no cover - reached after the fix only
            self.predict_dataloader_calls += 1
            return []

    class _FakeTrainer:
        def __init__(self):
            self.predict_calls = 0

        def predict(self, model, dataloaders=None):
            self.predict_calls += 1
            return [np.zeros((0, 1), dtype=np.float32)]

        class _Acc:
            def __init__(self):
                self.__class__.__name__ = "CPUAccelerator"

        accelerator = _Acc()

    est = PytorchLightningEstimator.__new__(PytorchLightningEstimator)
    est.model = object()  # truthy marker for "fitted"
    est.prediction_datamodule = _RecordingDM()
    est.datamodule_class = lambda **kw: _RecordingDM()
    est.datamodule_params = {"batch_size": 64}
    est.trainer = _FakeTrainer()

    X = np.zeros((4, 3), dtype=np.float32)

    # Pre-fix would raise UnboundLocalError here. We are only checking
    # the FIRST half of _predict_raw (binding + setup_predict); the
    # post-setup trainer path uses our mock so it returns immediately.
    try:
        est._predict_raw(X)
    except Exception as exc:
        # The fake trainer/dataloader plumbing may surface unrelated
        # errors. The fix is verified by reaching setup_predict at all.
        if isinstance(exc, UnboundLocalError):
            raise AssertionError("C7 fix missing: still raises UnboundLocalError") from exc

    assert len(est.prediction_datamodule.setup_predict_calls) == 1, "C7: pre-attached prediction_datamodule.setup_predict was never reached"


# ---------------------------------------------------------------------------
# C8: neural/flat.py:432 weight_sum==0 silent skip
# ---------------------------------------------------------------------------


def test_c8_zero_weight_sum_returns_zero_loss_without_sync(caplog) -> None:
    """When ``sample_weight.sum() == 0``, the per-sample weighted-loss
    path produces zero gradient. The post-2026-05-22 contract returns
    the safe-divide ``raw / clamp(weight_sum, min=1e-12)`` = 0.0 without
    issuing a per-batch WARN -- the previous WARN forced a GPU->CPU sync
    every batch for negligible operator value (the zero-gradient signal
    surfaces downstream as flat val loss). Sensor: loss is the expected
    zero scalar AND no WARN is emitted.
    """
    pytest.importorskip("torch")
    pytest.importorskip("lightning")
    import logging
    import torch

    from mlframe.training.neural.flat import MLPTorchModel

    # Construct a minimal module via __new__ to avoid the heavyweight
    # __init__ chain; populate just what _compute_weighted_loss reads.
    mod = MLPTorchModel.__new__(MLPTorchModel)
    mod.loss_fn = lambda p, y: torch.nn.functional.mse_loss(p, y)

    preds = torch.tensor([[2.0], [3.0], [4.0]])
    labels = torch.tensor([1.0, 2.0, 3.0])
    zero_w = torch.tensor([0.0, 0.0, 0.0])

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.flat"):
        out = MLPTorchModel._compute_weighted_loss(mod, preds, labels, zero_w)

    # Loss must still be a differentiable scalar of correct dtype.
    assert out.shape == torch.Size([])
    assert float(out) == 0.0

    # The 2026-05-22 contract DROPPED the per-batch WARN (GPU->CPU sync
    # cost was the dominant overhead). No WARN should fire on the
    # all-zero-weight path; the operator notices via flat val loss.
    warn_lines = [r for r in caplog.records if "sample_weight.sum()=0" in r.getMessage()]
    assert not warn_lines, (
        f"C8: per-batch zero-weight WARN was re-introduced; that re-adds a forced GPU->CPU sync per batch. Got: {[r.getMessage() for r in warn_lines]}"
    )


# ---------------------------------------------------------------------------
# C9: neural/recurrent.py per-sequence normalization default is "none"
# ---------------------------------------------------------------------------


def test_c9_default_sequence_preprocessing_preserves_magnitude() -> None:
    """Default ``sequence_preprocessing="none"`` must pass sequences
    through (cast to float32 only). Pre-audit the implicit hardcoded
    behaviour was per-sequence z-score on columns 1+ and astronomy-
    specific MJD delta on column 0 -- destroyed magnitude info that's
    discriminative in the general ML case.
    """
    pytest.importorskip("torch")
    import numpy as np
    from mlframe.training.neural.recurrent import _RecurrentWrapperBase

    seq = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]], dtype=np.float64)

    out_default = _RecurrentWrapperBase._preprocess_sequence(seq, mode="none")
    assert out_default.dtype == np.float32
    np.testing.assert_array_equal(out_default, seq.astype(np.float32))

    # Per-sequence z-score: column 0 should become standardised.
    out_zscore = _RecurrentWrapperBase._preprocess_sequence(seq, mode="per_sequence_zscore")
    assert abs(out_zscore[:, 0].mean()) < 1e-5
    assert abs(out_zscore[:, 0].std() - 1.0) < 1e-3

    # Astronomy legacy mode: column 0 becomes scaled delta, column 1
    # z-scored.
    out_astro = _RecurrentWrapperBase._preprocess_sequence(seq, mode="astronomy_mjd_delta")
    assert out_astro[0, 0] == 0.0  # first delta forced to 0
    # delta[1] = (seq[1,0] - seq[0,0]) / 10 = (2-1)/10 = 0.1
    assert abs(out_astro[1, 0] - 0.1) < 1e-5

    # Unknown mode raises.
    with pytest.raises(ValueError, match="unknown mode"):
        _RecurrentWrapperBase._preprocess_sequence(seq, mode="bogus")


def test_c9_config_default_is_none() -> None:
    """The default value of ``RecurrentConfig.sequence_preprocessing``
    is ``"none"`` -- no implicit per-sequence normalization."""
    from mlframe.training.neural._recurrent_config import RecurrentConfig

    cfg = RecurrentConfig()
    assert cfg.sequence_preprocessing == "none", f"C9: default must be 'none', got {cfg.sequence_preprocessing!r}"


# ---------------------------------------------------------------------------
# C10: _predict_guards.py NaN-guard refuse-by-default
# ---------------------------------------------------------------------------


def test_c10_nan_guard_refuses_when_unprimed_by_default() -> None:
    """Predict-time NaN guard must REFUSE to fit imputer/scaler on the
    current frame when no fit-time priming was done. Pre-fix the guard
    silently fit on the predict frame and persisted those test-set
    statistics on the model. Post-fix the guard raises
    :class:`NanGuardNotPrimedError`; callers wanting the legacy
    semantics opt in via ``fit_at_predict=True``.
    """
    from mlframe.training._predict_guards import (
        NanGuardNotPrimedError,
        _apply_nan_guard,
        prime_nan_guard_stats,
    )
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 4))
    X_predict_with_nan = rng.normal(size=(20, 4))
    X_predict_with_nan[:5, 1] = np.nan

    # Fit a model on clean data so the predict call is the only place
    # the guard could fit.
    model = Ridge().fit(X_train, rng.normal(size=100))

    # 1) Without priming, predict-time NaN guard must refuse.
    with pytest.raises(NanGuardNotPrimedError, match="prime_nan_guard_stats"):
        _apply_nan_guard(model, X_predict_with_nan, lambda X: model.predict(X), n_rows=20)

    # 2) After priming on train, predict-time guard works fine.
    prime_nan_guard_stats(model, X_train)
    out = _apply_nan_guard(
        model,
        X_predict_with_nan,
        lambda X: model.predict(X),
        n_rows=20,
    )
    assert out.shape == (20,)
    assert np.all(np.isfinite(out))

    # 3) Explicit opt-in to legacy fit-on-predict still works.
    model2 = Ridge().fit(X_train, rng.normal(size=100))
    out2 = _apply_nan_guard(
        model2,
        X_predict_with_nan,
        lambda X: model2.predict(X),
        n_rows=20,
        fit_at_predict=True,
    )
    assert out2.shape == (20,)
    assert np.all(np.isfinite(out2))
    # The legacy path also persists stats on the model.
    assert hasattr(model2, "_mlframe_nan_imputer")
    assert hasattr(model2, "_mlframe_nan_scaler")


# ---------------------------------------------------------------------------
# Wave 1.5: cross-process monkey-patching no longer applied at import time
# ---------------------------------------------------------------------------


def test_wave15_third_party_patches_not_applied_at_bare_import() -> None:
    """Audit 2026-05-17 Wave 1.5: bare ``import mlframe.training``
    must NOT mutate joblib / lightgbm / catboost / xgboost state.
    Patches are applied lazily by the suite entrypoint
    (``train_mlframe_models_suite``) and by the public factory
    functions ``make_pool`` / ``make_dmatrix`` / ``make_lgb_dataset``.

    Probes the import-time contract in a SUBPROCESS so the rebinding
    that ``importlib.reload`` would otherwise cause doesn't pollute the
    rest of the suite (per the test-pollution rule in CLAUDE.md). The
    subprocess does one bare ``import`` and prints the flag; any True
    value means a caller re-added an import-time apply call.
    """
    import os
    import subprocess
    import sys
    import textwrap

    import mlframe as _mlframe_pkg

    _src_root = os.path.dirname(os.path.dirname(_mlframe_pkg.__file__))
    _env = {**os.environ, "PYTHONPATH": _src_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    _probe = textwrap.dedent("""
        import sys
        import mlframe.training._model_factories as mf
        sys.stdout.write(repr(getattr(mf, "_THIRD_PARTY_PATCHES_APPLIED", "MISSING")))
    """)
    _res = subprocess.run(
        [sys.executable, "-c", _probe],
        capture_output=True,
        text=True,
        timeout=180,
        env=_env,
    )
    assert _res.returncode == 0, f"probe subprocess failed: {_res.stderr}"
    assert _res.stdout.strip() == "False", (
        f"Wave 1.5: third-party patches were applied at import time (probe printed {_res.stdout!r}). Did someone re-add the import-time call?"
    )


def test_wave15_factory_applies_patches_on_first_call() -> None:
    """The factory functions (``make_pool`` / ``make_dmatrix`` /
    ``make_lgb_dataset``) must flip the flag on first use."""
    pytest.importorskip("lightgbm")

    import mlframe.training._model_factories as mf

    # Reset for this test.
    mf._THIRD_PARTY_PATCHES_APPLIED = False

    # Construction triggers apply_third_party_patches_once.
    import numpy as np

    X = np.zeros((4, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int32)
    _ = mf.make_lgb_dataset(X, label=y)

    assert mf._THIRD_PARTY_PATCHES_APPLIED is True
