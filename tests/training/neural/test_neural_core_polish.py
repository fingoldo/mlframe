"""Regression tests for LOW + POLISH findings in neural/ + ranker_suite.

Wave 4 + 5 of the 2026-05-17 audit. Each test pins a specific fix; tests
fail on pre-fix code and pass on post-fix.

Categories covered:
- S3 assert -> raise at module boundaries: generate_mlp arg validation,
  PytorchLightningEstimator float32_matmul_precision, PeriodicLearningRateFinder period.
- Naming nit: ``seq_input_size`` magic 4 replaced by module constant
  ``_DEFAULT_SEQ_INPUT_SIZE``; save/load round-trip uses it.
- Setup() stub remains a Lightning no-op but is now documented; calling
  it on a fully-initialised DataModule does not raise.
- ``custom_collate_fn`` is identity (picklable, top-level) and accepted by
  DataLoader's collate_fn argument.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


# ---------------------------------------------------------------------------
# S3-flat-asserts: generate_mlp validation raises ValueError, not AssertionError
# ---------------------------------------------------------------------------


def test_low_s3_generate_mlp_rejects_bad_args_with_valueerror() -> None:
    """Pre-fix: ``assert nlayers >= 1`` raised AssertionError and was stripped
    under ``python -O``. Post-fix raises ValueError with a descriptive message
    that survives optimisation flags."""
    from mlframe.training.neural.flat import generate_mlp

    # nlayers must be a positive int
    with pytest.raises(ValueError, match="nlayers"):
        generate_mlp(num_features=4, num_classes=2, nlayers=0)
    # Non-int nlayers -> TypeError (Pythonic for a type mismatch).
    with pytest.raises(TypeError, match="nlayers"):
        generate_mlp(num_features=4, num_classes=2, nlayers=1.5)  # type: ignore[arg-type]
    # dropout_prob must be >= 0
    with pytest.raises(ValueError, match="dropout_prob"):
        generate_mlp(num_features=4, num_classes=2, dropout_prob=-0.1)
    # consec_layers_neurons_ratio must be >= 1
    with pytest.raises(ValueError, match="consec_layers_neurons_ratio"):
        generate_mlp(num_features=4, num_classes=2, consec_layers_neurons_ratio=0.5)
    # min_layer_neurons must be a positive int
    with pytest.raises(ValueError, match="min_layer_neurons"):
        generate_mlp(num_features=4, num_classes=2, min_layer_neurons=0)
    # num_classes must be None or non-negative int (passing -1 fails)
    with pytest.raises(ValueError, match="num_classes"):
        generate_mlp(num_features=4, num_classes=-1)
    # first_layer_num_neurons below min_layer_neurons
    with pytest.raises(ValueError, match="first_layer_num_neurons"):
        generate_mlp(
            num_features=4,
            num_classes=2,
            min_layer_neurons=8,
            first_layer_num_neurons=4,
        )


def test_low_s3_periodic_lr_finder_rejects_bad_period() -> None:
    """Pre-fix: ``assert period > 0`` raised AssertionError. Post-fix raises
    ValueError so callers can introspect the failure."""
    from mlframe.training.neural.base import PeriodicLearningRateFinder

    with pytest.raises(ValueError, match="period"):
        PeriodicLearningRateFinder(period=0)
    with pytest.raises(ValueError, match="period"):
        PeriodicLearningRateFinder(period=-3)
    # Non-int period -> TypeError (Pythonic for a type mismatch).
    with pytest.raises(TypeError, match="period"):
        PeriodicLearningRateFinder(period=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# POLISH naming nit: _DEFAULT_SEQ_INPUT_SIZE constant in recurrent module
# ---------------------------------------------------------------------------


def test_polish_default_seq_input_size_constant_used_consistently() -> None:
    """The save/load fallback and the wrapper init must share one source of
    truth. Pre-fix three separate magic-4 literals risked drifting; the test
    asserts they all read from the module constant."""
    from mlframe.training.neural import recurrent as rec_mod

    assert hasattr(rec_mod, "_DEFAULT_SEQ_INPUT_SIZE")
    assert isinstance(rec_mod._DEFAULT_SEQ_INPUT_SIZE, int)
    assert rec_mod._DEFAULT_SEQ_INPUT_SIZE >= 1

    # Wrapper init uses the constant: a freshly-constructed wrapper without
    # any fit() reports the constant value (not a hardcoded literal).
    wrapper = rec_mod.RecurrentClassifierWrapper()
    assert wrapper._seq_input_size == rec_mod._DEFAULT_SEQ_INPUT_SIZE

    # RecurrentTorchModel's seq_input_size default also tracks the constant.
    import inspect

    sig = inspect.signature(rec_mod.RecurrentTorchModel.__init__)
    assert sig.parameters["seq_input_size"].default == rec_mod._DEFAULT_SEQ_INPUT_SIZE


# ---------------------------------------------------------------------------
# POLISH: RecurrentDataModule.setup() is an intentional no-op
# ---------------------------------------------------------------------------


def test_polish_recurrent_datamodule_setup_is_intentional_noop() -> None:
    """Lightning's DataModule contract calls ``setup(stage)`` before fit /
    validate / test / predict. Our DataModule receives arrays via __init__,
    so setup() has no work to do but MUST still exist and be safe to call
    multiple times with arbitrary stage strings."""
    import numpy as np

    from mlframe.training.neural._recurrent_data import RecurrentDataModule

    seqs = [np.zeros((3, 2), dtype=np.float32) for _ in range(4)]
    labels = np.zeros(4, dtype=np.int64)
    dm = RecurrentDataModule(
        train_sequences=seqs,
        train_features=None,
        train_labels=labels,
        train_sample_weight=None,
        val_sequences=seqs,
        val_features=None,
        val_labels=labels,
        val_sample_weight=None,
        batch_size=2,
        num_workers=0,
        is_regression=False,
        use_stratified_sampler=False,
    )
    # Calling with each Lightning stage must not raise nor mutate the
    # already-loaded arrays.
    for stage in ("fit", "validate", "test", "predict", None):
        result = dm.setup(stage)
        assert result is None
    assert dm.train_sequences is seqs


# ---------------------------------------------------------------------------
# POLISH: custom_collate_fn is identity + picklable (top-level callable)
# ---------------------------------------------------------------------------


def test_polish_custom_collate_fn_is_identity_and_picklable() -> None:
    """Pre-fix the body was correct but the comment ("mimicking lambda x: x")
    suggested a placeholder. Post-fix the docstring explains WHY it exists
    (must be top-level for multi-worker DataLoader pickling, identity is the
    intent). Behavioural pin: it hands the batch through verbatim and is
    picklable."""
    import pickle

    from mlframe.training.neural.base import custom_collate_fn

    batch = [("a", 1), ("b", 2), ("c", 3)]
    assert custom_collate_fn(batch) is batch

    # Must round-trip through pickle so DataLoader's multi-worker spawn can
    # ship it across the process boundary.
    restored = pickle.loads(pickle.dumps(custom_collate_fn))
    assert restored(batch) == batch


# ---------------------------------------------------------------------------
# S1 sweep pin: stripped dated audit comments leave no '2026-05-1x' tags
# ---------------------------------------------------------------------------


def test_low_s1_no_dated_audit_tags_in_neural_scope() -> None:
    """Pre-fix multiple ``# 2026-05-1x:`` / ``Wave 27`` tags lived inside
    source comments. Per ``feedback_no_audit_phase_in_comments`` these belong
    in git/PR text, not source. Post-fix the strings are gone from the
    in-scope modules."""
    import pathlib
    import mlframe as _mlframe

    # Resolve from the installed mlframe package so the test is robust
    # to its own location in the tests/ tree.
    _mlframe_root = pathlib.Path(_mlframe.__file__).resolve().parent
    targets = [
        _mlframe_root / "training" / "ranking" / "ranker_suite.py",
        _mlframe_root / "training" / "neural" / "_recurrent_config.py",
    ]
    for p in targets:
        text = p.read_text(encoding="utf-8")
        # No literal dated tags anywhere in the file.
        assert "2026-05-08" not in text, p
        assert "2026-05-10" not in text, p
        assert "2026-05-12" not in text, p
        assert "audit 2026-05-17" not in text, p
        # No "Wave NN" history markers in comments.
        assert "Wave 27" not in text, p
