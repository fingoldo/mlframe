"""Meta-test: every transformer in feature_engineering/transformer/ that does
``np.asarray(X, dtype=np.float32)`` must first call ``validate_numeric_input``,
so the float32-precision warning added in commit d772f0c actually fires on real
user call paths and not just in standalone validator unit tests.

Without this compliance test a future transformer added to the package could
quietly skip the validator and re-introduce the silent precision-loss bug class
that the warning was meant to surface.
"""

from __future__ import annotations

import pathlib

import pytest


import mlframe as _mlframe

TRANSFORMER_DIR = pathlib.Path(_mlframe.__file__).resolve().parent / "feature_engineering" / "transformer"


def _list_transformer_modules() -> list[pathlib.Path]:
    """Every public transformer .py under transformer/ (skip _utils, __init__, _benchmarks)."""
    out = []
    for p in TRANSFORMER_DIR.glob("*.py"):
        if p.name.startswith("_"):
            continue
        if p.name == "__init__.py":
            continue
        out.append(p)
    return sorted(out)


def _file_does_float32_cast(src: str) -> bool:
    """True iff the module casts something to float32 by any common numpy / torch
    pattern. Previous pattern required BOTH ``dtype=np.float32`` AND ``np.asarray``
    in the source, which silently passed over every transformer that uses the
    far more common ``.astype(np.float32, copy=False)`` form (random_features,
    row_attention, residual_attention, ...). Result: 12 transformer modules
    skipped the compliance check with "nothing to validate" while their
    production code DID cast and was NOT calling ``validate_numeric_input``.

    Patterns matched (one is enough):
      * ``np.asarray(..., dtype=np.float32)`` -- the historical narrow pattern.
      * ``.astype(np.float32`` / ``.astype('float32')`` / ``.astype("float32")``
        -- the common in-place dtype-cast form used by most transformers.
      * ``np.array(..., dtype=np.float32)`` -- alternative array constructor.
      * ``torch.float32`` -- torch-side cast (transformer modules that emit
        Tensors).
    Compatible regression: any module that used to match the old pattern still
    matches via the literal-``np.asarray`` branch.
    """
    if "np.asarray" in src and "dtype=np.float32" in src:
        return True
    if "np.array" in src and "dtype=np.float32" in src:
        return True
    if ".astype(np.float32" in src:
        return True
    if ".astype('float32')" in src or '.astype("float32")' in src:
        return True
    if "torch.float32" in src:
        return True
    return False


def _file_calls_validate_numeric_input(src: str) -> bool:
    """True iff the module imports OR calls validate_numeric_input."""
    return "validate_numeric_input" in src


@pytest.mark.parametrize("path", _list_transformer_modules(), ids=lambda p: p.name)
def test_every_float32_caster_calls_validate_numeric_input(path):
    """Compliance check: any module that casts X to float32 must run the validator
    first so the precision warning fires (silently truncating int64 epoch-seconds
    to float32 would otherwise corrupt every kNN / SMOTE / RFF distance silently)."""
    src = path.read_text(encoding="utf-8")
    if not _file_does_float32_cast(src):
        pytest.skip(f"{path.name}: no np.asarray(..., dtype=np.float32) -- nothing to validate")
    assert _file_calls_validate_numeric_input(src), (
        f"{path.name} casts to float32 but does not call validate_numeric_input. "
        f"This bypasses the float32-precision warning (commit d772f0c). Add "
        f"`validate_numeric_input(X_train, name='X_train')` before the float32 cast."
    )


def test_meta_self_check_at_least_one_compliant_transformer():
    """Sanity: this compliance test would be vacuously true if no transformer
    matches the float32-cast pattern. At least one must, otherwise the test is
    dead code that gives false confidence."""
    modules = _list_transformer_modules()
    f32_callers = [p for p in modules if _file_does_float32_cast(p.read_text(encoding="utf-8"))]
    assert len(f32_callers) >= 5, f"expected >=5 transformers doing float32 cast (audit found 8+); got {len(f32_callers)}: {[p.name for p in f32_callers]}"
