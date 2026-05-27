"""PipelineCache byte budget = fraction of TOTAL host RAM (clamped to free).

The old "available - 8 GB" budget drifted up to a 64 GB cap and, on a box
where model training itself used 100 GB+, the cache + in-flight float64
transforms overran RAM (OOM at 174 GB). The budget is now a fraction of
TOTAL host RAM (host-predictable, tunable via
train_mlframe_models_suite(pipeline_cache_ram_budget_fraction=...)), still
clamped to currently-available RAM minus a floor.
"""
from __future__ import annotations

import os

import psutil

from mlframe.training._strategies_pipeline_cache import (
    _resolve_pipeline_cache_budget,
)

_GiB = 1024 ** 3


def _clear_env():
    for k in ("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT",
              "MLFRAME_PIPELINE_CACHE_RAM_FRACTION"):
        os.environ.pop(k, None)


def test_absolute_bytes_env_override_wins():
    _clear_env()
    os.environ["MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"] = "1234567"
    try:
        assert _resolve_pipeline_cache_budget(0.4) == 1234567
    finally:
        _clear_env()


def test_budget_never_exceeds_available_or_total_fraction():
    _clear_env()
    try:
        vm = psutil.virtual_memory()
        for frac in (0.1, 0.4, 0.6):
            b = _resolve_pipeline_cache_budget(frac)
            # Never more than the fraction of total ...
            assert b <= int(vm.total * frac) + _GiB, (frac, b)
            # ... and never more than (available - 4 GB floor), unless the
            # 2 GiB hard floor applies.
            assert b <= max(2 * _GiB, vm.available), (frac, b)
            assert b >= 2 * _GiB or b == int(vm.total * frac)
    finally:
        _clear_env()


def test_fraction_env_is_read_when_no_arg():
    _clear_env()
    os.environ["MLFRAME_PIPELINE_CACHE_RAM_FRACTION"] = "0.05"
    try:
        b_env = _resolve_pipeline_cache_budget()
        b_arg = _resolve_pipeline_cache_budget(0.05)
        # Same fraction whether via env or arg (allow tiny availability drift).
        assert abs(b_env - b_arg) <= 1 * _GiB
    finally:
        _clear_env()


def test_fraction_clamped_to_sane_band():
    _clear_env()
    try:
        # Negative / >0.9 fractions must not crash and stay within the band.
        b_neg = _resolve_pipeline_cache_budget(-1.0)
        b_huge = _resolve_pipeline_cache_budget(5.0)
        vm = psutil.virtual_memory()
        assert b_neg >= 2 * _GiB
        assert b_huge <= int(vm.total * 0.9) + _GiB
    finally:
        _clear_env()
