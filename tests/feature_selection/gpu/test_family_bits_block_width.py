"""The block width of the v2 family-null kernel follows the device limits and the override, and falls back to the constant."""

import mlframe.feature_selection.filters._family_maxt_cuda_bits as bits
import mlframe.feature_selection.filters._gpu_hw_launch as hw

_PASCAL = {
    "sm_count": 6, "max_threads_per_block": 1024, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "warp": 32,
    "shared_per_block": 49152, "regs_per_block": 65536, "regs_per_sm": 65536, "shared_per_sm": 98304, "reserved_shared_per_block": 0, "reg_alloc_unit": 256,
}


class _Kernel:
    """The two attributes of a compiled cupy kernel the width rule reads."""

    num_regs = 40
    shared_size_bytes = 0


def test_width_follows_the_shared_table(monkeypatch):
    """A 12.5 KB class table (n = 100000) holds 6 blocks of 256 on a Pascal SM: that is the width with the most resident threads; a tiny table reaches full occupancy at a narrower or equal width."""
    monkeypatch.delenv("MLFRAME_GPU_BLOCK_FAMILY_BITS", raising=False)
    monkeypatch.setattr(hw, "device_props", lambda: _PASCAL)
    assert bits.v2_block_width(_Kernel(), 12504, 250) == 256
    assert bits.v2_block_width(_Kernel(), 12504, 40) == 64  # no more columns than a 64-wide block can use


def test_override_and_fallback(monkeypatch):
    """The environment variable wins when valid; no device information keeps the historical constant."""
    monkeypatch.setattr(hw, "device_props", lambda: {})
    monkeypatch.delenv("MLFRAME_GPU_BLOCK_FAMILY_BITS", raising=False)
    assert bits.v2_block_width(_Kernel(), 12504, 250) == bits._THREADS_V2
    monkeypatch.setenv("MLFRAME_GPU_BLOCK_FAMILY_BITS", "512")
    assert bits.v2_block_width(_Kernel(), 12504, 250) == 512
    monkeypatch.setenv("MLFRAME_GPU_BLOCK_FAMILY_BITS", "100")  # not a multiple of 32: ignored
    assert bits.v2_block_width(_Kernel(), 12504, 250) == bits._THREADS_V2
