"""Register-pressure factories of the radix-select, interpolation and transpose kernels of ``_gpu_resident_select_kernels``.

The kernels are compiled on first use by the getters of that module (which stays under the module-size limit); the factories here compile the same sources so the register-pressure registry
(`tests/test_meta/test_cuda_kernel_register_pressure.py`) checks every one of them for spills on a device.
"""

from __future__ import annotations

from typing import Any

from . import _gpu_resident_select_kernels as _k


def kernel_radix_f64(cp: Any, real: Any = None) -> Any:
    """``radix_select_f64`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_F64_SRC, "radix_select_f64")


def kernel_radix_f32(cp: Any, real: Any = None) -> Any:
    """``radix_select_f32`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_F32_SRC, "radix_select_f32")


def kernel_radix_f32_bsearch(cp: Any, real: Any = None) -> Any:
    """``radix_select_f32_bsearch`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_F32_BSEARCH_SRC, "radix_select_f32_bsearch")


def kernel_radix_f32_v3(cp: Any, real: Any = None) -> Any:
    """``radix_select_f32_v3`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_F32_V3_SRC, "radix_select_f32_v3")


def kernel_interp_f64(cp: Any, real: Any = None) -> Any:
    """``radix_select_interp_f64`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_INTERP_F64_SRC, "radix_select_interp_f64")


def kernel_interp_f64_v2(cp: Any, real: Any = None) -> Any:
    """``radix_select_interp_f64_v2`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_INTERP_F64_V2_SRC, "radix_select_interp_f64_v2")


def kernel_interp_f64_v3(cp: Any, real: Any = None) -> Any:
    """``radix_select_interp_f64_v3`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_SELECT_INTERP_F64_V3_SRC, "radix_select_interp_f64_v3")


def kernel_transpose_f32(cp: Any, real: Any = None) -> Any:
    """``transpose_f32`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._TRANSPOSE_F32_SRC, "transpose_f32")


def kernel_transpose_f64(cp: Any, real: Any = None) -> Any:
    """``transpose_f64`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._TRANSPOSE_F64_SRC, "transpose_f64")


def kernel_transpose_i8(cp: Any, real: Any = None) -> Any:
    """``transpose_i8`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._TRANSPOSE_I8_SRC, "transpose_i8")


def kernel_transpose_i16(cp: Any, real: Any = None) -> Any:
    """``transpose_i16`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._TRANSPOSE_I16_SRC, "transpose_i16")


def kernel_interp(cp: Any, real: Any = None) -> Any:
    """``radix_interp`` (fixed element types; ``real`` unused); registry factory."""
    return cp.RawKernel(_k._RADIX_INTERP_SRC, "radix_interp")
