"""Architecture-dependent launch limits that CUDA kernel sources need at compile time.

A ``__launch_bounds__(T, B)`` with a literal ``B`` is tuned on one card: the threads one SM can hold are 2048 (sm_60 to 70, 80, 90), 1536 (sm_86, 87, 89, 120) or 1024 (sm_75), and a
hint above the limit is ignored with a ptxas warning. NVRTC defines ``__CUDA_ARCH__`` for the device the module is compiled for (cupy compiles per device), so a kernel in a
multi-GPU host with unlike cards gets the right limit for each. Use ``__launch_bounds__(T, MLF_THREADS_PER_SM / T)`` after putting ``ARCH_THREADS_HEADER`` in front of the source.
"""

ARCH_THREADS_HEADER = r"""
#if __CUDA_ARCH__ == 750
#define MLF_THREADS_PER_SM 1024
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ >= 1200
#define MLF_THREADS_PER_SM 1536
#else
#define MLF_THREADS_PER_SM 2048
#endif
"""

__all__ = ["ARCH_THREADS_HEADER"]
