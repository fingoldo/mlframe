"""Kernel prototypes for the offset-product operator: fused offset-grid scorer, closed-form shift estimators, permutation null, CUDA kernel source plus numpy emulation.

The CUDA source in ``offset_fused_cuda`` and ``offset_closed_form`` was never run on a GPU; only its numpy emulation is verified. Tests live in ``tests/feature_selection/fe/factory``.
"""
