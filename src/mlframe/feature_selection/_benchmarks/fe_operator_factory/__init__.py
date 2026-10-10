"""FE operator factory: the protocol, the helpers and the evidence scripts for turning a feature-engineering idea into an MRMR operator.

``README.md`` next to this file is the protocol. Sub-packages: ``common`` (shared binned-MI, downstream-error and path helpers), ``stat_study`` (statistics study of the
offset-product operator), ``kernel_prototypes`` (njit / CUDA-emulation kernels for that operator) and ``brainstorm`` (17 orthogonal operator classes screened with one harness).
"""
