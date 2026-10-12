"""Shared fakes and parametrisation tables for the MRMR regression test modules."""

from __future__ import annotations


class _FakeCudaFault(Exception):
    """Stand-in for numba's CudaAPIError/CudaDriverError, which derive from Exception directly, not RuntimeError."""


_ORTH_SCORER_ZOO_MODULES = [
    ("_orthogonal_hsic_fe", "hybrid_orth_mi_hsic_fe_with_recipes"),
    ("_orthogonal_lasso_fe", "hybrid_orth_mi_lasso_fe_with_recipes"),
    ("_orthogonal_routing_fe", "hybrid_orth_mi_conditional_routing_fe_with_recipes"),
    ("_orthogonal_scorer_auto_fe", "hybrid_orth_mi_ensemble_fe_with_recipes"),
    ("_orthogonal_three_gate_mi_fe", "hybrid_orth_mi_three_gate_fe_with_recipes"),
    ("_orthogonal_total_correlation_fe", "hybrid_orth_mi_tc_fe_with_recipes"),
    ("_orthogonal_adaptive_arity_fe", "hybrid_orth_mi_adaptive_arity_fe_with_recipes"),
    ("_orthogonal_adaptive_degree_fe", "hybrid_orth_mi_adaptive_degree_fe_with_recipes"),
    ("_orthogonal_bootstrap_mi_fe", "hybrid_orth_mi_bootstrap_fe_with_recipes"),
    ("_orthogonal_cmim_fe", "hybrid_orth_mi_cmim_fe_with_recipes"),
    ("_orthogonal_copula_mi_fe", "hybrid_orth_mi_copula_fe_with_recipes"),
    ("_orthogonal_dcor_fe", "hybrid_orth_mi_dcor_fe_with_recipes"),
    ("_orthogonal_elasticnet_fe", "hybrid_orth_mi_elasticnet_fe_with_recipes"),
]
