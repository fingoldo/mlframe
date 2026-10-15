"""Shared SHAP proxy-fidelity policy.

The composite fidelity score is ``w_spearman * spearman + w_recall * recall@k``; the trust-guard config, the revalidation helper and the fit params must agree on the weights.
"""

DEFAULT_TRUST_GUARD_FIDELITY_WEIGHTS: tuple[float, float] = (0.6, 0.4)  # (w_spearman, w_recall)
