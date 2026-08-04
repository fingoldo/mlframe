"""Optimizer dispatch for ``ShapProxiedFitMixin._run_search``.

Carved out of ``_shap_proxied_fit.py`` to keep it under the 1k LOC ceiling.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def run_search(self: Any, optimizer: str, phi: np.ndarray, base: np.ndarray, y: np.ndarray) -> list:
    """Dispatch to the chosen optimizer; returns list of (proxy_loss, feature_idx tuple)."""
    if optimizer == "bruteforce":
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

        return list(brute_force_top_n(
            phi,
            base,
            y,
            min_card=self.min_features,
            parallel=(phi.shape[1] >= 14),
            classification=self.classification,
            metric=self.metric,
            max_card=self.max_features,
            top_n=self.top_n,
        ))
    if optimizer == "bruteforce_gpu":
        # Size-aware dispatcher: defaults to CPU, routes to the cupy kernel only when the KTC
        # crossover says GPU wins, and auto-falls back to the CPU kernel on any cupy/OOM error
        # (catch + log once). Keeps zero crash risk on hosts that segfault importing cupy.
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_subsetrank import brute_force_top_n_dispatch

        return list(brute_force_top_n_dispatch(
            phi,
            base,
            y,
            min_card=self.min_features,
            parallel=True,
            prefer_gpu=True,
            classification=self.classification,
            metric=self.metric,
            max_card=self.max_features,
            top_n=self.top_n,
        ))
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_heuristics as heur

    if optimizer == "beam":
        return list(heur.beam_search(
            phi,
            base,
            y,
            beam_width=self.beam_width,
            min_card=self.min_features,
            classification=self.classification,
            metric=self.metric,
            max_card=self.max_features,
            top_n=self.top_n,
        ))
    if optimizer == "greedy_forward":
        return list(heur.greedy_forward(phi, base, y, classification=self.classification, metric=self.metric, max_card=self.max_features, top_n=self.top_n))
    if optimizer == "greedy_backward":
        return list(heur.greedy_backward(phi, base, y, classification=self.classification, metric=self.metric, min_card=self.min_features, top_n=self.top_n))
    if optimizer == "multistart":
        return list(heur.multistart_local(
            phi,
            base,
            y,
            rng=self._rng,
            classification=self.classification,
            metric=self.metric,
            max_card=self.max_features,
            top_n=self.top_n,
        ))
    if optimizer == "genetic":
        return list(heur.genetic(
            phi,
            base,
            y,
            rng=self._rng,
            classification=self.classification,
            metric=self.metric,
            max_card=self.max_features,
            top_n=self.top_n,
        ))
    if optimizer == "annealing":
        return list(heur.simulated_annealing(
            phi,
            base,
            y,
            rng=self._rng,
            classification=self.classification,
            metric=self.metric,
            max_card=self.max_features,
            top_n=self.top_n,
        ))
    if optimizer == "gradient":
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_gradient import gradient_top_n

        return list(gradient_top_n(phi, base, y, classification=self.classification, metric=self.metric, random_state=int(self.random_state), top_n=self.top_n))
    raise ValueError(f"Unknown optimizer={optimizer!r}")
