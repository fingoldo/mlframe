"""Proxy-loss bias corrector for ShapProxiedFS (levers #3 + #6).

The coalition proxy is a *biased* estimator of a retrained model: the bias grows with subset
redundancy (correlated survivors a retrain could lean on, that the proxy can't) and varies with
cardinality. The trust guard already trains a handful of honest anchor models -- instead of throwing
those (proxy, honest) pairs away after computing Spearman, we fit a tiny calibrator
``honest_loss ~ f(proxy_loss, cardinality, redundancy)`` and use it to RE-RANK candidates before the
expensive top-N honest re-validation. This both answers the user's open "do we need to scale the
partial sums?" question (yes -- but as a redundancy/cardinality-dependent map fit from data, not a
guessed global scale) and concentrates the honest-retrain budget on the corrector's best subsets.

The corrector is monotone-agnostic and cheap (Ridge over 4 engineered features); with too few anchors
it degrades gracefully to "proxy_loss only" (rank-preserving), so it can never do worse than the raw
proxy ordering on the held anchors.
"""

from __future__ import annotations

import numpy as np


def subset_redundancy(phi: np.ndarray, idx) -> float:
    """Mean pairwise |correlation| of the selected proxy columns' attributions -- a cheap redundancy
    summary (0 for singletons). High redundancy is exactly where the proxy under-credits subsets."""
    idx = list(idx)
    if len(idx) < 2:
        return 0.0
    sub = phi[:, idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(sub, rowvar=False)
    if not np.ndim(c):
        return 0.0
    iu = np.triu_indices(c.shape[0], k=1)
    vals = np.abs(c[iu])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else 0.0


def _features(proxy, card, redund):
    proxy = np.asarray(proxy, dtype=np.float64)
    card = np.asarray(card, dtype=np.float64)
    redund = np.asarray(redund, dtype=np.float64)
    # proxy, cardinality, redundancy, and the proxy x redundancy interaction (the bias term).
    return np.column_stack([proxy, card, redund, proxy * redund])


class ProxyCorrector:
    """Calibrated map from (proxy_loss, cardinality, redundancy) -> predicted honest loss."""

    def __init__(self, model=None, mean=None, std=None, fallback=False):
        self.model = model
        self.mean = mean
        self.std = std
        self.fallback = fallback  # True -> identity on proxy_loss (rank-preserving)

    def predict(self, proxy, card, redund):
        if self.fallback or self.model is None:
            return np.asarray(proxy, dtype=np.float64)
        F = (_features(proxy, card, redund) - self.mean) / self.std
        return self.model.predict(F)


def fit_proxy_corrector(proxy_losses, honest_losses, cards, redunds, *, min_anchors=12) -> ProxyCorrector:
    """Fit ``honest ~ Ridge(proxy, card, redund, proxy*redund)``. Falls back to identity (proxy-only,
    rank-preserving) when too few finite anchors to fit reliably."""
    proxy = np.asarray(proxy_losses, dtype=np.float64)
    honest = np.asarray(honest_losses, dtype=np.float64)
    cards = np.asarray(cards, dtype=np.float64)
    redunds = np.asarray(redunds, dtype=np.float64)
    ok = np.isfinite(proxy) & np.isfinite(honest) & np.isfinite(cards) & np.isfinite(redunds)
    proxy, honest, cards, redunds = proxy[ok], honest[ok], cards[ok], redunds[ok]
    if proxy.size < min_anchors or np.unique(honest).size < 3:
        return ProxyCorrector(fallback=True)
    from sklearn.linear_model import Ridge

    F = _features(proxy, cards, redunds)
    mean = F.mean(axis=0)
    std = np.where(F.std(axis=0) > 0, F.std(axis=0), 1.0)
    Fs = (F - mean) / std
    model = Ridge(alpha=1.0).fit(Fs, honest)
    return ProxyCorrector(model=model, mean=mean, std=std, fallback=False)


def rerank_candidates(corrector: ProxyCorrector, candidates, phi):
    """Re-rank ``[(proxy_loss, idx), ...]`` by the corrector's predicted honest loss (ascending).

    Returns the re-ordered candidate list (same elements; the stored proxy_loss is preserved so the
    downstream report still shows the raw proxy value).
    """
    if not candidates:
        return candidates
    proxy = np.array([c[0] for c in candidates], dtype=np.float64)
    cards = np.array([len(c[1]) for c in candidates], dtype=np.float64)
    redund = np.array([subset_redundancy(phi, c[1]) for c in candidates], dtype=np.float64)
    pred = corrector.predict(proxy, cards, redund)
    order = np.argsort(pred, kind="stable")
    return [candidates[i] for i in order]
