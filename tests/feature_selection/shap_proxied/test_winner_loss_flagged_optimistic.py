"""The chosen subset's holdout loss is the minimum over many candidates on one holdout, and must be flagged as such."""

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_refine import _mark_selection_optimistic


def test_only_the_winner_is_flagged():
    ranked = [{"features": (1, 2), "honest_loss": 0.30}, {"features": (3,), "honest_loss": 0.31}]
    _mark_selection_optimistic(ranked, (1, 2))
    assert ranked[0]["honest_loss_selection_optimistic"] is True
    assert "honest_loss_selection_optimistic" not in ranked[1]
