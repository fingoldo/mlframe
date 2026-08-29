"""RUX-2: a composer's caption must describe the panels it actually rendered.

The caption was one figure-level string written for each composer's DEFAULT template. A caller asking for a
narrower mix read a paragraph about panels that were not on their figure, and a caller asking for a token the
paragraph never covered got no explanation of it at all. Each composer now joins per-token fragments for the
tokens it rendered.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from mlframe.reporting.charts._captions import caption_for_tokens

COMPOSERS = ("binary", "ltr", "multiclass", "multilabel", "quantile", "regression", "temporal")


@pytest.mark.parametrize("module_name", COMPOSERS)
def test_every_token_has_a_caption_fragment(module_name):
    """The gate that keeps this from rotting: a new panel token ships with its sentence or fails here."""
    mod = importlib.import_module(f"mlframe.reporting.charts.{module_name}")
    builders = set(mod._TOKEN_BUILDERS)
    captions = set(mod._TOKEN_CAPTIONS)
    assert builders == captions, (
        f"{module_name}: tokens without a caption fragment: {sorted(builders - captions)}; " f"fragments with no token: {sorted(captions - builders)}"
    )


@pytest.mark.parametrize("module_name", COMPOSERS)
def test_fragments_are_sentences_not_labels(module_name):
    """A fragment is prose a reader can use, not a restatement of the token name."""
    mod = importlib.import_module(f"mlframe.reporting.charts.{module_name}")
    for token, fragment in mod._TOKEN_CAPTIONS.items():
        assert len(fragment) >= 60, f"{module_name}.{token}: fragment too short to explain anything"
        assert fragment.rstrip().endswith("."), f"{module_name}.{token}: fragment is not a sentence"


class TestTheJoiner:
    """``caption_for_tokens`` decides what a reader sees, so its edge cases are worth pinning."""

    FRAGS = {"A": "Panel A explains a thing.", "B": "Panel B explains another.", "C": "Panel C repeats."}

    def test_only_rendered_tokens_contribute(self):
        """The defect in one line: a token that was not rendered must not be described."""
        out = caption_for_tokens("Lead.", ["A"], self.FRAGS)
        assert "Panel A" in out and "Panel B" not in out

    def test_render_order_is_preserved(self):
        """The caption reads in the order the panels appear, not alphabetically."""
        out = caption_for_tokens("Lead.", ["B", "A"], self.FRAGS)
        assert out.index("Panel B") < out.index("Panel A")

    def test_a_repeated_token_is_described_once(self):
        """Two panels built from the same token share one explanation."""
        out = caption_for_tokens("Lead.", ["A", "A"], self.FRAGS)
        assert out.count("Panel A explains") == 1

    def test_an_unknown_token_is_skipped_not_raised(self):
        """A caption is a reading aid; refusing to build the figure over a missing sentence trades down."""
        out = caption_for_tokens("Lead.", ["A", "NOPE"], self.FRAGS)
        assert out == "Lead. Panel A explains a thing."

    def test_tail_lands_last(self):
        """A composer with a closing caveat keeps it after the per-panel sentences."""
        out = caption_for_tokens("Lead.", ["A"], self.FRAGS, tail="Caveat.")
        assert out.endswith("Caveat.")


def test_binary_caption_changes_with_the_template():
    """End-to-end on a real composer: two templates must not produce the same caption."""
    from mlframe.reporting.charts.binary import compose_binary_figure

    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) < 0.3).astype(int)
    s = np.clip(y * 0.4 + rng.random(n) * 0.6, 0.0, 1.0)

    ks_gain = compose_binary_figure(y, s, panels_template="KS GAIN").caption
    roc_pr = compose_binary_figure(y, s, panels_template="ROC PR").caption
    assert ks_gain != roc_pr
    assert "KS statistic" in ks_gain and "KS statistic" not in roc_pr
    assert "PR curve" in roc_pr and "PR curve" not in ks_gain
