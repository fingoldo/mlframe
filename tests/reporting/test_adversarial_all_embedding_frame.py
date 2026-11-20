"""An all-embedding frame gets an annotated skip, not a 0-feature fit (audit 2026-11-20, finding 20-F6)."""
import numpy as np
import pandas as pd

from mlframe.reporting.charts._drift_adversarial import adversarial_validation
from mlframe.reporting.spec import AnnotationPanelSpec


def test_embedding_only_frame_is_skipped_with_a_note():
    """Every column holds per-row lists: the panel must be the annotated skip, not a ValueError from a 0-column fit."""
    rng = np.random.default_rng(0)
    n = 300
    emb = [list(rng.standard_normal(4)) for _ in range(n)]
    train = pd.DataFrame({"emb": emb})
    test = pd.DataFrame({"emb": emb[: n // 2]})
    fig = adversarial_validation(train, test)
    annotation = fig.panels[0][0]
    assert isinstance(annotation, AnnotationPanelSpec)
    assert "skipped" in annotation.text
