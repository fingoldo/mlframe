"""Text-embedding loading helpers."""
from __future__ import annotations

from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def read_glove_embeddings(
    embeddings_dict: Optional[Dict[str, "Any"]] = None,
    fpath: str = "glove.6B.50d.txt",
) -> Dict[str, "Any"]:
    """Load GloVe vectors into a dict keyed by word.

    ``embeddings_dict`` is optional; pass an existing dict to merge into, or
    ``None`` to start fresh. The populated dict is always returned (previous
    revision mutated the argument in place and returned ``None``, which was
    easy to misuse).
    """
    import numpy as np
    from pyutilz.system import tqdmu as tqdm

    if embeddings_dict is None:
        embeddings_dict = {}

    skipped = 0
    with open(fpath, encoding="utf8") as f:
        for line in tqdm(f, miniters=100):
            values = line.split()
            if not values:
                # A blank line used to raise IndexError on values[0] and abort the whole file (audit 29-F18).
                skipped += 1
                continue
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
            except (ValueError, TypeError):
                # Multi-word keys and other non-numeric rows are skipped, and counted so the drop is visible (audit 29-F18).
                skipped += 1
                continue
            embeddings_dict[word] = coefs
    if skipped:
        logger.warning("read_glove_embeddings: skipped %d unparseable line(s) in %s", skipped, fpath)
    logger.debug("Total %s word vectors.", len(embeddings_dict))
    return embeddings_dict
