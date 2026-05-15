from typing import Any, Sequence, List, Optional, Callable, Dict
import logging


def read_glove_embeddings(
    embeddings_dict: Optional[Dict[str, "Any"]] = None,
    fpath: Optional[str] = "glove.6B.50d.txt",
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

    with open(fpath, encoding="utf8") as f:
        for line in tqdm(f, miniters=100):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
            except (ValueError, TypeError):
                continue
            embeddings_dict[word] = coefs
    logging.debug("Total %s word vectors.", len(embeddings_dict))
    return embeddings_dict
