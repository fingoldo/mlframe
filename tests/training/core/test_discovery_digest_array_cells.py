"""The discovery inputs digest must hash a val frame whose cells are arrays (an embedding column).

``pd.util.hash_pandas_object`` rejects array-valued cells with ``TypeError: unhashable type: 'numpy.ndarray'``, which failed the
composite discovery for every fuzz combo carrying an embedding column. The digest must also change when an embedding value changes,
otherwise a discovery cache would replay specs computed on different validation data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._phase_composite_discovery_gates import discovery_inputs_digest

_VAL_Y = np.array([0.0, 1.0, 0.0])


def _val_frame_with_embedding(first_row_value: float) -> pd.DataFrame:
    """Three-row frame with one numeric column and one object column whose cells are 4-dim float arrays.

    The digest only reads the val frame when targets accompany it, so every caller passes ``val_y`` as well."""
    emb = [np.array([first_row_value, 0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0, 10.0])]
    return pd.DataFrame({"num": [1.0, 2.0, 3.0], "emb": pd.Series(emb, dtype=object)})


def test_digest_accepts_array_valued_cells():
    """A val frame with an embedding column digests without raising."""
    digest = discovery_inputs_digest(val_df=_val_frame_with_embedding(0.5), val_y=_VAL_Y)

    assert isinstance(digest, str) and len(digest) > 0


def test_digest_changes_when_an_embedding_value_changes():
    """Changing one element of one embedding changes the digest, so a cached discovery is not replayed on different val data."""
    before = discovery_inputs_digest(val_df=_val_frame_with_embedding(0.5), val_y=_VAL_Y)
    after = discovery_inputs_digest(val_df=_val_frame_with_embedding(0.75), val_y=_VAL_Y)

    assert before != after


def test_digest_is_stable_for_identical_embedding_values():
    """Equal embedding contents give equal digests across separately built frames."""
    first = discovery_inputs_digest(val_df=_val_frame_with_embedding(0.5), val_y=_VAL_Y)
    second = discovery_inputs_digest(val_df=_val_frame_with_embedding(0.5), val_y=_VAL_Y)

    assert first == second
