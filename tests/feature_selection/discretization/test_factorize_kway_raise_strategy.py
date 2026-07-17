"""Wave 9.1 loop-iter-13 regression: ``_apply_factorize_kway`` MUST raise
under ``unknown_strategy='raise'`` when an intermediate chain lookup
hits a ``-1`` sentinel.

Pre-fix: the raise check ran ONLY after the loop finished, on the final
``running`` array. But the loop body computes
``pre_prune = running + vals_next * running_nuniq`` for the next step.
When ``running[i] == -1`` (unseen prefix), pre_prune becomes a negative
or small index that Python's ``np.ndarray[idx]`` wraps via standard
negative-index semantics, returning a real class code from the tail
of ``chain_lookups[step-1]``. The -1 sentinel is wiped out before the
post-loop check sees it -> contract violated, no error raised.

Concrete repro (3-way recipe, unknown_strategy='raise', input
combination NOT seen at fit):
  step 1: running = [-1]
  step 2: pre_prune = [-1] + 0*2 = [-1]
          chain_lookups[1][-1] = chain_lookups[1][last] = 1  (a valid class)
          running = [1]    <-- sentinel wiped!
  post-loop: (running < 0).any() is False -> NO raise
  output = [1]              <-- silently wrong class

Severity: medium. Only affects users who explicitly opt into
``unknown_strategy='raise'`` on k>=3-way factorize recipes, but for them
it's a violated documented contract: they wanted an error and got a
silently corrupted class fed into the downstream model.

Fix: add the ``(running < 0).any()`` check after EVERY chain step's
``running = chain_lookups[...]`` assignment, not just after the loop.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest


def test_raise_on_unseen_prefix_at_chain_step_1():
    """3-way recipe with chain_lookups[0] containing -1 (unseen 2-way
    prefix); transform input that lands on that slot must raise.
    """
    from mlframe.feature_selection.filters.engineered_recipes import (
        EngineeredRecipe,
        _apply_factorize_kway,
    )

    r = EngineeredRecipe(
        name="kway_test",
        kind="factorize",
        src_names=("a", "b", "c"),
        factorize_nbins=(2, 2, 2),
        unknown_strategy="raise",
        extra={
            "chain_lookups": [
                np.array([0, 1, -1, 0], dtype=np.int64),  # pre_prune=2 unseen
                np.array([0, 1, 0, 1], dtype=np.int64),
            ],
            "chain_nuniqs": [2, 2],
        },
    )
    X = pd.DataFrame({"a": [0], "b": [1], "c": [0]})  # pre_prune step1 = 2
    with pytest.raises(ValueError, match="unseen prefix"):
        _apply_factorize_kway(r, X)


def test_clip_and_sentinel_paths_still_work():
    """Negative control: clip and sentinel strategies should NOT raise -
    they handle unseen combinations gracefully.
    """
    from mlframe.feature_selection.filters.engineered_recipes import (
        EngineeredRecipe,
        _apply_factorize_kway,
    )

    base = EngineeredRecipe(
        name="kway_test",
        kind="factorize",
        src_names=("a", "b", "c"),
        factorize_nbins=(2, 2, 2),
        unknown_strategy="raise",
        extra={
            "chain_lookups": [
                np.array([0, 1, -1, 0], dtype=np.int64),
                np.array([0, 1, 0, 1], dtype=np.int64),
            ],
            "chain_nuniqs": [2, 2],
        },
    )
    X = pd.DataFrame({"a": [0], "b": [1], "c": [0]})
    for strat in ("clip", "sentinel"):
        out = _apply_factorize_kway(replace(base, unknown_strategy=strat), X)
        assert out is not None and len(out) == 1


def test_raise_on_unseen_at_chain_step_2():
    """4-way recipe where step 2 produces -1 - the original post-loop
    check would miss it because step 3 would wrap the -1 to a positive
    class via negative-index semantics.
    """
    from mlframe.feature_selection.filters.engineered_recipes import (
        EngineeredRecipe,
        _apply_factorize_kway,
    )

    r = EngineeredRecipe(
        name="kway_test_4way",
        kind="factorize",
        src_names=("a", "b", "c", "d"),
        factorize_nbins=(2, 2, 2, 2),
        unknown_strategy="raise",
        extra={
            "chain_lookups": [
                np.array([0, 1, 2, 0], dtype=np.int64),  # 2-way: 3 unique
                np.array([0, 1, -1, 0, 1, 0], dtype=np.int64),  # 3-way w/ unseen at slot 2
                np.array([0, 1, 0, 1], dtype=np.int64),  # 4-way
            ],
            "chain_nuniqs": [3, 2, 2],
        },
    )
    # Input a=0,b=1: step1 pre_prune=2, running=chain_lookups[0][2]=2
    # Step2: c=0, pre_prune=2+0*3=2, running=chain_lookups[1][2]=-1 (UNSEEN)
    # Pre-fix: step 3 d=0, pre_prune=-1+0*2=-1,
    #   chain_lookups[2][-1]=1, returns [1] without raising.
    X = pd.DataFrame({"a": [0], "b": [1], "c": [0], "d": [0]})
    with pytest.raises(ValueError, match="unseen prefix at chain step 2"):
        _apply_factorize_kway(r, X)
