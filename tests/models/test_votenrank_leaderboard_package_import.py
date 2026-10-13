"""The votenrank leaderboard subpackage re-exports ``Leaderboard`` from its implementation module.

The package ``__init__`` had no test importing it, so its single import line showed as uncovered and the changed-file coverage
gate failed on any commit that touched it.
"""

from __future__ import annotations


def test_leaderboard_reexported_from_subpackage():
    """``Leaderboard`` imported from the package is the implementation class."""
    from mlframe.votenrank.leaderboard import Leaderboard
    from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard as LeaderboardImpl

    assert Leaderboard is LeaderboardImpl
