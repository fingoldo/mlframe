"""What a feature selector returns when no feature is significant against its null.

An empty selection breaks ``transform`` and the final model's fit, so a selector never returns an empty support silently. When
no feature is significant it sets ``no_significant_features_`` on the instance, warns, and returns the policy's choice:

* ``"best_feature"`` - the single best feature by MI (a cheap, short final model; the default).
* ``"all_orig_features"`` - every original feature, so the final model gets a chance to find something on its own.

The policy is a global setting; an instance can override it with ``no_significant_policy``.
"""

from __future__ import annotations

NO_SIGNIFICANT_POLICIES = ("best_feature", "all_orig_features")

_policy = "best_feature"


def get_no_significant_policy() -> str:
    """The global policy used when an instance does not override it."""
    return _policy


def set_no_significant_policy(policy: str) -> None:
    """Set the global policy; raises ``ValueError`` on an unknown name."""
    global _policy
    if policy not in NO_SIGNIFICANT_POLICIES:
        raise ValueError(f"no_significant policy must be one of {NO_SIGNIFICANT_POLICIES}; got {policy!r}")
    _policy = policy


def resolve_no_significant_policy(instance) -> str:
    """The policy for one selector instance: its own ``no_significant_policy`` when set, else the global one."""
    own = getattr(instance, "no_significant_policy", None)
    if own is None:
        return _policy
    if own not in NO_SIGNIFICANT_POLICIES:
        raise ValueError(f"no_significant_policy must be one of {NO_SIGNIFICANT_POLICIES}; got {own!r}")
    return own
