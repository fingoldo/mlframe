"""What a feature selector returns when no feature is significant against its null.

An empty selection breaks ``transform`` and the final model's fit, so a selector never returns an empty support silently. When
no feature is significant it sets ``no_significant_features_`` on the instance, warns, and returns the policy's choice:

* ``"best_features"`` - the features that passed the significance gates (returned as selected); when none passed, the single best feature by MI, so the support is never empty (the default).
* ``"all_orig_features"`` - every original feature, so the final model gets a chance to find something on its own.

The policy is set per instance with ``no_significant_policy`` (default ``best_features``).
"""

from __future__ import annotations

NO_SIGNIFICANT_POLICIES = ("best_features", "all_orig_features")

_DEFAULT_POLICY = "best_features"


def resolve_no_significant_policy(instance: object) -> str:
    """The policy for one selector instance: its own ``no_significant_policy`` when set, else the default."""
    own: str | None = getattr(instance, "no_significant_policy", None)
    if own is None:
        return _DEFAULT_POLICY
    if own not in NO_SIGNIFICANT_POLICIES:
        raise ValueError(f"no_significant_policy must be one of {NO_SIGNIFICANT_POLICIES}; got {own!r}")
    return str(own)
