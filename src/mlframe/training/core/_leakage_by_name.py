"""Features named like the targets: a post-outcome column the correlation-based leakage check cannot see.

``analyze_feature_distribution`` flags leakage by |corr(feature, y)| >= 0.99. A column that is only KNOWN after the
outcome need not correlate with it at all: a production run fed ``target_end_date`` (decomposed into
``target_end_date_month`` / ``_sin`` / ...) to models whose targets were ``target_total_hours`` / ``target_total_charge``,
and BaselineDiagnostics ranked ``target_end_date_month`` in their top-3 features. The contract end date is known only
after the job is done, so those metrics are optimistic.

Naming is the available signal: the targets of a run share a prefix (``target_`` here), and a FEATURE carrying that same
prefix was named by the same convention, so it is at least worth a warning. Flagging is deliberate, never an auto-drop:
a project may legitimately name a feature this way.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

# Shortest prefix worth trusting: two characters plus the separator, so "y_" style namespaces are covered but an
# accidental one-letter overlap between unrelated names is not.
_MIN_PREFIX_LEN = 3
_SEPARATORS = ("_", "-", ".")


def _common_prefix(names: Sequence[str]) -> str:
    """Longest common prefix of ``names`` truncated at its last separator, or "" when there is no usable one."""
    if len(names) < 2:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while prefix and not name.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            return ""
    cut = max((prefix.rfind(sep) for sep in _SEPARATORS), default=-1)
    prefix = prefix[: cut + 1] if cut >= 0 else prefix
    return prefix if len(prefix) >= _MIN_PREFIX_LEN else ""


def target_named_features(feature_names: Iterable[str], target_names: Sequence[str]) -> List[str]:
    """Feature columns that carry the targets' shared naming prefix (or a target's own name) without being targets."""
    targets = [str(t) for t in target_names if t]
    if not targets:
        return []
    target_set = set(targets)
    prefix = _common_prefix(sorted(targets))
    derived_prefixes = tuple(f"{t}{sep}" for t in targets for sep in _SEPARATORS)
    out: List[str] = []
    for name in feature_names:
        col = str(name)
        if col in target_set:
            continue
        if (prefix and col.startswith(prefix)) or col.startswith(derived_prefixes):
            out.append(col)
    return out
