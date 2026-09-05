"""Blind aggregation: read the tables before knowing which row is the author's own method.

The benchmark is designed and run by the author of one of the arms it judges. No fabrication is needed for
that to bias a result -- it is enough to look at a leaderboard, notice one's own method third, and go
looking for a reason the two above it had an unfair advantage. That search is honest work, and it is
performed asymmetrically, which is precisely what makes it bias.

The mitigation is mechanical rather than moral. Arms are relabelled `arm_a`, `arm_b`, ... in a mapping the
aggregation and plotting code never reads; the analysis is run, the numbers are committed, and only then is
the mapping applied. Any explanation found while blind applies to whichever arm it turns out to be.

The label assignment is deterministic given the run's arm set, so a blinded table can be regenerated
exactly. It is deliberately NOT alphabetical: `arm_a` would be `ace` every time, and a run's second table
would be readable from memory. The order is a hash of the arm name and a per-run salt, so labels shuffle
between runs while staying stable inside one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["blind_labels", "apply_blinding", "write_mapping", "load_mapping", "unblind_text", "MAPPING_SUFFIX", "NEVER_BLINDED"]

MAPPING_SUFFIX = ".blinding.json"
# The null hypothesis is never blinded: every contrast is stated against it, so hiding it would make the
# tables unreadable without hiding anything an author could act on -- `all-features` is nobody's method.
NEVER_BLINDED: Tuple[str, ...] = ("all-features",)


def _label_for(index: int) -> str:
    """Return the blinded label for a position: ``arm_a``, ``arm_b``, ..., ``arm_aa``."""
    letters = string.ascii_lowercase
    if index < len(letters):
        return f"arm_{letters[index]}"
    first, second = divmod(index, len(letters))
    return f"arm_{letters[first - 1]}{letters[second]}"


def blind_labels(arms: Iterable[str], salt: str = "") -> Dict[str, str]:
    """Return ``{real arm name: blinded label}``, deterministic in the arm set and the salt.

    Args:
        arms: The arms appearing in a run.
        salt: Per-run salt. Two runs with different salts assign different labels to the same arm, which is
            what stops a reader from carrying yesterday's mapping into today's table from memory.

    Returns:
        A mapping that leaves :data:`NEVER_BLINDED` names unchanged.
    """
    real = sorted({str(arm) for arm in arms})
    blinded = [arm for arm in real if arm not in NEVER_BLINDED]
    ordered = sorted(blinded, key=lambda name: hashlib.blake2b(f"{salt}::{name}".encode(), digest_size=8).hexdigest())
    mapping = {name: _label_for(position) for position, name in enumerate(ordered)}
    mapping.update({name: name for name in real if name in NEVER_BLINDED})
    return mapping


def apply_blinding(records: Sequence[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return copies of ``records`` with arm names replaced by their blinded labels.

    Copies rather than in-place edits: the caller keeps the real records, and an analysis that accidentally
    reached for them would then be visibly reading unblinded data rather than silently doing so.

    Raises:
        KeyError: If a record names an arm the mapping does not cover, which means the mapping was built
            from a different run and the blinding would be inconsistent across the table.
    """
    out: List[Dict[str, Any]] = []
    for record in records:
        arm = str(record.get("arm", ""))
        if arm not in mapping:
            raise KeyError(f"arm {arm!r} is not in the blinding mapping; it was built from a different run")
        blinded = dict(record)
        blinded["arm"] = mapping[arm]
        out.append(blinded)
    return out


def write_mapping(results_path: str, mapping: Dict[str, str]) -> str:
    """Write the mapping beside its results file and return the path.

    Kept in its own file, never inside the results, so the aggregation code can be given the records without
    also being given the key.
    """
    path = f"{results_path}{MAPPING_SUFFIX}"
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_mapping(results_path: str) -> Optional[Dict[str, str]]:
    """Return the mapping beside a results file, or ``None`` when the run was not blinded."""
    path = f"{results_path}{MAPPING_SUFFIX}"
    try:
        with open(path, encoding="utf-8") as handle:
            return {str(key): str(value) for key, value in json.load(handle).items()}
    except (OSError, ValueError) as exc:
        logger.info("no blinding mapping at %s: %s", path, exc)
        return None


def unblind_text(text: str, mapping: Dict[str, str]) -> str:
    """Return ``text`` with blinded labels replaced by the real arm names.

    Applied once, to a committed report, after the numbers are fixed. Longer labels are substituted first so
    ``arm_a`` cannot consume the prefix of ``arm_aa``.
    """
    reverse = {label: name for name, label in mapping.items() if label != name}
    for label in sorted(reverse, key=len, reverse=True):
        text = text.replace(label, reverse[label])
    return text
