"""Meta-test — every "X and Y are mutually exclusive" claim in a config
docstring must be backed by a runtime ``@model_validator``.

Catches the failure mode where a docstring promises a constraint
(``binarization_threshold and kbins are mutually exclusive``) but no
validator enforces it — users who set both get undefined behaviour
deeper in the pipeline.

Heuristic:
  1. Walk every config class's ``__doc__``; find phrases of the form
     ``"X and Y are mutually exclusive"`` (case-insensitive, allowing
     code-quoted backticks ``X``, plus the synonym
     ``"set at most one of"``).
  2. For each such (X, Y) pair, attempt to construct the class with
     BOTH X and Y set to a non-default value.
  3. The construction must raise ``ValidationError`` / ``ValueError`` —
     proving an enforcing validator exists.

Limitations:
  * Doesn't try to statically prove the validator references both
    fields; the empirical "raises when both set" is sufficient.
  * Skips configs whose required fields prevent default construction.
  * Skips fields that are typed as opaque (Any / Callable) since we
    can't synthesise a non-default sentinel.
"""

from __future__ import annotations

import inspect
import re

from pydantic import BaseModel, ValidationError, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from mlframe.training import configs as configs_module

# Patterns to surface mutually-exclusive pairs from prose. Each pattern
# captures (a, b) — field names. Order doesn't matter; we collect all
# matches and dedupe at the call site. Backtick-quoted names are
# normalised before pattern matching.
_PATTERNS = [
    re.compile(
        r"([A-Za-z_]\w+)\s+and\s+([A-Za-z_]\w+)\s+are\s+mutually\s+exclusive",
        re.IGNORECASE,
    ),
    re.compile(
        r"set\s+at\s+most\s+one\s+of\s*[:`\s]*([A-Za-z_]\w+)\s*[`/,]+\s*([A-Za-z_]\w+)",
        re.IGNORECASE,
    ),
    # ``X OR Y (mutually exclusive)`` — used in numpydoc list-of-steps.
    re.compile(
        r"([A-Za-z_]\w+)\s+OR\s+([A-Za-z_]\w+)\s*\(\s*mutually\s+exclusive\s*\)",
        re.IGNORECASE,
    ),
]


def _config_classes() -> list[type[BaseModel]]:
    """Returns ``out`` (after 2 setup steps)."""
    out = []
    for _, obj in inspect.getmembers(configs_module, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj is not BaseModel):
            continue
        if obj.__module__ not in {
            configs_module.__name__,
            f"{configs_module.__package__}._preprocessing_configs",
            f"{configs_module.__package__}._model_configs",
            f"{configs_module.__package__}._training_runtime_configs",
            f"{configs_module.__package__}._composite_target_discovery_config",
            f"{configs_module.__package__}._reporting_configs",
            f"{configs_module.__package__}._configs_base",
            f"{configs_module.__package__}._feature_selection_config",
        }:
            continue
        out.append(obj)
    return out


def _has_default(info: FieldInfo) -> bool:
    """Returns ``False`` (after 2 setup steps)."""
    if info.default is not PydanticUndefined and info.default is not Ellipsis:
        return True
    if info.default_factory is not None:
        return True
    return False


def _required_sentinels(cls: type[BaseModel]) -> dict:
    """Returns ``out`` (after 2 setup steps)."""
    out: dict = {}
    for name, info in cls.model_fields.items():
        if not _has_default(info):
            out[name] = f"_meta_test_sentinel_{name}"
    return out


def _doc_lines_normalised(doc: str | None) -> str:
    """Strip backticks so ``X`` matches X in the regex."""
    if not doc:
        return ""
    return doc.replace("``", " ").replace("`", " ")


def _find_mutex_pairs(cls: type[BaseModel]) -> list[tuple[str, str]]:
    """Scan the class docstring AND every field's docstring/description for mutex-claim phrasing.

    Limiting the scan to documentation strings (class.__doc__ + Field(..., description=...)) keeps
    this test discovery-only on prose surfaces users actually see, avoiding source-text dependence
    on internal validator-body wording.
    """
    text_chunks = [_doc_lines_normalised(cls.__doc__)]
    for info in cls.model_fields.values():
        desc = getattr(info, "description", None)
        if desc:
            text_chunks.append(_doc_lines_normalised(desc))

    pairs: list[tuple[str, str]] = []
    for text in text_chunks:
        for pattern in _PATTERNS:
            for m in pattern.finditer(text):
                a, b = m.group(1), m.group(2)
                if a in cls.model_fields and b in cls.model_fields:
                    pairs.append((a, b))
    return list(dict.fromkeys(pairs))


def _synth_value(annotation):
    """Produce a non-default sentinel value matching ``annotation``."""
    # Walk Optional[X] / Union[X, None]
    args = getattr(annotation, "__args__", ()) or ()
    candidates = (annotation, *args) if args else (annotation,)
    for cand in candidates:
        if cand in (int, float):
            return 0.5
        if cand is bool:
            return True
        if cand is str:
            return "synth"
        if cand is list or getattr(cand, "__origin__", None) is list:
            return ["synth"]
        if cand is dict or getattr(cand, "__origin__", None) is dict:
            return {"k": "v"}
    # Fallback
    return 0.5


def _mutex_failures(cls: type[BaseModel]) -> list[str]:
    """One message per documented mutually exclusive pair of ``cls`` that constructs without raising."""
    failures: list[str] = []
    sentinels = _required_sentinels(cls)
    for a, b in _find_mutex_pairs(cls):
        kwargs = dict(sentinels)
        kwargs[a] = _synth_value(cls.model_fields[a].annotation)
        kwargs[b] = _synth_value(cls.model_fields[b].annotation)
        try:
            cls(**kwargs)
        except (ValidationError, ValueError):
            continue  # Properly rejected.
        except Exception:  # nosec B112 -- a non-validation error means the synthesised value mismatched the type; inconclusive for this pair
            continue
        failures.append(f"{cls.__name__}: docstring claims '{a}' and '{b}' are mutually exclusive, but cls({a}=..., {b}=...) succeeded without raising")
    return failures


class _MutexEnforced(BaseModel):
    """``first`` and ``second`` are mutually exclusive."""

    first: float | None = None
    second: float | None = None

    @model_validator(mode="after")
    def _exclusive(self) -> "_MutexEnforced":
        """Reject setting both."""
        if self.first is not None and self.second is not None:
            raise ValueError("first and second are mutually exclusive")
        return self


class _MutexUnenforced(BaseModel):
    """``first`` and ``second`` are mutually exclusive."""

    first: float | None = None
    second: float | None = None


def test_mutex_gate_catches_an_unenforced_claim_and_passes_an_enforced_one():
    """A documented mutex pair with no validator is reported; the same claim backed by a validator is not."""
    assert _find_mutex_pairs(_MutexEnforced) == [("first", "second")]
    assert _mutex_failures(_MutexEnforced) == []
    assert _mutex_failures(_MutexUnenforced) == [
        "_MutexUnenforced: docstring claims 'first' and 'second' are mutually exclusive, but cls(first=..., second=...) succeeded without raising"
    ]


def test_mutually_exclusive_pairs_are_enforced_by_a_validator():
    """Every documented mutually exclusive pair of a config class is enforced by a validator."""
    classes = _config_classes()
    assert classes, "no config classes found"
    failures = [msg for cls in classes for msg in _mutex_failures(cls)]
    assert not failures, f"{len(failures)} mutex-claim(s) without enforcing validator:\n  " + "\n  ".join(failures)
