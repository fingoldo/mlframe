"""Helpers and defaults shared by the reporting siblings, kept in a leaf module.

Leaf module: the probabilistic, diagnostics and shared-API siblings import these, and this module imports nothing from the reporting package.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence
import numpy as np
import pandas as pd

try:
    from IPython.display import display as _ipython_display
except ImportError:  # pragma: no cover
    _ipython_display = None  # type: ignore[assignment]

logger = logging.getLogger("mlframe.training.reporting._reporting")


DEFAULT_PLOT_SAMPLE_SIZE = 5000


DEFAULT_REPORT_NDIGITS = 2


DEFAULT_CALIB_REPORT_NDIGITS = 2


DEFAULT_NBINS = 10


DEFAULT_FIGSIZE: tuple[int, int] = (15, 5)


DEFAULT_RANDOM_SEED = 42


def _reporting_field_default(field_name: str):
    """Field default of ReportingConfig, used to detect a left-at-default value the operator never customized. Cached on first call; None if the config is unavailable."""
    cache = _reporting_field_default.__dict__
    if field_name not in cache:
        try:
            from ..configs import ReportingConfig

            default = ReportingConfig.model_fields[field_name].default
        except (ImportError, KeyError) as e:
            # Narrowed from a bare `except Exception` (which also silently absorbed e.g. an AttributeError
            # from a pydantic version bump changing model_fields' shape) to the two genuine "config
            # unavailable" cases (import-cycle avoidance, a renamed/removed field), now logged so a real
            # bug is at least discoverable -- the result is memoized, so an unnoticed failure previously
            # disabled panel_emphasis="data_aware"'s default-detection for the rest of the process with
            # zero trace in the logs.
            logger.debug("_reporting_field_default(%r): config unavailable (%s); not cached, retried on the next call.", field_name, e)
            return None
        cache[field_name] = default
    return cache[field_name]


def _maybe_display(obj):
    """Display ``obj`` in Jupyter; no-op in scripts/CI.

    ``UnicodeEncodeError`` guard: outside Jupyter, IPython's ``display``
    falls back to printing ``repr(obj)`` to stdout. On a Windows console
    under the cp1251 / cp1252 code page, a frame carrying non-Latin-1 cell
    values (emoji, CJK, combining-mark cyrillic -- the ``weird_cat_content``
    fuzz axis) raises ``UnicodeEncodeError: 'charmap' codec can't encode``
    and crashes the whole report. The display is cosmetic, so swallow the
    encode error rather than abort the run. Surfaced by fuzz (4 combos with
    unicode-heavy categorical columns)."""
    if _ipython_display is not None:
        try:
            _ipython_display(obj)
        except UnicodeEncodeError:
            logger.debug("_maybe_display: skipped display() (console encoding cannot represent the object).")
        return
    # No IPython kernel (script / CI): the styled fairness / error-segmentation tables would vanish silently. Log a
    # plain text rendering instead so the only error-segmentation artifact survives in script logs (INV-52).
    text = _frame_to_text(obj)
    if text is not None:
        logger.info("%s", text)


def _frame_to_text(obj) -> Optional[str]:
    """Best-effort plain-text rendering of a DataFrame / pandas Styler for the no-IPython log fallback.

    Returns ``None`` for objects that are not frame-like (nothing to log)."""
    data = getattr(obj, "data", obj)  # a pandas Styler exposes the underlying frame at ``.data``
    to_string = getattr(data, "to_string", None)
    if not callable(to_string):
        return None
    try:
        return str(to_string())
    except Exception as exc:
        logger.debug("_frame_to_text: to_string() rendering failed: %s", exc)
        return None


def _style_with_caption(df, caption: str):
    """Return ``df.style.set_caption(caption)`` when the pandas Styler is
    usable, else the bare ``df``.

    ``DataFrame.style`` requires the optional ``jinja2`` dependency; pandas
    raises ``AttributeError: The '.style' accessor requires jinja2`` on import
    failure. The caption is purely cosmetic Jupyter-HTML decoration, so a
    headless / CI environment without jinja2 must NOT crash the whole report
    over it -- fall back to the plain frame (which ``_maybe_display`` /
    ``display`` render as a normal repr). Surfaced by fuzz c0005 (mlp combo,
    fairness-report styling) on a box without jinja2 installed.
    """
    try:
        return df.style.set_caption(caption)
    except (AttributeError, ImportError) as _exc:
        logger.debug("_style_with_caption fell back to its default after %s", type(_exc).__name__)
        return df


def _labels_are_arange(classes: Sequence | np.ndarray | None, probs: np.ndarray) -> bool:
    """True when class labels are exactly 0..K-1, so the batched-kernel per-class ICE
    (indicator ``targets == column_index``) is bit-identical to the report's per-class
    recompute (``targets == class_name``) and can be indexed by class_id. Non-0-indexed
    labels (e.g. [1,2,3] / strings) make the column index differ from the label -> unsafe."""
    k = probs.shape[1] if hasattr(probs, "shape") and probs.ndim == 2 else (len(classes) if classes is not None else 0)
    return bool(classes is not None and len(classes) == k and all(isinstance(c, (int, np.integer)) and int(c) == i for i, c in enumerate(classes)))


def _canonical_multilabel_y(targets) -> np.ndarray:
    """Coerce a multilabel-shaped target to a clean ``(N, K) ndarray``.

    Accepts the three shapes the suite encounters in production:
    1. ``np.ndarray`` 2-D ``(N, K)`` — passed through.
    2. ``pd.DataFrame`` — ``.values``.
    3. ``np.ndarray`` 1-D ``object`` dtype where each cell is a
       length-K array-like (the polars ``pl.List(pl.Int8)`` /
       ``pl.List(pl.Float32)`` -> pandas object roundtrip). Stacked
       to 2-D via ``np.stack``.
    Otherwise returns ``np.asarray(targets)`` unchanged (1-D / scalar
    / unhandled — caller decides).

    For float-typed cells (``pl.List(pl.Float32)`` source), the
    output is cast to ``int64`` via ``>= 0.5`` threshold:
    downstream metrics expect ``{0, 1}`` indicators.

    Extracted from the inline canonicalization at evaluation.py so
    ``mlframe.training.baselines.dummy`` and other consumers share
    one path.
    """
    if isinstance(targets, pd.DataFrame):
        targets_arr = targets.values
    elif isinstance(targets, np.ndarray):
        targets_arr = targets
    else:
        targets_arr = np.asarray(targets)

    if targets_arr.dtype == object and targets_arr.ndim == 1 and targets_arr.shape[0] > 0:
        first = targets_arr[0]
        if hasattr(first, "shape") or (hasattr(first, "__len__") and not isinstance(first, (str, bytes))):
            try:
                targets_arr = np.stack([np.asarray(c) for c in targets_arr], axis=0)
            except Exception as exc:
                # Unstackable (jagged / mixed shape) — leave as-is so the
                # caller's exception path surfaces the underlying issue.
                logger.debug("_canonical_multilabel_y: per-row stack failed, leaving target unstacked: %s", exc)
                return np.asarray(targets_arr)

    # Coerce float-indicator multilabel to int via threshold.
    if targets_arr.ndim == 2 and targets_arr.dtype.kind == "f":
        targets_arr = (targets_arr >= 0.5).astype(np.int64)

    return np.asarray(targets_arr)


def _unwrap_booster(model: Any) -> Any:
    """Peel common mlframe wrappers to reach the underlying lgb/xgb/catboost estimator that holds iteration history.

    Returns the first object in the wrapper chain exposing a recognised history accessor (or the original model).
    """
    seen = set()
    cur = model
    for _ in range(6):  # bounded: wrapper nesting is shallow (composite -> calibrated -> base)
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if any(hasattr(cur, a) for a in ("evals_result_", "evals_result", "get_evals_result")):
            return cur
        nxt = None
        for attr in ("base_estimator", "estimator", "best_estimator_", "model_", "_model", "regressor_", "classifier_"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                break
        if nxt is None:
            break
        cur = nxt
    return cur


def _canonicalize_split_names(split_names: list) -> dict:
    """Map booster eval-set names to ``compose_training_curve_figure``'s canonical ``train`` / ``val`` keys.

    Keys already recognised by the alias set (``train`` / ``valid`` / ``validation`` / ``learn`` / ...) pass through.
    lightgbm's positional ``valid_0`` / ``valid_1`` / ... default names are not in that set; by the mlframe convention
    that the FIRST eval set is the training fold and the LAST is the holdout, the lowest-index ``valid_N`` maps to
    ``train`` and the highest to ``val``. When an explicit train key (e.g. lgb's ``training``) is already present, a
    lone positional ``valid_N`` is the holdout and maps to ``val`` (not a second train). Leftovers keep their name.

    A LONE positional entry with no explicit train key is the ``val`` set, not ``train``: mlframe's own
    ``_setup_eval_set`` (the path every default suite fit goes through) registers exactly ONE eval set for
    lgb/xgb/cb when no shard evaluation is configured, and that one set is always the VALIDATION data
    (``fit_params["eval_set"] = (val_df, val_target)`` / ``[(val_df, val_target)]`` -- never a train re-eval).
    Mapping it to ``train`` produced a training-curve chart with a single series labelled "train" that was
    actually the validation curve early stopping was watching -- and because ``_metric_panel``'s "val optimum"
    star marker requires BOTH ``train`` and ``val`` present, the mislabelling also silently dropped the star.
    """
    out: dict = {}
    recognised_train = {"train", "training", "learn"}
    recognised_val = {"val", "valid", "validation", "test", "eval", "holdout"}
    has_train = False
    has_val = False
    positional = []  # (index, original_name) for the lgb ``valid_<N>`` / xgb ``validation_<N>`` families
    for name in split_names:
        low = str(name).strip().lower()
        if low in recognised_train:
            has_train = True
            continue
        if low in recognised_val:
            has_val = True
            continue
        for prefix in ("valid_", "validation_"):
            if low.startswith(prefix) and low[len(prefix) :].isdigit():
                positional.append((int(low[len(prefix) :]), name))
                break
    if positional:
        positional.sort()
        if has_train:
            # An explicit train split already exists; the (lone or last) positional is the holdout.
            if not has_val:
                out[positional[-1][1]] = "val"
        elif len(positional) > 1:
            out[positional[0][1]] = "train"
            out[positional[-1][1]] = "val"
        else:
            # A single positional entry with no train key anywhere: mlframe's own default fit path never
            # registers a lone TRAIN-only eval set, so this one is the validation data.
            out[positional[0][1]] = "val"
    return out


def model_name_for_title(target_type: Any) -> str:
    """Short report-title tag from the target_type (combined-HTML page heading)."""
    return str(target_type) if target_type else "Model"


_SHIM_CLASS_SUFFIXES = ("WithDatasetReuse", "WithDMatrixReuse", "WithReuse", "WithEvalSetScaling")


def display_estimator_name(name: str) -> str:
    """Strip internal shim-wrapper suffixes from an estimator class name for public display.

    ``LGBMRegressorWithDatasetReuse`` -> ``LGBMRegressor``. Applied repeatedly so stacked shims collapse. A bare
    suffix (the whole name is a shim marker) is left untouched -- there is no real estimator name to recover.
    """
    out = str(name)
    changed = True
    while changed:
        changed = False
        for suffix in _SHIM_CLASS_SUFFIXES:
            if out.endswith(suffix) and len(out) > len(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def _fi_png_path(base: str) -> str:
    """``base`` resolved to its .png destination under the active per-format layout.

    Composing ``base + "_fiplot.png"`` wrote the plot beside the png/ directory every other chart
    went into, which is how a production output folder ended up with loose files at its top level.
    """
    from mlframe.reporting.renderers.save import resolve_output_path

    return resolve_output_path(base + "_fiplot", "matplotlib", "png", multi_output=False)


def _account_panel_render(metrics: Any, target_type: Any, rendered_tag: Any, panel_failures: list, plot_file: Any) -> None:
    """Record in ``metrics["charts"]`` whether the multi-target panel grid rendered, failed, or had nothing to render.

    INV-48: account which panel grids rendered so a run can assert chart presence. The no-crash contract holds -- a
    failed render is logged + swallowed inside the dispatcher and recorded as "failed". INV-56: also stamp the base path
    so the accounting points at the on-disk artifact, not just a tag.

    "Nothing to render" is NOT a failure. ``render_multi_target_panels`` returns None both for "nothing rendered because
    the branch matched and crashed" and for "no branch matched at all", and the second is the normal case for a
    regression target: ``binary_panels`` defaults non-empty, so the guard opens for every report, and a regression run
    then recorded ``regression_panels`` as FAILED on every single report -- for a grid that does not exist. It is
    recorded as skipped instead, through the one helper every writer of ``charts["skipped"]`` shares: this path used to
    append to a list there while the diagnostics assigned into a dict, which ended a production run with a TypeError.
    """
    if not isinstance(metrics, dict):
        return
    charts = metrics.setdefault("charts", {"saved": [], "failed": []})
    _target_type_lc = (target_type or "").lower()
    which = "binary" if _target_type_lc == "binary_classification" else (_target_type_lc if _target_type_lc != "" else "panels")
    if rendered_tag:
        charts["saved"].append(f"{rendered_tag}_panels")
        charts.setdefault("paths", []).append(f"{plot_file}_{rendered_tag}_panels")
    elif panel_failures:
        charts["failed"].append(f"{which}_panels")
        # Distinguishes an actual render-time exception (branch matched, then crashed) from a plain no-op -- a batch
        # run needs to count how many reports dropped a whole panel set.
        charts.setdefault("panel_exceptions", []).extend(panel_failures)
    else:
        from mlframe.reporting.shared import record_skipped as _record_skipped

        _record_skipped(charts, f"{which}_panels", "no panel grid applies to this target type")


def _stamp_binary_decile_table(metrics: dict, targets: Any, probs: Any) -> None:
    """Store ``binary_decile_table`` in ``metrics`` from the positive-class score; best-effort, a failure leaves the key absent."""
    try:
        from mlframe.reporting.charts.binary import binary_decile_table

        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
            score = probs_arr[:, 1]
        elif probs_arr.ndim == 1:
            score = probs_arr
        elif probs_arr.ndim == 2 and probs_arr.shape[1] == 1:
            score = probs_arr.ravel()
        else:
            return
        metrics["binary_decile_table"] = binary_decile_table(np.asarray(targets).ravel(), score)
    except Exception:  # best-effort: metrics["binary_decile_table"] simply stays absent
        logger.exception("binary_decile_table computation failed; continuing.")


def _train_envelope_bounds(y_train_envelope_stats) -> tuple:
    """Return ``(y_min, y_max, y_std)`` of the train envelope stats, or ``(None, None, None)`` when they are absent."""
    if y_train_envelope_stats is None:
        return None, None, None
    return (
        getattr(y_train_envelope_stats, "y_min", None),
        getattr(y_train_envelope_stats, "y_max", None),
        getattr(y_train_envelope_stats, "y_std", None),
    )


def _slugify_class(name: str) -> str:
    """Filesystem-safe class-name slug for per-class plot filenames."""
    try:
        from pyutilz.strings import slugify

        slug = slugify(name)
    except Exception as e:
        logger.debug("slugify() failed for name=%r: %s", name, e)
        slug = ""
    if not slug:
        slug = "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")
    return slug if slug != "" else "class"
