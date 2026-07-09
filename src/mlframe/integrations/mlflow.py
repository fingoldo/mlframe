"""MLflow logging helpers: flatten/report sklearn classification_report dicts, embed HTML in the run UI, and idempotently get-or-create runs by name."""

from __future__ import annotations

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple
import html
import re
from enum import Enum
import mlflow

# Matches ``scheme://user:password@`` prefixes so we can scrub them before
# printing or logging exception text that bubbled up from mlflow.
_USERINFO_RE = re.compile(r"(?i)([a-z][a-z0-9+\-.]*://)[^/@\s]+@")


def _strip_userinfo(text: object) -> str:
    """Redact any ``scheme://user:password@`` userinfo from ``text`` before it is logged, so tracking-server credentials embedded in a raised exception don't leak into log files."""
    return _USERINFO_RE.sub(r"\1***@", str(text))

########################################################################################################################################################################################################################################
# MLFLOW
########################################################################################################################################################################################################################################

def flatten_classification_report(cr: dict, separate_metrics=("accuracy","balanced_accuracy","brier_score_loss","roc_auc"),source:str="")->dict:
    """Flatten a sklearn ``classification_report(output_dict=True)`` into a single flat dict of MLflow-metric-name -> value.

    Scalar metrics named in ``separate_metrics`` (e.g. ``accuracy``) are popped out under ``source + metric``; every
    remaining per-class / averaged sub-dict is expanded to ``source + <"class "+label or "macro avg"/"weighted avg">
    + "_" + metric``, matching MLflow's flat metric-name requirement (no nested structures).
    """
    res = {}
    for metric in separate_metrics:
        if metric in cr:
            res[source + metric] = cr.pop(metric)
    for class_or_avg, metrics_dict in cr.items():
        prefix = class_or_avg if class_or_avg in ("macro avg", "weighted avg") else "class " + str(class_or_avg)
        for metric, value in metrics_dict.items():
            res[source + prefix + "_" + metric] = value
    return res

def log_classification_report_to_mlflow(cr: dict, step: int,separate_metrics=("accuracy",),source:str=""):
    """Logging all metrics from a dict-like classification_report as flat MLFlow entries."""

    for metric in separate_metrics:
        if metric in cr:
            mlflow.log_metric(source + metric, cr.pop(metric), step=step)
    for class_or_avg, metrics_dict in cr.items():
        prefix = class_or_avg if class_or_avg in ("macro avg", "weighted avg") else "class " + str(class_or_avg)
        for metric, value in metrics_dict.items():
            mlflow.log_metric(source + prefix + "_" + metric, value, step=step)

def embed_website_to_mlflow(url:str,fname:str="url",extension:str='.html',width:int=700,height:int=450)->None:
    """Creates a html file with desired url embedded to be shown nicely in MLFlow UI."""

    safe_url = html.escape(url, quote=True)
    safe_width = int(width)
    safe_height = int(height)
    website_embed = f"""<!DOCTYPE html>
    <html>
    <iframe src="{safe_url}" style='width: {safe_width}px; height: {safe_height}px' sandbox='allow-same-origin allow-scripts'>
    </iframe>
    </html>"""

    if fname.lower().endswith(extension.lower()):
        extension = ""

    with open(fname + extension, "w", encoding="utf-8") as f:
        f.write(website_embed)

def get_or_create_mlflow_run(run_name: str, parent_run_id: Optional[str] = None, experiment_name: Optional[str] = None, experiment_id: Optional[str] = None, tags: Optional[dict] = None) -> Tuple[object, bool]:
    """Tries to find a run by name within current mlflow experiment.
    If not found, creates new one.
    """
    if tags is None:
        tags = {}
    # Escape embedded double-quotes/backslashes so user-controlled run_name can't
    # break out of the mlflow DSL filter literal.
    def _dsl_escape(s: object) -> str:
        """Escape backslashes and double-quotes so ``s`` is safe to embed inside an mlflow search-runs DSL string literal."""
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    filter_string = f'run_name = "{_dsl_escape(run_name)}"'
    if parent_run_id:
        filter_string += f' and tag.mlflow.parentRunId = "{_dsl_escape(parent_run_id)}"'

    runs = mlflow.search_runs(experiment_names=[experiment_name] if experiment_name else None, filter_string=filter_string, output_format="list",)
    if runs:
        for run in runs:
            return run, True
        return None, False
    else:
        if experiment_name:
            mlflow.set_experiment(experiment_name=experiment_name)
        run_tags = {"mlflow.parentRunId": parent_run_id} if parent_run_id else None
        if tags:
            if run_tags is None:
                run_tags = tags
            else:
                run_tags.update(tags)

        nfailed = 0
        if not run_name:
            logger.warning("empty run name!!!")

        while True:
            try:
                run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id, tags=run_tags)
            except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                nfailed += 1
                if nfailed > 5:
                    # Wave 41 (2026-05-20): preserve traceback before final-give-up return;
                    # caller sees only (None, False) otherwise.
                    logger.exception("mlflow.start_run failed after %d retries", nfailed)
                    return None, False
                scrubbed = _strip_userinfo(e)
                if "already active" in str(e):
                    active = mlflow.active_run()
                    if active is not None:
                        logger.warning("%s active run_id=%s", scrubbed, active.info.run_id)
                        mlflow.end_run()
                    else:
                        logger.warning(scrubbed)
                else:
                    logger.error(scrubbed)
                    raise
            else:
                mlflow.end_run()
                break
        return run, False

def create_mlflow_run_label(params: Optional[dict] = None, category: Optional[str] = None) -> str:
    """Build a compact, human-readable run label like ``"category:key1=val1,key2=val2"`` from a params dict, skipping falsy values.

    ``Enum`` values render as their member name and bare ``type`` values render as the class name, so the label stays
    short and readable instead of showing Python's default ``repr``.
    """
    if params is None:
        params = {}
    label_parts = []
    for key, value in params.items():
        if value:
            if isinstance(value, Enum):
                label_parts.append(f"{key}={value.name}")
            else:
                if type(value) is type:
                    label_parts.append(f"{key}={value.__name__}")
                else:
                    label_parts.append(f"{key}={value}")
    label = ",".join(label_parts)
    if category:
        if label:
            label = f"{category}:{label}"
        else:
            label = f"{category}"
    return label
