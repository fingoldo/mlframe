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

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import
import html
import re
import pandas as pd, numpy as np
from enum import Enum
import mlflow

# Matches ``scheme://user:password@`` prefixes so we can scrub them before
# printing or logging exception text that bubbled up from mlflow.
_USERINFO_RE = re.compile(r"(?i)([a-z][a-z0-9+\-.]*://)[^/@\s]+@")


def _strip_userinfo(text: object) -> str:
    return _USERINFO_RE.sub(r"\1***@", str(text))

########################################################################################################################################################################################################################################
# MLFLOW
########################################################################################################################################################################################################################################

def flatten_classification_report(cr: dict, separate_metrics=("accuracy","balanced_accuracy","brier_score_loss","roc_auc"),source:str="")->dict:
    res={}
    for metric in separate_metrics:
        if metric in cr:
            res[source+metric]= cr.pop(metric)
    for class_or_avg, metrics_dict in cr.items():
        prefix=class_or_avg if class_or_avg in ('macro avg', 'weighted avg') else 'class '+str(class_or_avg)
        for metric, value in metrics_dict.items():
            res[source+prefix + "_" + metric]= value
    return res
            
def log_classification_report_to_mlflow(cr: dict, step: int,separate_metrics=("accuracy",),source:str=""):
    """Logging all metrics from a dict-like classification_report as flat MLFlow entries."""

    for metric in separate_metrics:
        if metric in cr:
            mlflow.log_metric(source+metric, cr.pop(metric), step=step)
    for class_or_avg, metrics_dict in cr.items():
        prefix=class_or_avg if class_or_avg in ('macro avg', 'weighted avg') else 'class '+str(class_or_avg)
        for metric, value in metrics_dict.items():
            mlflow.log_metric(source+prefix + "_" + metric, value, step=step)

def embed_website_to_mlflow(url:str,fname:str="url",extension:str='.html',width:int=700,height:int=450)->None:
    """Creates a html file with desired url embedded to be shown nicely in MLFlow UI."""

    safe_url = html.escape(url, quote=True)
    safe_width = int(width)
    safe_height = int(height)
    website_embed = f'''<!DOCTYPE html>
    <html>
    <iframe src="{safe_url}" style='width: {safe_width}px; height: {safe_height}px' sandbox='allow-same-origin allow-scripts'>
    </iframe>
    </html>'''

    if fname.lower().endswith(extension.lower()):
        extension=""

    with open(fname+extension, "w", encoding="utf-8") as f:
        f.write(website_embed)

def get_or_create_mlflow_run(run_name: str, parent_run_id: str = None, experiment_name: str = None, experiment_id: str = None,tags:dict={}) -> Tuple[object, bool]:
    """Tries to find a run by name within current mlflow experiment.
    If not found, creates new one.
    """
    # Escape embedded double-quotes/backslashes so user-controlled run_name can't
    # break out of the mlflow DSL filter literal.
    def _dsl_escape(s: object) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    filter_string=f'run_name = "{_dsl_escape(run_name)}"'
    if parent_run_id:
        filter_string+=f' and tag.mlflow.parentRunId = "{_dsl_escape(parent_run_id)}"'

    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_string, output_format="list",)
    if runs:
        for run in runs:
            return run, True
    else:
        if experiment_name:
            mlflow.set_experiment(experiment_name=experiment_name)
        run_tags={"mlflow.parentRunId": parent_run_id} if parent_run_id else None
        if tags:
            if run_tags is None:
                run_tags=tags
            else:
                run_tags.update(tags)
        
        nfailed=0
        if not run_name:
            print("empty run name!!!")

        while True:
            try:                
                run = mlflow.start_run(
                    run_name=run_name, experiment_id=experiment_id, tags=run_tags
                )
            except Exception as e:
                nfailed+=1
                if nfailed>5:
                    return None,False
                scrubbed = _strip_userinfo(e)
                if 'already active' in str(e):
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
    
def create_mlflow_run_label(params: dict={}, category: str = None) -> str:
    label = []
    for key, value in params.items():
        if value:
            if isinstance(value, Enum):
                label.append(f"{key}={value.name}")
            else:
                if type(value) == type:
                    label.append(f"{key}={value.__name__}")
                else:
                    label.append(f"{key}={value}")
    label = ",".join(label)
    if category:
        if label:
            label = f"{category}:{label}"
        else:
            label = f"{category}"
    return label