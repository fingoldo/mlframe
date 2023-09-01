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
import pandas as pd, numpy as np
from enum import Enum
import mlflow

########################################################################################################################################################################################################################################
# MLFLOW
########################################################################################################################################################################################################################################

def flatten_classification_report(cr: dict, separate_metrics=("accuracy","balanced_accuracy","roc_auc"),source:str="")->dict:
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

    website_embed = f'''<!DOCTYPE html>
    <html>
    <iframe src="{url}" style='width: {width}px; height: {height}px' sandbox='allow-same-origin allow-scripts'>
    </iframe>
    </html>'''

    if fname[:-len(extension)].lower()==extension:
        extension=""

    with open(fname+extension, "w") as f:
        f.write(website_embed)

def get_or_create_mlflow_run(run_name: str, parent_run_id: str = None, experiment_name: str = None, experiment_id: str = None,tags:dict={}) -> Tuple[object, bool]:
    """Tries to find a run by name within current mlflow experiment.
    If not found, creates new one.
    """
    filter_string=f'run_name = "{run_name}"'
    if parent_run_id:
        filter_string+=f' and tag.mlflow.parentRunId = "{parent_run_id}"'

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
                if 'already active' in str(e):
                    raise(e)
                    run = mlflow.active_run()
                    print(str(e),"active run_id=",run.info.run_id)
                    mlflow.end_run()
                else:
                    logger.error(e)
                    raise(e) 
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