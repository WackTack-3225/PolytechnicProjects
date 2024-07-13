"""Project pipelines."""
from __future__ import annotations
from typing import Dict

# Import my pipelines here so I can declare them here
from model.pipelines import data_processing as dp
from model.pipelines import ml_models as ml

from kedro.pipeline import Pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    ml_pipeline = ml.create_pipeline()
    return {
        "dp": data_processing_pipeline,
        "ml": ml_pipeline,
        "__default__": data_processing_pipeline + ml_pipeline
    }
