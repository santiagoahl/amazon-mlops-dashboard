"""
create Pydantic models
"""
from typing import List
from pathlib import Path
import os
from pydantic import BaseModel, validator

#root_dir = Path(__file__).resolve().parent

ROOT_DIR = "/home/santi/current-projects/public-apis/amazon-mlops-dashboard"

def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v

class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    root_dir: str = ROOT_DIR
    
    # API 
    path_api_responses: str = os.path.join(root_dir, "data/raw/api-calls")
    api_queries: str = os.path.join(root_dir, "api_queries.json")
    
    
    # Data
    data_raw: str = os.path.join(root_dir, "data/raw/csv/all_countries_tennis_data.csv")  # PATH_MERGED_RAW_CSVS
    data_clean: str = os.path.join(root_dir, "data/pre-processed/cleaned/tennis_data_cleaned.csv")
    data_process: str = os.path.join(root_dir, "data/pre-processed/cleaned/tennis_data_processed.csv")
    data_augmented: str = os.path.join(root_dir, "data/pre-processed/augmented/tennis_data.csv")
    data_final: str =  os.path.join(root_dir, "data/results/inferences/tennis_data.csv")
    distributions: str = os.path.join(root_dir, "data/distributions/distributions.pkl")
    
    # Model
    model: str =  os.path.join(root_dir, "models/tennis_demand_model.pkl")
    mlflow_tracking_uri: str = "http://127.0.0.1:8000"
    input_notebook: str =  os.path.join(root_dir, "notebooks/analyze_results.ipynb")
    output_notebook: str =  os.path.join(root_dir, "notebooks/results.ipynb")
     

class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    drop_columns: List[str] = ["Id"]
    label: str = "Species"
    test_size: float = 0.3

    _validated_test_size = validator("test_size", allow_reuse=True)(
        must_be_non_negative
    )


class ModelParams(BaseModel):
    """Specify the parameters of the `train` flow"""

    C: List[float] = [0.1, 1, 10, 100, 1000]
    gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]

    _validated_fields = validator("*", allow_reuse=True, each_item=True)(
        must_be_non_negative
    )
