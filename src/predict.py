"This script computes inferences"
"""Python script to train Tennis Demmand models"""

# Utils
from config import Location, ModelParams
from prefect import flow, task
from datetime import date, datetime
from typing import Union, Any
import logging

# Data Processing
import numpy as np
import pandas as pd
import data_preprocessing as dp

# Machine Learning
from sklearn.model_selection import GridSearchCV, train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
import mlflow

# System
import os
import sys
import joblib
import json


TIMESTAMP = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
DEV_NAME = "santiagoal"
OUTPUT_VAR_NAME = "sales_volume"

mlflow_client = mlflow.MlflowClient(tracking_uri=Location().mlflow_tracking_uri)
os.environ["LOGNAME"] = DEV_NAME

# Set Logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y:%m:%d %H:%M:%S",
    filename=os.path.join(Location().root_dir, "data/logs/inference/output.log")
)

logger = logging.getLogger(name="Logger")
logger.setLevel(logging.INFO)

def import_data(location: Location = Location()) -> pd.DataFrame:   
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    df_raw = dp.extract_json_df()
    return df_raw


##@task
def prepare_data(df: pd.DataFrame = None, location: Location = Location()) -> pd.DataFrame:   
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    df_clean = dp.clean_data(df)
    return df_clean


##@task
def compute_inferences(input_df: Union[pd.DataFrame, np.array], location: Location = Location()) -> Any:   
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
   
    # Import model
    with open(location.model, "rb") as m:
        model = joblib.load(m)
    
    input_df.drop("sales_volume", axis=1, inplace=True)
    # Compute compute_inferences
    predictions = model.predict(input_df)
    return predictions


##@task
def include_predictions(
            input_df: Union[pd.DataFrame, np.array], 
            predictions: np.array, 
            location: Location = Location()
        ) -> pd.DataFrame:   
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    output_df = input_df.copy()
    output_df[OUTPUT_VAR_NAME] = predictions
    output_df[OUTPUT_VAR_NAME] = output_df[OUTPUT_VAR_NAME].map(lambda x: int(x))  # Convert to integers
    ordered_cols = [col for col in output_df.columns if col != OUTPUT_VAR_NAME] + [OUTPUT_VAR_NAME]
    output_df = output_df[ordered_cols]
    output_df.to_csv(location.data_final, index=False)
    return output_df


#@flow
def predict(location: Location = Location()) -> None:   
    """
    Description.

    Parameters
    ----------
    arg1 : type
        Description
    arg2 : type
        Description
    arg3 : type
        Description

    Returns:
        type:
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    logger.info("Importing Data...")
    df_raw = import_data()
    logger.info("Preparing Data for Inferences...")
    input_df = prepare_data(df=df_raw)
    logger.info("Computing Inferences...")
    predictions = compute_inferences(input_df=input_df)
    logger.info("Saving compute_inferences...")
    include_predictions(input_df=df_raw, predictions=predictions)
    logger.info(f"Inferences Succesfully Computed. \ncompute_inferences Saved in {location.data_final}")
    return None


if __name__=="__main__":
    predict()