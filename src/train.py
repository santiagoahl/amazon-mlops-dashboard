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
    filename=os.path.join(Location().root_dir, "data/logs/train/output.log")
)

logger = logging.getLogger(name="Logger")
logger.setLevel(logging.INFO)

#@task
def get_processed_data(data_location: str) -> dict[str, Any]:
    """Get processed data from a specified location

    Parameters
    ----------
    data_location : str
        Location to get the data
    """
    
    data = pd.read_csv(data_location)
    X, y = get_X_y(data, label=OUTPUT_VAR_NAME) 
    
    data_split = split_train_test(X, y, test_size=0.2)
    return data_split

#@task
def train_model(
    model_params: ModelParams,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Any:
    """Train the model

    Parameters
    ----------
    model_params : ModelParams
        Parameters for the model
    X_train : pd.DataFrame
        Features for training
    y_train : pd.Series
        Label for training
    """
    
    grid = GridSearchCV(SVR(), model_params.model_dump(), refit=True, verbose=3)
    grid.fit(X_train, y_train.ravel())                    
    return grid


#@task
def predict(grid: GridSearchCV, X_test: pd.DataFrame) -> np.array:
    """_summary_

    Parameters
    ----------
    grid : GridSearchCV
    X_test : pd.DataFrame
        Features for testing
    """
    return grid.predict(X_test)


#@task
def save_model(model: GridSearchCV, save_path: str) -> None:
    """Save model to a specified location

    Parameters
    ----------
    model : GridSearchCV
    save_path : str
    """
    joblib.dump(model, save_path)


#@task
def save_predictions(predictions: np.array, save_path: str) -> None:
    """Save predictions to a specified location

    Parameters
    ----------
    predictions : np.array
    save_path : str
    """
    joblib.dump(predictions, save_path)

#@task
def get_X_y(data: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """Get features and label

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    label : str
        Name of the label
    """
    X = data.drop(columns=label, axis=1)
    y = data[label].values.reshape(-1, 1)
    return X, y


#@task
def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int) -> dict[str, Any]:
    """_summary_

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Target
    test_size : int
        Size of the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.reshape(-1, 1),
        "y_test": y_test.reshape(-1, 1),
    }
    
def evaluate_regression_model(
        model,
        data: dict
    ) -> dict:   
    """
    Report Regression-Related Metrics to evaluate model performance.

    Parameters
    ----------
    model
        Model to test.
    data: dict
        Dictionary with the following items:        
        X_test : type
            Test input data.
        y_test : type
            Test output data.
    
    Returns:
        dict: Results
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    
    # TODO: replace with current model if the performance is surpassed
    X_test, y_test = (data["X_test"], data["y_test"])
    
    y_pred = predict(model, X_test)
    return {
            "mae_score": mean_absolute_error(y_true=y_test, y_pred=y_pred),
            "mse_score": mean_squared_error(y_true=y_test, y_pred=y_pred),
            "rmse_score": root_mean_squared_error(y_true=y_test, y_pred=y_pred),
            "r2_score": r2_score(y_true=y_test, y_pred=y_pred)
        }
#@flow
def train(
    location: Location = Location(),
    model_params: ModelParams = ModelParams(),
) -> None:
    """Flow to train the model

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    model_params : ModelParams, optional
        Model Hyperparameter grid for training the model, by default ModelParams()
    """
    logger.info("Importing Train Data...")
    data = get_processed_data(location.data_augmented)
    logger.info("Training Models...")
    model = train_model(model_params, data["X_train"], data["y_train"])
    logger.info("Saving Models...")
    save_model(model, save_path=location.model)
    #save_predictions(predictions, save_path=location.data_final)
    with open(os.path.join(location.root_dir, "models/tennis_demand_model.pkl"), "rb") as m:
        model = joblib.load(m)
    data = get_processed_data(location.data_process)
    logger.info("Evaluating Model...")
    metrics = evaluate_regression_model(model, data)
    params = model.get_params()
    logger.info("Saving Results...")

    mlflow.set_experiment(
        experiment_name=f"Train Models Script - MLflow Integration"
    )
    #f"SVR - Augmented Dataset ({TIMESTAMP})"
    with mlflow.start_run(run_name=f"SVR - Augmented Dataset ({TIMESTAMP})") as mf:
        mlflow.set_tag("developer", DEV_NAME)
        mlflow.set_tag("dataset", "v1.0")
        mlflow.set_tag("runtype", "model training")
        
        mlflow.log_params(
                params=params
            )
        mlflow.log_metrics(
                metrics=metrics
            )
        mlflow.sklearn.log_model(
            sk_model=model,
            input_example=data["X_test"],
            artifact_path="svr_model"
        )
    logger.info(f"Model Succesfully trained\nModel Saved in {location.model}")
    return metrics

if __name__ == "__main__":
    train()
