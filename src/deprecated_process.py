"""Python script to process the data"""

from config import Location, ProcessConfig
from prefect import flow, task

# File Management
import joblib
import json

# Machine Learning Modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Data and numerical processing
from scipy.stats import norm
import pandas as pd
import numpy as np

from typing import *
import re

@task #Delete
def get_raw_data(data_location: str) -> pd.DataFrame:
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return pd.read_csv(data_location)

@task
def evaluate_regression_model(
        model, 
        y_true: Union[np.array, pd.Series], 
        y_pred: Union[np.array, pd.Series]
    ) -> dict:   
    """
    Save the performance of a (fitted) regression model leveraging R2, MSE, RMSE and MAE metrics.

    Args:
        model (abc.ABCMeta): Fitted model.
        y_pred (np.array | pd.Series): Model predictions.
        y_true (np.array | pd.Series): Actual values.
    
    Returns:
        dict: Performance metrics
    
    Example:
        >>> evaluate_regression_model(
                model=lasso_regressor,
                y_pred=y_pred,
                y_true=y_val
            )
        {'r2': -0.04734884276611817,
        'mae': np.float64(83.54344966689207),
        'mse': np.float64(16866.74842992118),
        'rmse': np.float64(129.87204637612044)}

    """
    
    performance_metrics = {
        "r2": r2_score(y_true, y_pred), 
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": root_mean_squared_error(y_true, y_pred) ** 2,
        "rmse": root_mean_squared_error(y_true, y_pred)
    }
    
    return performance_metrics

@task # Delete
def clean_data(df: pd.DataFrame) -> pd.DataFrame:   
    """
    Cleans and prepares dataframe to make predictions.

    Args:
        df (pd.DataFrame): Raw data to be cleaned.
    
    Returns:
        pd.DataFrame
    
    Example:
        >>> ('arg1', 'arg2')
        'output'
    """
    
    # Process price format

    df[['product_price']] = df[[
        'product_price'
        ]].map(lambda price_raw: float(price_raw[1:]) if price_raw != None else price_raw)

    df[['product_original_price']] = df[[
        'product_original_price'
        ]].map(lambda price_raw: float(price_raw[1:]) if price_raw != None else price_raw)

    df[['product_minimum_offer_price']] = df[[
        'product_minimum_offer_price'
        ]].map(lambda price_raw: float(price_raw[1:]) if price_raw != None else price_raw
    )
        
        
    # Convert to float

    df['product_star_rating'] = df['product_star_rating'].astype(float);

    df["coupon_discount"] = df["coupon_text"].map(
        lambda coupon_txt: 
            re.search(pattern="\d{1,2}(\.+\d{1,2})*", string=coupon_txt).group()
            if type(coupon_txt) != float
            else '0.0'
    )

    # conver to float

    df['coupon_discount'] = df['coupon_discount'].map(
        lambda discount_str: float(discount_str)
    ) 

    df.drop(labels=["coupon_text"], axis=1);

    # Process categorical data

    df["is_prime"] = pd.get_dummies(
        df["is_prime"], 
        dtype=float
        )[True]

    df["climate_pledge_friendly"] = pd.get_dummies(
        df["climate_pledge_friendly"], 
        dtype=float
        )[True]

    df["has_variations"] = pd.get_dummies(
        df["has_variations"], 
        dtype=float
        )[True]

    # Reorder columns to leave the predict variable at the end
    
    input_cols = [
        'product_price',
        'product_original_price',
        'product_star_rating',
        'product_num_ratings',
        'product_minimum_offer_price',
        'is_prime',
        'climate_pledge_friendly',
        'has_variations',
        'coupon_discount'
    ]

    df = df[
        input_cols  + ['sales_volume']
        ]
    # Imputation of null values
    df[input_cols] = df[input_cols].fillna(0.0)
    return df

@task  # Delete
def extract_json_df(json_file_paths: list) -> pd.DataFrame:   
    """
    Collects a list of json files, extracts the data and merges it.

    Args:
        json_file_paths (list): Files to examinate.
    
    Returns:
        pd.DataFrame: Merged data
    
    Example:
        >>> extract_json_df("../data/api-calls/tenis_products_49.json")
        pd.DataFrame
    """
    json_files = []
    
    for filepath in json_file_paths:
        with open(filepath, 'r') as f:
            json_files.append(
                json.load(f)
            )
        
    dataframes_list = [
        pd.DataFrame(json.loads(file['response'])['data']['products']) 
        for file in json_files
    ]
    
    data_merged = pd.concat(dataframes_list)
    
    return data_merged

@task  # Delete
def gaussian_noise(df: pd.DataFrame, target_column: str) -> pd.DataFrame:   
    """
    Modifies data by adding Gaussian Noise depending on the noise that is desired to be added

    Args:
        df (pd.Series|pd.DataFrame): Data to be transformed.
        targe_column (str): Column to be transformed with Gaussian Noise
    
    Returns:
        pd.DataFrame: Dataframe with Gaussian Noise implemented 
    
    Example:
        >>> gaussian_noise(
                df=pd.DataFrame([
                    '50+ bought in past month', '200+ bought in past month'
                    ]),
                target_column="sales_volume"
            )
        pd.DataFrame([78, 354])
    """
    
    # Mean and variance to add Gaussian Noise
    noise_mapping = {
        None: [30, 30 * 0.5],
        'List: ': [30, 30 * 0.5],
        '0': [30, 30 * 0.5],
        'No featured offers available': [30, 30 * 0.5],
        '50+ bought in past month': [50, 50 * 0.5],
        '100+ bought in past month': [100, 100 * 0.5],
        '200+ bought in past month': [100, 100 * 0.5],
        '300+ bought in past month': [100, 100 * 0.5],
        '400+ bought in past month': [100, 100 * 0.5],
        '500+ bought in past month': [200, 200 * 0.5],
        '700+ bought in past month': [100, 100 * 0.5],
        '800+ bought in past month': [100, 100 * 0.5],
        '900+ bought in past month': [100, 100 * 0.5],
        '1K+ bought in past month': [3000, 3000 * 0.5]
        }
    
    # Define column names, these columns will help to add the Gaussian Noise
    target_column_numerical = target_column + "_numerical"
    target_column_cleaned = target_column + "_cleaned"
    gaussian_noise_column = "gaussian_noise"
    
    #noises_mapping = {key: normal.rvs(loc=value[0], scale=value[1], sample)}
    # Clean output variable -> Non-info values are supposed to be replaced by 0

    df[target_column_cleaned] = df[target_column].map(
        lambda value: value if re.match(pattern='\d+', string=str(value))
        else '0'
    )
    
    target_column_vals_raw = list(df[target_column_cleaned].unique())
    
    # Get the numerical values from the target_column 
    target_column_vals = {
        val_raw: re.search(pattern='\d+', string=val_raw).group() 
        for val_raw in target_column_vals_raw
    }
    
    # Fix issue with 1K value to prevent it to be replaced by 1 instead of 1000
    target_column_vals = {
        key: 1000 if value == '1' else int(value) 
        for key, value in target_column_vals.items()
    }  
    
    # Generate Gaussian Noise for each possible key of the Noise Mapping
    gaussian_noise_map = lambda cat_value: abs(norm.rvs(
            loc=noise_mapping[cat_value][0],
            scale=noise_mapping[cat_value][1],
            size=1
        )[0])
    
    
    # Generate column with numerical values using the target column
    df[[target_column_numerical]] = df[[target_column_cleaned]].replace(target_column_vals) 
    
    # Generate Noise    
    df[gaussian_noise_column] = df[target_column_cleaned].apply(gaussian_noise_map)
    
    # Redefine the target column adding Gaussian Noise
    df[target_column] = df[target_column_numerical] + df[gaussian_noise_column]
    
    df[[target_column]] = df[[target_column]].map(lambda x: int(x))
    
    df.drop(
        [target_column_numerical, gaussian_noise_column, target_column_cleaned],
        axis=1,
        inplace=True
    )
    return df

@task  # Delete
def drop_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop unimportant columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    columns : list
        Columns to drop
    """
    return data.drop(columns=columns)


@task  # Delete
def get_X_y(data: pd.DataFrame, label: str):# -> tuple[DataFrame, Series]:
    """Get features and label

    Parameters
    ----------
    data : pd.DataFrame
        Data to process
    label : str
        Name of the label
    """
    X = data.drop(columns=label)
    y = data[label]
    return X, y


@task  # Delete
def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: int):
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
        X, y, test_size=test_size, random_state=0
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    
@task  # Delete
def augment_data(df: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:   
    """
    Creates synthetic samples using KDE.

    Args:
        df (pd.DataFrame): DataFrame to increase samples.
        scale (float): Augmentation multiplier, default 1.0. Use float numbers
            greater or equal than 1.0 to augment the number of samples, 2.0 to 
            duplicate and so on.
    
    Returns:
        pd.DataFrame: Augmented data
    
    Example:
        >>> 
    """
    
    
    return aug_df


@task  # Delete
def save_processed_data(data: dict, save_location: str) -> None:
    """Save processed data

    Parameters
    ----------
    data : dict
        Data to process
    save_location : str
        Where to save the data
    """
    joblib.dump(data, save_location)


@flow
def process(
    location: Location = Location(),
    config: ProcessConfig = ProcessConfig(),
):
    """Flow to process the ata

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    config : ProcessConfig, optional
        Configurations for processing data, by default ProcessConfig()
    """
    data = get_raw_data(location.data_raw)
    processed = drop_columns(data, config.drop_columns)
    X, y = get_X_y(processed, config.label)
    split_data = split_train_test(X, y, config.test_size)
    save_processed_data(split_data, location.data_process)


if __name__ == "__main__":
    process(config=ProcessConfig(test_size=0.1))
