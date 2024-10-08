"""Python script to process the data"""

import joblib
import pandas as pd
from config import Location, ProcessConfig
from prefect import flow, task
from sklearn.model_selection import train_test_split
from typing import *
from scipy.stats import norm
import re


@task
def get_raw_data(data_location: str) -> pd.DataFrame:
    """Read raw data

    Parameters
    ----------
    data_location : str
        The location of the raw data
    """
    return pd.read_csv(data_location)

@task
def gaussian_noise(df: pd.DataFrame, target_column: str) -> pd.DataFrame:   
    """
    Modifies data by adding Gaussian Noise depending on the noise that is desired to be added.

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
        'No featured offers available': [30, 30 * 0.5],
        '50+ bought in past month': [50, 50 * 0.5],
        '100+ bought in past month': [100, 100 * 0.5],
        'List: ': [30, 30 * 0.5],
        '200+ bought in past month': [100, 100 * 0.5],
        '300+ bought in past month': [200, 200 * 0.5],
        None: [30, 30 * 0.5],
        '500+ bought in past month': [300, 300 * 0.5],
        '800+ bought in past month': [100, 100 * 0.5],
        '900+ bought in past month': [100, 100 * 0.5],
        '1K+ bought in past month': [3000, 3000 * 0.5],
        }
    
    # Define column names, these columns will help to add the Gaussian Noise
    target_column_numerical = target_column + "_numerical"
    target_column_cleaned = target_column + "_cleaned"
    
    #noises_mapping = {key: normal.rvs(loc=value[0], scale=value[1], sample)}
    # Clean output variable -> Non-info values are supposed to be replaced by 0

    df[target_column_cleaned] = df[target_column].map(
        lambda value: '0' if value in [None, 'List: ', 'No featured offers available'] 
        else value
    )
    
    # Get the numerical values from the target_column 
    target_column_vals = {
        val_raw: re.search(pattern='\d+', string=val_raw).group() 
        for val_raw in target_column_vals_raw
    }
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
    df[[target_column_numerical]] = df[[target_column+"_cleaned"]].replace(target_column_vals) 
    
    # Generate Noise    
    print(df[[target_column]].unique())
    df['gaussian_noise'] = df[target_column].apply(gaussian_noise_map)
    
    # Redefine the target column adding Gaussian Noise
    df[target_column] = df[target_column_numerical] + df['gaussian_noise']
    
    df[[target_column]] = df[[target_column]].map(lambda x: int(x))
    
    df.drop(
        [target_column_numerical, "gaussian_noise", target_column_cleaned],
        axis=1,
        inplace=True
    )
    return df

@task
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


@task
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


@task
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


@task
def save_processed_data(data: dict, save_location: str):
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
