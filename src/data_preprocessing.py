# This script cleans, transforms and adds gaussian noise for oversampling purposes

import logging.config
from config import Location, ProcessConfig
from prefect import flow, task

# File Management
import joblib
import json
import os
import sys

# Machine Learning Modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Data and numerical processing
from scipy.stats import norm
import pandas as pd
import numpy as np

from typing import *
import re
import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y:%m:%d %H:%M:%S",
    stream=sys.stderr,
    filename=os.path.join(Location().root_dir, "data/logs/preprocess/output.log")
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

##@task
def extract_json_df(location: Location = Location()) -> pd.DataFrame:   
    """
    Collects a list of json files, extracts the data and merges it.

    Parameters:
    ----------
        json_file_paths (list): Files to examinate.
    
    Returns:
        pd.DataFrame: Merged data
    
    Example:
        >>> extract_json_df("../data/api-calls/tenis_products_49.json")
        pd.DataFrame
    """
    
    df = pd.read_csv(location.data_raw)

    json_file_paths = os.listdir(location.path_api_responses)
    json_file_paths = [
            os.path.join(location.path_api_responses, f) 
            for f in json_file_paths
        ]
    json_files = []
    
    for filepath in json_file_paths:
        if not filepath.endswith(".json"):
            continue
        with open(filepath, 'r') as f:
            json_files.append(
                json.load(f)
            )
        
    dataframes_list = [
        pd.DataFrame(file['response']) 
        for file in json_files
    ]
    
    data_merged = pd.concat(dataframes_list)
    data_merged.to_csv(
        location.data_raw,
        index=False
    )
    
    return df


#@task
def clean_data(df: pd.DataFrame = None, location: Location = Location()) -> pd.DataFrame:
    """
    Cleans and prepares the dataframe to make predictions.

    Returns:
        pd.DataFrame: Cleaned data ready for prediction.
    """
    
    # TODO: Delete zero records
    # Load raw data
    # Read CSV in case user have passed no df to
    if df is None:
        try:
            df = pd.read_csv(location.data_raw)
            logger.info(f"DataFrame loaded from: {location.data_raw}")
        except FileNotFoundError:
            raise ValueError(f"File not found at the specified location: {location.data_raw}")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the file: {e}")

    # Helper function to clean price columns
    def parse_price(price_raw) -> float:
        try:
            return float(price_raw[1:]) if isinstance(price_raw, str) and price_raw.startswith('$') else 0.0
        except Exception:
            return 0.0

    # Clean price columns
    price_columns = ['product_price', 'product_original_price', 'product_minimum_offer_price']
    for col in price_columns:
        df[col] = df[col].apply(parse_price)

    # Convert star rating to float
    df['product_star_rating'] = pd.to_numeric(df['product_star_rating'], errors='coerce').fillna(0.0)

    # Extract and clean coupon discount
    df["coupon_discount"] = df["coupon_text"].apply(
        lambda coupon_txt: float(re.search(r"\d{1,2}(\.\d{1,2})*", coupon_txt).group()) 
        if isinstance(coupon_txt, str) and re.search(r"\d{1,2}(\.\d{1,2})*", coupon_txt) 
        else 0.0
    )

    # Drop unnecessary columns
    df = df.drop(columns=["coupon_text"], errors="ignore")

    # Process categorical data into numerical (is_prime, climate_pledge_friendly, has_variations)
    categorical_columns = ["is_prime", "climate_pledge_friendly", "has_variations"]
    for col in categorical_columns:
        df[col] = pd.get_dummies(df[col], dtype=float).get(True, 0.0)

    # Reorder columns to leave the target variable at the end
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
    df = df[input_cols + ['sales_volume']]

    # Fill missing values
    df[input_cols] = df[input_cols].fillna(0.0)

    # Save cleaned data
    df.to_csv(location.data_clean, index=False)

    return df


#@task
def gaussian_noise(target_column, location: Location = Location()) -> pd.DataFrame:   
    """
    Modifies data by adding Gaussian Noise depending on the noise that is desired to be added

    Parameters:
    ----------
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
        
    df = pd.read_csv(location.data_clean)
    # Mean and variance to add Gaussian Noise
    noise_mapping = {
        None: [30, 30 * 0.5],
        'List: ': [30, 30 * 0.5],
        '0': [30, 30 * 0.5],
        'No featured offers available': [30, 30 * 0.5],

        # French stores
        'Plus de 50 achetés au cours du mois dernier': [50, 50 * 0.5],
        'Plus de 100 achetés au cours du mois dernier': [50, 50 * 0.5],
        'Plus de 200 achetés au cours du mois dernier': [200, 200 * 0.5],
        'Plus de 300 achetés au cours du mois dernier': [300, 300 * 0.5],
        'Plus de 400 achetés au cours du mois dernier': [400, 400 * 0.5],
        'Plus de 500 achetés au cours du mois dernier': [500, 500 * 0.5],
        'Plus de 600 achetés au cours du mois dernier': [600, 600 * 0.5],
        'Plus de 700 achetés au cours du mois dernier': [700, 700 * 0.5],
        'Plus de 800 achetés au cours du mois dernier': [800, 800 * 0.5],
        'Plus de 900 achetés au cours du mois dernier': [900, 900 * 0.5],
        'Plus de 1 k achetés au cours du mois dernier': [3000, 3000 * 0.5],
        'Plus de 2 k achetés au cours du mois dernier': [2000, 2000 * 0.5],
        'Plus de 3 k achetés au cours du mois dernier': [3000, 3000 * 0.5],
        'Plus de 4 k achetés au cours du mois dernier': [4000, 4000 * 0.5],
        'Plus de 5 k achetés au cours du mois dernier': [5000, 5000 * 0.5],
        'Plus de 6 k achetés au cours du mois dernier': [6000, 6000 * 0.5],
        'Plus de 7 k achetés au cours du mois dernier': [7000, 7000 * 0.5],
        'Plus de 8 k achetés au cours du mois dernier': [8000, 8000 * 0.5],
        'Plus de 9 k achetés au cours du mois dernier': [9000, 9000 * 0.5],
        'Plus de 10 k achetés au cours du mois dernier': [10000, 10000 * 0.5],

        # English stores
        '50+ bought in past month': [50, 50 * 0.5],
        '100+ bought in past month': [50, 50 * 0.5],
        '200+ bought in past month': [100, 100 * 0.5],
        '300+ bought in past month': [100, 100 * 0.5],
        '400+ bought in past month': [100, 100 * 0.5],
        '500+ bought in past month': [100, 100 * 0.5],
        '600+ bought in past month': [100, 100 * 0.5],
        '700+ bought in past month': [100, 100 * 0.5],
        '800+ bought in past month': [100, 100 * 0.5],
        '900+ bought in past month': [100, 100 * 0.5],
        '1K+ bought in past month': [3000, 3000 * 0.5],
        '2K+ bought in past month': [2000, 2000 * 0.5],
        '3K+ bought in past month': [3000, 3000 * 0.5],
        '4K+ bought in past month': [4000, 4000 * 0.5],
        '5K+ bought in past month': [5000, 5000 * 0.5],
        '6K+ bought in past month': [6000, 6000 * 0.5],
        '7K+ bought in past month': [7000, 7000 * 0.5],
        '8K+ bought in past month': [8000, 8000 * 0.5],
        '9K+ bought in past month': [9000, 9000 * 0.5],
        '10K+ bought in past month': [10000, 10000 * 0.5],
        '15K+ bought in past month': [15000, 15000 * 0.5]
    }

    
    # Define column names, these columns will help to add the Gaussian Noise
    target_column_numerical = target_column + "_numerical"
    target_column_cleaned = target_column + "_cleaned"
    gaussian_noise_column = "gaussian_noise"
    
    #noises_mapping = {key: normal.rvs(loc=value[0], scale=value[1], sample)}
    # Clean output variable -> Non-info values are supposed to be replaced by 0

    df[target_column_cleaned] = df[target_column].map(
        lambda value: value if re.match(pattern='\d', string=str(value))
        else '0'
    )
    
    target_column_vals_raw = list(df[target_column_cleaned].unique())
    
    # Get the numerical values from the target_column 
    target_column_vals = {
        val_raw: re.search(pattern='\d', string=val_raw).group() 
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
    
    df[target_column] = df[target_column].map(lambda x: int(x))
    
    df.drop(
        [target_column_numerical, gaussian_noise_column, target_column_cleaned],
        axis=1,
        inplace=True
    )
    
    df.to_csv(
        location.data_process,
        index=False    
    )
    return None


#@task
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


def knn_linear(
            df_original: pd.DataFrame,
            df_augmented: pd.DataFrame,
            k: int=10
        ) -> None:
    """
    Estimate y using similar (x, y) samples using KNN
    
    Parameters:
    ----------
        df_original (pd.DataFrame): DataFrame without sythetic samples.
        df_augmented (pd.DataFrame): DataFrame with sythetic samples. 
        k (int): Main parameter to run KNN.
        
    Returns:
        pd.DataFrame: Augmented data with estimated y values.
    """
    
    X = df_original.drop("sales_volume", axis=1).to_numpy()
    y = df_original["sales_volume"].to_numpy()
    
    y_hat_array = []
    for x in df_augmented.values:
        
        # Calculate the K Nearest Neighbors
        euclidean_distances = np.linalg.norm(X-x, axis=1)
        sorted_indices = np.argsort(euclidean_distances, axis=0)[:k]
        
        # Get the y values of the associated KNN
        y_knn = y[sorted_indices]
        
        # Compute estimation of y as y_hat
        y_hat = y_knn.mean()
        y_hat_array.append(int(y_hat))

    y_hat_array = np.array([y_hat_array]).reshape(len(df_augmented),)
    
    # Include y_hat as df column
    df_augmented["sales_volume"] = y_hat_array
    return df_augmented

#@task
def augment_data(aug_factor: float = 10.0, location: Location = Location()) -> pd.DataFrame:   
    """
    Creates synthetic samples using KDE.

    Parameters:
    ----------
        df (pd.DataFrame): DataFrame to increase samples.
        aug_factor (float): Augmentation multiplier, default 1.0. Use float numbers
            greater or equal than 1.0 to augment the number of samples, 2.0 to 
            duplicate and so on.
    
    Returns:
        pd.DataFrame: Augmented data
    
    Example:
        >>> 
    """
    
    distributions = joblib.load(location.distributions) 
    df = pd.read_csv(location.data_process)
    
    input_samples = []
    n_samples = int(len(df) * aug_factor)

    # Generate syntetic samples using estimated PDFs
    for distribution_obj in distributions:
        #logger.info(distribution_obj["random_variable"])
        pdf = distribution_obj["scipy_pdf_obj"]

        if distribution_obj["kind"] == "parametric":
            # Generate synthetic samples using rvs() for parametric distributions
            syntetic_samples = pdf.rvs(size=n_samples).reshape(n_samples,)
        elif distribution_obj["kind"] == "non-parametric":
            # Generate synthetic samples using resample() for non-parametric distributions
            syntetic_samples = pdf.resample(size=n_samples).reshape(n_samples,)

        # Append the generated samples to the input_samples list
        input_samples.append(syntetic_samples)

    # Stack all generated samples into a single numpy array
    input_samples = np.column_stack(input_samples)
    
    df_augmented = pd.DataFrame(
            data=input_samples, 
            columns=df.columns.drop("sales_volume")
        )
    
    # Fill estimated y samples
    df_augmented = knn_linear(df, df_augmented)
    
    # Generate output samples
    df_augmented.to_csv(location.data_augmented, index=False)
    return None


#@task
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
    

#@task
def process_data(location: Location = Location()) -> None:   
    """
    Description.
    
    """
    logger.info("Extracting Raw Data...")
    extract_json_df()
    logger.info("Cleaning Data...")
    clean_data()
    logger.info("Formatting output var and including Gaussian Noise...")
    gaussian_noise("sales_volume")
    logger.info("Augmenting Data...")
    
    # TODO: Include a distribution estimation module
    augment_data()
    logger.info(f"Data Succesfully Processed \nData Saved in {location.data_augmented}")
    return None

def main() -> None:
    process_data()

if __name__=="__main__":
    main()
