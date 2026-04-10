"""
Data Cleaning Utilities
Module for data cleaning and preprocessing functions for car price analysis
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def string_similarity(str1, str2):
    """
    Calculate string similarity ratio between two strings
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        float: Similarity ratio (0-1)
    """
    return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()


def fill_from_group(group, cols, numeric_cols=None):
    """
    Fill missing values in a group from other observations
    
    Args:
        group: DataFrame group
        cols: List of columns to fill
        numeric_cols: List of numeric columns to use mean for
    
    Returns:
        DataFrame: Group with filled values
    """
    if numeric_cols is None:
        numeric_cols = []
    
    for col in cols:
        if group[col].isnull().any():
            if col in numeric_cols:
                fill_value = group[col].mean()
            else:
                mode_val = group[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else None
            
            if pd.notna(fill_value):
                group[col] = group[col].fillna(fill_value)
    
    return group


def fill_missing_from_make_model(df, fill_cols=None, numeric_cols=None):
    """
    Fill missing values by grouping on make and model
    
    Args:
        df: Input DataFrame
        fill_cols: List of columns to fill
        numeric_cols: List of numeric columns
    
    Returns:
        DataFrame: DataFrame with filled values
    """
    if fill_cols is None:
        fill_cols = ['engine', 'transmission', 'fuel_type', 'body_type', 
                     'drive_train', 'cylinders', 'doors', 'horsepower', 
                     'trim', 'exterior_color', 'condition']
    
    if numeric_cols is None:
        numeric_cols = ['horsepower', 'cylinders', 'doors', 'city_mpg', 'highway_mpg']
    
    logger.info(f"Filling missing values from make/model groups...")
    grouped = df.groupby(['make', 'model'], group_keys=False)
    df = grouped.apply(lambda x: fill_from_group(x, fill_cols, numeric_cols))
    
    logger.info("✅ Filled from make/model groups")
    return df


def fill_missing_from_make(df, fill_cols=None, numeric_cols=None):
    """
    Fill remaining missing values at make (brand) level
    
    Args:
        df: Input DataFrame
        fill_cols: List of columns to fill
        numeric_cols: List of numeric columns
    
    Returns:
        DataFrame: DataFrame with filled values
    """
    if fill_cols is None:
        fill_cols = ['engine', 'transmission', 'fuel_type', 'body_type', 
                     'drive_train', 'cylinders', 'doors', 'horsepower', 
                     'condition']
    
    if numeric_cols is None:
        numeric_cols = ['horsepower', 'cylinders', 'doors', 'city_mpg', 'highway_mpg']
    
    logger.info(f"Filling remaining missing values from make level...")
    grouped_make = df.groupby('make', group_keys=False)
    df = grouped_make.apply(lambda x: fill_from_group(x, fill_cols, numeric_cols))
    
    logger.info("✅ Filled from make groups")
    return df


def fill_missing_at_global(df, fill_cols=None, numeric_cols=None):
    """
    Fill remaining missing values using global statistics
    
    Args:
        df: Input DataFrame
        fill_cols: List of columns to fill
        numeric_cols: List of numeric columns
    
    Returns:
        DataFrame: DataFrame with filled values
    """
    if fill_cols is None:
        fill_cols = ['body_type', 'engine', 'transmission', 'fuel_type', 
                     'drive_train', 'trim', 'condition']
    
    if numeric_cols is None:
        numeric_cols = ['city_mpg', 'highway_mpg']
    
    logger.info(f"Filling remaining missing values at global level...")
    
    # Categorical columns - use mode
    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                logger.info(f"  Filled {col} with mode: {mode_val[0]}")
    
    # Numeric columns - use mean
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            logger.info(f"  Filled {col} with mean: {mean_val:.2f}")
    
    return df


def calculate_derived_columns(df, current_year=2026):
    """
    Calculate derived columns (car_age, miles_per_year, market metrics)
    
    Args:
        df: Input DataFrame
        current_year: Current year for age calculation
    
    Returns:
        DataFrame: DataFrame with calculated columns
    """
    logger.info("Calculating derived columns...")
    
    # car_age: current_year - year
    if 'car_age' not in df.columns or df['car_age'].isnull().any():
        df['car_age'] = df['car_age'].fillna(current_year - df['year'])
        logger.info(f"  Calculated car_age = {current_year} - year")
    
    # miles_per_year: miles / car_age
    if 'miles_per_year' not in df.columns or df['miles_per_year'].isnull().any():
        df['miles_per_year'] = df['miles_per_year'].fillna(
            df.apply(lambda row: row['miles'] / row['car_age'] if row['car_age'] > 0 else 0, axis=1)
        )
        logger.info(f"  Calculated miles_per_year = miles / car_age")
    
    # avg_price_model: average price for each make/model
    if 'avg_price_model' not in df.columns or df['avg_price_model'].isnull().any():
        avg_price_per_model = df.groupby(['make', 'model'])['price'].transform('mean')
        df['avg_price_model'] = df['avg_price_model'].fillna(avg_price_per_model)
        logger.info(f"  Calculated avg_price_model = mean(price) by make/model")
    
    # price_vs_market: price - avg_price_model
    if 'price_vs_market' not in df.columns or df['price_vs_market'].isnull().any():
        df['price_vs_market'] = df['price_vs_market'].fillna(
            df['price'] - df['avg_price_model']
        )
        logger.info(f"  Calculated price_vs_market = price - avg_price_model")
    
    return df


def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a column
    
    Args:
        df: Input DataFrame
        column: Column name
        method: 'iqr' for interquartile range, 'zscore' for z-score
        threshold: IQR multiplier (default 1.5) or z-score threshold (default 3)
    
    Returns:
        DataFrame: DataFrame without outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        removed = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        logger.info(f"Removed {removed} outliers from {column} using IQR method")
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        removed = len(z_scores[z_scores > threshold])
        df = df[np.abs(stats.zscore(df[column])) <= threshold]
        logger.info(f"Removed {removed} outliers from {column} using z-score method")
    
    return df


def standardize_columns(df, columns_mapping=None):
    """
    Standardize column names and types
    
    Args:
        df: Input DataFrame
        columns_mapping: Dictionary of column name mappings
    
    Returns:
        DataFrame: DataFrame with standardized columns
    """
    if columns_mapping:
        df = df.rename(columns=columns_mapping)
        logger.info(f"Renamed columns: {columns_mapping}")
    
    # Ensure key columns have correct types
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    if 'miles' in df.columns:
        df['miles'] = pd.to_numeric(df['miles'], errors='coerce')
    
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    
    return df


def clean_dataset(df, remove_outliers_flag=False, outlier_column='price', 
                  current_year=2026):
    """
    Comprehensive dataset cleaning pipeline
    
    Args:
        df: Input DataFrame
        remove_outliers_flag: Whether to remove outliers
        outlier_column: Column to check for outliers
        current_year: Current year for calculations
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    logger.info("Starting comprehensive data cleaning...")
    
    # Step 1: Fill from make/model and make level
    df = fill_missing_from_make_model(df)
    df = fill_missing_from_make(df)
    df = fill_missing_at_global(df)
    
    # Step 2: Calculate derived columns
    df = calculate_derived_columns(df, current_year)
    
    # Step 3: Remove duplicates if any
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate rows")
    
    # Step 4: Optional outlier removal
    if remove_outliers_flag and outlier_column in df.columns:
        df = remove_outliers(df, outlier_column, method='iqr', threshold=1.5)
    
    logger.info(f"✅ Data cleaning complete! Final size: {len(df)} rows")
    return df


def get_data_summary(df):
    """
    Get comprehensive summary of dataset quality
    
    Args:
        df: Input DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'complete_columns': (df.notna().sum() == len(df)).sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include='object').columns.tolist(),
    }
    return summary


if __name__ == "__main__":
    # Example usage
    print("Data Cleaning Module")
    print("Import this module and use the functions for data preprocessing")
