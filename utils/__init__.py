"""
Utils package for data science project
Contains utilities for data cleaning, feature engineering, and modeling
"""

from .data_cleaning import (
    fill_missing_from_make_model,
    fill_missing_from_make,
    fill_missing_at_global,
    calculate_derived_columns,
    clean_dataset,
    remove_outliers,
    get_data_summary
)

from .modeling import (
    CarPriceFeatureEngineer,
    CarPriceXGBModel,
    full_modeling_pipeline
)

__version__ = "1.0.0"
__author__ = "Data Science Team"

__all__ = [
    'fill_missing_from_make_model',
    'fill_missing_from_make',
    'fill_missing_at_global',
    'calculate_derived_columns',
    'clean_dataset',
    'remove_outliers',
    'get_data_summary',
    'CarPriceFeatureEngineer',
    'CarPriceXGBModel',
    'full_modeling_pipeline'
]
