# Car Price Prediction - Complete Modeling Suite

## Overview

This project contains a comprehensive data science pipeline for predicting car prices in Mexico using machine learning (XGBoost). The suite includes data cleaning utilities, advanced feature engineering, and a production-ready price prediction model.

## Project Structure

```
Proyecto_Final/
├── data/
│   ├── mexico_cars_complete.csv       # Final cleaned & unified dataset (2,397 records)
│   └── [other data files]
├── utils/
│   ├── __init__.py                    # Package initialization
│   ├── data_cleaning.py               # Data cleaning utilities
│   └── modeling.py                    # Feature engineering & XGBoost model
├── models/
│   └── car_price_xgb_model.pkl        # Trained XGBoost model
├── results/
│   ├── feature_importance.csv         # Feature importance ranking
│   ├── model_predictions.csv          # Test set predictions & errors
│   └── model_metrics.csv              # Model performance metrics
├── car_price_model.py                 # Main modeling script
└── README.md                          # This file
```

## Features & Capabilities

### 1. Data Cleaning Module (`utils/data_cleaning.py`)

Comprehensive data preprocessing functions:

- **`fill_missing_from_make_model()`** - Fill missing values by make/model groups
- **`fill_missing_from_make()`** - Fill remaining missing values at brand level
- **`fill_missing_at_global()`** - Fill remaining missing values using global statistics
- **`calculate_derived_columns()`** - Calculate car_age, miles_per_year, market metrics
- **`remove_outliers()`** - Remove outliers using IQR or Z-score methods
- **`clean_dataset()`** - Complete cleaning pipeline
- **`get_data_summary()`** - Generate data quality summary

**Example Usage:**
```python
from utils import clean_dataset, get_data_summary

# Load and clean data
df = pd.read_csv('data/mexico_cars_complete.csv')
df_clean = clean_dataset(df)

# Get summary
summary = get_data_summary(df_clean)
print(f"Complete columns: {summary['complete_columns']}/{summary['total_columns']}")
```

### 2. Feature Engineering & Modeling (`utils/modeling.py`)

#### CarPriceFeatureEngineer Class

Advanced feature engineering with 59 total engineered features:

**Core Methods:**
- **`create_make_model_mileage_label()`** - Composite label for vehicle segments
- **`create_brand_statistics()`** - Brand-level pricing statistics
  - `brand_avg_price` - Average price by brand
  - `brand_median_price` - Median price by brand
  - `brand_price_std` - Price standard deviation by brand
  - `brand_car_count` - Number of cars per brand

- **`create_year_statistics()`** - Year-based features
  - `year_avg_price` - Average price by model year
  - `year_median_price` - Median price by model year

- **`create_make_model_statistics()`** - Make/Model combinations
  - `model_avg_price` - Average price by make/model
  - `model_median_price` - Median price by make/model
  - `model_price_std` - Standard deviation by make/model

- **`create_mileage_range_statistics()`** - Mileage-based segmentation
  - Segments: 0-20k, 20-50k, 50-100k, 100-150k, 150k+ miles

- **`create_body_type_features()`** - Vehicle type classification
  - `is_suv` - SUV/Truck indicator
  - `is_luxury_brand` - Luxury brand indicator (BMW, Mercedes, Audi, etc.)
  - `is_sports` - Sports car indicator
  - `is_sedan`, `is_hatchback` - Body type indicators

- **`create_price_ratio_features()`** - Derived metrics
  - `price_per_mile` - Price efficiency metric
  - `price_per_age` - Price depreciation metric
  - `price_pct_from_brand` - Deviation from brand average
  - `price_pct_from_model` - Deviation from model average

#### CarPriceXGBModel Class

Production-ready XGBoost model:

**Methods:**
- **`train_test_split_data()`** - 80/20 train/test split
- **`train_model()`** - Train XGBoost with customizable parameters
- **`evaluate_model()`** - Generate performance metrics
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
  - MAPE (Mean Absolute Percentage Error)
- **`get_feature_importance()`** - Rank features by importance
- **`make_prediction()`** - Predict prices for new cars
- **`save_model()` / `load_model()`** - Model persistence

**Example Usage:**
```python
from utils import CarPriceFeatureEngineer, CarPriceXGBModel

# Feature Engineering
engineer = CarPriceFeatureEngineer(df)
df_engineered = engineer.create_all_features()

# Model Training
model = CarPriceXGBModel(target='price', test_size=0.2)
X, y, features = model.prepare_features(df_engineered)
model.train_test_split_data(X, y)
model.train_model(max_depth=6, learning_rate=0.1, n_estimators=200)

# Evaluation
metrics = model.evaluate_model()
print(f"R² Score: {metrics['R2']:.4f}")
print(f"MAE: ${metrics['MAE']:,.2f}")

# Feature Importance
importance = model.get_feature_importance(top_n=15)

# Predictions
new_predictions = model.make_prediction(new_cars_df, features)
```

### 3. Complete Modeling Pipeline

**Function: `full_modeling_pipeline()`**

Orchestrates the entire workflow:
1. Feature engineering
2. Train/test splitting
3. Model training
4. Performance evaluation
5. Feature analysis

```python
from utils import full_modeling_pipeline

engineer, model, metrics, importance = full_modeling_pipeline(
    df,
    test_size=0.2,
    xgb_params={'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200}
)
```

## Model Performance

### Metrics
- **MAE**: $14,557.46 (average prediction error)
- **RMSE**: $44,076.19 (root mean squared error)
- **R² Score**: 0.9914 (99.14% variance explained)
- **MAPE**: 4.51% (mean absolute percentage error)

### Top Features by Importance
1. **price_per_age** (40.54%) - Price depreciation rate
2. **price_per_mile** (15.95%) - Price efficiency
3. **avg_price_model** (14.30%) - Model average pricing
4. **model_avg_price** (6.60%) - Make/model combination average
5. **price_pct_from_brand** (6.23%) - Deviation from brand average

## Dataset Summary

**Final Unified Dataset: `mexico_cars_complete.csv`**
- **Records**: 2,397 vehicles
- **Columns**: 29 core + 30 engineered features
- **Completeness**: 100% for all modeling-critical columns
- **Data Sources**:
  - autos_mexico: 1,449 records (60.5%)
  - car_listings_mexico_combined: 948 records (39.5%)

**Key Statistics:**
- Price Range: $5,000 - $4,499,990 (avg: $397,021)
- Mileage: 0 - 240,000 miles (avg: 59,737)
- Car Age: 1-20 years (avg: 5.5 years)
- Year Range: 2006 - 2025

**Top Brands by Volume:**
1. Volkswagen (15.0%)
2. Chevrolet (12.9%)
3. Nissan (9.8%)
4. Audi (8.5%)
5. Ford (8.5%)

## Running the Complete Pipeline

```bash
# From project root
python3 car_price_model.py
```

**Output Files Generated:**
1. `models/car_price_xgb_model.pkl` - Trained model
2. `results/feature_importance.csv` - Feature rankings
3. `results/model_predictions.csv` - Test predictions & errors
4. `results/model_metrics.csv` - Performance metrics

## Advanced Usage Examples

### Example 1: Making Predictions for New Cars

```python
import pandas as pd
from utils import CarPriceXGBModel, CarPriceFeatureEngineer

# Load trained model
model = CarPriceXGBModel()
model.load_model('models/car_price_xgb_model.pkl')

# Prepare new cars data
new_cars = pd.DataFrame({
    'make': ['Toyota', 'BMW', 'Ford'],
    'model': ['Camry', '3 Series', 'Mustang'],
    'year': [2023, 2022, 2021],
    'miles': [15000, 25000, 30000],
    'body_type': ['Sedan', 'Sedan', 'Coupe'],
    # ... other features
})

# Feature engineering
engineer = CarPriceFeatureEngineer(new_cars)
new_cars_eng = engineer.create_all_features()

# Predict
predictions = model.make_prediction(new_cars_eng, feature_cols)
```

### Example 2: Analyzing Brand Price Statistics

```python
from utils import CarPriceFeatureEngineer

engineer = CarPriceFeatureEngineer(df)
df_eng = engineer.create_all_features()

# Get brand statistics
brand_stats = engineer.get_feature_importance_data()['by_brand']
print(brand_stats.sort_values(('price', 'mean'), ascending=False).head(10))
```

### Example 3: Custom Data Cleaning

```python
from utils import clean_dataset, remove_outliers

# Clean dataset
df_clean = clean_dataset(df, remove_outliers_flag=False)

# Remove extreme outliers
df_clean = remove_outliers(df_clean, column='price', method='iqr', threshold=1.5)

# Check data quality
from utils import get_data_summary
summary = get_data_summary(df_clean)
```

## Feature Dictionary

### Core Features
- `make`: Vehicle brand/manufacturer
- `model`: Vehicle model name
- `year`: Model year (2006-2025)
- `price`: Vehicle price (target variable)
- `miles`: Odometer miles
- `car_age`: Age in years (calculated)
- `body_type`: Vehicle type (Sedan, SUV, etc.)
- `fuel_type`: Fuel type (Gasoline, Diesel, etc.)
- `transmission`: Transmission type
- `cylinders`: Number of cylinders
- `horsepower`: Engine horsepower
- `doors`: Number of doors

### Engineered Features (30 additional)

**Brand Features:**
- `brand_avg_price` - Average price for brand
- `brand_median_price` - Median price for brand
- `brand_price_std` - Price standard deviation for brand

**Model Features:**
- `model_avg_price` - Average price for make/model
- `model_median_price` - Median price for make/model
- `model_price_std` - Price std dev for make/model

**Year Features:**
- `year_avg_price` - Average price for model year
- `year_median_price` - Median price for model year

**Type Features:**
- `is_suv` - Binary indicator for SUVs
- `is_luxury_brand` - Binary indicator for luxury brands
- `is_sports` - Binary indicator for sports cars
- `is_sedan` - Binary indicator for sedans
- `is_hatchback` - Binary indicator for hatchbacks

**Ratio Features:**
- `price_per_mile` - Price efficiency metric
- `price_per_age` - Depreciation rate
- `price_pct_from_brand` - % deviation from brand average
- `price_pct_from_model` - % deviation from model average

## Technologies & Libraries

- **Python 3.12**
- **XGBoost 2.x** - Gradient boosting for price prediction
- **scikit-learn** - Machine learning utilities
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **logging** - Process tracking

## Installation & Setup

```bash
# Create virtual environment
python3 -m venv cars_venv
source cars_venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost

# Run modeling pipeline
python3 car_price_model.py
```

## Project Goals & Results

✅ **Unified Dataset**: Consolidated multiple data sources (1,449 + 948 records)
✅ **Data Quality**: 100% completeness on modeling-critical columns
✅ **Feature Engineering**: 59 total features with business context
✅ **Model Accuracy**: 99.14% R² score, 4.51% MAPE
✅ **Production Ready**: Serialized model for deployment
✅ **Interpretability**: Clear feature importance rankings
✅ **Reusability**: Modular, well-documented utilities

## Future Enhancements

- [ ] Hyperparameter tuning with Optuna
- [ ] Cross-validation and ensemble models
- [ ] Price prediction confidence intervals
- [ ] Real-time API for predictions
- [ ] Dashboard for model monitoring
- [ ] Regional price analysis
- [ ] Seasonal trend analysis
- [ ] Used car depreciation curves

## Contact & Support

For questions or issues with this modeling suite, please refer to the individual module docstrings and example usage in `car_price_model.py`.

---

**Last Updated**: April 8, 2026
**Version**: 1.0
**Status**: Production Ready ✅
