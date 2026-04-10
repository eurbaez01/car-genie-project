"""
Car Price Prediction Model - Script Version
Converted from the illustrative notebook for production use
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def fill_missing_from_make_model(df):
    """Fill missing values by grouping on make and model"""
    fill_cols = ['engine', 'transmission', 'fuel_type', 'body_type', 
                 'drive_train', 'cylinders', 'doors', 'horsepower', 
                 'trim', 'exterior_color', 'condition']
    numeric_cols = ['horsepower', 'cylinders', 'doors', 'city_mpg', 'highway_mpg']
    
    print("Filling missing values from make/model groups...")
    
    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            if col in numeric_cols:
                # Fill with group mean
                df[col] = df.groupby(['make', 'model'])[col].transform(
                    lambda x: x.fillna(x.mean()) if not x.isnull().all() else x
                )
            else:
                # Fill with group mode
                df[col] = df.groupby(['make', 'model'])[col].transform(
                    lambda x: x.fillna(x.mode()[0]) if len(x.mode()) > 0 and not x.isnull().all() else x
                )
                print(f"  Filled {col} with group mode")
    
    print("✅ Filled from make/model groups")
    return df


def fill_missing_from_make(df):
    """Fill remaining missing values at make (brand) level"""
    fill_cols = ['engine', 'transmission', 'fuel_type', 'body_type', 
                 'drive_train', 'cylinders', 'doors', 'horsepower', 'condition']
    numeric_cols = ['horsepower', 'cylinders', 'doors', 'city_mpg', 'highway_mpg']
    
    print("Filling remaining missing values from make level...")
    
    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            if col in numeric_cols:
                # Fill with make-level mean
                df[col] = df.groupby('make')[col].transform(
                    lambda x: x.fillna(x.mean()) if not x.isnull().all() else x
                )
            else:
                # Fill with make-level mode
                df[col] = df.groupby('make')[col].transform(
                    lambda x: x.fillna(x.mode()[0]) if len(x.mode()) > 0 and not x.isnull().all() else x
                )
                print(f"  Filled {col} with make-level mode")
    
    print("✅ Filled from make groups")
    return df


def fill_missing_at_global(df):
    """Fill remaining missing values using global statistics"""
    fill_cols = ['body_type', 'engine', 'transmission', 'fuel_type', 
                 'drive_train', 'trim', 'condition']
    numeric_cols = ['city_mpg', 'highway_mpg']
    
    print("Filling remaining missing values at global level...")
    
    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                print(f"  Filled {col} with mode: {mode_val[0]}")
    
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"  Filled {col} with mean: {mean_val:.2f}")
    
    return df


def calculate_derived_columns(df, current_year=2026):
    """Calculate derived columns (car_age, miles_per_year, market metrics)"""
    print("Calculating derived columns...")
    
    if 'car_age' not in df.columns or df['car_age'].isnull().any():
        df['car_age'] = df['car_age'].fillna(current_year - df['year'])
        print(f"  Calculated car_age = {current_year} - year")
    
    if 'miles_per_year' not in df.columns or df['miles_per_year'].isnull().any():
        df['miles_per_year'] = df['miles_per_year'].fillna(
            df.apply(lambda row: row['miles'] / row['car_age'] if row['car_age'] > 0 else 0, axis=1)
        )
        print("  Calculated miles_per_year = miles / car_age")
    
    if 'avg_price_model' not in df.columns or df['avg_price_model'].isnull().any():
        avg_price_per_model = df.groupby(['make', 'model'])['price'].transform('mean')
        df['avg_price_model'] = df['avg_price_model'].fillna(avg_price_per_model)
        print("  Calculated avg_price_model = mean(price) by make/model")
    
    if 'price_vs_market' not in df.columns or df['price_vs_market'].isnull().any():
        df['price_vs_market'] = df['price_vs_market'].fillna(
            df['price'] - df['avg_price_model']
        )
        print("  Calculated price_vs_market = price - avg_price_model")
    
    return df


def clean_dataset(df):
    """Comprehensive dataset cleaning pipeline"""
    print("Starting comprehensive data cleaning...")
    
    df = fill_missing_from_make_model(df)
    df = fill_missing_from_make(df)
    df = fill_missing_at_global(df)
    df = calculate_derived_columns(df)
    
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} duplicate rows")
    
    print(f"✅ Data cleaning complete! Final size: {len(df)} rows")
    return df


def create_make_model_mileage_label(df):
    """Create a composite label: make_model_mileage"""
    print("Creating make_model_mileage label...")
    
    df['mileage_range'] = pd.cut(
        df['miles'],
        bins=[0, 20000, 50000, 100000, 150000, 250000],
        labels=['0-20k', '20-50k', '50-100k', '100-150k', '150k+']
    )
    
    df['make_model_mileage'] = (
        df['make'] + '_' + 
        df['model'].str.replace(' ', '_') + '_' + 
        df['mileage_range'].astype(str)
    )
    
    print(f"Created {df['make_model_mileage'].nunique()} unique make_model_mileage groups")
    return df


def create_brand_statistics(df):
    """Create aggregated statistics by brand (make)"""
    print("Calculating brand statistics...")
    
    df['brand_avg_price'] = df['make'].map(df.groupby('make')['price'].mean())
    df['brand_median_price'] = df['make'].map(df.groupby('make')['price'].median())
    df['brand_price_std'] = df['make'].map(df.groupby('make')['price'].std())
    df['brand_car_count'] = df['make'].map(df.groupby('make').size())
    
    print("Created brand statistics features")
    return df


def create_year_statistics(df):
    """Create aggregated statistics by year"""
    print("Calculating year statistics...")
    
    df['year_avg_price'] = df['year'].map(df.groupby('year')['price'].mean())
    df['year_median_price'] = df['year'].map(df.groupby('year')['price'].median())
    
    print("Created year statistics features")
    return df


def create_make_model_statistics(df):
    """Create aggregated statistics by make and model"""
    print("Calculating make/model statistics...")
    
    df['model_avg_price'] = df.groupby(['make', 'model'])['price'].transform('mean')
    df['model_median_price'] = df.groupby(['make', 'model'])['price'].transform('median')
    df['model_price_std'] = df.groupby(['make', 'model'])['price'].transform('std')
    
    print("Created make/model statistics features")
    return df


def create_mileage_range_statistics(df):
    """Create statistics by mileage ranges"""
    print("Calculating mileage range statistics...")
    
    df['mileage_range_avg_price'] = df['mileage_range'].map(
        df.groupby('mileage_range')['price'].mean()
    )
    
    print("Created mileage range statistics features")
    return df


def create_body_type_features(df):
    """Create features for different body types (SUVs, Luxury, etc.)"""
    print("Creating body type classification features...")
    
    suv_types = ['SUV', 'Truck']
    luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Porsche', 'Jaguar', 'Cadillac', 'Lexus']
    sports_types = ['Sports', 'Coupe', 'Convertible']
    
    df['is_suv'] = df['body_type'].isin(suv_types).astype(int)
    df['is_luxury_brand'] = df['make'].isin(luxury_brands).astype(int)
    df['is_sports'] = df['body_type'].isin(sports_types).astype(int)
    df['is_sedan'] = (df['body_type'] == 'Sedan').astype(int)
    df['is_hatchback'] = (df['body_type'] == 'Hatchback').astype(int)
    
    df['suv_avg_price'] = df['is_suv'].map({
        1: df[df['is_suv'] == 1]['price'].mean(),
        0: df[df['is_suv'] == 0]['price'].mean()
    })
    
    df['luxury_brand_avg_price'] = df['is_luxury_brand'].map({
        1: df[df['is_luxury_brand'] == 1]['price'].mean(),
        0: df[df['is_luxury_brand'] == 0]['price'].mean()
    })
    
    print("Created body type classification features")
    return df


def create_price_ratio_features(df):
    """Create ratio-based features for price analysis"""
    print("Creating price ratio features...")
    
    df['price_per_mile'] = df['price'] / (df['miles'] + 1)
    df['price_per_age'] = df['price'] / (df['car_age'] + 0.1)
    
    print("Created price ratio features")
    return df


def encode_categorical_features(df):
    """Encode categorical features using label encoding"""
    print("Encoding categorical features...")
    
    categorical_cols = ['make', 'model', 'body_type', 'transmission', 'fuel_type', 
                       'drive_train', 'condition', 'exterior_color', 'trim', 
                       'make_model_mileage', 'mileage_range']
    
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded {col} with {len(le.classes_)} unique values")
    
    print("✅ Categorical encoding complete")
    return df, encoders


def create_all_features(df):
    """Apply all feature engineering steps"""
    print("Starting comprehensive feature engineering...")
    
    df = create_make_model_mileage_label(df)
    df = create_brand_statistics(df)
    df = create_year_statistics(df)
    df = create_make_model_statistics(df)
    df = create_mileage_range_statistics(df)
    df = create_body_type_features(df)
    df = create_price_ratio_features(df)
    df, encoders = encode_categorical_features(df)
    
    print(f"✅ Feature engineering complete! Final shape: {df.shape}")
    return df, encoders


def prepare_features(df, target_col='price'):
    """Prepare features for modeling"""
    print("Preparing features for modeling...")
    
    # Define feature columns (exclude target and non-numeric)
    exclude_cols = [target_col, 'title', 'url']  # Add any text columns to exclude
    feature_cols = []
    
    for col in df.columns:
        if col not in exclude_cols:
            # Include numeric columns and encoded categorical columns
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32'] or '_encoded' in col:
                feature_cols.append(col)
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Selected {len(feature_cols)} features for modeling")
    print(f"Target variable: {target_col}")
    return X, y, feature_cols


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def train_xgb_model(X_train, y_train, xgb_params=None):
    """Train XGBoost model"""
    if xgb_params is None:
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(**xgb_params, random_state=42)
    model.fit(X_train, y_train)
    
    print("✅ Model training complete")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'y_pred': y_pred
    }
    
    print(f"  MAE (Mean Absolute Error):     ${mae:,.2f}")
    print(f"  RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"  R² Score:                      {r2:.4f}")
    print(f"  MAPE (Mean Absolute % Error):  {mape:.2f}%")
    
    return metrics


def get_feature_importance(model, feature_cols, top_n=15):
    """Get feature importance ranking"""
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    for i, (_, row) in enumerate(importance.head(top_n).iterrows(), 1):
        print(f"  {i}. {row['feature']:<30} {row['importance']:.4f}")
    
    return importance


def make_predictions(model, new_data, feature_cols, encoders):
    """Make predictions on new car data"""
    print("Making predictions on new data...")
    
    # Apply the same preprocessing and feature engineering
    new_data_processed = clean_dataset(new_data.copy())
    new_data_processed, _ = create_all_features(new_data_processed)
    
    # Ensure all required features are present
    missing_features = set(feature_cols) - set(new_data_processed.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default values
        for feat in missing_features:
            new_data_processed[feat] = 0
    
    # Select only the features used in training
    X_new = new_data_processed[feature_cols]
    
    # Make predictions
    predictions = model.predict(X_new)
    
    print(f"✅ Generated {len(predictions)} predictions")
    return predictions


def main():
    """Main modeling pipeline"""
    
    print("\n" + "=" * 100)
    print("CAR PRICE PREDICTION - COMPLETE MODELING PIPELINE")
    print("=" * 100)
    
    # 1. Load Data
    print("\n1️⃣  LOADING DATA")
    print("-" * 100)
    
    data_path = 'data/modeling_data/mexico_cars_complete.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        print("Please ensure mexico_cars_complete.csv exists in the data folder")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # 2. Data Cleaning
    print("\n2️⃣  DATA CLEANING")
    print("-" * 100)
    
    df_clean = clean_dataset(df)
    
    # 3. Feature Engineering
    print("\n3️⃣  FEATURE ENGINEERING")
    print("-" * 100)
    
    df_engineered, encoders = create_all_features(df_clean)
    
    # 4. Model Training
    print("\n4️⃣  MODEL TRAINING")
    print("-" * 100)
    
    # Prepare features and split data
    X, y, feature_cols = prepare_features(df_engineered)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    # Train model
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1
    }
    
    model = train_xgb_model(X_train, y_train, xgb_params)
    
    # 5. Model Evaluation
    print("\n5️⃣  MODEL EVALUATION")
    print("-" * 100)
    
    metrics = evaluate_model(model, X_test, y_test)
    importance = get_feature_importance(model, feature_cols)
    
    # 6. Save Results
    print("\n6️⃣  SAVING RESULTS")
    print("-" * 100)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save model
    with open('models/car_price_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved: models/car_price_xgb_model.pkl")
    
    # Save feature importance
    importance.to_csv('results/feature_importance.csv', index=False)
    print("✅ Feature importance saved: results/feature_importance.csv")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual_price': y_test.values,
        'predicted_price': metrics['y_pred'],
        'error': y_test.values - metrics['y_pred'],
        'error_pct': ((y_test.values - metrics['y_pred']) / y_test.values * 100)
    })
    
    predictions_df.to_csv('results/model_predictions.csv', index=False)
    print("✅ Predictions saved: results/model_predictions.csv")
    
    # Save metrics
    metrics_summary = {
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2_Score': metrics['R2'],
        'MAPE': metrics['MAPE'],
        'Train_Size': len(X_train),
        'Test_Size': len(X_test),
        'Total_Records': len(df),
        'Total_Features': len(feature_cols)
    }
    
    pd.DataFrame([metrics_summary]).to_csv('results/model_metrics.csv', index=False)
    print("✅ Metrics saved: results/model_metrics.csv")
    
    # 7. Summary
    print("\n7️⃣  SUMMARY")
    print("-" * 100)
    print(f"""
    ✨ MODELING COMPLETE!
    
    Dataset: {len(df):,} records with {len(df_engineered.columns)} engineered features
    
    Model Performance:
      • MAE:  ${metrics['MAE']:,.2f}
      • RMSE: ${metrics['RMSE']:,.2f}
      • R²:   {metrics['R2']:.4f}
      • MAPE: {metrics['MAPE']:.2f}%
    
    Files Generated:
      ✅ models/car_price_xgb_model.pkl
      ✅ results/feature_importance.csv
      ✅ results/model_predictions.csv
      ✅ results/model_metrics.csv""")


if __name__ == "__main__":
    main()