"""
Modeling Utilities
Feature engineering and XGBoost model for car price prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarPriceFeatureEngineer:
    """Feature engineering for car price prediction"""
    
    def __init__(self, df):
        """
        Initialize feature engineer
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_stats = {}
        logger.info("Initialized CarPriceFeatureEngineer")
    
    def create_make_model_mileage_label(self):
        """
        Create a composite label: make_model_mileage
        Groups cars by make, model, and mileage range
        
        Returns:
            DataFrame: With new make_model_mileage column
        """
        logger.info("Creating make_model_mileage label...")
        
        # Create mileage bins
        self.df['mileage_range'] = pd.cut(
            self.df['miles'],
            bins=[0, 20000, 50000, 100000, 150000, 250000],
            labels=['0-20k', '20-50k', '50-100k', '100-150k', '150k+']
        )
        
        # Create composite label
        self.df['make_model_mileage'] = (
            self.df['make'] + '_' + 
            self.df['model'].str.replace(' ', '_') + '_' + 
            self.df['mileage_range'].astype(str)
        )
        
        logger.info(f"Created {self.df['make_model_mileage'].nunique()} unique make_model_mileage groups")
        return self.df
    
    def create_brand_statistics(self):
        """Create aggregated statistics by brand (make)"""
        logger.info("Calculating brand statistics...")
        
        brand_stats = self.df.groupby('make').agg({
            'price': ['mean', 'median', 'std', 'count'],
            'miles': 'mean',
            'year': 'mean',
            'car_age': 'mean',
            'cylinders': 'mean',
            'horsepower': 'mean'
        }).round(2)
        
        self.feature_stats['by_brand'] = brand_stats
        
        # Add as features to dataframe
        self.df['brand_avg_price'] = self.df['make'].map(
            self.df.groupby('make')['price'].mean()
        )
        self.df['brand_median_price'] = self.df['make'].map(
            self.df.groupby('make')['price'].median()
        )
        self.df['brand_price_std'] = self.df['make'].map(
            self.df.groupby('make')['price'].std()
        )
        self.df['brand_car_count'] = self.df['make'].map(
            self.df.groupby('make').size()
        )
        
        logger.info("Created brand statistics features")
        return self.df
    
    def create_year_statistics(self):
        """Create aggregated statistics by year"""
        logger.info("Calculating year statistics...")
        
        year_stats = self.df.groupby('year').agg({
            'price': ['mean', 'median'],
            'miles': 'mean',
            'cylinders': 'mean',
            'car_age': 'mean'
        }).round(2)
        
        self.feature_stats['by_year'] = year_stats
        
        # Add as features
        self.df['year_avg_price'] = self.df['year'].map(
            self.df.groupby('year')['price'].mean()
        )
        self.df['year_median_price'] = self.df['year'].map(
            self.df.groupby('year')['price'].median()
        )
        
        logger.info("Created year statistics features")
        return self.df
    
    def create_make_model_statistics(self):
        """Create aggregated statistics by make and model"""
        logger.info("Calculating make/model statistics...")
        
        model_stats = self.df.groupby(['make', 'model']).agg({
            'price': ['mean', 'median', 'std'],
            'miles': 'mean',
            'car_age': 'mean',
            'cylinders': 'mean',
            'horsepower': 'mean',
            'year': ['min', 'max']
        }).round(2)
        
        self.feature_stats['by_make_model'] = model_stats
        
        # Add as features
        self.df['model_avg_price'] = self.df.groupby(['make', 'model'])['price'].transform('mean')
        self.df['model_median_price'] = self.df.groupby(['make', 'model'])['price'].transform('median')
        self.df['model_price_std'] = self.df.groupby(['make', 'model'])['price'].transform('std')
        
        logger.info("Created make/model statistics features")
        return self.df
    
    def create_mileage_range_statistics(self):
        """Create statistics by mileage ranges"""
        logger.info("Calculating mileage range statistics...")
        
        mileage_stats = self.df.groupby('mileage_range').agg({
            'price': ['mean', 'median', 'std', 'count'],
            'car_age': 'mean',
            'year': 'mean'
        }).round(2)
        
        self.feature_stats['by_mileage_range'] = mileage_stats
        
        # Add as features
        self.df['mileage_range_avg_price'] = self.df['mileage_range'].map(
            self.df.groupby('mileage_range')['price'].mean()
        )
        
        logger.info("Created mileage range statistics features")
        return self.df
    
    def create_body_type_features(self):
        """Create features for different body types (SUVs, Luxury, etc.)"""
        logger.info("Creating body type classification features...")
        
        # Define body type categories
        suv_types = ['SUV', 'Truck']
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Porsche', 'Jaguar', 'Cadillac', 'Lexus']
        sports_types = ['Sports', 'Coupe', 'Convertible']
        
        # Create binary features
        self.df['is_suv'] = self.df['body_type'].isin(suv_types).astype(int)
        self.df['is_luxury_brand'] = self.df['make'].isin(luxury_brands).astype(int)
        self.df['is_sports'] = self.df['body_type'].isin(sports_types).astype(int)
        self.df['is_sedan'] = (self.df['body_type'] == 'Sedan').astype(int)
        self.df['is_hatchback'] = (self.df['body_type'] == 'Hatchback').astype(int)
        
        # Calculate average prices for each category
        self.df['suv_avg_price'] = self.df['is_suv'].map({
            1: self.df[self.df['is_suv'] == 1]['price'].mean(),
            0: self.df[self.df['is_suv'] == 0]['price'].mean()
        })
        
        self.df['luxury_brand_avg_price'] = self.df['is_luxury_brand'].map({
            1: self.df[self.df['is_luxury_brand'] == 1]['price'].mean(),
            0: self.df[self.df['is_luxury_brand'] == 0]['price'].mean()
        })
        
        logger.info("Created body type classification features")
        return self.df
    
    def create_price_ratio_features(self):
        """Create ratio-based features for price analysis"""
        logger.info("Creating price ratio features...")
        
        # Price to miles ratio
        self.df['price_per_mile'] = self.df['price'] / (self.df['miles'] + 1)
        
        # Price to age ratio
        self.df['price_per_age'] = self.df['price'] / (self.df['car_age'] + 0.1)
        
        # Price deviation from brand average (as percentage)
        self.df['price_pct_from_brand'] = (
            (self.df['price'] - self.df['brand_avg_price']) / self.df['brand_avg_price'] * 100
        )
        
        # Price deviation from model average
        self.df['price_pct_from_model'] = (
            (self.df['price'] - self.df['model_avg_price']) / self.df['model_avg_price'] * 100
        )
        
        logger.info("Created price ratio features")
        return self.df
    
    def encode_categorical_features(self, categorical_cols=None):
        """
        Encode categorical features
        
        Args:
            categorical_cols: List of categorical columns to encode
        """
        if categorical_cols is None:
            categorical_cols = ['make', 'model', 'body_type', 'fuel_type', 
                               'transmission', 'engine', 'drive_train']
        
        logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
        
        for col in categorical_cols:
            if col in self.df.columns:
                encoder = LabelEncoder()
                self.df[f'{col}_encoded'] = encoder.fit_transform(
                    self.df[col].astype(str)
                )
                self.encoders[col] = encoder
        
        logger.info(f"Encoded {len(self.encoders)} categorical features")
        return self.df
    
    def create_all_features(self):
        """Create all engineered features"""
        logger.info("Creating all engineered features...")
        
        self.create_make_model_mileage_label()
        self.create_brand_statistics()
        self.create_year_statistics()
        self.create_make_model_statistics()
        self.create_mileage_range_statistics()
        self.create_body_type_features()
        self.create_price_ratio_features()
        self.encode_categorical_features()
        
        logger.info(f"✅ Feature engineering complete! Total columns: {len(self.df.columns)}")
        return self.df
    
    def get_feature_importance_data(self):
        """Get feature statistics for analysis"""
        return self.feature_stats


class CarPriceXGBModel:
    """XGBoost model for car price prediction"""
    
    def __init__(self, target='price', test_size=0.2, random_state=42):
        """
        Initialize XGBoost model
        
        Args:
            target: Target column name
            test_size: Test set size
            random_state: Random state for reproducibility
        """
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        logger.info("Initialized CarPriceXGBModel")
    
    def prepare_features(self, df, feature_cols=None):
        """
        Prepare features for modeling
        
        Args:
            df: Input DataFrame with engineered features
            feature_cols: List of feature columns to use
        
        Returns:
            X, y: Features and target
        """
        logger.info("Preparing features for modeling...")
        
        if feature_cols is None:
            # Use all numeric columns except target and identifiers
            exclude_cols = [self.target, 'make_model_mileage', 'mileage_range', 
                           'id', 'title', 'vin', 'data_source', 'currency', 'state', 'city']
            feature_cols = [col for col in df.columns 
                          if df[col].dtype in [np.int64, np.float64]
                          and col not in exclude_cols
                          and col != self.target]
        
        X = df[feature_cols].copy()
        y = df[self.target].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        logger.info(f"Using {len(feature_cols)} features for modeling")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def train_test_split_data(self, X, y):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
        """
        logger.info(f"Splitting data: {self.test_size*100:.1f}% test set...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
    
    def train_model(self, X_train=None, y_train=None, **xgb_params):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features (uses self.X_train if None)
            y_train: Training target (uses self.y_train if None)
            **xgb_params: XGBoost parameters
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        logger.info("Training XGBoost model...")
        
        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Update with provided parameters
        default_params.update(xgb_params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(X_train, y_train, verbose=0)
        
        logger.info("✅ Model training complete!")
    
    def evaluate_model(self, X_test=None, y_test=None):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features (uses self.X_test if None)
            y_test: Test target (uses self.y_test if None)
        
        Returns:
            dict: Evaluation metrics
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        logger.info("Evaluating model performance...")
        
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'y_pred': y_pred
        }
        
        logger.info(f"\n📊 Model Performance Metrics:")
        logger.info(f"  MAE (Mean Absolute Error):  ${mae:,.2f}")
        logger.info(f"  RMSE (Root Mean Squared Error): ${rmse:,.2f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  MAPE (Mean Absolute % Error): {mape:.2f}%")
        
        return self.metrics
    
    def get_feature_importance(self, top_n=15):
        """
        Get feature importance from the model
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame: Feature importance ranking
        """
        if self.model is None:
            logger.warning("Model not trained yet!")
            return None
        
        importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n🎯 Top {top_n} Most Important Features:")
        for idx, row in importance.head(top_n).iterrows():
            print(f"  {row['feature']:35s} {row['importance']*100:6.2f}%")
        
        return importance
    
    def make_prediction(self, sample_df, feature_cols):
        """
        Make price prediction for new samples
        
        Args:
            sample_df: DataFrame with features
            feature_cols: List of feature columns
        
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            logger.warning("Model not trained yet!")
            return None
        
        X_sample = sample_df[feature_cols].fillna(sample_df[feature_cols].mean())
        predictions = self.model.predict(X_sample)
        
        return predictions
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            logger.warning("Model not trained yet!")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"✅ Model loaded from {filepath}")


def full_modeling_pipeline(df, test_size=0.2, xgb_params=None):
    """
    Complete modeling pipeline from feature engineering to evaluation
    
    Args:
        df: Input DataFrame
        test_size: Test set size
        xgb_params: XGBoost parameters
    
    Returns:
        tuple: (feature_engineer, model, metrics)
    """
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE MODELING PIPELINE")
    logger.info("=" * 80)
    
    # Feature Engineering
    engineer = CarPriceFeatureEngineer(df)
    df_engineered = engineer.create_all_features()
    
    # Model Preparation
    model = CarPriceXGBModel(target='price', test_size=test_size)
    X, y, feature_cols = model.prepare_features(df_engineered)
    model.train_test_split_data(X, y)
    
    # Model Training
    if xgb_params is None:
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    model.train_model(**xgb_params)
    
    # Model Evaluation
    metrics = model.evaluate_model()
    
    # Feature Importance
    importance = model.get_feature_importance(top_n=15)
    
    logger.info("=" * 80)
    logger.info("✅ MODELING PIPELINE COMPLETE!")
    logger.info("=" * 80)
    
    return engineer, model, metrics, importance


if __name__ == "__main__":
    print("Modeling Utilities Module")
    print("Import this module and use the classes for car price prediction")
