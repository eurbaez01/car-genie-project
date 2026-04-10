"""
Utilidades de Modelado
Ingeniería de características y modelo XGBoost para la predicción de precios de autos
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
    """Ingeniería de características para la predicción de precios de autos"""

    def __init__(self, df):
        """
        Inicializa el ingeniero de características

        Args:
            df: DataFrame de entrada
        """
        self.df = df.copy()
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_stats = {}
        logger.info("Initialized CarPriceFeatureEngineer")
    
    def create_make_model_mileage_label(self):
        """
        Crea una etiqueta compuesta: make_model_mileage
        Agrupa autos por marca, modelo y rango de kilometraje

        Returns:
            DataFrame: Con la nueva columna make_model_mileage
        """
        logger.info("Creando etiqueta make_model_mileage...")

        # Crear bins de kilometraje
        self.df['mileage_range'] = pd.cut(
            self.df['miles'],
            bins=[0, 20000, 50000, 100000, 150000, 250000],
            labels=['0-20k', '20-50k', '50-100k', '100-150k', '150k+']
        )
        
        # Crear etiqueta compuesta
        self.df['make_model_mileage'] = (
            self.df['make'] + '_' + 
            self.df['model'].str.replace(' ', '_') + '_' + 
            self.df['mileage_range'].astype(str)
        )
        
        logger.info(f"Created {self.df['make_model_mileage'].nunique()} unique make_model_mileage groups")
        return self.df
    
    def create_brand_statistics(self):
        """Crea estadísticas agregadas por marca"""
        logger.info("Calculando estadísticas por marca...")

        brand_stats = self.df.groupby('make').agg({
            'price': ['mean', 'median', 'std', 'count'],
            'miles': 'mean',
            'year': 'mean',
            'car_age': 'mean',
            'cylinders': 'mean',
            'horsepower': 'mean'
        }).round(2)
        
        self.feature_stats['by_brand'] = brand_stats
        
        # Agregar como características al dataframe
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
        """Crea estadísticas agregadas por año"""
        logger.info("Calculando estadísticas por año...")

        year_stats = self.df.groupby('year').agg({
            'price': ['mean', 'median'],
            'miles': 'mean',
            'cylinders': 'mean',
            'car_age': 'mean'
        }).round(2)
        
        self.feature_stats['by_year'] = year_stats
        
        # Agregar como características
        self.df['year_avg_price'] = self.df['year'].map(
            self.df.groupby('year')['price'].mean()
        )
        self.df['year_median_price'] = self.df['year'].map(
            self.df.groupby('year')['price'].median()
        )
        
        logger.info("Created year statistics features")
        return self.df
    
    def create_make_model_statistics(self):
        """Crea estadísticas agregadas por marca y modelo"""
        logger.info("Calculando estadísticas por marca/modelo...")
        
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
        """Crea estadísticas por rangos de kilometraje"""
        logger.info("Calculando estadísticas por rango de kilometraje...")
        
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
        """Crea características para diferentes tipos de carrocería (SUVs, Lujo, etc.)"""
        logger.info("Creando características de clasificación por tipo de carrocería...")

        # Definir categorías de tipo de carrocería
        suv_types = ['SUV', 'Truck']
        luxury_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Porsche', 'Jaguar', 'Cadillac', 'Lexus']
        sports_types = ['Sports', 'Coupe', 'Convertible']

        # Crear características binarias
        self.df['is_suv'] = self.df['body_type'].isin(suv_types).astype(int)
        self.df['is_luxury_brand'] = self.df['make'].isin(luxury_brands).astype(int)
        self.df['is_sports'] = self.df['body_type'].isin(sports_types).astype(int)
        self.df['is_sedan'] = (self.df['body_type'] == 'Sedan').astype(int)
        self.df['is_hatchback'] = (self.df['body_type'] == 'Hatchback').astype(int)
        
        # Calcular precios promedio para cada categoría
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
        """Crea características basadas en ratios para el análisis de precios"""
        logger.info("Creando características de ratio de precio...")

        # Ratio precio / kilometraje
        self.df['price_per_mile'] = self.df['price'] / (self.df['miles'] + 1)

        # Ratio precio / antigüedad
        self.df['price_per_age'] = self.df['price'] / (self.df['car_age'] + 0.1)

        # Desviación del precio respecto al promedio de la marca (en porcentaje)
        self.df['price_pct_from_brand'] = (
            (self.df['price'] - self.df['brand_avg_price']) / self.df['brand_avg_price'] * 100
        )
        
        # Desviación del precio respecto al promedio del modelo
        self.df['price_pct_from_model'] = (
            (self.df['price'] - self.df['model_avg_price']) / self.df['model_avg_price'] * 100
        )
        
        logger.info("Created price ratio features")
        return self.df
    
    def encode_categorical_features(self, categorical_cols=None):
        """
        Codifica características categóricas

        Args:
            categorical_cols: Lista de columnas categóricas a codificar
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
        """Crea todas las características engineered"""
        logger.info("Creando todas las características engineered...")
        
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
        """Obtiene estadísticas de características para análisis"""
        return self.feature_stats


class CarPriceXGBModel:
    """Modelo XGBoost para la predicción de precios de autos"""

    def __init__(self, target='price', test_size=0.2, random_state=42):
        """
        Inicializa el modelo XGBoost

        Args:
            target: Nombre de la columna objetivo
            test_size: Tamaño del conjunto de prueba
            random_state: Semilla aleatoria para reproducibilidad
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
        Prepara las características para el modelado

        Args:
            df: DataFrame de entrada con características engineered
            feature_cols: Lista de columnas de características a usar

        Returns:
            X, y: Características y objetivo
        """
        logger.info("Preparando características para el modelado...")

        if feature_cols is None:
            # Usar todas las columnas numéricas excepto objetivo e identificadores
            exclude_cols = [self.target, 'make_model_mileage', 'mileage_range',
                           'id', 'title', 'vin', 'data_source', 'currency', 'state', 'city']
            feature_cols = [col for col in df.columns
                          if df[col].dtype in [np.int64, np.float64]
                          and col not in exclude_cols
                          and col != self.target]

        X = df[feature_cols].copy()
        y = df[self.target].copy()

        # Manejar valores NaN restantes
        X = X.fillna(X.mean())
        
        logger.info(f"Using {len(feature_cols)} features for modeling")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def train_test_split_data(self, X, y):
        """
        Divide los datos en conjuntos de entrenamiento y prueba

        Args:
            X: Características
            y: Objetivo
        """
        logger.info(f"Dividiendo datos: {self.test_size*100:.1f}% conjunto de prueba...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
    
    def train_model(self, X_train=None, y_train=None, **xgb_params):
        """
        Entrena el modelo XGBoost

        Args:
            X_train: Características de entrenamiento (usa self.X_train si es None)
            y_train: Objetivo de entrenamiento (usa self.y_train si es None)
            **xgb_params: Parámetros de XGBoost
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        logger.info("Training XGBoost model...")
        
        # Parámetros por defecto de XGBoost
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
        
        # Actualizar con los parámetros proporcionados
        default_params.update(xgb_params)
        
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(X_train, y_train, verbose=0)
        
        logger.info("✅ Model training complete!")
    
    def evaluate_model(self, X_test=None, y_test=None):
        """
        Evalúa el rendimiento del modelo

        Args:
            X_test: Características de prueba (usa self.X_test si es None)
            y_test: Objetivo de prueba (usa self.y_test si es None)

        Returns:
            dict: Métricas de evaluación
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
        
        # Calcular MAPE (Error Porcentual Absoluto Medio)
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
        Obtiene la importancia de características del modelo

        Args:
            top_n: Número de características principales a retornar

        Returns:
            DataFrame: Ranking de importancia de características
        """
        if self.model is None:
            logger.warning("¡El modelo aún no ha sido entrenado!")
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
        Realiza predicciones de precio para nuevas muestras

        Args:
            sample_df: DataFrame con características
            feature_cols: Lista de columnas de características

        Returns:
            np.array: Predicciones
        """
        if self.model is None:
            logger.warning("¡El modelo aún no ha sido entrenado!")
            return None
        
        X_sample = sample_df[feature_cols].fillna(sample_df[feature_cols].mean())
        predictions = self.model.predict(X_sample)
        
        return predictions
    
    def save_model(self, filepath):
        """Guarda el modelo en un archivo"""
        if self.model is None:
            logger.warning("¡El modelo aún no ha sido entrenado!")
            return
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Carga el modelo desde un archivo"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"✅ Model loaded from {filepath}")


def full_modeling_pipeline(df, test_size=0.2, xgb_params=None):
    """
    Pipeline completo de modelado desde ingeniería de características hasta evaluación

    Args:
        df: DataFrame de entrada
        test_size: Tamaño del conjunto de prueba
        xgb_params: Parámetros de XGBoost

    Returns:
        tuple: (feature_engineer, model, metrics)
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE COMPLETO DE MODELADO")
    logger.info("=" * 80)

    # Ingeniería de Características
    engineer = CarPriceFeatureEngineer(df)
    df_engineered = engineer.create_all_features()

    # Preparación del Modelo
    model = CarPriceXGBModel(target='price', test_size=test_size)
    X, y, feature_cols = model.prepare_features(df_engineered)
    model.train_test_split_data(X, y)

    # Entrenamiento del Modelo
    if xgb_params is None:
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    model.train_model(**xgb_params)
    
    # Evaluación del Modelo
    metrics = model.evaluate_model()

    # Importancia de Características
    importance = model.get_feature_importance(top_n=15)

    logger.info("=" * 80)
    logger.info("✅ ¡PIPELINE DE MODELADO COMPLETO!")
    logger.info("=" * 80)
    
    return engineer, model, metrics, importance


if __name__ == "__main__":
    print("Módulo de Utilidades de Modelado")
    print("Importa este módulo y usa las clases para la predicción de precios de autos")
