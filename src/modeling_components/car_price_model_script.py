"""
Modelo de Predicción de Precios de Autos - Versión Script
Convertido del notebook ilustrativo para uso en producción
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar estilo para gráficas
plt.style.use('default')
sns.set_palette("husl")


def fill_missing_from_make_model(df):
    """Rellena valores faltantes agrupando por marca y modelo"""
    fill_cols = ['engine', 'transmission', 'fuel_type', 'body_type',
                 'drive_train', 'cylinders', 'doors', 'horsepower',
                 'trim', 'exterior_color', 'condition']
    numeric_cols = ['horsepower', 'cylinders', 'doors', 'city_mpg', 'highway_mpg']

    print("Rellenando valores faltantes desde grupos marca/modelo...")

    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            if col in numeric_cols:
                # Rellenar con la media del grupo
                df[col] = df.groupby(['make', 'model'])[col].transform(
                    lambda x: x.fillna(x.mean()) if not x.isnull().all() else x
                )
            else:
                # Rellenar con la moda del grupo
                df[col] = df.groupby(['make', 'model'])[col].transform(
                    lambda x: x.fillna(x.mode()[0]) if len(x.mode()) > 0 and not x.isnull().all() else x
                )
                print(f"  Rellenado {col} con moda del grupo")

    print("✅ Rellenado desde grupos marca/modelo")
    return df


def fill_missing_from_make(df):
    """Rellena los valores faltantes restantes a nivel de marca"""
    fill_cols = ['engine', 'transmission', 'fuel_type', 'body_type',
                 'drive_train', 'cylinders', 'doors', 'horsepower', 'condition']
    numeric_cols = ['horsepower', 'cylinders', 'doors', 'city_mpg', 'highway_mpg']

    print("Rellenando valores faltantes restantes desde nivel de marca...")

    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            if col in numeric_cols:
                # Rellenar con la media a nivel de marca
                df[col] = df.groupby('make')[col].transform(
                    lambda x: x.fillna(x.mean()) if not x.isnull().all() else x
                )
            else:
                # Rellenar con la moda a nivel de marca
                df[col] = df.groupby('make')[col].transform(
                    lambda x: x.fillna(x.mode()[0]) if len(x.mode()) > 0 and not x.isnull().all() else x
                )
                print(f"  Rellenado {col} con moda a nivel de marca")

    print("✅ Rellenado desde grupos de marca")
    return df


def fill_missing_at_global(df):
    """Rellena los valores faltantes restantes usando estadísticas globales"""
    fill_cols = ['body_type', 'engine', 'transmission', 'fuel_type',
                 'drive_train', 'trim', 'condition']
    numeric_cols = ['city_mpg', 'highway_mpg']

    print("Rellenando valores faltantes restantes a nivel global...")
    
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
    """Calcula columnas derivadas (car_age, miles_per_year, métricas de mercado)"""
    print("Calculando columnas derivadas...")

    if 'car_age' not in df.columns or df['car_age'].isnull().any():
        df['car_age'] = df['car_age'].fillna(current_year - df['year'])
        print(f"  Calculado car_age = {current_year} - year")

    if 'miles_per_year' not in df.columns or df['miles_per_year'].isnull().any():
        df['miles_per_year'] = df['miles_per_year'].fillna(
            df.apply(lambda row: row['miles'] / row['car_age'] if row['car_age'] > 0 else 0, axis=1)
        )
        print("  Calculado miles_per_year = miles / car_age")

    if 'avg_price_model' not in df.columns or df['avg_price_model'].isnull().any():
        avg_price_per_model = df.groupby(['make', 'model'])['price'].transform('mean')
        df['avg_price_model'] = df['avg_price_model'].fillna(avg_price_per_model)
        print("  Calculado avg_price_model = mean(price) por marca/modelo")

    if 'price_vs_market' not in df.columns or df['price_vs_market'].isnull().any():
        df['price_vs_market'] = df['price_vs_market'].fillna(
            df['price'] - df['avg_price_model']
        )
        print("  Calculado price_vs_market = price - avg_price_model")

    return df


def clean_dataset(df):
    """Pipeline completo de limpieza del dataset"""
    print("Iniciando limpieza exhaustiva de datos...")
    
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
    """Crea una etiqueta compuesta: make_model_mileage"""
    print("Creando etiqueta make_model_mileage...")
    
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
    """Crea estadísticas agregadas por marca"""
    print("Calculando estadísticas por marca...")
    
    df['brand_avg_price'] = df['make'].map(df.groupby('make')['price'].mean())
    df['brand_median_price'] = df['make'].map(df.groupby('make')['price'].median())
    df['brand_price_std'] = df['make'].map(df.groupby('make')['price'].std())
    df['brand_car_count'] = df['make'].map(df.groupby('make').size())
    
    print("Created brand statistics features")
    return df


def create_year_statistics(df):
    """Crea estadísticas agregadas por año"""
    print("Calculando estadísticas por año...")
    
    df['year_avg_price'] = df['year'].map(df.groupby('year')['price'].mean())
    df['year_median_price'] = df['year'].map(df.groupby('year')['price'].median())
    
    print("Created year statistics features")
    return df


def create_make_model_statistics(df):
    """Crea estadísticas agregadas por marca y modelo"""
    print("Calculando estadísticas por marca/modelo...")
    
    df['model_avg_price'] = df.groupby(['make', 'model'])['price'].transform('mean')
    df['model_median_price'] = df.groupby(['make', 'model'])['price'].transform('median')
    df['model_price_std'] = df.groupby(['make', 'model'])['price'].transform('std')
    
    print("Created make/model statistics features")
    return df


def create_mileage_range_statistics(df):
    """Crea estadísticas por rangos de kilometraje"""
    print("Calculando estadísticas por rango de kilometraje...")
    
    df['mileage_range_avg_price'] = df['mileage_range'].map(
        df.groupby('mileage_range')['price'].mean()
    )
    
    print("Created mileage range statistics features")
    return df


def create_body_type_features(df):
    """Crea características para diferentes tipos de carrocería (SUVs, Lujo, etc.)"""
    print("Creando características de clasificación por tipo de carrocería...")

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
    """Crea características basadas en ratios para el análisis de precios"""
    print("Creando características de ratio de precio...")
    
    df['price_per_mile'] = df['price'] / (df['miles'] + 1)
    df['price_per_age'] = df['price'] / (df['car_age'] + 0.1)
    
    print("Created price ratio features")
    return df


def encode_categorical_features(df):
    """Codifica características categóricas usando label encoding"""
    print("Codificando características categóricas...")
    
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
    """Aplica todos los pasos de ingeniería de características"""
    print("Iniciando ingeniería exhaustiva de características...")
    
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
    """Prepara las características para el modelado"""
    print("Preparando características para el modelado...")

    # Definir columnas de características (excluir objetivo y no numéricas)
    exclude_cols = [target_col, 'title', 'url']  # Agregar columnas de texto a excluir
    feature_cols = []

    for col in df.columns:
        if col not in exclude_cols:
            # Incluir columnas numéricas y categóricas codificadas
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32'] or '_encoded' in col:
                feature_cols.append(col)
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Selected {len(feature_cols)} features for modeling")
    print(f"Target variable: {target_col}")
    return X, y, feature_cols


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def train_xgb_model(X_train, y_train, xgb_params=None):
    """Entrena el modelo XGBoost"""
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
    """Evalúa el rendimiento del modelo"""
    print("Evaluando rendimiento del modelo...")
    
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
    """Obtiene el ranking de importancia de características"""
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    for i, (_, row) in enumerate(importance.head(top_n).iterrows(), 1):
        print(f"  {i}. {row['feature']:<30} {row['importance']:.4f}")
    
    return importance


def make_predictions(model, new_data, feature_cols, encoders):
    """Realiza predicciones sobre nuevos datos de autos"""
    print("Generando predicciones sobre nuevos datos...")

    # Aplicar el mismo preprocesamiento e ingeniería de características
    new_data_processed = clean_dataset(new_data.copy())
    new_data_processed, _ = create_all_features(new_data_processed)

    # Asegurar que todas las características requeridas estén presentes
    missing_features = set(feature_cols) - set(new_data_processed.columns)
    if missing_features:
        print(f"Advertencia: Características faltantes: {missing_features}")
        # Agregar características faltantes con valores por defecto
        for feat in missing_features:
            new_data_processed[feat] = 0

    # Seleccionar solo las características usadas en el entrenamiento
    X_new = new_data_processed[feature_cols]

    # Realizar predicciones
    predictions = model.predict(X_new)
    
    print(f"✅ Generated {len(predictions)} predictions")
    return predictions


def main():
    """Pipeline principal de modelado"""

    print("\n" + "=" * 100)
    print("PREDICCIÓN DE PRECIOS DE AUTOS - PIPELINE COMPLETO DE MODELADO")
    print("=" * 100)

    # 1. Cargar Datos
    print("\n1️⃣  CARGANDO DATOS")
    print("-" * 100)
    
    data_path = 'data/modeling_data/mexico_cars_complete.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        print("Please ensure mexico_cars_complete.csv exists in the data folder")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    
    # 2. Limpieza de Datos
    print("\n2️⃣  LIMPIEZA DE DATOS")
    print("-" * 100)

    df_clean = clean_dataset(df)

    # 3. Ingeniería de Características
    print("\n3️⃣  INGENIERÍA DE CARACTERÍSTICAS")
    print("-" * 100)

    df_engineered, encoders = create_all_features(df_clean)

    # 4. Entrenamiento del Modelo
    print("\n4️⃣  ENTRENAMIENTO DEL MODELO")
    print("-" * 100)

    # Preparar características y dividir datos
    X, y, feature_cols = prepare_features(df_engineered)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Entrenar modelo
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
    
    # 5. Evaluación del Modelo
    print("\n5️⃣  EVALUACIÓN DEL MODELO")
    print("-" * 100)
    
    metrics = evaluate_model(model, X_test, y_test)
    importance = get_feature_importance(model, feature_cols)
    
    # 6. Guardar Resultados
    print("\n6️⃣  GUARDANDO RESULTADOS")
    print("-" * 100)

    # Crear directorios
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Guardar modelo
    with open('models/car_price_xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved: models/car_price_xgb_model.pkl")
    
    # Guardar importancia de características
    importance.to_csv('results/feature_importance.csv', index=False)
    print("✅ Importancia de características guardada: results/feature_importance.csv")

    # Guardar predicciones
    predictions_df = pd.DataFrame({
        'actual_price': y_test.values,
        'predicted_price': metrics['y_pred'],
        'error': y_test.values - metrics['y_pred'],
        'error_pct': ((y_test.values - metrics['y_pred']) / y_test.values * 100)
    })
    
    predictions_df.to_csv('results/model_predictions.csv', index=False)
    print("✅ Predictions saved: results/model_predictions.csv")
    
    # Guardar métricas
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
    
    # 7. Resumen
    print("\n7️⃣  RESUMEN")
    print("-" * 100)
    print(f"""
    ✨ ¡MODELADO COMPLETO!

    Dataset: {len(df):,} registros con {len(df_engineered.columns)} características engineered

    Rendimiento del Modelo:
      • MAE:  ${metrics['MAE']:,.2f}
      • RMSE: ${metrics['RMSE']:,.2f}
      • R²:   {metrics['R2']:.4f}
      • MAPE: {metrics['MAPE']:.2f}%

    Archivos Generados:
      ✅ models/car_price_xgb_model.pkl
      ✅ results/feature_importance.csv
      ✅ results/model_predictions.csv
      ✅ results/model_metrics.csv""")


if __name__ == "__main__":
    main()