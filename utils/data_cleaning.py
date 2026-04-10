"""
Utilidades de Limpieza de Datos
Módulo con funciones de limpieza y preprocesamiento de datos para análisis de precios de autos
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def string_similarity(str1, str2):
    """
    Calcula el ratio de similitud entre dos cadenas de texto

    Args:
        str1: Primera cadena
        str2: Segunda cadena

    Returns:
        float: Ratio de similitud (0-1)
    """
    return SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()


def fill_from_group(group, cols, numeric_cols=None):
    """
    Rellena valores faltantes en un grupo a partir de otras observaciones

    Args:
        group: Grupo del DataFrame
        cols: Lista de columnas a rellenar
        numeric_cols: Lista de columnas numéricas para usar la media

    Returns:
        DataFrame: Grupo con valores rellenados
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
    Rellena valores faltantes agrupando por marca y modelo

    Args:
        df: DataFrame de entrada
        fill_cols: Lista de columnas a rellenar
        numeric_cols: Lista de columnas numéricas

    Returns:
        DataFrame: DataFrame con valores rellenados
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
    Rellena los valores faltantes restantes a nivel de marca

    Args:
        df: DataFrame de entrada
        fill_cols: Lista de columnas a rellenar
        numeric_cols: Lista de columnas numéricas

    Returns:
        DataFrame: DataFrame con valores rellenados
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
    Rellena los valores faltantes restantes usando estadísticas globales

    Args:
        df: DataFrame de entrada
        fill_cols: Lista de columnas a rellenar
        numeric_cols: Lista de columnas numéricas

    Returns:
        DataFrame: DataFrame con valores rellenados
    """
    if fill_cols is None:
        fill_cols = ['body_type', 'engine', 'transmission', 'fuel_type', 
                     'drive_train', 'trim', 'condition']
    
    if numeric_cols is None:
        numeric_cols = ['city_mpg', 'highway_mpg']
    
    logger.info(f"Rellenando valores faltantes restantes a nivel global...")

    # Columnas categóricas - usar moda
    for col in fill_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                logger.info(f"  Filled {col} with mode: {mode_val[0]}")
    
    # Columnas numéricas - usar media
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            logger.info(f"  Filled {col} with mean: {mean_val:.2f}")
    
    return df


def calculate_derived_columns(df, current_year=2026):
    """
    Calcula columnas derivadas (car_age, miles_per_year, métricas de mercado)

    Args:
        df: DataFrame de entrada
        current_year: Año actual para el cálculo de antigüedad

    Returns:
        DataFrame: DataFrame con columnas calculadas
    """
    logger.info("Calculando columnas derivadas...")

    # car_age: current_year - year
    if 'car_age' not in df.columns or df['car_age'].isnull().any():
        df['car_age'] = df['car_age'].fillna(current_year - df['year'])
        logger.info(f"  Calculado car_age = {current_year} - year")

    # miles_per_year: miles / car_age
    if 'miles_per_year' not in df.columns or df['miles_per_year'].isnull().any():
        df['miles_per_year'] = df['miles_per_year'].fillna(
            df.apply(lambda row: row['miles'] / row['car_age'] if row['car_age'] > 0 else 0, axis=1)
        )
        logger.info(f"  Calculado miles_per_year = miles / car_age")

    # avg_price_model: precio promedio por marca/modelo
    if 'avg_price_model' not in df.columns or df['avg_price_model'].isnull().any():
        avg_price_per_model = df.groupby(['make', 'model'])['price'].transform('mean')
        df['avg_price_model'] = df['avg_price_model'].fillna(avg_price_per_model)
        logger.info(f"  Calculado avg_price_model = mean(price) por marca/modelo")

    # price_vs_market: price - avg_price_model
    if 'price_vs_market' not in df.columns or df['price_vs_market'].isnull().any():
        df['price_vs_market'] = df['price_vs_market'].fillna(
            df['price'] - df['avg_price_model']
        )
        logger.info(f"  Calculado price_vs_market = price - avg_price_model")
    
    return df


def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Elimina valores atípicos de una columna

    Args:
        df: DataFrame de entrada
        column: Nombre de la columna
        method: 'iqr' para rango intercuartil, 'zscore' para z-score
        threshold: Multiplicador IQR (por defecto 1.5) o umbral z-score (por defecto 3)

    Returns:
        DataFrame: DataFrame sin valores atípicos
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
    Estandariza nombres y tipos de columnas

    Args:
        df: DataFrame de entrada
        columns_mapping: Diccionario con mapeo de nombres de columnas

    Returns:
        DataFrame: DataFrame con columnas estandarizadas
    """
    if columns_mapping:
        df = df.rename(columns=columns_mapping)
        logger.info(f"Columnas renombradas: {columns_mapping}")

    # Asegurar que las columnas clave tengan los tipos correctos
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
    Pipeline completo de limpieza del dataset

    Args:
        df: DataFrame de entrada
        remove_outliers_flag: Si se deben eliminar valores atípicos
        outlier_column: Columna a revisar para valores atípicos
        current_year: Año actual para los cálculos

    Returns:
        DataFrame: DataFrame limpio
    """
    logger.info("Iniciando limpieza exhaustiva de datos...")

    # Paso 1: Rellenar desde grupos marca/modelo y marca
    df = fill_missing_from_make_model(df)
    df = fill_missing_from_make(df)
    df = fill_missing_at_global(df)

    # Paso 2: Calcular columnas derivadas
    df = calculate_derived_columns(df, current_year)

    # Paso 3: Eliminar duplicados si existen
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        logger.info(f"Eliminadas {initial_len - len(df)} filas duplicadas")

    # Paso 4: Eliminación opcional de valores atípicos
    if remove_outliers_flag and outlier_column in df.columns:
        df = remove_outliers(df, outlier_column, method='iqr', threshold=1.5)
    
    logger.info(f"✅ Data cleaning complete! Final size: {len(df)} rows")
    return df


def get_data_summary(df):
    """
    Obtiene un resumen completo de la calidad del dataset

    Args:
        df: DataFrame de entrada

    Returns:
        dict: Estadísticas resumen
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
    # Ejemplo de uso
    print("Módulo de Limpieza de Datos")
    print("Importa este módulo y usa las funciones para el preprocesamiento de datos")
