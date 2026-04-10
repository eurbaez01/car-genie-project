"""
Predicción de Precios de Autos - Ejemplo Completo de Modelado
Demuestra el pipeline completo desde la carga de datos hasta el entrenamiento y evaluación del modelo
"""

import pandas as pd
import numpy as np
import sys
import os

# Agregar utils al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import (
    clean_dataset,
    get_data_summary,
    CarPriceFeatureEngineer,
    CarPriceXGBModel,
    full_modeling_pipeline
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    # 2. Resumen de Datos
    print("\n2️⃣  RESUMEN DE DATOS")
    print("-" * 100)
    
    summary = get_data_summary(df)
    print(f"Total records: {summary['total_records']:,}")
    print(f"Total columns: {summary['total_columns']}")
    print(f"Fully complete columns: {summary['complete_columns']}/{summary['total_columns']}")
    print(f"Numeric columns: {len(summary['numeric_columns'])}")
    print(f"Categorical columns: {len(summary['categorical_columns'])}")
    
    # 3. Info de Datos
    print("\n3️⃣  ESTADÍSTICAS CLAVE")
    print("-" * 100)
    
    print(f"Price Range:     ${df['price'].min():>12,.0f} - ${df['price'].max():>12,.0f} (avg: ${df['price'].mean():12,.0f})")
    print(f"Mileage Range:   {df['miles'].min():>13,.0f} - {df['miles'].max():>13,.0f} (avg: {df['miles'].mean():13,.0f})")
    print(f"Car Age Range:   {df['car_age'].min():>11.1f} - {df['car_age'].max():>11.1f} years (avg: {df['car_age'].mean():8.1f})")
    print(f"Year Range:      {int(df['year'].min()):>11d} - {int(df['year'].max()):>11d}")
    print(f"\nTop 5 Brands:")
    for i, (brand, count) in enumerate(df['make'].value_counts().head(5).items(), 1):
        print(f"  {i}. {brand:20s} {count:4d} records ({count/len(df)*100:5.1f}%)")
    
    # 4. Ejecutar Pipeline Completo
    print("\n4️⃣  EJECUTANDO PIPELINE COMPLETO DE MODELADO")
    print("-" * 100)

    # Parámetros de XGBoost
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
    
    engineer, model, metrics, importance = full_modeling_pipeline(
        df, 
        test_size=0.2, 
        xgb_params=xgb_params
    )
    
    # 5. Resultados
    print("\n5️⃣  RESULTADOS DEL MODELO")
    print("-" * 100)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  MAE (Mean Absolute Error):        ${metrics['MAE']:>12,.2f}")
    print(f"  RMSE (Root Mean Squared Error):   ${metrics['RMSE']:>12,.2f}")
    print(f"  R² Score:                         {metrics['R2']:>14.4f}")
    print(f"  MAPE (Mean Absolute % Error):     {metrics['MAPE']:>13.2f}%")
    
    # 6. Estadísticas de Características
    print("\n6️⃣  ESTADÍSTICAS DE CARACTERÍSTICAS")
    print("-" * 100)
    
    feature_stats = engineer.get_feature_importance_data()
    
    if 'by_brand' in feature_stats:
        print("\n🏢 Brand Statistics (Top 5):")
        brand_stats = feature_stats['by_brand'].sort_values(('price', 'mean'), ascending=False)
        for idx, (brand, row) in enumerate(brand_stats.head(5).iterrows(), 1):
            avg_price = row[('price', 'mean')]
            count = row[('price', 'count')]
            print(f"  {idx}. {brand:20s} Avg: ${avg_price:>12,.0f} ({int(count):3.0f} cars)")
    
    if 'by_make_model' in feature_stats:
        print("\n🚗 Model Statistics (Top 5 by avg price):")
        model_stats = feature_stats['by_make_model'].sort_values(('price', 'mean'), ascending=False)
        for idx, ((make, model_name), row) in enumerate(model_stats.head(5).iterrows(), 1):
            avg_price = row[('price', 'mean')]
            print(f"  {idx}. {make:15s} {model_name:20s} Avg: ${avg_price:>12,.0f}")
    
    if 'by_year' in feature_stats:
        print("\n📅 Year Statistics (Last 5 years):")
        year_stats = feature_stats['by_year'].sort_index(ascending=False)
        for year, row in year_stats.head(5).iterrows():
            avg_price = row[('price', 'mean')]
            print(f"  {int(year)} Avg Price: ${avg_price:>12,.0f}")
    
    # 7. Guardar Modelo
    print("\n7️⃣  GUARDANDO MODELO")
    print("-" * 100)
    
    model_path = 'models/car_price_xgb_model.pkl'
    os.makedirs('models', exist_ok=True)
    model.save_model(model_path)
    print(f"✅ Model saved: {model_path}")
    
    # 8. Exportar Importancia de Características
    print("\n8️⃣  EXPORTANDO RESULTADOS")
    print("-" * 100)

    # Guardar importancia de características
    importance.to_csv('results/feature_importance.csv', index=False)
    print(f"✅ Feature importance saved: results/feature_importance.csv")
    
    # Guardar predicciones
    predictions_df = pd.DataFrame({
        'actual_price': model.y_test.values,
        'predicted_price': metrics['y_pred'],
        'error': model.y_test.values - metrics['y_pred'],
        'error_pct': ((model.y_test.values - metrics['y_pred']) / model.y_test.values * 100)
    })
    
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv('results/model_predictions.csv', index=False)
    print(f"✅ Predictions saved: results/model_predictions.csv")
    
    # Guardar métricas
    metrics_summary = {
        'MAE': metrics['MAE'],
        'RMSE': metrics['RMSE'],
        'R2_Score': metrics['R2'],
        'MAPE': metrics['MAPE'],
        'Train_Size': len(model.X_train),
        'Test_Size': len(model.X_test),
        'Total_Records': len(df),
        'Total_Features': len(model.X_train.columns)
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv('results/model_metrics.csv', index=False)
    print(f"✅ Metrics saved: results/model_metrics.csv")
    
    # 9. Resumen
    print("\n9️⃣  RESUMEN")
    print("-" * 100)
    print(f"""
    ✨ ¡MODELADO COMPLETO!

    Dataset: {len(df):,} registros con {len(engineer.df.columns)} características engineered

    Rendimiento del Modelo:
      • MAE:  ${metrics['MAE']:,.2f}
      • RMSE: ${metrics['RMSE']:,.2f}
      • R²:   {metrics['R2']:.4f}
      • MAPE: {metrics['MAPE']:.2f}%

    Archivos Generados:
      ✅ models/car_price_xgb_model.pkl
      ✅ results/feature_importance.csv
      ✅ results/model_predictions.csv
      ✅ results/model_metrics.csv

    Próximos Pasos:
      • Usar model.make_prediction() para nuevas predicciones de autos
      • Analizar feature_importance.csv para identificar variables clave
      • Revisar model_predictions.csv para análisis de errores
    """)
    
    print("=" * 100 + "\n")
    
    return engineer, model, metrics


if __name__ == "__main__":
    engineer, model, metrics = main()
