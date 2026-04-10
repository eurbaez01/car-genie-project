"""
Car Genie — Aplicación web Flask
Integra: estimación de precios de autos, análisis de depreciación, recomendaciones con Claude AI
"""

import os
import json
from typing import Optional
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import sys, os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from modeling_components.car_depreciation_estimator import CarDepreciationEstimator
from modeling_components.car_recommender import CarRecommender

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Cargar dataset una vez al iniciar la aplicación
# ---------------------------------------------------------------------------
DATA_PATH = "data/modeling_data/mexico_cars_complete.csv"
df = pd.DataFrame()
estimator = None
MAKES = []
MODELS_BY_MAKE = {}
YEARS = []

def _load_data():
    global df, estimator, MAKES, MODELS_BY_MAKE, YEARS
    try:
        df = pd.read_csv(DATA_PATH)
        # Asegurarse de que car_age exista
        if 'car_age' not in df.columns:
            df['car_age'] = 2026 - df['year']
        df = df.dropna(subset=['price', 'year', 'make', 'model'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['miles'] = pd.to_numeric(df['miles'], errors='coerce').fillna(0)
        df = df[df['price'] > 0]

        estimator = CarDepreciationEstimator(df)
        estimator.calculate_brand_depreciation()   # pre-compute brand rates

        MAKES = sorted(df['make'].dropna().unique().tolist())
        MODELS_BY_MAKE = {
            make: sorted(df[df['make'] == make]['model'].dropna().unique().tolist())
            for make in MAKES
        }
        YEARS = sorted(df['year'].dropna().astype(int).unique().tolist(), reverse=True)
        print(f"Loaded {len(df)} cars | {len(MAKES)} makes")
    except Exception as e:
        print(f"Warning: could not load data — {e}")

_load_data()

# ---------------------------------------------------------------------------
# Tabla de precios de lista — cargada desde data/car_retail_prices.csv
# Generada por generate_retail_prices.py (fórmula marca/año/carrocería + inventario 0 km)
# ---------------------------------------------------------------------------
RETAIL_CSV = "data/modeling_data/car_retail_prices.csv"
retail_df: pd.DataFrame = pd.DataFrame()

def _load_retail_prices():
    global retail_df
    try:
        retail_df = pd.read_csv(RETAIL_CSV)
        print(f"Retail prices loaded: {len(retail_df)} entries")
    except FileNotFoundError:
        print(f"Warning: {RETAIL_CSV} not found — run generate_retail_prices.py first")

_load_retail_prices()


def get_retail_price(make: str, model: str, year: int) -> "Optional[float]":
    """
    Retorna el precio de lista (MSRP) estimado en MXN para una marca/modelo/año.

    Prioridad de coincidencia:
      1. Coincidencia exacta marca + modelo + año (cualquier tipo de carrocería → mediana)
      2. Misma marca + modelo, año más cercano
      3. Misma marca, aplicar depreciación promedio de la marca desde el año más cercano
    """
    if retail_df.empty:
        return None

    year = int(year)

    # 1. Coincidencia exacta (puede tener múltiples tipos de carrocería — tomar mediana)
    exact = retail_df[(retail_df['make'] == make) &
                      (retail_df['model'] == model) &
                      (retail_df['year'] == year)]
    if not exact.empty:
        return float(exact['retail_price_mxn'].median())

    # 2. Misma marca/modelo, año más cercano
    model_rows = retail_df[(retail_df['make'] == make) & (retail_df['model'] == model)]
    if not model_rows.empty:
        closest_yr = model_rows.iloc[(model_rows['year'] - year).abs().argsort()[:1]]['year'].values[0]
        ref_price  = float(model_rows[model_rows['year'] == closest_yr]['retail_price_mxn'].median())
        yr_diff    = int(closest_yr) - year
        brand_rate = estimator.brand_rates.get(make)
        ind_rates  = estimator.get_industry_standard_rates()
        price = ref_price
        for _ in range(abs(yr_diff)):
            rate = (brand_rate / 100) if brand_rate else (ind_rates.get(5, 8.0) / 100)
            # adjust toward requested year
            price = price * (1 - rate) if yr_diff > 0 else price / max(1 - rate, 0.01)
        return round(price, 0)

    # 3. Solo misma marca — retornar mediana para el año más cercano
    make_rows = retail_df[retail_df['make'] == make]
    if not make_rows.empty:
        return float(make_rows['retail_price_mxn'].median())

    return None


# ---------------------------------------------------------------------------
# Estimación de valor a partir de estadísticas del dataset
# ---------------------------------------------------------------------------

def estimate_value(make, model, year, miles, condition="Good", fuel_type=None):
    """Retorna (estimated_price, confidence_pct, sample_size) a partir de listados similares."""
    if df.empty:
        return None, 0, 0

    year = int(year)
    miles = float(miles)

    # Filtrado progresivo: exacto → ±2 años → misma marca
    filters = [
        df[(df['make'] == make) & (df['model'] == model) & (df['year'] == year)],
        df[(df['make'] == make) & (df['model'] == model) & (df['year'].between(year - 2, year + 2))],
        df[(df['make'] == make) & (df['model'] == model)],
        df[df['make'] == make],
    ]
    subset = pd.DataFrame()
    for f in filters:
        if len(f) >= 3:
            subset = f
            break
    if subset.empty:
        subset = df

    # Ajuste por kilometraje: ±$4,711 por cada 10,000 millas vs mediana del subconjunto
    median_miles = subset['miles'].median()
    mileage_adj = -((miles - median_miles) / 10_000) * 4_711

    # Ajuste por antigüedad usando tasas de depreciación de la industria del estimador
    industry_rates = estimator.get_industry_standard_rates()
    car_age = 2026 - year
    median_year = int(round(subset['year'].median()))
    base_age = 2026 - median_year
    base_price = subset['price'].median()

    age_adj = 0.0
    if car_age < base_age:
        for a in range(car_age + 1, base_age + 1):
            rate = industry_rates.get(min(a, 10), 2.0) / 100
            age_adj += base_price * rate
    elif car_age > base_age:
        val = base_price
        for a in range(base_age + 1, car_age + 1):
            rate = industry_rates.get(min(a, 10), 2.0) / 100
            val -= val * rate
        age_adj = val - base_price

    # Multiplicador por condición
    condition_mult = {"Excellent": 1.08, "Good": 1.0, "Fair": 0.90, "Poor": 0.78}.get(condition, 1.0)

    estimated = max((base_price + mileage_adj + age_adj) * condition_mult, 10_000)

    # Confianza basada en el tamaño de la muestra
    n = len(subset)
    confidence = min(50 + n * 2, 97)

    return round(estimated, 0), confidence, n


def get_depreciation_timeline(make, model, year, estimated_price):
    """Retorna el valor año a año, filas del mejor momento para vender, y recomendación de venta."""
    current_age = 2026 - int(year)

    # ── El precio de lista solo se usa como referencia visual en el banner ────
    retail = get_retail_price(make, model, year)
    reference_price = retail if retail else estimated_price

    # ── La proyección de depreciación parte del valor estimado actual ─────────
    sell_result = estimator.get_best_time_to_sell(
        initial_price=estimated_price,
        make=make,
        model=model,
        current_age=current_age,
        threshold_pct=60.0,
        years_ahead=30,
    )
    projection = sell_result['projection']  # reuse the projection already computed

    # ── Datos del gráfico: medianas históricas (pasado) + valores proyectados futuros ──
    chart = []
    past_data = (
        df[(df['make'] == make) & (df['model'] == model)]
        .groupby('year')['price']
        .median()
        .to_dict()
    )
    for y in sorted(past_data.keys()):
        chart.append({"year": int(y), "value": round(past_data[y], 0), "type": "historical"})

    # projection tiene 'future_year' = 2026 + (car_age - current_age)
    for _, row in projection.iterrows():
        future_yr = int(row['future_year'])
        if future_yr > 2026:          # solo agregar años genuinamente futuros al gráfico
            chart.append({"year": future_yr, "value": round(row['projected_value'], 0), "type": "projected"})

    # ── Filas de la tabla del mejor momento para vender (próximos 7 años proyectados) ─
    best_sell = []
    prev_val  = estimated_price
    best_year = sell_result['best_sell_year']

    for _, row in projection.iterrows():
        future_yr = int(row['future_year'])
        if future_yr < 2026:
            continue                  # skip rows before now
        val       = round(row['projected_value'], 0)
        retention = round((val / estimated_price) * 100, 1)
        annual_loss = round(prev_val - val, 0) if future_yr > 2026 else 0
        best_sell.append({
            "year":        future_yr,
            "value":       val,
            "retention":   retention,
            "annual_loss": annual_loss,
            "is_best":     future_yr == best_year,
        })
        prev_val = val
        if len(best_sell) >= 7:
            break

    # ── Métricas resumen ─────────────────────────────────────────────────────
    last_val = best_sell[-1]['value'] if best_sell else estimated_price
    total_depr_pct = round((1 - last_val / estimated_price) * 100, 1)
    avg_annual_loss = round((estimated_price - last_val) / len(best_sell), 0) if best_sell else 0

    # ── Recalcular recomendación de venta relativa al precio de LISTA ────────
    # retention_pct = cuánto del precio de lista conserva el auto en el año de venta
    value_at_sell = sell_result['value_at_sell']
    retention_vs_retail = round((value_at_sell / reference_price) * 100, 2)

    # crosses_below_year: primer año en que el valor proyectado cae por debajo del 60% del precio de lista
    retail_threshold = reference_price * 0.60
    crosses_below_year = None
    for _, row in projection.iterrows():
        if int(row['future_year']) >= 2026 and row['projected_value'] < retail_threshold:
            crosses_below_year = int(row['future_year'])
            break

    # best_sell_year: último año en que el valor aún es >= 60% del precio de lista
    best_sell_year_retail = (crosses_below_year - 1) if crosses_below_year else sell_result['best_sell_year']
    years_from_now_retail = max(0, best_sell_year_retail - 2026)

    return {
        "chart": chart,
        "best_sell": best_sell,
        "total_depreciation_pct": total_depr_pct,
        "avg_annual_loss": avg_annual_loss,
        "retail_price": retail,
        "retail_source": "zero_mile_inventory" if (not retail_df.empty and not retail_df[(retail_df['make']==make)&(retail_df['model']==model)&(retail_df['year']==int(year))&(retail_df['source']=='zero_mile_inventory')].empty) else "formula_estimate",
        "sell_recommendation": {
            "best_sell_year":     best_sell_year_retail,
            "best_sell_age":      sell_result['best_sell_age'],
            "years_from_now":     years_from_now_retail,
            "value_at_sell":      value_at_sell,
            "retention_pct":      retention_vs_retail,
            "crosses_below_year": crosses_below_year,
            "threshold_pct":      60.0,
            "reference_price":    reference_price,
        },
    }


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", makes=MAKES, years=YEARS)


@app.route("/api/models/<make>")
def get_models(make):
    return jsonify(MODELS_BY_MAKE.get(make, []))


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    make = data.get("make", "")
    model = data.get("model", "")
    year = data.get("year", 2020)
    miles = data.get("miles", 50000)
    condition = data.get("condition", "Good")
    fuel_type = data.get("fuel_type", "")

    if not make or not model:
        return jsonify({"error": "make and model are required"}), 400

    estimated, confidence, sample_n = estimate_value(make, model, year, miles, condition, fuel_type)
    if estimated is None:
        return jsonify({"error": "Could not estimate value — no data available"}), 400

    depr = get_depreciation_timeline(make, model, year, estimated)

    return jsonify({
        "estimated_value": estimated,
        "value_range": [round(estimated * 0.93, 0), round(estimated * 1.07, 0)],
        "confidence": confidence,
        "sample_size": sample_n,
        "total_depreciation_pct": depr["total_depreciation_pct"],
        "avg_annual_loss": depr["avg_annual_loss"],
        "chart": depr["chart"],
        "best_sell": depr["best_sell"],
        "sell_recommendation": depr["sell_recommendation"],
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please describe what you need"}), 400

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured on server"}), 500

    try:
        rec = CarRecommender(api_key=api_key)
        result = rec.recommend_from_natural_language(text, max_recommendations=3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
