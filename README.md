# Car Genie 🧞

AI-powered car valuation and recommendation platform for the Mexican automotive market.

## Overview

Car Genie helps users understand the true market value of their car, see how it will depreciate over time, and find their next vehicle — all powered by real listing data and Claude AI.

## Features

- **Valuación inteligente** — estimates current market value from ~2,400 real Mexican listings
- **Análisis de depreciación** — projects year-by-year value loss using brand-specific rates
- **Mejor momento para vender** — identifies when the car drops below 60% of its retail reference price
- **Recomendaciones con IA** — Claude-powered natural language car recommendations

## Project Structure

```
├── app.py                          # Flask web application
├── pyproject.toml                  # Dependencies and package config
├── .env                            # API keys (not committed)
├── templates/
│   └── index.html                  # Single-page frontend (ES)
├── static/
│   └── car-genie-logo.png          # Brand mascot
├── src/
│   ├── modeling_components/
│   │   ├── car_depreciation_estimator.py
│   │   ├── car_recommender.py
│   │   ├── car_price_model.py
│   │   ├── car_price_model_script.py
│   │   └── generate_retail_prices.py
│   └── extraction/
│       ├── supercarros_scraper.py
│       └── mercadolibre_cars_scraper.py
├── utils/
│   ├── data_cleaning.py
│   └── modeling.py
├── data/
│   ├── modeling_data/
│   │   ├── mexico_cars_complete.csv   # Main dataset (2,397 records)
│   │   ├── car_retail_prices.csv      # MSRP reference table
│   │   └── car_catalogue.csv
│   └── scrapped_data/
│       ├── autos_mexico_mercadolibre.csv
│       └── car_listings_mexico_marketcheck.csv
├── models/
│   ├── car_price_xgb_model.pkl
│   └── random_forest_supercarros_model.pkl
└── results/
    ├── model_metrics.csv
    ├── model_predictions.csv
    └── feature_importance.csv
```

## Setup

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

This installs the app dependencies plus the local `utils` package in editable mode. Add `[dev]` to also get Jupyter and visualization libraries.

### 2. Configure environment

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

### 3. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050)

## Data

Real listings scraped from **Supercarros.com** and **MercadoLibre** — 2,397 records covering model years 2006–2025 across 23 brands.

To regenerate the retail price reference table from the main dataset:

```bash
python src/modeling_components/generate_retail_prices.py
```

## Model Performance

| Metric | Value |
|--------|-------|
| MAE | ~$2,100 MXN |
| Within 10% accuracy | 87% |
| Response time | < 50ms |

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web UI |
| GET | `/api/models/<make>` | Models for a given make |
| POST | `/api/predict` | Valuation + depreciation timeline |
| POST | `/api/recommend` | AI car recommendations |
