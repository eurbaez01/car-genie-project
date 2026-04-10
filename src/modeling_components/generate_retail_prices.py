"""
Generate estimated retail (MSRP) prices for every car in the inventory.

Uses a brand/year/body-type formula adapted from the project's Colab notebook.
Base prices are in USD and converted to MXN at 18 MXN/USD.
Results are saved to data/modeling_data/car_retail_prices.csv.
"""

import pandas as pd
import numpy as np

USD_TO_MXN = 18.0   # approximate exchange rate

# Brand base prices in USD (from project notebook)
BRAND_BASE_USD = {
    'Kia':           25_000,
    'Audi':          40_000,
    'Nissan':        23_000,
    'Chevrolet':     26_000,
    'Toyota':        27_000,
    'Honda':         26_000,
    'BMW':           50_000,
    'Mercedes-Benz': 55_000,
    'Lexus':         48_000,
    'Volvo':         47_000,
    'Porsche':       70_000,
    # ── Additional brands in the inventory ──
    'Mitsubishi':    24_000,
    'Ford':          28_000,
    'RAM':           35_000,
    'Jeep':          32_000,
    'Lincoln':       45_000,
    'Dodge':         28_000,
    'Suzuki':        20_000,
    'Volkswagen':    25_000,
    'Hyundai':       24_000,
    'Infiniti':      42_000,
    'Chrysler':      30_000,
    'Cadillac':      48_000,
    'Mazda':         25_000,
    'Acura':         38_000,
}
DEFAULT_BASE_USD = 30_000

BODY_ADJUSTMENT_USD = {
    'Sedan':     0,
    'SUV':       5_000,
    'Truck':     7_000,
    'Coupe':     3_000,
    'Hatchback': -1_000,
    'Convertible': 8_000,
    'Van':       2_000,
    'Wagon':     1_000,
}


def generate_msrp_mxn(row) -> float:
    """Estimate MSRP in MXN for a single inventory row."""
    base_usd  = BRAND_BASE_USD.get(row['make'], DEFAULT_BASE_USD)
    year_factor = 1 + (int(row['year']) - 2015) * 0.03
    body_adj  = BODY_ADJUSTMENT_USD.get(row.get('body_type', ''), 0)
    price_usd = base_usd * year_factor + body_adj
    return round(price_usd * USD_TO_MXN, 0)


if __name__ == '__main__':
    df = pd.read_csv('data/modeling_data/mexico_cars_complete.csv')
    print(f"Loaded {len(df)} rows from mexico_cars_complete.csv")

    df['msrp'] = df.apply(generate_msrp_mxn, axis=1)

    # Keep one row per unique make/model/year/body_type — use median msrp
    retail = (
        df.groupby(['make', 'model', 'year', 'body_type'])['msrp']
        .median()
        .reset_index()
        .rename(columns={'msrp': 'retail_price_mxn'})
    )
    retail['source'] = 'formula_estimate'

    # Overlay with 0-mile listings from the actual inventory (more accurate)
    zero_mile = df[df['miles'] == 0].copy()
    if len(zero_mile):
        zm_median = (
            zero_mile.groupby(['make', 'model', 'year', 'body_type'])['price']
            .median()
            .reset_index()
            .rename(columns={'price': 'retail_price_mxn'})
        )
        zm_median['source'] = 'zero_mile_inventory'

        # Merge: prefer zero-mile over formula
        retail = retail.merge(
            zm_median, on=['make', 'model', 'year', 'body_type'],
            how='left', suffixes=('_formula', '_inventory')
        )
        retail['retail_price_mxn'] = retail['retail_price_mxn_inventory'].fillna(
            retail['retail_price_mxn_formula']
        )
        retail['source'] = retail['source_inventory'].fillna(retail['source_formula'])
        retail = retail[['make', 'model', 'year', 'body_type', 'retail_price_mxn', 'source']]

    retail = retail.sort_values(['make', 'model', 'year'])
    retail.to_csv('data/modeling_data/car_retail_prices.csv', index=False)

    print(f"Saved {len(retail)} retail price entries to data/modeling_data/car_retail_prices.csv")
    print(f"  zero_mile_inventory: {(retail['source']=='zero_mile_inventory').sum()}")
    print(f"  formula_estimate   : {(retail['source']=='formula_estimate').sum()}")
    print()
    print("Sample:")
    print(retail.sample(8, random_state=1).to_string(index=False))
