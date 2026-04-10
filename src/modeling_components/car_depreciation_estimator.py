"""
Car Depreciation Analysis and Estimation
Functions to analyze and estimate car depreciation over time
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarDepreciationEstimator:
    """Estimates car depreciation rates and values over time"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with car data

        Args:
            df: DataFrame with car data including price, year, make, model
        """
        self.df = df.copy()
        self.current_year = 2026
        self.df['car_age'] = self.current_year - self.df['year']
        self.depreciation_rates = {}
        self.brand_rates = {}
        self.model_rates = {}

        logger.info(f"Initialized depreciation estimator with {len(df)} cars")

    def calculate_age_based_depreciation(self) -> pd.DataFrame:
        """
        Calculate average prices and depreciation rates by car age

        Returns:
            DataFrame with depreciation analysis by age
        """
        logger.info("Calculating age-based depreciation...")

        # Group by age
        age_stats = self.df.groupby('car_age').agg({
            'price': ['mean', 'median', 'std', 'count', 'min', 'max'],
            'year': 'first'
        }).round(2)

        age_stats.columns = ['_'.join(col).strip() for col in age_stats.columns.values]
        age_stats = age_stats.reset_index()

        # Calculate year-over-year depreciation rates
        age_stats = age_stats.sort_values('car_age')
        age_stats['prev_price_mean'] = age_stats['price_mean'].shift(1)
        age_stats['depreciation_rate'] = (
            (age_stats['prev_price_mean'] - age_stats['price_mean']) /
            age_stats['prev_price_mean'] * 100
        ).round(2)

        # Remove anomalous rates (depreciation > 100% or < -50%)
        age_stats.loc[
            (age_stats['depreciation_rate'] > 100) |
            (age_stats['depreciation_rate'] < -50),
            'depreciation_rate'
        ] = np.nan

        self.age_depreciation = age_stats
        logger.info(f"Calculated depreciation for {len(age_stats)} age groups")

        return age_stats

    def calculate_brand_depreciation(self) -> pd.DataFrame:
        """
        Calculate depreciation rates by brand using robust statistics

        Returns:
            DataFrame with brand-level depreciation analysis
        """
        logger.info("Calculating brand-based depreciation...")

        # Remove price outliers (beyond 3 standard deviations)
        df_no_outliers = self.df.copy()
        df_no_outliers['price_zscore'] = (
            (df_no_outliers['price'] - df_no_outliers['price'].mean()) /
            df_no_outliers['price'].std()
        )
        df_no_outliers = df_no_outliers[np.abs(df_no_outliers['price_zscore']) <= 3]

        brand_stats = df_no_outliers.groupby(['make', 'car_age']).agg({
            'price': ['median', 'count']  # Use median instead of mean for robustness
        }).round(2)

        brand_stats.columns = ['_'.join(col).strip() for col in brand_stats.columns.values]
        brand_stats = brand_stats.reset_index()

        # Calculate depreciation rates for each brand
        brand_depr_rates = {}

        for brand in brand_stats['make'].unique():
            brand_data = brand_stats[brand_stats['make'] == brand].sort_values('car_age')

            if len(brand_data) >= 3:  # Need at least 3 data points
                brand_data = brand_data.sort_values('car_age')
                brand_data['prev_price'] = brand_data['price_median'].shift(1)
                brand_data['depr_rate'] = (
                    (brand_data['prev_price'] - brand_data['price_median']) /
                    brand_data['prev_price'] * 100
                )

                # Filter out extreme depreciation rates (>200% or <-50%)
                valid_rates = brand_data['depr_rate'][
                    (brand_data['depr_rate'] <= 200) &
                    (brand_data['depr_rate'] >= -50) &
                    (~np.isnan(brand_data['depr_rate']))
                ]

                if len(valid_rates) > 0:
                    avg_rate = valid_rates.mean()
                    brand_depr_rates[brand] = avg_rate

        self.brand_rates = brand_depr_rates

        # Create summary DataFrame
        brand_summary = pd.DataFrame([
            {'brand': brand, 'avg_depreciation_rate': rate}
            for brand, rate in brand_depr_rates.items()
        ]).sort_values('avg_depreciation_rate')

    def calculate_mileage_depreciation(self) -> pd.DataFrame:
        """
        Calculate depreciation rates by mileage using regression analysis

        Returns:
            DataFrame with mileage-based depreciation analysis
        """
        logger.info("Calculating mileage-based depreciation...")

        # Remove outliers
        df_clean = self.df.copy()
        df_clean['price_zscore'] = (df_clean['price'] - df_clean['price'].mean()) / df_clean['price'].std()
        df_clean = df_clean[np.abs(df_clean['price_zscore']) <= 3]

        # Use regression to find relationship between mileage and price
        from sklearn.linear_model import LinearRegression

        # Prepare data for regression
        X = df_clean['miles'].values.reshape(-1, 1)
        y = df_clean['price'].values

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)

        # Calculate depreciation rate per mile
        slope = reg.coef_[0]  # Price change per mile
        intercept = reg.intercept_

        # Calculate R-squared
        r_squared = reg.score(X, y)

        # Create depreciation analysis
        mileage_analysis = {
            'method': 'linear_regression',
            'slope_per_mile': slope,
            'loss_per_mile': abs(slope),  # Loss is positive
            'loss_per_1000_miles': abs(slope) * 1000,
            'intercept': intercept,
            'r_squared': r_squared,
            'sample_size': len(df_clean),
            'mileage_range': f"{df_clean['miles'].min():,.0f} - {df_clean['miles'].max():,.0f} miles",
            'price_range': f"${df_clean['price'].min():,.0f} - ${df_clean['price'].max():,.0f}"
        }

        # Store the depreciation rate
        self.mileage_depr_rate = abs(slope) * 1000  # Loss per 1,000 miles

        logger.info(f"Calculated depreciation: ${self.mileage_depr_rate:.2f} per 1,000 miles (R²={r_squared:.3f})")

        return pd.DataFrame([mileage_analysis])

    def estimate_car_loss_by_mileage(self, initial_value: float, current_mileage: float = 0,
                                    target_mileage: float = None, mileage_increment: float = 10000) -> pd.DataFrame:
        """
        Estimate car value loss as mileage increases

        Args:
            initial_value: Starting value of the car
            current_mileage: Current mileage (default: 0)
            target_mileage: Target mileage to project to (optional)
            mileage_increment: Miles to increment by for each projection step

        Returns:
            DataFrame with mileage projections and value losses
        """
        logger.info(f"Estimating mileage-based depreciation from ${initial_value:,.0f} starting at {current_mileage:,.0f} miles")

        # Calculate mileage depreciation rates if not already done
        if not hasattr(self, 'mileage_depr_rate') or self.mileage_depr_rate is None:
            mileage_df = self.calculate_mileage_depreciation()
            if mileage_df.empty:
                logger.warning("No mileage depreciation data available")
                return pd.DataFrame()

        # Use calculated depreciation rate per 1,000 miles
        loss_per_1000_miles = self.mileage_depr_rate

        # If no target mileage specified, project reasonable range
        if target_mileage is None:
            target_mileage = min(current_mileage + 100000, 200000)  # Up to 100k more or 200k total

        # Create mileage projections
        mileage_steps = []
        current_value = initial_value

        mileage = current_mileage
        step = 0

        while mileage <= target_mileage:
            mileage_steps.append({
                'step': step,
                'mileage': mileage,
                'value': current_value,
                'loss_from_start': initial_value - current_value,
                'remaining_percentage': (current_value / initial_value) * 100
            })

            # Calculate loss for next increment
            if step > 0:  # Don't depreciate the first step
                loss_amount = (mileage_increment / 1000) * loss_per_1000_miles
                current_value = max(current_value - loss_amount, initial_value * 0.1)  # Floor at 10% of original value

            mileage += mileage_increment
            step += 1

        result_df = pd.DataFrame(mileage_steps)

        # Add additional calculated columns
        result_df['loss_amount'] = result_df['loss_from_start'].diff().fillna(0)
        result_df['loss_percentage'] = (result_df['loss_amount'] / result_df['value'].shift(1)) * 100
        result_df['loss_percentage'] = result_df['loss_percentage'].fillna(0)

        logger.info(f"Projected {len(result_df)} mileage steps up to {target_mileage:,.0f} miles")
        return result_df

    def get_industry_standard_rates(self) -> Dict[int, float]:
        """
        Get industry-standard depreciation rates by age

        Returns:
            Dictionary mapping age to depreciation rate (%)
        """
        # Based on industry data (KBB, Edmunds, etc.)
        # These are approximate annual depreciation rates
        industry_rates = {
            1: 20.0,   # First year: 20% depreciation
            2: 15.0,   # Second year: 15%
            3: 12.0,   # Third year: 12%
            4: 10.0,   # Fourth year: 10%
            5: 8.0,    # Fifth year: 8%
            6: 7.0,    # Sixth year: 7%
            7: 6.0,    # Seventh year: 6%
            8: 5.0,    # Eighth year: 5%
            9: 4.0,    # Ninth year: 4%
            10: 3.0    # Tenth year: 3%
        }

        # For ages beyond 10, use 2% per year
        for age in range(11, 21):
            industry_rates[age] = 2.0

        return industry_rates

    def estimate_car_value_over_time(self,
                                   initial_price: float,
                                   make: str = None,
                                   model: str = None,
                                   current_age: int = 0,
                                   years_ahead: int = 10) -> pd.DataFrame:
        """
        Estimate car value depreciation over time

        Args:
            initial_price: Original MSRP or current price
            make: Car brand (optional, for brand-specific rates)
            model: Car model (optional)
            current_age: Current age of the car in years
            years_ahead: How many years to project

        Returns:
            DataFrame with value projections by year
        """
        logger.info(f"Estimating depreciation for {make} {model} starting at age {current_age}")

        # Get appropriate depreciation rates
        if make and make in self.brand_rates:
            # Use brand-specific rate if available
            annual_rate = self.brand_rates[make] / 100
            rate_type = 'brand_specific'
        else:
            # Use industry standard rates
            industry_rates = self.get_industry_standard_rates()
            rate_type = 'industry_standard'

        # Create projection
        projections = []
        current_value = initial_price

        for year in range(current_age, current_age + years_ahead + 1):
            if year == current_age:
                depreciation_rate = 0.0  # No depreciation at current age
            elif rate_type == 'brand_specific':
                depreciation_rate = annual_rate
            else:
                depreciation_rate = industry_rates.get(min(year, max(industry_rates.keys())), 2.0) / 100

            # Calculate new value
            if year > current_age:
                depreciation_amount = current_value * depreciation_rate
                current_value = current_value - depreciation_amount

            projections.append({
                'year': self.current_year - year,  # Calendar year
                'car_age': year,
                'projected_value': max(0, current_value),  # Don't go below 0
                'depreciation_rate': depreciation_rate * 100,
                'depreciation_amount': depreciation_amount if year > current_age else 0,
                'rate_type': rate_type
            })

        result_df = pd.DataFrame(projections)
        logger.info(f"Projected {len(result_df)} years of depreciation")

        return result_df

    def get_best_time_to_sell(self,
                             initial_price: float,
                             make: str = None,
                             model: str = None,
                             current_age: int = 0,
                             threshold_pct: float = 50.0,
                             years_ahead: int = 30) -> Dict[str, Any]:
        """
        Find the last year the car retains at least threshold_pct of its value.

        The "best time to sell" is defined as the final year where the projected
        value is still above the threshold (default 50% of initial_price).  After
        that point depreciation has eroded more than half the car's worth, so
        selling sooner captures more value.

        Args:
            initial_price: Current or original price of the car (MXN).
            make: Car brand — used for brand-specific depreciation rates.
            model: Car model (informational, not used in calculation).
            current_age: Current age of the car in years (0 = brand new).
            threshold_pct: Value-retention target (default 50 %).
            years_ahead: Maximum years to project forward.

        Returns:
            Dictionary with:
              - best_sell_year      : calendar year to sell (int)
              - best_sell_age       : car age at that point (int)
              - years_from_now      : how many years until that point (int)
              - value_at_sell       : projected value at sell year (float)
              - retention_pct       : actual retention at sell year (float)
              - threshold_pct       : the threshold used (float)
              - crosses_below_year  : first year value drops *below* threshold (int | None)
              - projection          : full DataFrame used (for charting / debugging)
        """
        projection = self.estimate_car_value_over_time(
            initial_price=initial_price,
            make=make,
            model=model,
            current_age=current_age,
            years_ahead=years_ahead,
        )

        # estimate_car_value_over_time stores 'year' as (current_year - car_age),
        # i.e. the model year going BACKWARD.  For sell-timing we need the
        # FUTURE calendar year: current_year + years elapsed since now.
        projection = projection.copy()
        projection['future_year'] = self.current_year + (projection['car_age'] - current_age)

        threshold_value = initial_price * (threshold_pct / 100)

        # Rows where value is still at or above the threshold
        above = projection[projection['projected_value'] >= threshold_value]

        if above.empty:
            # Already below threshold — recommend selling now
            best_row = projection.iloc[0]
            crosses_below = int(projection.iloc[0]['future_year'])
        else:
            best_row = above.iloc[-1]   # last future year still at/above threshold
            below = projection[projection['projected_value'] < threshold_value]
            crosses_below = int(below.iloc[0]['future_year']) if not below.empty else None

        best_sell_year = int(best_row['future_year'])
        best_sell_age  = int(best_row['car_age'])
        years_from_now = max(0, best_sell_age - current_age)
        value_at_sell  = round(float(best_row['projected_value']), 2)
        retention      = round((value_at_sell / initial_price) * 100, 2)

        logger.info(
            f"Best time to sell {make or ''} {model or ''}: "
            f"future year {best_sell_year} (age {best_sell_age}, "
            f"{years_from_now} yrs from now), "
            f"value ${value_at_sell:,.0f} ({retention:.1f}% retained)"
        )

        return {
            'best_sell_year':     best_sell_year,
            'best_sell_age':      best_sell_age,
            'years_from_now':     years_from_now,
            'value_at_sell':      value_at_sell,
            'retention_pct':      retention,
            'threshold_pct':      threshold_pct,
            'crosses_below_year': crosses_below,
            'projection':         projection,
        }

    def calculate_total_loss_by_year(self,
                                   initial_price: float,
                                   make: str = None,
                                   years: int = 10) -> pd.DataFrame:
        """
        Calculate total depreciation loss by year

        Args:
            initial_price: Original price
            make: Car brand
            years: Number of years to calculate

        Returns:
            DataFrame with annual loss amounts
        """
        projection = self.estimate_car_value_over_time(
            initial_price=initial_price,
            make=make,
            years_ahead=years
        )

        # Calculate losses
        losses = []
        for i in range(1, len(projection)):
            current = projection.iloc[i]
            previous = projection.iloc[i-1]

            loss_amount = previous['projected_value'] - current['projected_value']
            loss_percentage = (loss_amount / previous['projected_value']) * 100

            losses.append({
                'year': current['year'],
                'car_age': current['car_age'],
                'loss_amount': loss_amount,
                'loss_percentage': loss_percentage,
                'remaining_value': current['projected_value'],
                'remaining_percentage': (current['projected_value'] / initial_price) * 100
            })

        return pd.DataFrame(losses)

    def plot_depreciation_curve(self,
                              initial_price: float,
                              make: str = None,
                              years: int = 10,
                              save_path: str = None):
        """
        Plot depreciation curve for a car

        Args:
            initial_price: Original price
            make: Car brand
            years: Years to plot
            save_path: Path to save plot (optional)
        """
        projection = self.estimate_car_value_over_time(
            initial_price=initial_price,
            make=make,
            years_ahead=years
        )

        plt.figure(figsize=(12, 8))

        # Plot value over time
        plt.subplot(2, 1, 1)
        plt.plot(projection['car_age'], projection['projected_value'], 'b-', linewidth=2, marker='o')
        plt.title(f'Car Depreciation Projection\n{make or "Average Car"} - Initial Price: ${initial_price:,.0f}')
        plt.xlabel('Car Age (Years)')
        plt.ylabel('Projected Value ($)')
        plt.grid(True, alpha=0.3)

        # Format y-axis
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot depreciation rate
        plt.subplot(2, 1, 2)
        plt.plot(projection['car_age'], projection['depreciation_rate'], 'r-', linewidth=2, marker='s')
        plt.title('Annual Depreciation Rate')
        plt.xlabel('Car Age (Years)')
        plt.ylabel('Depreciation Rate (%)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def get_depreciation_summary(self) -> Dict:
        """
        Get comprehensive depreciation summary

        Returns:
            Dictionary with depreciation statistics
        """
        # Calculate age-based depreciation
        age_depr = self.calculate_age_based_depreciation()

        # Calculate brand depreciation
        brand_depr = self.calculate_brand_depreciation()

        # Get industry rates
        industry_rates = self.get_industry_standard_rates()

        summary = {
            'total_cars': len(self.df),
            'year_range': f"{self.df['year'].min()}-{self.df['year'].max()}",
            'price_range': f"${self.df['price'].min():,.0f}-${self.df['price'].max():,.0f}",
            'age_based_depreciation': age_depr.to_dict('records'),
            'brand_depreciation': brand_depr.to_dict('records'),
            'industry_standard_rates': industry_rates,
            'fastest_depreciating_brands': brand_depr.head(5).to_dict('records'),
            'slowest_depreciating_brands': brand_depr.tail(5).to_dict('records')
        }

        return summary


def estimate_car_loss_by_year(df: pd.DataFrame,
                            initial_price: float,
                            make: str = None,
                            current_age: int = 0,
                            years: int = 10) -> pd.DataFrame:
    """
    Convenience function to estimate car depreciation loss by year

    Args:
        df: Car dataset DataFrame
        initial_price: Original car price
        make: Car brand (optional)
        current_age: Current car age
        years: Years to project

    Returns:
        DataFrame with annual depreciation losses
    """
    estimator = CarDepreciationEstimator(df)
    return estimator.calculate_total_loss_by_year(initial_price, make, years)


def estimate_car_loss_by_mileage(df: pd.DataFrame,
                               initial_value: float,
                               current_mileage: float = 0,
                               target_mileage: float = None,
                               mileage_increment: float = 10000) -> pd.DataFrame:
    """
    Convenience function to estimate car depreciation loss by mileage

    Args:
        df: Car dataset DataFrame
        initial_value: Starting value of the car
        current_mileage: Current mileage (default: 0)
        target_mileage: Target mileage to project to (optional)
        mileage_increment: Miles to increment by for each projection step

    Returns:
        DataFrame with mileage projections and value losses
    """
    estimator = CarDepreciationEstimator(df)
    return estimator.estimate_car_loss_by_mileage(initial_value, current_mileage, target_mileage, mileage_increment)


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/modeling_data/mexico_cars_complete.csv')

    # Create estimator
    estimator = CarDepreciationEstimator(df)

    # Get depreciation summary
    summary = estimator.get_depreciation_summary()

    print("🚗 CAR DEPRECIATION ANALYSIS")
    print("=" * 50)
    print(f"Total cars analyzed: {summary['total_cars']}")
    print(f"Year range: {summary['year_range']}")
    print(f"Price range: {summary['price_range']}")

    print("\n🏎️  TOP 5 FASTEST DEPRECIATING BRANDS:")
    for brand in summary['fastest_depreciating_brands']:
        print(f"  {brand['brand']}: {brand['avg_depreciation_rate']:.1f}%/year")

    print("\n🐌 TOP 5 SLOWEST DEPRECIATING BRANDS:")
    for brand in summary['slowest_depreciating_brands']:
        print(f"  {brand['brand']}: {brand['avg_depreciation_rate']:.1f}%/year")

    # Example depreciation calculation
    print("\n📊 EXAMPLE: Depreciation projection for a $500,000 car")
    losses = estimate_car_loss_by_year(df, 500000, "BMW", years=5)
    print(losses.head())

    # Plot example
    try:
        estimator.plot_depreciation_curve(500000, "BMW", years=10)
    except:
        print("Could not display plot (may be running in headless environment)")