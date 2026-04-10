"""
Análisis y Estimación de Depreciación de Autos
Funciones para analizar y estimar la depreciación de autos a lo largo del tiempo
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
    """Estima las tasas de depreciación y los valores de los autos a lo largo del tiempo"""

    def __init__(self, df: pd.DataFrame):
        """
        Inicializa con datos de autos

        Args:
            df: DataFrame con datos de autos que incluye precio, año, marca, modelo
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
        Calcula los precios promedio y las tasas de depreciación por antigüedad del auto

        Returns:
            DataFrame con el análisis de depreciación por antigüedad
        """
        logger.info("Calculando depreciación por antigüedad...")

        # Agrupar por antigüedad
        age_stats = self.df.groupby('car_age').agg({
            'price': ['mean', 'median', 'std', 'count', 'min', 'max'],
            'year': 'first'
        }).round(2)

        age_stats.columns = ['_'.join(col).strip() for col in age_stats.columns.values]
        age_stats = age_stats.reset_index()

        # Calcular tasas de depreciación año a año
        age_stats = age_stats.sort_values('car_age')
        age_stats['prev_price_mean'] = age_stats['price_mean'].shift(1)
        age_stats['depreciation_rate'] = (
            (age_stats['prev_price_mean'] - age_stats['price_mean']) /
            age_stats['prev_price_mean'] * 100
        ).round(2)

        # Eliminar tasas anómalas (depreciación > 100% o < -50%)
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
        Calcula las tasas de depreciación por marca usando estadísticas robustas

        Returns:
            DataFrame con el análisis de depreciación a nivel de marca
        """
        logger.info("Calculando depreciación por marca...")

        # Eliminar valores atípicos de precio (más de 3 desviaciones estándar)
        df_no_outliers = self.df.copy()
        df_no_outliers['price_zscore'] = (
            (df_no_outliers['price'] - df_no_outliers['price'].mean()) /
            df_no_outliers['price'].std()
        )
        df_no_outliers = df_no_outliers[np.abs(df_no_outliers['price_zscore']) <= 3]

        brand_stats = df_no_outliers.groupby(['make', 'car_age']).agg({
            'price': ['median', 'count']  # Usar mediana en lugar de media para mayor robustez
        }).round(2)

        brand_stats.columns = ['_'.join(col).strip() for col in brand_stats.columns.values]
        brand_stats = brand_stats.reset_index()

        # Calcular tasas de depreciación para cada marca
        brand_depr_rates = {}

        for brand in brand_stats['make'].unique():
            brand_data = brand_stats[brand_stats['make'] == brand].sort_values('car_age')

            if len(brand_data) >= 3:  # Se necesitan al menos 3 puntos de datos
                brand_data = brand_data.sort_values('car_age')
                brand_data['prev_price'] = brand_data['price_median'].shift(1)
                brand_data['depr_rate'] = (
                    (brand_data['prev_price'] - brand_data['price_median']) /
                    brand_data['prev_price'] * 100
                )

                # Filtrar tasas de depreciación extremas (>200% o <-50%)
                valid_rates = brand_data['depr_rate'][
                    (brand_data['depr_rate'] <= 200) &
                    (brand_data['depr_rate'] >= -50) &
                    (~np.isnan(brand_data['depr_rate']))
                ]

                if len(valid_rates) > 0:
                    avg_rate = valid_rates.mean()
                    brand_depr_rates[brand] = avg_rate

        self.brand_rates = brand_depr_rates

        # Crear DataFrame resumen
        brand_summary = pd.DataFrame([
            {'brand': brand, 'avg_depreciation_rate': rate}
            for brand, rate in brand_depr_rates.items()
        ]).sort_values('avg_depreciation_rate')

    def calculate_mileage_depreciation(self) -> pd.DataFrame:
        """
        Calcula las tasas de depreciación por kilometraje usando análisis de regresión

        Returns:
            DataFrame con el análisis de depreciación basado en kilometraje
        """
        logger.info("Calculando depreciación por kilometraje...")

        # Eliminar valores atípicos
        df_clean = self.df.copy()
        df_clean['price_zscore'] = (df_clean['price'] - df_clean['price'].mean()) / df_clean['price'].std()
        df_clean = df_clean[np.abs(df_clean['price_zscore']) <= 3]

        # Usar regresión para encontrar la relación entre kilometraje y precio
        from sklearn.linear_model import LinearRegression

        # Preparar datos para la regresión
        X = df_clean['miles'].values.reshape(-1, 1)
        y = df_clean['price'].values

        # Ajustar regresión lineal
        reg = LinearRegression()
        reg.fit(X, y)

        # Calcular tasa de depreciación por milla
        slope = reg.coef_[0]  # Cambio de precio por milla
        intercept = reg.intercept_

        # Calcular R-cuadrada
        r_squared = reg.score(X, y)

        # Crear análisis de depreciación
        mileage_analysis = {
            'method': 'linear_regression',
            'slope_per_mile': slope,
            'loss_per_mile': abs(slope),  # La pérdida es positiva
            'loss_per_1000_miles': abs(slope) * 1000,
            'intercept': intercept,
            'r_squared': r_squared,
            'sample_size': len(df_clean),
            'mileage_range': f"{df_clean['miles'].min():,.0f} - {df_clean['miles'].max():,.0f} miles",
            'price_range': f"${df_clean['price'].min():,.0f} - ${df_clean['price'].max():,.0f}"
        }

        # Guardar la tasa de depreciación
        self.mileage_depr_rate = abs(slope) * 1000  # Pérdida por cada 1,000 millas

        logger.info(f"Calculated depreciation: ${self.mileage_depr_rate:.2f} per 1,000 miles (R²={r_squared:.3f})")

        return pd.DataFrame([mileage_analysis])

    def estimate_car_loss_by_mileage(self, initial_value: float, current_mileage: float = 0,
                                    target_mileage: float = None, mileage_increment: float = 10000) -> pd.DataFrame:
        """
        Estima la pérdida de valor del auto a medida que aumenta el kilometraje

        Args:
            initial_value: Valor inicial del auto
            current_mileage: Kilometraje actual (por defecto: 0)
            target_mileage: Kilometraje objetivo a proyectar (opcional)
            mileage_increment: Millas a incrementar en cada paso de proyección

        Returns:
            DataFrame con proyecciones de kilometraje y pérdidas de valor
        """
        logger.info(f"Estimating mileage-based depreciation from ${initial_value:,.0f} starting at {current_mileage:,.0f} miles")

        # Calcular tasas de depreciación por kilometraje si aún no se han calculado
        if not hasattr(self, 'mileage_depr_rate') or self.mileage_depr_rate is None:
            mileage_df = self.calculate_mileage_depreciation()
            if mileage_df.empty:
                logger.warning("No hay datos de depreciación por kilometraje disponibles")
                return pd.DataFrame()

        # Usar la tasa de depreciación calculada por cada 1,000 millas
        loss_per_1000_miles = self.mileage_depr_rate

        # Si no se especifica un kilometraje objetivo, proyectar un rango razonable
        if target_mileage is None:
            target_mileage = min(current_mileage + 100000, 200000)  # Hasta 100k más o 200k en total

        # Crear proyecciones de kilometraje
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

            # Calcular pérdida para el siguiente incremento
            if step > 0:  # No depreciar el primer paso
                loss_amount = (mileage_increment / 1000) * loss_per_1000_miles
                current_value = max(current_value - loss_amount, initial_value * 0.1)  # Mínimo 10% del valor original

            mileage += mileage_increment
            step += 1

        result_df = pd.DataFrame(mileage_steps)

        # Agregar columnas calculadas adicionales
        result_df['loss_amount'] = result_df['loss_from_start'].diff().fillna(0)
        result_df['loss_percentage'] = (result_df['loss_amount'] / result_df['value'].shift(1)) * 100
        result_df['loss_percentage'] = result_df['loss_percentage'].fillna(0)

        logger.info(f"Projected {len(result_df)} mileage steps up to {target_mileage:,.0f} miles")
        return result_df

    def get_industry_standard_rates(self) -> Dict[int, float]:
        """
        Obtiene las tasas de depreciación estándar de la industria por antigüedad

        Returns:
            Diccionario que mapea la antigüedad a la tasa de depreciación (%)
        """
        # Basado en datos de la industria (KBB, Edmunds, etc.)
        # Estas son tasas de depreciación anual aproximadas
        industry_rates = {
            1: 20.0,   # Primer año: 20% de depreciación
            2: 15.0,   # Segundo año: 15%
            3: 12.0,   # Tercer año: 12%
            4: 10.0,   # Cuarto año: 10%
            5: 8.0,    # Quinto año: 8%
            6: 7.0,    # Sexto año: 7%
            7: 6.0,    # Séptimo año: 6%
            8: 5.0,    # Octavo año: 5%
            9: 4.0,    # Noveno año: 4%
            10: 3.0    # Décimo año: 3%
        }

        # Para antigüedades mayores a 10 años, usar 2% anual
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
        Estima la depreciación del valor del auto a lo largo del tiempo

        Args:
            initial_price: Precio de lista original o precio actual
            make: Marca del auto (opcional, para tasas específicas por marca)
            model: Modelo del auto (opcional)
            current_age: Antigüedad actual del auto en años
            years_ahead: Cuántos años proyectar hacia adelante

        Returns:
            DataFrame con proyecciones de valor por año
        """
        logger.info(f"Estimating depreciation for {make} {model} starting at age {current_age}")

        # Obtener las tasas de depreciación apropiadas
        if make and make in self.brand_rates:
            # Usar tasa específica por marca si está disponible
            annual_rate = self.brand_rates[make] / 100
            rate_type = 'brand_specific'
        else:
            # Usar tasas estándar de la industria
            industry_rates = self.get_industry_standard_rates()
            rate_type = 'industry_standard'

        # Crear proyección
        projections = []
        current_value = initial_price

        for year in range(current_age, current_age + years_ahead + 1):
            if year == current_age:
                depreciation_rate = 0.0  # Sin depreciación en la antigüedad actual
            elif rate_type == 'brand_specific':
                depreciation_rate = annual_rate
            else:
                depreciation_rate = industry_rates.get(min(year, max(industry_rates.keys())), 2.0) / 100

            # Calcular nuevo valor
            if year > current_age:
                depreciation_amount = current_value * depreciation_rate
                current_value = current_value - depreciation_amount

            projections.append({
                'year': self.current_year - year,  # Año calendario
                'car_age': year,
                'projected_value': max(0, current_value),  # No bajar de 0
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
        Encuentra el último año en que el auto conserva al menos threshold_pct de su valor.

        El "mejor momento para vender" se define como el último año en que el valor proyectado
        todavía está por encima del umbral (por defecto 50% del precio inicial). A partir de
        ese punto la depreciación ha erosionado más de la mitad del valor del auto, por lo que
        vender antes captura más valor.

        Args:
            initial_price: Precio actual u original del auto (MXN).
            make: Marca del auto — usada para tasas de depreciación específicas por marca.
            model: Modelo del auto (informativo, no se usa en el cálculo).
            current_age: Antigüedad actual del auto en años (0 = nuevo).
            threshold_pct: Objetivo de retención de valor (por defecto 50%).
            years_ahead: Máximo de años a proyectar hacia adelante.

        Returns:
            Diccionario con:
              - best_sell_year      : año calendario para vender (int)
              - best_sell_age       : antigüedad del auto en ese momento (int)
              - years_from_now      : cuántos años faltan para ese punto (int)
              - value_at_sell       : valor proyectado en el año de venta (float)
              - retention_pct       : retención real en el año de venta (float)
              - threshold_pct       : el umbral utilizado (float)
              - crosses_below_year  : primer año en que el valor cae *por debajo* del umbral (int | None)
              - projection          : DataFrame completo usado (para gráficas / depuración)
        """
        projection = self.estimate_car_value_over_time(
            initial_price=initial_price,
            make=make,
            model=model,
            current_age=current_age,
            years_ahead=years_ahead,
        )

        # estimate_car_value_over_time almacena 'year' como (current_year - car_age),
        # es decir, el año del modelo yendo HACIA ATRÁS. Para el momento de venta necesitamos
        # el año calendario FUTURO: current_year + años transcurridos desde ahora.
        projection = projection.copy()
        projection['future_year'] = self.current_year + (projection['car_age'] - current_age)

        threshold_value = initial_price * (threshold_pct / 100)

        # Filas donde el valor aún está en o por encima del umbral
        above = projection[projection['projected_value'] >= threshold_value]

        if above.empty:
            # Ya está por debajo del umbral — recomendar vender ahora
            best_row = projection.iloc[0]
            crosses_below = int(projection.iloc[0]['future_year'])
        else:
            best_row = above.iloc[-1]   # último año futuro aún en o por encima del umbral
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
        Calcula la pérdida total por depreciación año a año

        Args:
            initial_price: Precio original
            make: Marca del auto
            years: Número de años a calcular

        Returns:
            DataFrame con los montos de pérdida anual
        """
        projection = self.estimate_car_value_over_time(
            initial_price=initial_price,
            make=make,
            years_ahead=years
        )

        # Calcular pérdidas
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
        Grafica la curva de depreciación de un auto

        Args:
            initial_price: Precio original
            make: Marca del auto
            years: Años a graficar
            save_path: Ruta para guardar la gráfica (opcional)
        """
        projection = self.estimate_car_value_over_time(
            initial_price=initial_price,
            make=make,
            years_ahead=years
        )

        plt.figure(figsize=(12, 8))

        # Graficar valor a lo largo del tiempo
        plt.subplot(2, 1, 1)
        plt.plot(projection['car_age'], projection['projected_value'], 'b-', linewidth=2, marker='o')
        plt.title(f'Proyección de Depreciación\n{make or "Auto Promedio"} - Precio Inicial: ${initial_price:,.0f}')
        plt.xlabel('Antigüedad del Auto (Años)')
        plt.ylabel('Valor Proyectado ($)')
        plt.grid(True, alpha=0.3)

        # Formatear eje Y
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Graficar tasa de depreciación
        plt.subplot(2, 1, 2)
        plt.plot(projection['car_age'], projection['depreciation_rate'], 'r-', linewidth=2, marker='s')
        plt.title('Tasa de Depreciación Anual')
        plt.xlabel('Antigüedad del Auto (Años)')
        plt.ylabel('Tasa de Depreciación (%)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()

    def get_depreciation_summary(self) -> Dict:
        """
        Obtiene un resumen completo de depreciación

        Returns:
            Diccionario con estadísticas de depreciación
        """
        # Calcular depreciación por antigüedad
        age_depr = self.calculate_age_based_depreciation()

        # Calcular depreciación por marca
        brand_depr = self.calculate_brand_depreciation()

        # Obtener tasas de la industria
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
    Función conveniente para estimar la pérdida por depreciación del auto año a año

    Args:
        df: DataFrame con datos de autos
        initial_price: Precio original del auto
        make: Marca del auto (opcional)
        current_age: Antigüedad actual del auto
        years: Años a proyectar

    Returns:
        DataFrame con pérdidas anuales por depreciación
    """
    estimator = CarDepreciationEstimator(df)
    return estimator.calculate_total_loss_by_year(initial_price, make, years)


def estimate_car_loss_by_mileage(df: pd.DataFrame,
                               initial_value: float,
                               current_mileage: float = 0,
                               target_mileage: float = None,
                               mileage_increment: float = 10000) -> pd.DataFrame:
    """
    Función conveniente para estimar la pérdida por depreciación del auto según el kilometraje

    Args:
        df: DataFrame con datos de autos
        initial_value: Valor inicial del auto
        current_mileage: Kilometraje actual (por defecto: 0)
        target_mileage: Kilometraje objetivo a proyectar (opcional)
        mileage_increment: Millas a incrementar en cada paso de proyección

    Returns:
        DataFrame con proyecciones de kilometraje y pérdidas de valor
    """
    estimator = CarDepreciationEstimator(df)
    return estimator.estimate_car_loss_by_mileage(initial_value, current_mileage, target_mileage, mileage_increment)


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv('data/modeling_data/mexico_cars_complete.csv')

    # Crear estimador
    estimator = CarDepreciationEstimator(df)

    # Obtener resumen de depreciación
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

    # Ejemplo de cálculo de depreciación
    print("\n📊 EJEMPLO: Proyección de depreciación para un auto de $500,000")
    losses = estimate_car_loss_by_year(df, 500000, "BMW", years=5)
    print(losses.head())

    # Ejemplo de gráfica
    try:
        estimator.plot_depreciation_curve(500000, "BMW", years=10)
    except:
        print("No se pudo mostrar la gráfica (puede estar corriendo en un entorno sin pantalla)")