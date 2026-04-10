"""
Sistema de Recomendación de Autos con la API de Claude
Proporciona recomendaciones personalizadas de autos impulsadas por IA basadas en las preferencias del cliente
"""

import os
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from anthropic import Anthropic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarRecommender:
    """
    Sistema de recomendación de autos impulsado por IA usando la API de Claude
    """

    def __init__(self, api_key: Optional[str] = None, car_data_path: str = "data/modeling_data/mexico_cars_complete.csv"):
        """
        Inicializa el recomendador de autos

        Args:
            api_key: Clave de API de Anthropic (si es None, usa la variable de entorno)
            car_data_path: Ruta al dataset de autos
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        self.client = Anthropic(api_key=self.api_key)

        # Cargar datos de autos
        try:
            self.car_data = pd.read_csv(car_data_path)
            logger.info(f"Loaded {len(self.car_data)} cars from {car_data_path}")
        except FileNotFoundError:
            logger.warning(f"Car data file not found: {car_data_path}")
            self.car_data = pd.DataFrame()

        # Cargar estimador de depreciación para análisis de valor
        self.depreciation_data = self._load_depreciation_data()

    def _load_depreciation_data(self) -> Dict[str, Any]:
        """Carga datos de análisis de depreciación para las recomendaciones"""
        try:
            from modeling_components.car_depreciation_estimator import CarDepreciationEstimator
            estimator = CarDepreciationEstimator(self.car_data)
            brand_depr = estimator.calculate_brand_depreciation()
            return {
                'brand_rates': dict(zip(brand_depr['brand'], brand_depr['avg_depreciation_rate'])),
                'mileage_rate': 4711.08  # $ per 1000 miles
            }
        except Exception as e:
            logger.warning(f"Could not load depreciation data: {e}")
            return {}

    def get_car_recommendations(self,
                              client_profile: Dict[str, Any],
                              max_recommendations: int = 3,
                              include_value_analysis: bool = True) -> Dict[str, Any]:
        """
        Obtiene recomendaciones personalizadas de autos usando Claude AI

        Args:
            client_profile: Diccionario con preferencias del cliente
            max_recommendations: Número máximo de recomendaciones a retornar
            include_value_analysis: Si se debe incluir análisis de depreciación

        Returns:
            Diccionario con recomendaciones y análisis
        """
        # Preparar datos de contexto
        market_context = self._prepare_market_context(client_profile)

        # Crear prompt para Claude
        prompt = self._create_recommendation_prompt(client_profile, market_context, max_recommendations)

        try:
            # Llamar a la API de Claude
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                temperature=0.7,
                system="You are an expert car consultant with deep knowledge of the Mexican automotive market. Provide personalized, practical car recommendations based on client needs and market data.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            recommendations = self._parse_claude_response(response.content[0].text)

            # Agregar análisis de valor si se solicita
            if include_value_analysis and recommendations:
                recommendations = self._add_value_analysis(recommendations, client_profile)

            return {
                'success': True,
                'recommendations': recommendations,
                'client_profile': client_profile,
                'market_context': market_context
            }

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': []
            }

    def _prepare_market_context(self, client_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara el contexto de datos del mercado para Claude"""
        if self.car_data.empty:
            return {}

        # Filtrar autos según el presupuesto y preferencias del cliente
        budget_max = client_profile.get('budget_max', float('inf'))
        filtered_cars = self.car_data[self.car_data['price'] <= budget_max].copy()

        if not filtered_cars.empty:
            # Estadísticas básicas del mercado
            market_stats = {
                'total_cars_available': len(filtered_cars),
                'price_range': {
                    'min': int(filtered_cars['price'].min()),
                    'max': int(filtered_cars['price'].max()),
                    'median': int(filtered_cars['price'].median())
                },
                'popular_brands': filtered_cars['make'].value_counts().head(5).to_dict(),
                'popular_body_types': filtered_cars['body_type'].value_counts().head(3).to_dict() if 'body_type' in filtered_cars.columns else {},
                'fuel_types': filtered_cars['fuel_type'].value_counts().to_dict() if 'fuel_type' in filtered_cars.columns else {},
                'avg_mileage': int(filtered_cars['miles'].mean()) if 'miles' in filtered_cars.columns else None
            }

            # Contexto de confiabilidad/depreciación por marca
            if self.depreciation_data:
                market_stats['brand_depreciation_insights'] = {
                    'best_value_retention': sorted(self.depreciation_data.get('brand_rates', {}).items(),
                                                 key=lambda x: x[1])[:3],
                    'fastest_depreciation': sorted(self.depreciation_data.get('brand_rates', {}).items(),
                                                 key=lambda x: x[1], reverse=True)[:3]
                }

            return market_stats

        return {}

    def _create_recommendation_prompt(self, client_profile: Dict[str, Any],
                                    market_context: Dict[str, Any],
                                    max_recommendations: int) -> str:
        """Crea el prompt de Claude para recomendaciones de autos"""

        prompt_parts = [
            "Based on the following client profile and market data, recommend the best cars for this client.",
            "",
            "CLIENT PROFILE:"
        ]

        # Preferencias del cliente
        for key, value in client_profile.items():
            if value is not None:
                prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")

        prompt_parts.extend([
            "",
            "MARKET CONTEXT:"
        ])

        # Datos del mercado
        if market_context:
            prompt_parts.extend([
                f"- Total cars available in budget: {market_context.get('total_cars_available', 'N/A')}",
                f"- Price range: ${market_context.get('price_range', {}).get('min', 'N/A'):,} - ${market_context.get('price_range', {}).get('max', 'N/A'):,}",
                f"- Popular brands: {', '.join(market_context.get('popular_brands', {}).keys())}",
                f"- Popular body types: {', '.join(market_context.get('popular_body_types', {}).keys())}"
            ])

            if 'brand_depreciation_insights' in market_context:
                best_value = [brand for brand, _ in market_context['brand_depreciation_insights']['best_value_retention']]
                prompt_parts.append(f"- Best value retention brands: {', '.join(best_value)}")

        prompt_parts.extend([
            "",
            f"Provide exactly {max_recommendations} car recommendations in the following JSON format:",
            "{",
            '  "recommendations": [',
            '    {',
            '      "rank": 1,',
            '      "make": "Brand Name",',
            '      "model": "Model Name",',
            '      "year_range": "2020-2023",',
            '      "estimated_price": 350000,',
            '      "key_features": ["feature1", "feature2", "feature3"],',
            '      "why_recommended": "Brief explanation of why this car fits the client",',
            '      "pros": ["pro1", "pro2"],',
            '      "cons": ["con1", "con2"],',
            '      "alternative_considerations": "Any alternatives to consider"',
            '    }',
            '  ]',
            "}",
            "",
            "Guidelines:",
            "- Focus on cars available in the Mexican market",
            "- Consider reliability, maintenance costs, and resale value",
            "- Match recommendations to client's specific needs and budget",
            "- Be practical and realistic about market conditions",
            "- Include both pros and cons for balanced recommendations"
        ])

        return "\n".join(prompt_parts)

    def _parse_claude_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parsea la respuesta JSON de Claude"""
        try:
            # Extraer JSON de la respuesta
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                return parsed.get('recommendations', [])
            else:
                logger.warning("No se encontró JSON en la respuesta de Claude")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return []

    def _add_value_analysis(self, recommendations: List[Dict[str, Any]],
                          client_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Agrega análisis de depreciación y valor a las recomendaciones"""
        if not self.depreciation_data:
            return recommendations

        for rec in recommendations:
            try:
                brand = rec.get('make', '')
                price = rec.get('estimated_price', 0)

                # Agregar información de depreciación
                if brand in self.depreciation_data.get('brand_rates', {}):
                    annual_depr = self.depreciation_data['brand_rates'][brand]
                    rec['value_analysis'] = {
                        'annual_depreciation_rate': round(annual_depr, 2),
                        'estimated_5year_value': round(price * (1 - annual_depr/100 * 5), 0),
                        'depreciation_category': 'Lenta' if annual_depr < 15 else 'Moderada' if annual_depr < 25 else 'Rápida'
                    }

                # Agregar información de impacto por kilometraje
                if self.depreciation_data.get('mileage_rate'):
                    rec['mileage_impact'] = {
                        'loss_per_1000_miles': self.depreciation_data['mileage_rate'],
                        'estimated_50000_mile_value': round(price * 0.7, 0)  # Estimación aproximada
                    }

            except Exception as e:
                logger.warning(f"Could not add value analysis for {rec.get('make', 'Unknown')}: {e}")

        return recommendations

    def recommend_from_natural_language(self,
                                        user_text: str,
                                        max_recommendations: int = 3) -> Dict[str, Any]:
        """
        Obtiene recomendaciones de autos directamente desde una descripción en lenguaje natural.

        Args:
            user_text: Texto libre describiendo lo que necesita el cliente
            max_recommendations: Número de recomendaciones a retornar

        Returns:
            Diccionario con recomendaciones y perfil extraído
        """
        market_summary = ""
        if not self.car_data.empty:
            brands = ", ".join(self.car_data['make'].value_counts().head(8).index.tolist())
            market_summary = f"\nAvailable market brands include: {brands}."

        system_prompt = (
            "You are an expert car consultant for the Mexican automotive market. "
            "A client will describe what they need in natural language. "
            "First extract the key requirements, then recommend the best cars available in Mexico."
            + market_summary
        )

        prompt = (
            f"Client description:\n\"{user_text}\"\n\n"
            f"Based on this description, provide exactly {max_recommendations} car recommendations. "
            "Return a JSON object with two keys:\n"
            "1. \"extracted_profile\": the client preferences you inferred (budget, use, family size, etc.)\n"
            "2. \"recommendations\": an array of car objects, each with:\n"
            "   - rank, make, model, year_range, estimated_price (MXN),\n"
            "     key_features (list), why_recommended, pros (list), cons (list)\n\n"
            "Guidelines:\n"
            "- Focus on cars available in Mexico\n"
            "- Be practical and realistic about budget and availability\n"
            "- If budget is not mentioned, make reasonable assumptions and state them\n"
            "- Return ONLY the JSON object, no extra text"
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            raw = response.content[0].text
            start_idx = raw.find('{')
            end_idx = raw.rfind('}') + 1
            parsed = json.loads(raw[start_idx:end_idx])

            recommendations = parsed.get('recommendations', [])
            extracted_profile = parsed.get('extracted_profile', {})

            if self.depreciation_data and recommendations:
                recommendations = self._add_value_analysis(recommendations, extracted_profile)

            return {
                'success': True,
                'extracted_profile': extracted_profile,
                'recommendations': recommendations,
            }

        except Exception as e:
            logger.error(f"Error in natural language recommendation: {e}")
            return {'success': False, 'error': str(e), 'recommendations': []}

    def get_client_questionnaire(self) -> Dict[str, Any]:
        """Obtiene un cuestionario estructurado para perfilar al cliente"""
        return {
            'budget': {
                'min_budget': 'Minimum budget in MXN',
                'max_budget': 'Maximum budget in MXN',
                'down_payment': 'Available down payment'
            },
            'usage': {
                'primary_use': 'City driving, highway, off-road, etc.',
                'daily_mileage': 'Average daily kilometers',
                'passengers': 'Number of regular passengers',
                'cargo_needs': 'Cargo space requirements'
            },
            'preferences': {
                'fuel_type': 'Gasoline, diesel, hybrid, electric',
                'transmission': 'Manual, automatic, CVT',
                'body_type': 'Sedan, SUV, hatchback, pickup, etc.',
                'brand_preferences': 'Preferred or avoided brands',
                'key_features': 'Must-have features (A/C, ABS, etc.)'
            },
            'lifestyle': {
                'family_size': 'Number of family members',
                'maintenance_budget': 'Monthly maintenance budget',
                'reliability_importance': 'How important is reliability (1-10)',
                'resale_value_importance': 'How important is resale value (1-10)'
            }
        }


def recommend_car_for_client(client_profile: Dict[str, Any],
                           api_key: Optional[str] = None,
                           max_recommendations: int = 3) -> Dict[str, Any]:
    """
    Función conveniente para obtener recomendaciones de autos

    Args:
        client_profile: Preferencias y requisitos del cliente
        api_key: Clave de API de Anthropic
        max_recommendations: Número de recomendaciones a retornar

    Returns:
        Diccionario con recomendaciones
    """
    recommender = CarRecommender(api_key=api_key)
    return recommender.get_car_recommendations(client_profile, max_recommendations)


# Ejemplo de uso
if __name__ == "__main__":
    # Perfil de cliente de ejemplo
    client_example = {
        'name': 'Juan Pérez',
        'budget_max': 500000,
        'budget_min': 200000,
        'primary_use': 'City driving and family transport',
        'family_size': 4,
        'fuel_type_preference': 'Gasoline',
        'reliability_importance': 9,
        'key_requirements': ['Air conditioning', 'ABS', 'Multiple airbags', 'Good fuel economy']
    }

    # Normalmente esto se llamaría con una clave de API válida
    print("Sistema de recomendación de autos inicializado")
    print("Para usar: establece la variable de entorno ANTHROPIC_API_KEY y llama a recommend_car_for_client()")
    print("\nEstructura de perfil de cliente de ejemplo:")
    print(json.dumps(client_example, indent=2))