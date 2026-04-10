#!/usr/bin/env python3
"""
Simple example of using the Car Recommender with Claude API
"""

import os
from car_recommender import recommend_car_for_client

def main():
    """Example usage of car recommendation system"""

    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        print("Get your key from: https://console.anthropic.com/")
        return

    # Example client: Young professional looking for reliable city car
    client_profile = {
        'name': 'Sofia Martinez',
        'budget_max': 400000,  # MXN
        'budget_min': 200000,
        'primary_use': 'City commuting and weekend trips',
        'family_size': 2,
        'fuel_type_preference': 'Gasoline',
        'reliability_importance': 9,
        'key_requirements': [
            'Good fuel economy',
            'Modern safety features',
            'Comfortable for daily driving',
            'Easy maintenance'
        ],
        'lifestyle': 'Urban professional, values reliability and efficiency'
    }

    print("🚗 GETTING CAR RECOMMENDATIONS FOR SOFIA MARTINEZ")
    print("=" * 60)
    print(f"Budget: ${client_profile['budget_min']:,.0f} - ${client_profile['budget_max']:,.0f}")
    print(f"Use: {client_profile['primary_use']}")
    print(f"Family: {client_profile['family_size']} people")
    print(f"Priority: Reliability ({client_profile['reliability_importance']}/10)")

    try:
        # Get recommendations
        result = recommend_car_for_client(client_profile, max_recommendations=3)

        if result['success']:
            print(f"\n✅ FOUND {len(result['recommendations'])} RECOMMENDATIONS:\n")

            for i, rec in enumerate(result['recommendations'], 1):
                print(f"🏆 RECOMMENDATION #{i}")
                print(f"   {rec.get('make', 'Unknown')} {rec.get('model', 'Unknown')}")
                print(f"   💰 Price: ${rec.get('estimated_price', 0):,.0f}")
                print(f"   📅 Year range: {rec.get('year_range', 'N/A')}")
                print(f"   🎯 Why recommended: {rec.get('why_recommended', 'N/A')}")

                if rec.get('key_features'):
                    print(f"   ⭐ Key features: {', '.join(rec['key_features'][:3])}")

                if rec.get('pros'):
                    print(f"   ✅ Pros: {', '.join(rec['pros'][:2])}")

                if rec.get('cons'):
                    print(f"   ⚠️  Cons: {', '.join(rec['cons'][:2])}")

                if 'value_analysis' in rec:
                    va = rec['value_analysis']
                    print(f"   📊 Value: {va.get('depreciation_category', 'Unknown')} depreciation")
                    print(f"       5-year estimated value: ${va.get('estimated_5year_value', 0):,.0f}")

                print()

        else:
            print(f"❌ Error getting recommendations: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your ANTHROPIC_API_KEY is valid and you have internet connection")


if __name__ == "__main__":
    main()