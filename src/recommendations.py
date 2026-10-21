"""
Weather Recommendations Module for WeatherWise Analytics.
Translates machine learning model outputs (rain probability, predicted temperature)
into actionable daily guidance and safety recommendations.
"""

from typing import Any, Dict


def generate_weather_recommendations(
    rain_probability: float,
    predicted_max_temp: float,
    current_temp_diff: float,
    month: int
) -> Dict[str, Any]:
    """
    Translates model predictions into understandable, actionable weather recommendations.
    
    Decision Rules:
    - Umbrella: Thresholds at 30% and 60% rain probability
    - Clothing: Evaluated across 5 temperature brackets (<15°C, 15-22°C, 22-32°C, 32-40°C, >40°C)
    - Outdoor Activity: Joint assessment of precipitation risk and heat/cold stress
    """
    prob_pct = round(rain_probability * 100, 1)

    # 1. Umbrella Recommendation
    if rain_probability >= 0.60:
        umbrella_status = "Essential — Strong Rain Probability"
        umbrella_advice = f"High precipitation likelihood ({prob_pct}%). Carry a sturdy umbrella and waterproof outerwear before heading out."
        umbrella_badge = "High Risk"
        umbrella_color = "#E76F51"
    elif rain_probability >= 0.30:
        umbrella_status = "Recommended — Moderate Rain Likelihood"
        umbrella_advice = f"Moderate precipitation likelihood ({prob_pct}%). Keeping a compact umbrella in your vehicle or bag is advised."
        umbrella_badge = "Moderate Risk"
        umbrella_color = "#F4A261"
    else:
        umbrella_status = "Not Required — Dry Conditions Likely"
        umbrella_advice = f"Low precipitation likelihood ({prob_pct}%). Normal outdoor commute without rain gear is supported."
        umbrella_badge = "Low Risk"
        umbrella_color = "#00A896"

    # 2. Temperature and Clothing Advice
    if predicted_max_temp < 15.0:
        temp_category = "Cold Weather Alert"
        clothing_advice = "Heavy winter coat, woolen layers, scarf, and thermal wear recommended, especially during morning and evening hours."
        temp_badge = "Cold Wave"
        temp_color = "#028090"
    elif predicted_max_temp < 22.0:
        temp_category = "Pleasant / Cool Weather"
        clothing_advice = "Light jacket, pullover, or cardigan over comfortable everyday attire."
        temp_badge = "Pleasant"
        temp_color = "#00A896"
    elif predicted_max_temp < 32.0:
        temp_category = "Moderate / Comfortable"
        clothing_advice = "Breathable cotton clothing, sunglasses, and standard comfortable attire."
        temp_badge = "Comfortable"
        temp_color = "#2A9D8F"
    elif predicted_max_temp < 40.0:
        temp_category = "Hot Summer Conditions"
        clothing_advice = "Light-colored loose fabrics, cap or hat, sunscreen, and regular hydration."
        temp_badge = "Hot"
        temp_color = "#F4A261"
    else:
        temp_category = "Severe Heatwave Advisory"
        clothing_advice = "High thermal stress. Avoid direct sun exposure between 12:00 PM and 4:00 PM. Drink plenty of water and wear UV-protective headgear."
        temp_badge = "Extreme Heat"
        temp_color = "#E76F51"

    # 3. Outdoor Activity Feasibility Index
    if rain_probability >= 0.60 or predicted_max_temp >= 42.0 or predicted_max_temp < 8.0:
        activity_status = "Adverse / Unfavorable"
        activity_reason = "Elevated risk of heavy precipitation or thermal discomfort. Consider indoor venues for sports, gatherings, or extended travel."
        activity_color = "#E76F51"
    elif rain_probability >= 0.30 or predicted_max_temp >= 38.0 or current_temp_diff > 18.0:
        activity_status = "Proceed with Caution"
        activity_reason = "Weather conditions are viable but unstable. Monitor sky conditions, plan shaded breaks, and allow schedule flexibility."
        activity_color = "#F4A261"
    else:
        activity_status = "Highly Favorable"
        activity_reason = "Stable meteorological conditions predicted. Highly suitable for travel, sports, commuting, and outdoor construction."
        activity_color = "#00A896"

    # 4. Seasonal Context
    seasons = {
        (12, 1, 2): "Winter Season (Western disturbances can cause winter showers; morning fog common)",
        (3, 4, 5): "Pre-Monsoon / Summer Season (High daytime temperatures, localized dust storms)",
        (6, 7, 8, 9): "Southwest Monsoon Season (Peak rainfall window, humid conditions, intermittent downpours)",
        (10, 11): "Post-Monsoon / Autumn Transition (Decreasing humidity, clear skies, dropping night temperatures)"
    }
    seasonal_summary = "Standard seasonal window"
    for months, text in seasons.items():
        if month in months:
            seasonal_summary = text
            break

    return {
        "umbrella": {
            "status": umbrella_status,
            "advice": umbrella_advice,
            "badge": umbrella_badge,
            "color": umbrella_color,
            "rain_probability": prob_pct,
        },
        "temperature": {
            "predicted_max": round(predicted_max_temp, 1),
            "category": temp_category,
            "advice": clothing_advice,
            "badge": temp_badge,
            "color": temp_color,
        },
        "activity": {
            "status": activity_status,
            "reason": activity_reason,
            "color": activity_color,
        },
        "seasonal_context": seasonal_summary,
        "disclaimer": (
            "Notice: These recommendations are educational and analytical estimates derived from "
            "historical Delhi weather patterns (1951-2024). They are not official meteorological "
            "forecasts issued by the India Meteorological Department (IMD)."
        )
    }
