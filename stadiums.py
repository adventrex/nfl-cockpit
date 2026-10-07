"""stadiums.py — stadium coordinates + Open-Meteo forecast (pure; no Streamlit)."""
import requests

STADIUMS = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4010, "retractable"), "BAL": (39.2780, -76.6227, "open"),
    "BUF": (42.7738, -78.7870, "open"), "CAR": (35.2258, -80.8528, "open"), "CHI": (41.8623, -87.6167, "open"),
    "CIN": (39.0955, -84.5161, "open"), "CLE": (41.5061, -81.6995, "open"), "DAL": (32.7473, -97.0945, "retractable"),
    "DEN": (39.7439, -105.0201, "open"), "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "open"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"), "JAX": (30.3240, -81.6373, "open"),
    "KC": (39.0489, -94.4839, "open"), "LV": (36.0909, -115.1833, "dome"), "LAC": (33.9535, -118.3390, "dome"),
    "LA": (33.9535, -118.3390, "dome"), "MIA": (25.9580, -80.2389, "open"), "MIN": (44.9735, -93.2575, "dome"),
    "NE": (42.0909, -71.2643, "open"), "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8135, -74.0745, "open"),
    "NYJ": (40.8135, -74.0745, "open"), "PHI": (39.9008, -75.1675, "open"), "PIT": (40.4468, -80.0158, "open"),
    "SF": (37.4032, -121.9698, "open"), "SEA": (47.5952, -122.3316, "open"), "TB": (27.9759, -82.5033, "open"),
    "TEN": (36.1665, -86.7713, "open"), "WAS": (38.9077, -76.8645, "open"),
}


def get_forecast(team, date_obj):
    s = STADIUMS.get(team)
    if not s:
        return None
    lat, lon, kind = s
    if kind != "open":
        return {"desc": "dome", "is_closed": True, "temp": 72, "wind": 0, "rain": 0}
    try:
        d = date_obj.strftime("%Y-%m-%d")
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
            "latitude": lat, "longitude": lon, "daily": "temperature_2m_max,precipitation_probability_max,windspeed_10m_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "auto", "start_date": d, "end_date": d})
        dd = r.json()["daily"]
        return {"desc": f"{dd['temperature_2m_max'][0]}F wind {dd['windspeed_10m_max'][0]} rain {dd['precipitation_probability_max'][0]}%",
                "is_closed": False, "temp": dd["temperature_2m_max"][0], "wind": dd["windspeed_10m_max"][0], "rain": dd["precipitation_probability_max"][0]}
    except Exception:
        return {"desc": "n/a", "is_closed": False, "temp": 70, "wind": 5, "rain": 0}
