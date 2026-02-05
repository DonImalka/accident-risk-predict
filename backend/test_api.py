import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = {
    "road_type": "highway",
    "num_lanes": 2,
    "curvature": 0.0,
    "speed_limit": 100.0,
    "lighting": "daylight",
    "weather": "clear",
    "road_signs_present": 1,
    "public_road": 1,
    "time_of_day": "morning",
    "holiday": 0,
    "school_season": 0,
    "num_reported_accidents": 0
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
