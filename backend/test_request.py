import requests
import json

url = "http://127.0.0.1:8080/predict"
data = {
    "road_type": "Highway",
    "num_lanes": 4,
    "curvature": 0.1,
    "speed_limit": 100,
    "lighting": "Daylight",
    "weather": "Clear",
    "road_signs_present": 1,
    "public_road": 1,
    "time_of_day": "Day",
    "holiday": 0,
    "school_season": 0,
    "num_reported_accidents": 5
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
