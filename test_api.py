#!/usr/bin/env python
"""Quick API test script"""
import requests
import json

# Test cases
test_cases = [
    {
        "name": "Low Risk",
        "data": {
            "age": 40, "sex": 0, "cp": 3, "trestbps": 110, "chol": 150, 
            "fbs": 0, "restecg": 0, "thalach": 90, "exang": 0, 
            "oldpeak": 0.5, "slope": 0, "ca": 0, "thal": 1
        }
    },
    {
        "name": "Moderate Risk",
        "data": {
            "age": 55, "sex": 1, "cp": 1, "trestbps": 130, "chol": 200, 
            "fbs": 0, "restecg": 1, "thalach": 120, "exang": 0, 
            "oldpeak": 1.0, "slope": 1, "ca": 1, "thal": 2
        }
    },
    {
        "name": "High Risk",
        "data": {
            "age": 65, "sex": 1, "cp": 0, "trestbps": 160, "chol": 280, 
            "fbs": 1, "restecg": 2, "thalach": 150, "exang": 1, 
            "oldpeak": 2.0, "slope": 2, "ca": 2, "thal": 3
        }
    }
]

print("Testing API endpoint...")
for tc in test_cases:
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=tc["data"],
            timeout=5
        )
        result = response.json()
        prob = round(result["probability"] * 100, 2)
        risk = result["risk_level"]
        explanation = result.get("explanation", "N/A")
        print(f"{tc['name']:15s}: {prob:6.2f}% -> {risk:10s} | {explanation}")
    except Exception as e:
        print(f"{tc['name']:15s}: ERROR - {e}")
