#!/usr/bin/env python
"""Direct app test"""
import sys
sys.path.insert(0, "c:/Users/manu7/VSCode/ML project")

from bcknd.main import app, HeartInput
from fastapi.testclient import TestClient

client = TestClient(app)

# Test the endpoint
data = {
    "age": 40, "sex": 0, "cp": 3, "trestbps": 110, "chol": 150,
    "fbs": 0, "restecg": 0, "thalach": 90, "exang": 0,
    "oldpeak": 0.5, "slope": 0, "ca": 0, "thal": 1
}

try:
    response = client.post("/predict", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Risk Level: {result['risk_level']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Explanation: {result.get('explanation', 'N/A')}")
    print(f"SHAP Values: {len(result['shap_values'])} values")
    print("Success!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
