#!/usr/bin/env python
"""Test all three risk categories with explanations"""
from bcknd.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

test_cases = [
    ('Low', {'age': 40, 'sex': 0, 'cp': 3, 'trestbps': 110, 'chol': 150, 'fbs': 0, 'restecg': 0, 'thalach': 90, 'exang': 0, 'oldpeak': 0.5, 'slope': 0, 'ca': 0, 'thal': 1}),
    ('Moderate', {'age': 55, 'sex': 1, 'cp': 1, 'trestbps': 130, 'chol': 200, 'fbs': 0, 'restecg': 1, 'thalach': 120, 'exang': 0, 'oldpeak': 1.0, 'slope': 1, 'ca': 1, 'thal': 2}),
    ('High', {'age': 65, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 280, 'fbs': 1, 'restecg': 2, 'thalach': 150, 'exang': 1, 'oldpeak': 2.0, 'slope': 2, 'ca': 2, 'thal': 3}),
]

print('='*100)
print(f"{'Category':10s} {'Probability':>12s} {'Risk Level':>12s} Explanation")
print('='*100)
for name, data in test_cases:
    resp = client.post('/predict', json=data)
    r = resp.json()
    prob = round(r['probability'] * 100, 2)
    risk = r['risk_level']
    expl = r.get('explanation', 'N/A')
    print(f"{name:10s} {prob:>11.2f}% {risk:>12s} {expl}")
print('='*100)
