"""Test SHAP directly"""
import sys
sys.path.insert(0, 'c:\\Users\\manu7\\VSCode\\ML project')

from bcknd.main import predict, HeartInput

# Test data
data = HeartInput(
    age=65,
    sex=1,
    cp=0,
    trestbps=160,
    chol=280,
    fbs=1,
    restecg=2,
    thalach=150,
    exang=1,
    oldpeak=2.0,
    slope=2,
    ca=2,
    thal=3
)

print("Testing prediction with SHAP...")
result = predict(data)

print("\n=== RESULT ===")
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Model Type: {result['model_type']}")

print("\n=== SHAP VALUES ===")
if result['shap_values']:
    print(f"Type: {type(result['shap_values'])}")
    print(f"Keys: {result['shap_values'].keys() if isinstance(result['shap_values'], dict) else 'Not a dict'}")
    if isinstance(result['shap_values'], dict):
        print(f"Base Value: {result['shap_values'].get('base_value')}")
        print(f"Model Prediction: {result['shap_values'].get('model_prediction')}")
        print(f"Top Features: {len(result['shap_values'].get('top_features', []))} features")
        if result['shap_values'].get('top_features'):
            print("\nTop 3 Features:")
            for feat in result['shap_values']['top_features'][:3]:
                print(f"  {feat['rank']}. {feat['feature']}: {feat['shap_contribution']} ({feat['impact']})")
else:
    print("SHAP values are None!")

print("\n=== EXPLANATION ===")
if result['explanation']:
    print(f"Keys: {result['explanation'].keys()}")
    if 'error' in result['explanation']:
        print(f"ERROR: {result['explanation']['error']}")
    if 'clinical_insights' in result['explanation']:
        print(f"Clinical Insights: {len(result['explanation']['clinical_insights'])} insights")
else:
    print("Explanation is empty!")
