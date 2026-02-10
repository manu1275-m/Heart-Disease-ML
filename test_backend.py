import sys
sys.path.insert(0, '.')

from bcknd.main import predict, HeartInput

# Test prediction
data = HeartInput(
    age=65, sex=1, cp=0, trestbps=160, chol=280, 
    fbs=1, restecg=2, thalach=150, exang=1, 
    oldpeak=2.0, slope=2, ca=2, thal=3
)

try:
    result = predict(data)
    print("✅ Backend is working!")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Has SHAP: {result['shap_values'] is not None}")
    
    if result['shap_values']:
        print(f"Top features count: {len(result['shap_values'].get('top_features', []))}")
        print("\nTop 3 features:")
        for feat in result['shap_values']['top_features'][:3]:
            print(f"  {feat['rank']}. {feat['feature']}: {feat['shap_contribution']} ({feat['impact']})")
    else:
        print("❌ No SHAP values in response")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
