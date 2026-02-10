from pathlib import Path
from typing import Any, List, Optional, cast, Annotated

import numpy as np  # type: ignore[import-unresolved]
import pandas as pd  # type: ignore[import-unresolved]
import pickle
import shap  # type: ignore[import-unresolved]
from fastapi import FastAPI, HTTPException  # type: ignore[import-unresolved]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-unresolved]
from pydantic import BaseModel, Field  # type: ignore[import-unresolved]

app = FastAPI()

# Allow local frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).with_name("model.pkl")

model: Optional[Any] = None
feature_names: Optional[List[str]] = None
explainer = None
shap_type: Optional[str] = None
load_error: Optional[Exception] = None

def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Place model.pkl in backend/."
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def _get_feature_names(pipeline) -> Optional[List[str]]:
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        cols = preprocessor.transformers_[0][2]
        return list(cols)
    except Exception:
        return None


def _build_explainer(core_model):
    try:
        # For tree-based models (RF, GB), use TreeExplainer
        if hasattr(core_model, 'estimators_'):
            return shap.TreeExplainer(core_model), "tree"
        # For logistic regression, compute contributions manually
        elif hasattr(core_model, 'coef_'):
            return core_model, "logistic_coefficients"
        return None, None
    except Exception:
        return None, None


try:
    model = _load_model()
    feature_names = _get_feature_names(model)
    model_core = cast(Any, getattr(model, "named_steps", {}).get("model", model))
    explainer, shap_type = _build_explainer(model_core)
except Exception as exc:
    load_error = exc
    model = None
    feature_names = None
    explainer = None
    shap_type = None


class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: Annotated[int, Field(ge=0, le=1)]
    oldpeak: float
    slope: int
    ca: int
    thal: int


def _ensure_model_loaded():
    if model is None:
        detail = "Model not loaded. Place model.pkl in backend/ and restart."
        if load_error:
            detail += f" Load error: {load_error}"
        raise HTTPException(status_code=503, detail=detail)


def _engineer_features(base_payload: dict) -> dict:
    """Apply feature engineering to convert base 13 features to 23 engineered features."""
    engineered = {}
    
    # Copy base features
    engineered['age'] = base_payload['age']
    engineered['sex'] = base_payload['sex']
    engineered['cp'] = base_payload['cp']
    engineered['trestbps'] = base_payload['trestbps']
    engineered['chol'] = base_payload['chol']
    engineered['fbs'] = base_payload['fbs']
    engineered['restecg'] = base_payload['restecg']
    engineered['thalach'] = base_payload['thalach']
    engineered['exang'] = base_payload['exang']
    engineered['oldpeak'] = base_payload['oldpeak']
    engineered['slope'] = base_payload['slope']
    engineered['ca'] = base_payload['ca']
    engineered['thal'] = base_payload['thal']
    
    # Age features
    engineered['age_squared'] = (engineered['age'] ** 2) / 100
    
    # BP features
    engineered['trestbps_squared'] = (engineered['trestbps'] ** 2) / 1000
    # BP category: <120=0, 120-140=1, >140=2
    if engineered['trestbps'] < 120:
        engineered['bp_category'] = 0.0
    elif engineered['trestbps'] <= 140:
        engineered['bp_category'] = 1.0
    else:
        engineered['bp_category'] = 2.0
    
    # Cholesterol features
    engineered['chol_squared'] = (engineered['chol'] ** 2) / 10000
    # Chol category: <200=0, 200-240=1, >240=2
    if engineered['chol'] < 200:
        engineered['chol_category'] = 0.0
    elif engineered['chol'] <= 240:
        engineered['chol_category'] = 1.0
    else:
        engineered['chol_category'] = 2.0
    
    # Heart rate reserve (age-adjusted)
    age_predicted_max_hr = 220 - engineered['age']
    if age_predicted_max_hr > 0:
        hr_reserve = (age_predicted_max_hr - engineered['thalach']) / age_predicted_max_hr
        engineered['hr_reserve'] = max(0.0, min(1.0, hr_reserve))
    else:
        engineered['hr_reserve'] = 0.5
    
    # Oldpeak features
    engineered['oldpeak_squared'] = (engineered['oldpeak'] ** 2) / 10
    
    # Interaction features
    engineered['age_sex_interaction'] = (engineered['age'] * engineered['sex']) / 100
    engineered['bp_chol_interaction'] = (engineered['trestbps'] * engineered['chol']) / 10000
    engineered['exang_oldpeak_interaction'] = engineered['exang'] * engineered['oldpeak']
    
    return engineered


@app.get("/")
def root():
    if model is None:
        return {"status": "degraded", "error": str(load_error)}
    return {"status": "ok", "model": "gradient_boosting"}


@app.post("/predict")
def predict(data: HeartInput):
    _ensure_model_loaded()

    payload = data.model_dump()
    original_payload = payload.copy()

    # Clamp and encode values
    payload['oldpeak'] = float(np.clip(payload['oldpeak'], 0.1, 6.0))
    if payload['oldpeak'] < 2.0:
        payload['oldpeak'] = 0.1
    else:
        payload['oldpeak'] = 2.0 + (payload['oldpeak'] - 2.0) * 0.45
    
    payload['slope'] = int(np.clip(payload['slope'], 0, 3))
    payload['ca'] = int(np.clip(payload['ca'], 0, 4))
    payload['thal'] = int(np.clip(payload['thal'], 1, 3))
    payload['trestbps'] = int(np.clip(payload['trestbps'], 94, 200))
    payload['thalach'] = int(np.clip(payload['thalach'], 71, 202))
    
    # Store original BP/CP/exang for risk adjustment
    original_trestbps = int(original_payload.get("trestbps", 120))
    original_cp = int(np.clip(data.cp, 0, 3))
    original_exang = int(np.clip(data.exang, 0, 1))
    
    # Encode restecg and cp
    restecg_val = int(np.clip(payload['restecg'], 0, 2))
    restecg_encoding = {0: 0.2, 1: 1.0, 2: 2.0}
    payload['restecg'] = restecg_encoding.get(restecg_val, 0.2)
    
    cp_val = int(np.clip(payload['cp'], 0, 3))
    cp_encoding = {0: 0.2, 1: 0.8, 2: 0.8, 3: 2.5}
    payload['cp'] = cp_encoding.get(cp_val, 0.2)
    
    # Engineer features
    engineered = _engineer_features(payload)
    
    # Get ordered feature names from model
    ordered_features = feature_names or list(engineered.keys())
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([engineered], columns=ordered_features)
    
    # Fill any NaN values with mean
    input_df = input_df.fillna(input_df.mean())
    
    assert model is not None
    model_infer = cast(Any, model)
    proba = float(model_infer.predict_proba(input_df)[0][1])
    
    # ========================================================================
    # Risk Classification Based on Probability Thresholds Only
    # ========================================================================
    if proba >= 0.65:
        risk_level = "high"
    elif proba >= 0.41:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # ========================================================================
    # SHAP Explanations - only data needed for visualization
    # ========================================================================
    shap_values = None
    
    if explainer is not None and shap_type == "tree":
        try:
            shap_values_array = explainer.shap_values(input_df)
            if isinstance(shap_values_array, list):
                shap_values_array = shap_values_array[1]
            
            all_contributions = []
            for i in range(len(ordered_features)):
                feature_name = ordered_features[i]
                shap_value = float(shap_values_array[0, i])
                feature_value = float(input_df[feature_name].values[0])
                all_contributions.append({
                    'feature': feature_name,
                    'shap_value': shap_value,
                    'feature_value': feature_value,
                    'abs_shap': abs(shap_value)
                })
            
            all_contributions.sort(key=lambda x: x['abs_shap'], reverse=True)
            top_5_features = all_contributions[:5]
            
            shap_values = {
                'top_features': []
            }
            
            for i, contrib in enumerate(top_5_features):
                feature_name = contrib['feature']
                shap_val = contrib['shap_value']
                feature_val = contrib['feature_value']
                
                # Map to original feature if it's an engineered feature
                original_feature_name = feature_name
                if '_squared' in feature_name:
                    original_feature_name = feature_name.replace('_squared', '')
                    if original_feature_name == 'trestbps':
                        original_feature_name = "Resting BP (squared term)"
                    elif original_feature_name == 'chol':
                        original_feature_name = "Cholesterol (squared term)"
                    elif original_feature_name == 'oldpeak':
                        original_feature_name = "ST Depression (squared term)"
                elif '_category' in feature_name:
                    original_feature_name = feature_name.replace('_category', ' category')
                elif '_interaction' in feature_name:
                    original_feature_name = feature_name.replace('_', ' ')
                elif feature_name == 'hr_reserve':
                    original_feature_name = 'Heart Rate Reserve (age-adjusted)'
                
                direction = "INCREASES" if shap_val > 0 else "DECREASES"
                magnitude = abs(shap_val)
                
                if magnitude > 0.5:
                    impact_level = "VERY HIGH"
                elif magnitude > 0.2:
                    impact_level = "HIGH"
                elif magnitude > 0.1:
                    impact_level = "MODERATE"
                else:
                    impact_level = "LOW"
                
                shap_values['top_features'].append({
                    'rank': i + 1,
                    'feature': original_feature_name,
                    'value': round(feature_val, 2),
                    'shap_contribution': round(shap_val, 4),
                    'impact': f"{impact_level} {direction}",
                    'direction': direction.lower()
                })
        except Exception:
            shap_values = None
    
    return {
        "prediction": 1 if proba >= 0.5 else 0,
        "probability": round(proba, 4),
        "risk_level": risk_level,
        "shap_values": shap_values,
        "explanation": {},
        "features": ordered_features[:13],
        "model_type": shap_type
    }


@app.get("/status")
def status():
    if model is None:
        return {"status": "degraded", "error": str(load_error)}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": shap_type,
        "features": len(feature_names or []),
        "features_list": feature_names
    }
