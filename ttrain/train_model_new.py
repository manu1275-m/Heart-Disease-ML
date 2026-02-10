from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "HeartDiseaseTrain-Test.csv"
MODEL_PATH = ROOT / "bcknd" / "model.pkl"

# API schema uses these names
FEATURE_COLS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COL = "target"


def load_and_preprocess_data(path: Path) -> pd.DataFrame:
    """Load data and convert categorical columns to numeric matching API schema."""
    if not path.exists():
        raise FileNotFoundError(f"Missing data at {path}")
    
    df = pd.read_csv(path)
    
    # Create new dataframe with API column names
    processed = pd.DataFrame()
    
    # Age (unchanged)
    processed['age'] = df['age']
    
    # Sex: Male=1, Female=0
    processed['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    # Chest pain type: encode for risk ordering
    cp_map = {
        'Typical angina': 0,
        'Atypical angina': 1,
        'Non-anginal pain': 2,
        'Asymptomatic': 3
    }
    processed['cp'] = df['chest_pain_type'].map(cp_map)
    
    # Resting BP (unchanged)
    processed['trestbps'] = df['resting_blood_pressure']
    
    # Cholesterol (unchanged)
    processed['chol'] = df['cholestoral']
    
    # Fasting blood sugar: >120=1, <=120=0
    processed['fbs'] = df['fasting_blood_sugar'].map({
        'Greater than 120 mg/ml': 1,
        'Lower than 120 mg/ml': 0
    })
    
    # Rest ECG: Normal=0, ST-T=1, LVH=2
    ecg_map = {
        'Normal': 0,
        'ST-T wave abnormality': 1,
        'Left ventricular hypertrophy': 2
    }
    processed['restecg'] = df['rest_ecg'].map(ecg_map)
    
    # Max heart rate (unchanged)
    processed['thalach'] = df['Max_heart_rate']
    
    # Exercise induced angina: Yes=1, No=0
    processed['exang'] = df['exercise_induced_angina'].map({'Yes': 1, 'No': 0})
    
    # Oldpeak (unchanged)
    processed['oldpeak'] = df['oldpeak']
    
    # Slope: Up=0, Flat=1, Down=2
    slope_map = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }
    processed['slope'] = df['slope'].map(slope_map)
    
    # CA (vessels colored): Zero=0, One=1, Two=2, Three=3
    ca_map = {
        'Zero': 0,
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Four': 3
    }
    processed['ca'] = df['vessels_colored_by_flourosopy'].map(ca_map)
    
    # Thal: Normal=1, Fixed Defect=2, Reversible Defect=3
    thal_map = {
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversable Defect': 3,
        'No': 1
    }
    processed['thal'] = df['thalassemia'].map(thal_map)
    
    # Target (INVERTED: in this dataset 0=disease, 1=healthy, but API expects 1=disease, 0=healthy)
    processed['target'] = 1 - df['target']  # Flip 0->1 and 1->0
    
    return processed


def build_pipeline(feature_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)], remainder="drop"
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        verbose=0
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train() -> None:
    print(f"Loading data from {DATA_PATH}")
    df = load_and_preprocess_data(DATA_PATH)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining model on {len(X_train)} samples...")
    pipeline = build_pipeline(FEATURE_COLS)
    
    # Cross-validation to check for overfitting
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
