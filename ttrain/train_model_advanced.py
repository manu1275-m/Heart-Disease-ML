from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pickle

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "HeartDiseaseTrain-Test.csv"
MODEL_PATH = ROOT / "bcknd" / "model.pkl"

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
    """Load data with advanced feature engineering."""
    if not path.exists():
        raise FileNotFoundError(f"Missing data at {path}")
    
    df = pd.read_csv(path)
    processed = {}
    
    # Age features with polynomial and interaction terms
    processed['age'] = df['age']
    processed['age_squared'] = (df['age'] ** 2) / 100  # Normalize
    
    # Sex
    processed['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    # Chest pain type with risk encoding
    cp_map = {
        'Typical angina': 0.2,
        'Atypical angina': 0.8,
        'Non-anginal pain': 0.8,
        'Asymptomatic': 2.5
    }
    processed['cp'] = df['chest_pain_type'].map(cp_map)
    
    # Resting BP with additional features
    trestbps = df['resting_blood_pressure']
    processed['trestbps'] = trestbps
    processed['trestbps_squared'] = (trestbps ** 2) / 1000
    processed['bp_category'] = pd.cut(trestbps, bins=[0, 120, 140, 300], labels=[0, 1, 2], include_lowest=True).astype(float)
    
    # Cholesterol with additional features
    chol = df['cholestoral']
    processed['chol'] = chol
    processed['chol_squared'] = (chol ** 2) / 10000
    processed['chol_category'] = pd.cut(chol, bins=[0, 200, 240, 500], labels=[0, 1, 2], include_lowest=True).astype(float)
    
    # Fasting blood sugar
    processed['fbs'] = df['fasting_blood_sugar'].map({
        'Greater than 120 mg/ml': 1,
        'Lower than 120 mg/ml': 0
    })
    
    # Rest ECG
    ecg_map = {
        'Normal': 0.2,
        'ST-T wave abnormality': 1.0,
        'Left ventricular hypertrophy': 2.0
    }
    processed['restecg'] = df['rest_ecg'].map(ecg_map)
    
    # Max heart rate with additional features
    thalach = df['Max_heart_rate']
    processed['thalach'] = thalach
    
    # Heart rate reserve (age-adjusted max HR)
    age_predicted_max_hr = 220 - df['age']
    hr_reserve = (age_predicted_max_hr - thalach) / age_predicted_max_hr
    hr_reserve = hr_reserve.clip(0, 1).fillna(0.5)
    processed['hr_reserve'] = hr_reserve
    
    # Exercise induced angina
    processed['exang'] = df['exercise_induced_angina'].map({'Yes': 1, 'No': 0})
    
    # ST depression induced by exercise (oldpeak)
    oldpeak = df['oldpeak'].clip(0.1, 6.0)
    oldpeak = oldpeak.where(oldpeak >= 2.0, 0.1)
    oldpeak = oldpeak.where(oldpeak < 2.0, 2.0 + (oldpeak - 2.0) * 0.45)
    processed['oldpeak'] = oldpeak
    processed['oldpeak_squared'] = (oldpeak ** 2) / 10
    
    # Slope
    slope_map = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }
    processed['slope'] = df['slope'].map(slope_map)
    
    # Colored vessels
    ca_map = {
        'Zero': 0,
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Four': 3
    }
    processed['ca'] = df['vessels_colored_by_flourosopy'].map(ca_map)
    
    # Thalassemia
    thal_map = {
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversable Defect': 3,
        'No': 1
    }
    processed['thal'] = df['thalassemia'].map(thal_map)
    
    # Interaction features
    processed['age_sex_interaction'] = (df['age'] * processed['sex']) / 100
    processed['bp_chol_interaction'] = (trestbps * chol) / 10000
    processed['exang_oldpeak_interaction'] = processed['exang'] * oldpeak
    
    # Target (inverted)
    processed['target'] = 1 - df['target']
    
    # Define the correct feature order to maintain consistency
    feature_order = [
        'age', 'age_squared', 'sex', 'cp', 'trestbps', 'trestbps_squared', 'bp_category',
        'chol', 'chol_squared', 'chol_category', 'fbs', 'restecg', 'thalach', 'hr_reserve',
        'exang', 'oldpeak', 'oldpeak_squared', 'slope', 'ca', 'thal',
        'age_sex_interaction', 'bp_chol_interaction', 'exang_oldpeak_interaction', 'target'
    ]
    
    # Convert to DataFrame with correct column order
    result_df = pd.DataFrame({col: processed[col] for col in feature_order})
    return result_df


def build_advanced_pipeline(feature_cols: List[str]) -> Pipeline:
    """Build logistic regression pipeline with standardization."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop"
    )
    
    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
        C=0.1,  # Stronger regularization
        random_state=42,
        penalty="l2",
        tol=1e-4
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_rf_pipeline(feature_cols: List[str]) -> Pipeline:
    """Build Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop"
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def build_gb_pipeline(feature_cols: List[str]) -> Pipeline:
    """Build Gradient Boosting pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop"
    )
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        loss="log_loss",
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_and_evaluate() -> Pipeline:
    """Train multiple models and return the best one."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    df = load_and_preprocess_data(DATA_PATH)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Features: {len([c for c in df.columns if c != 'target'])}")
    print(f"Target distribution:\n{df['target'].value_counts()}\n")
    
    # Use all engineered features and handle missing values
    all_feature_cols = [c for c in df.columns if c != 'target']
    X = df[all_feature_cols]
    y = df['target']
    
    # Fill any remaining NaN values
    X = X.fillna(X.mean())
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train target distribution:\n{y_train.value_counts()}\n")
    
    results = {}
    best_model = None
    best_f1 = 0
    
    # ============================================================
    # Train Logistic Regression
    # ============================================================
    print("=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)
    lr_pipeline = build_advanced_pipeline(all_feature_cols)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        lr_pipeline, X_train, y_train, cv=skf, 
        scoring='f1', n_jobs=-1
    )
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
    
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    f1_lr = f1_score(y_test, y_pred_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    
    results['LogisticRegression'] = {
        'f1': f1_lr, 'accuracy': acc_lr, 'auc': auc_lr,
        'model': lr_pipeline, 'predictions': y_pred_lr, 'probas': y_proba_lr
    }
    print(f"LR Accuracy: {acc_lr:.4f}, F1: {f1_lr:.4f}, AUC: {auc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr))
    
    if f1_lr > best_f1:
        best_f1 = f1_lr
        best_model = lr_pipeline
    
    # ============================================================
    # Train Random Forest
    # ============================================================
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)
    rf_pipeline = build_rf_pipeline(all_feature_cols)
    
    cv_scores = cross_val_score(
        rf_pipeline, X_train, y_train, cv=skf,
        scoring='f1', n_jobs=-1
    )
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
    
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    
    f1_rf = f1_score(y_test, y_pred_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    
    results['RandomForest'] = {
        'f1': f1_rf, 'accuracy': acc_rf, 'auc': auc_rf,
        'model': rf_pipeline, 'predictions': y_pred_rf, 'probas': y_proba_rf
    }
    print(f"RF Accuracy: {acc_rf:.4f}, F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf))
    
    if f1_rf > best_f1:
        best_f1 = f1_rf
        best_model = rf_pipeline
    
    # ============================================================
    # Train Gradient Boosting
    # ============================================================
    print("=" * 70)
    print("TRAINING GRADIENT BOOSTING")
    print("=" * 70)
    gb_pipeline = build_gb_pipeline(all_feature_cols)
    
    cv_scores = cross_val_score(
        gb_pipeline, X_train, y_train, cv=skf,
        scoring='f1', n_jobs=-1
    )
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
    
    gb_pipeline.fit(X_train, y_train)
    y_pred_gb = gb_pipeline.predict(X_test)
    y_proba_gb = gb_pipeline.predict_proba(X_test)[:, 1]
    
    f1_gb = f1_score(y_test, y_pred_gb)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    auc_gb = roc_auc_score(y_test, y_proba_gb)
    
    results['GradientBoosting'] = {
        'f1': f1_gb, 'accuracy': acc_gb, 'auc': auc_gb,
        'model': gb_pipeline, 'predictions': y_pred_gb, 'probas': y_proba_gb
    }
    print(f"GB Accuracy: {acc_gb:.4f}, F1: {f1_gb:.4f}, AUC: {auc_gb:.4f}")
    print(classification_report(y_test, y_pred_gb))
    
    if f1_gb > best_f1:
        best_f1 = f1_gb
        best_model = gb_pipeline
    
    # ============================================================
    # Select and save best model
    # ============================================================
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    for name, metrics in results.items():
        print(f"{name:20s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    
    best_name = [n for n, m in results.items() if m['f1'] == best_f1][0]
    print(f"\n[OK] Best model selected: {best_name} (F1: {best_f1:.4f})")
    
    # Save the best model
    if best_model is not None:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)
        print(f"[OK] Saved best model to {MODEL_PATH}")
    else:
        raise ValueError("No model was trained successfully")
    
    return best_model


if __name__ == "__main__":
    train_and_evaluate()
