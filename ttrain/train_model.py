from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "training" / "heart.csv"
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


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data at {path}")
    return pd.read_csv(path)


def build_pipeline(feature_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)], remainder="drop"
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=4,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train() -> None:
    df = load_data(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(FEATURE_COLS)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    print("Classification report:\n", report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
