"""Train the pre-contact Wellness Tourism XGBoost model.

Run from repository root: python -u model_building/train.py
Requires artifacts/train.csv and artifacts/test.csv from prep.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
TARGET = "ProdTaken"
ID_COL = "CustomerID"

NUMERIC_COLUMNS = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_COLUMNS = ["Occupation", "Gender", "MaritalStatus", "Designation"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
EXCLUDED_CONTACT_COLUMNS = {
    "TypeofContact", "DurationOfPitch", "NumberOfFollowups",
    "ProductPitched", "PitchSatisfactionScore",
}


def normalize_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the spelling conventions expected by the deployed form."""
    frame = frame.copy()
    corrections = {
        "Gender": {"Fe Male": "Female"},
        "MaritalStatus": {"Unmarried": "Single"},
        "Occupation": {"Free Lancer": "Freelancer"},
    }
    for column, mapping in corrections.items():
        frame[column] = frame[column].replace(mapping)
    return frame


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_path = ARTIFACTS / "train.csv"
    test_path = ARTIFACTS / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing prepared CSV files. Run model_building/prep.py first.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    expected = set(FEATURE_COLUMNS + [ID_COL, TARGET])
    for name, frame in (("train", train), ("test", test)):
        missing = expected.difference(frame.columns)
        unexpected = EXCLUDED_CONTACT_COLUMNS.intersection(frame.columns)
        if missing or unexpected:
            raise ValueError(f"{name}: missing={sorted(missing)}, contact leakage={sorted(unexpected)}")
        if len(frame) + 0 == 0:
            raise ValueError(f"{name} dataset is empty")

    train = normalize_categories(train)
    test = normalize_categories(test)
    X_train = train[FEATURE_COLUMNS].copy()
    X_test = test[FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce")
    y_train = pd.to_numeric(train[TARGET], errors="raise").astype(int)
    y_test = pd.to_numeric(test[TARGET], errors="raise").astype(int)
    if set(y_train.unique()) != {0, 1} or not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must be a binary target with 0 and 1 in training data")

    print(f"Training rows: {len(train)}, test rows: {len(test)}", flush=True)
    print(f"Predictors ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}", flush=True)
    print(f"scikit-learn={sklearn.__version__}; xgboost={xgboost.__version__}; joblib={joblib.__version__}", flush=True)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    if positive == 0:
        raise ValueError("No positive training examples found")
    scale_pos_weight = negative / positive

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
        )),
    ])
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )

    # IMPORTANT: These lines are indented inside main(), just like the training block.
    mlflow.set_tracking_uri("sqlite:///" + str((ROOT / "mlflow.db").resolve()))
    mlflow.set_experiment("Wellness Tourism XGBoost")

    with mlflow.start_run(run_name="Pre_Contact_XGBoost_Training"):
        print("Starting GridSearchCV training...", flush=True)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_test.nunique() == 2 else None,
            "cv_best_f1": float(search.best_score_),
        }
        mlflow.log_params(search.best_params_)
        mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

        joblib.dump(best_model, MODELS / "best_model.joblib")
        (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        # Include both common schema key spellings to ease integration with the app.
        categories = {
            column: [str(x) for x in best_model.named_steps["preprocessor"]
                     .named_transformers_["cat"].named_steps["onehot"].categories_[i]]
            for i, column in enumerate(CATEGORICAL_COLUMNS)
        }
        schema = {
            "columns": FEATURE_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_features": NUMERIC_COLUMNS,
            "categorical_features": CATEGORICAL_COLUMNS,
            "categories": categories,
            "target": TARGET,
            "id_column": ID_COL,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
        }
        (MODELS / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

        predictions = pd.DataFrame({
            ID_COL: test[ID_COL].values,
            "Actual": y_test.values,
            "Predicted": y_pred,
            "PurchaseProbability": y_prob,
        })
        predictions.to_csv(ARTIFACTS / "customer_predictions.csv", index=False)

        mlflow.log_artifact(str(MODELS / "metrics.json"))
        mlflow.log_artifact(str(MODELS / "feature_schema.json"))
        print("Best hyperparameters:", search.best_params_, flush=True)
        print("Test metrics:", json.dumps(metrics, indent=2), flush=True)
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred), flush=True)
        print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0), flush=True)
        print("Saved model, metrics, schema, and customer predictions.", flush=True)


if __name__ == "__main__":
    main()
