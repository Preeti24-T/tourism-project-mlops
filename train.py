from pathlib import Path
import json
import joblib
import mlflow
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

ROOT = Path(__file__).resolve().parents[1]

def main():

    # Load prepared datasets
    train = pd.read_csv(ROOT / "artifacts/train.csv")
    test = pd.read_csv(ROOT / "artifacts/test.csv")

    # Preserve CustomerIDs for prediction output
    test_ids = test["CustomerID"].copy()

    # CustomerID is not a predictor
    X_train = train.drop(
        columns=["CustomerID", "ProdTaken"]
    )
    y_train = train["ProdTaken"].astype(int)

    X_test = test.drop(
        columns=["CustomerID", "ProdTaken"]
    )
    y_test = test["ProdTaken"].astype(int)

    # Identify feature types
    num_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Create preprocessing
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"),
         num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ]), cat_cols)
    ])

    # Handle class imbalance
    weight = (
        (y_train == 0).sum() /
        (y_train == 1).sum()
    )

    # Create XGBoost model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=weight,
            random_state=42,
            n_jobs=1
        ))
    ])

    # Tune hyperparameters
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0]
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    # Configure MLflow
    mlflow.set_tracking_uri(
        "sqlite:///" + str(
            (ROOT / "mlflow.db").resolve()
        )
    )

    mlflow.set_experiment(
        "Wellness Tourism XGBoost"
    )

    with mlflow.start_run(
        run_name="Automated_XGBoost_Training"
    ):

        # Train model
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Predict
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(
            X_test
        )[:, 1]

        # Evaluate
        metrics = {
            "accuracy": accuracy_score(
                y_test, y_pred
            ),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test, y_prob
            ),
            "best_cv_f1": search.best_score_
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        # Log tuning candidates
        results = pd.DataFrame(search.cv_results_)

        for i, row in results.iterrows():
            with mlflow.start_run(
                run_name=f"Candidate_{i+1}",
                nested=True
            ):
                mlflow.log_params(row["params"])
                mlflow.log_metric(
                    "mean_cv_f1",
                    float(row["mean_test_score"])
                )

        # Save model and metadata
        model_dir = ROOT / "models"
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "best_model.joblib"

        joblib.dump(best_model, model_path)

        schema = {
            "columns": X_train.columns.tolist(),
            "numerical": num_cols,
            "categorical": cat_cols
        }

        (model_dir / "feature_schema.json").write_text(
            json.dumps(schema, indent=2)
        )

        (model_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )

        mlflow.log_artifact(str(model_path))

        # Save predictions with CustomerID
        predictions = pd.DataFrame({
            "CustomerID": test_ids.to_numpy(),
            "Actual_ProdTaken": y_test.to_numpy(),
            "Predicted_ProdTaken": y_pred,
            "Purchase_Probability": y_prob
        })

        predictions.to_csv(
            ROOT / "artifacts/customer_predictions.csv",
            index=False
        )

        print("Best parameters:", search.best_params_)
        print("Metrics:", metrics)
        print(classification_report(
            y_test, y_pred, zero_division=0
        ))
        print("Customer predictions saved.")

if __name__ == "__main__":
    main()
