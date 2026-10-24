
# ============================================================
# XGBOOST MODEL TRAINING WITH MLFLOW
# ============================================================

from pathlib import Path
import json
import joblib
import mlflow
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Project root
ROOT = Path(__file__).resolve().parents[1]


def main():

    # --------------------------------------------------------
    # 1. Load prepared datasets
    # --------------------------------------------------------
    train = pd.read_csv(ROOT / "artifacts/train.csv")
    test = pd.read_csv(ROOT / "artifacts/test.csv")

    # Preserve CustomerIDs for prediction reporting.
    test_ids = test["CustomerID"].copy()

    # --------------------------------------------------------
    # 2. Separate features and target
    # --------------------------------------------------------
    X_train = train.drop(
        columns=["CustomerID", "ProdTaken"]
    )

    y_train = train["ProdTaken"].astype(int)

    X_test = test.drop(
        columns=["CustomerID", "ProdTaken"]
    )

    y_test = test["ProdTaken"].astype(int)

    # --------------------------------------------------------
    # 3. Validate pre-contact feature selection
    # --------------------------------------------------------
    post_contact_features = [
        "TypeofContact",
        "DurationOfPitch",
        "NumberOfFollowups",
        "ProductPitched",
        "PitchSatisfactionScore"
    ]

    # Ensure excluded columns are not present.
    assert not set(post_contact_features).intersection(
        X_train.columns
    )

    assert not set(post_contact_features).intersection(
        X_test.columns
    )

    # Ensure the model has exactly 13 predictors.
    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13

    # Ensure training and testing columns match.
    assert X_train.columns.tolist() == X_test.columns.tolist()

    print("Feature validation passed.")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # --------------------------------------------------------
    # 4. Identify numerical and categorical variables
    # --------------------------------------------------------
    num_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    assert len(num_cols) == 9
    assert len(cat_cols) == 4

    # --------------------------------------------------------
    # 5. Create preprocessing pipeline
    # --------------------------------------------------------
    # Replace missing numerical values with median.
    numerical_transformer = SimpleImputer(
        strategy="median"
    )

    # Replace missing categorical values with most
    # frequent value and apply one-hot encoding.
    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # --------------------------------------------------------
    # 6. Handle target class imbalance
    # --------------------------------------------------------
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError("No positive class found.")

    class_weight = negative_count / positive_count

    print("Scale positive weight:", class_weight)

    # --------------------------------------------------------
    # 7. Create XGBoost pipeline
    # --------------------------------------------------------
    model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=class_weight,
                random_state=42,
                n_jobs=1
            )
        )
    ])

    # --------------------------------------------------------
    # 8. Hyperparameter tuning
    # --------------------------------------------------------
    # Same search space as the original notebook.
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
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    # --------------------------------------------------------
    # 9. Configure MLflow tracking
    # --------------------------------------------------------
    # Use SQLite instead of the old filesystem tracking URI.
    # Configure MLflow to use SQLite tracking.
# The database is created within the project directory.
mlflow.set_tracking_uri(
    "sqlite:///" + str((ROOT / "mlflow.db").resolve())
)

mlflow.set_experiment("Wellness Tourism XGBoost")

    # --------------------------------------------------------
    # 10. Train, evaluate and track the model
    # --------------------------------------------------------
    with mlflow.start_run(
        run_name="Pre_Contact_XGBoost_Training"
    ):

        # Train model with hyperparameter search.
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Generate predictions.
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics.
        metrics = {
            "accuracy": float(
                accuracy_score(y_test, y_pred)
            ),
            "precision": float(
                precision_score(
                    y_test, y_pred, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, y_pred, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_test, y_pred, zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, y_prob)
            ),
            "best_cv_f1": float(search.best_score_)
        }

        # Log best hyperparameters and evaluation metrics.
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        # Log individual GridSearchCV candidates.
        cv_results = pd.DataFrame(search.cv_results_)

        for i, row in cv_results.iterrows():
            with mlflow.start_run(
                run_name=f"Candidate_{i + 1}",
                nested=True
            ):
                mlflow.log_params(row["params"])
                mlflow.log_metric(
                    "mean_cv_f1",
                    float(row["mean_test_score"])
                )

        # ----------------------------------------------------
        # 11. Save trained model and metadata
        # ----------------------------------------------------
        model_dir = ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "best_model.joblib"

        joblib.dump(best_model, model_path)

        feature_schema = {
            "columns": X_train.columns.tolist(),
            "numerical": num_cols,
            "categorical": cat_cols
        }

        (model_dir / "feature_schema.json").write_text(
            json.dumps(feature_schema, indent=2)
        )

        (model_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )

        # Log saved model artifact in MLflow.
        mlflow.log_artifact(str(model_path))

        # ----------------------------------------------------
        # 12. Save customer-level predictions
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 13. Display final model results
        # ----------------------------------------------------
        print("\nBest hyperparameters:")
        print(search.best_params_)

        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2))

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        print("\nModel saved:", model_path)
        print("Customer predictions saved successfully.")


if __name__ == "__main__":
    main()
