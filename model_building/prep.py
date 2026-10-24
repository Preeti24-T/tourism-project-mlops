
# ============================================================
# DATA PREPARATION SCRIPT
# Wellness Tourism Purchase Prediction
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root directory
ROOT = Path(__file__).resolve().parents[1]


def main():

    # --------------------------------------------------------
    # 1. Load original customer dataset
    # --------------------------------------------------------
    df = pd.read_csv(ROOT / "data/tourism.csv")

    original_count = len(df)

    print("Original customer records:", original_count)

    # --------------------------------------------------------
    # 2. Clean categorical values
    # --------------------------------------------------------
    # Remove leading/trailing spaces from text columns.
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_cols:
        df[col] = df[col].str.strip()

    # Standardize inconsistent categories.
    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female"
    })

    df["MaritalStatus"] = df["MaritalStatus"].replace({
        "Unmarried": "Single"
    })

    df["Occupation"] = df["Occupation"].replace({
        "Free Lancer": "Freelancer"
    })

    # Standardize contact labels in the original data.
    # TypeofContact will not be used for model training.
    df["TypeofContact"] = df["TypeofContact"].replace({
        "Self Inquiry": "Self Enquiry"
    })

    # --------------------------------------------------------
    # 3. Validate customer records and target
    # --------------------------------------------------------
    assert len(df) == original_count

    if df["ProdTaken"].isna().any():
        raise ValueError("Target contains missing values.")

    if df["CustomerID"].isna().any():
        raise ValueError("CustomerID contains missing values.")

    # --------------------------------------------------------
    # 4. Exclude post-contact features from model inputs
    # --------------------------------------------------------
    excluded_columns = [
        "ProdTaken",
        "CustomerID",
        "Unnamed: 0",
        "TypeofContact",
        "DurationOfPitch",
        "NumberOfFollowups",
        "ProductPitched",
        "PitchSatisfactionScore"
    ]

    X = df.drop(
        columns=excluded_columns,
        errors="ignore"
    ).copy()

    y = df["ProdTaken"].astype(int)

    # Check expected model input structure.
    assert X.shape[1] == 13

    print("Selected model features:", X.columns.tolist())

    # --------------------------------------------------------
    # 5. Stratified train-test split
    # --------------------------------------------------------
    # Split row indices to preserve CustomerID alignment.
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # 6. Prepare training dataset
    # --------------------------------------------------------
    train = X.iloc[train_idx].copy()

    # Add CustomerID for identification, not training.
    train.insert(
        0,
        "CustomerID",
        df["CustomerID"].iloc[train_idx].to_numpy()
    )

    # Add target column.
    train["ProdTaken"] = y.iloc[train_idx].to_numpy()

    # --------------------------------------------------------
    # 7. Prepare testing dataset
    # --------------------------------------------------------
    test = X.iloc[test_idx].copy()

    test.insert(
        0,
        "CustomerID",
        df["CustomerID"].iloc[test_idx].to_numpy()
    )

    test["ProdTaken"] = y.iloc[test_idx].to_numpy()

    # --------------------------------------------------------
    # 8. Save prepared datasets
    # --------------------------------------------------------
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        output_dir / "train.csv",
        index=False
    )

    test.to_csv(
        output_dir / "test.csv",
        index=False
    )

    # --------------------------------------------------------
    # 9. Final validation
    # --------------------------------------------------------
    assert len(train) + len(test) == original_count
    assert len(train) == 3302
    assert len(test) == 826

    # Each saved dataset has 13 predictors,
    # CustomerID and ProdTaken.
    assert train.shape[1] == 15
    assert test.shape[1] == 15

    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("All customer records preserved.")
    print("Pre-contact feature validation passed.")


if __name__ == "__main__":
    main()

