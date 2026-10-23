from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def get_dataset_path() -> Path:
    candidates = [
        ROOT / "tourism.csv",
        ROOT / "data" / "tourism.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Expected one of: tourism.csv or data/tourism.csv"
    )


def main():

    # Read all original customer records
    df = pd.read_csv(get_dataset_path())

    original_count = len(df)

    # Clean text labels without deleting records
    categorical = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical:
        df[col] = df[col].str.strip()

    df["Gender"] = df["Gender"].replace({
        "Fe Male": "Female"
    })

    df["MaritalStatus"] = df["MaritalStatus"].replace({
        "Unmarried": "Single"
    })

    df["Occupation"] = df["Occupation"].replace({
        "Free Lancer": "Freelancer"
    })

    # Preserve CustomerID and all rows
    assert len(df) == original_count

    if df["ProdTaken"].isna().any():
        raise ValueError("Target contains missing values")

    # Separate model inputs and target
    # CustomerID is kept in output datasets
    # but not used as a model feature
    X = df.drop(
        columns=[
            "ProdTaken",
            "CustomerID",
            "Unnamed: 0"
        ],
        errors="ignore"
    )

    y = df["ProdTaken"].astype(int)

    # Split row positions to preserve ID alignment
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create training dataset
    train = X.iloc[train_idx].copy()

    train.insert(
        0,
        "CustomerID",
        df["CustomerID"].iloc[train_idx].to_numpy()
    )

    train["ProdTaken"] = y.iloc[train_idx].to_numpy()

    # Create testing dataset
    test = X.iloc[test_idx].copy()

    test.insert(
        0,
        "CustomerID",
        df["CustomerID"].iloc[test_idx].to_numpy()
    )

    test["ProdTaken"] = y.iloc[test_idx].to_numpy()

    # Save data
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)

    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)

    # Validate record counts
    assert len(train) + len(test) == original_count

    print("Original rows:", original_count)
    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("All rows preserved successfully.")


if __name__ == "__main__":
    main()
