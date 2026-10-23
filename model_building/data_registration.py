from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "CustomerID", "ProdTaken", "Age",
    "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome"
}


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
    df = pd.read_csv(get_dataset_path())

    missing_columns = EXPECTED - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty")

    if df["CustomerID"].isna().any():
        raise ValueError("Missing CustomerID values")

    if not df["ProdTaken"].isin([0, 1]).all():
        raise ValueError("Invalid target values")

    print("Dataset shape:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate IDs:",
          df["CustomerID"].duplicated().sum())

    print("Target distribution:")
    print(df["ProdTaken"].value_counts())

    print("Data registration successful.")


if __name__ == "__main__":
    main()
