from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths and configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "kkbox"
    / "processed"
    / "kkbox_model_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "kkbox"
    / "processed"
)

SPLIT_PATH = OUTPUT_DIR / "split_assignments.parquet"
SUMMARY_PATH = OUTPUT_DIR / "preprocessing_summary.json"

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

ID_COL = "msno"
TARGET_COL = "is_churn"

# Treat these codes as categorical
CATEGORICAL_COLS = [
    "city",
    "gender",
    "registered_via",
    "registration_month",
]

# The remaining predictors below are
# identified dynamically after excluding ID,
# target, and categorical columns


# Load and validate modelling dataset

def load_model_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modelling dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    required_cols = {
        ID_COL,
        TARGET_COL,
        *CATEGORICAL_COLS,
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}"
        )

    if df[ID_COL].isna().any():
        raise ValueError("Customer IDs contain null values.")

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "Modelling dataset contains duplicate customer IDs."
        )

    if df[TARGET_COL].isna().any():
        raise ValueError("Target contains null values.")

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary values 0 and 1."
        )

    if not df[TARGET_COL].value_counts().index.isin(
        [0, 1]
    ).all():
        raise ValueError("Unexpected target values.")

    return df


# Define predictor columns

def get_feature_columns(df):
    excluded_cols = {
        ID_COL,
        TARGET_COL,
    }

    numeric_cols = [
        col
        for col in df.columns
        if col not in excluded_cols
        and col not in CATEGORICAL_COLS
    ]

    # Confirm the expected predictor types.
    non_numeric = [
        col
        for col in numeric_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if non_numeric:
        raise TypeError(
            "Expected numeric predictors, but found "
            f"non-numeric columns: {non_numeric}"
        )

    return numeric_cols, CATEGORICAL_COLS.copy()


# Prepare predictor data
def prepare_predictors(X):
    """
    Apply deterministic type handling only.

    No statistical values are learned here.
    Imputation, encoding and scaling are fitted later
    using training data only.
    """

    X = X.copy()

    categorical_cols = [
        col
        for col in CATEGORICAL_COLS
        if col in X.columns
    ]

    numeric_cols = [
        col
        for col in X.columns
        if col not in categorical_cols
    ]

    # Convert category codes to strings so that
    # OneHotEncoder treats them as categories
    for col in categorical_cols:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    # Convert infinite numerical values to missing.
    for col in numeric_cols:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

        X[col] = X[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return X


# Build model-specific preprocessing pipeline
def build_preprocessor(
    numeric_cols,
    categorical_cols,
    scale_numeric=True,
):
    """
    Create a preprocessing transformer.

    scale_numeric=True:
        Median imputation + sparse-compatible scaling.
        Use for Logistic Regression.

    scale_numeric=False:
        Median imputation without scaling.
        Use for XGBoost.
    """

    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(
                        with_mean=False,
                    ),
                ),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_cols,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    return preprocessor


# Create shared stratified data partitions
def create_splits(df):
    """
    Generate reproducible, stratified 70/15/15
    customer partitions.

    Both models will use these same assignments.
    """

    customer_ids = df[ID_COL]
    y = df[TARGET_COL]

    # First split: 70% train, 30% remainder.
    train_ids, remainder_ids, y_train, y_remainder = (
        train_test_split(
            customer_ids,
            y,
            test_size=1 - TRAIN_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Split remainder equally into validation and test.
    validation_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_remainder,
    )

    assignments = pd.concat(
        [
            pd.DataFrame({
                ID_COL: train_ids,
                "split": "train",
            }),
            pd.DataFrame({
                ID_COL: validation_ids,
                "split": "validation",
            }),
            pd.DataFrame({
                ID_COL: test_ids,
                "split": "test",
            }),
        ],
        ignore_index=True,
    )

    # Validate that every customer is assigned once
    if len(assignments) != len(df):
        raise ValueError(
            "Split assignment count does not match dataset."
        )

    if assignments[ID_COL].duplicated().any():
        raise ValueError(
            "A customer appears in more than one split."
        )

    if set(assignments[ID_COL]) != set(customer_ids):
        raise ValueError(
            "Split assignments do not cover the full cohort."
        )

    return assignments


# Summarise and save split assignments
def build_summary(df, assignments):
    validation = df[
        [ID_COL, TARGET_COL]
    ].merge(
        assignments,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if validation["split"].isna().any():
        raise ValueError(
            "Some customers have no split assignment."
        )

    summary = {
        "random_state": RANDOM_STATE,
        "split_proportions": {
            "train": TRAIN_SIZE,
            "validation": VALIDATION_SIZE,
            "test": TEST_SIZE,
        },
        "total_customers": int(len(df)),
        "target_distribution": {},
        "splits": {},
    }

    for target_value, count in (
        df[TARGET_COL].value_counts().sort_index().items()
    ):
        summary["target_distribution"][str(target_value)] = {
            "count": int(count),
            "proportion": round(
                float(count / len(df)),
                6,
            ),
        }

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        part = validation[
            validation["split"] == split_name
        ]

        churn_count = int(
            (part[TARGET_COL] == 1).sum()
        )

        summary["splits"][split_name] = {
            "customers": int(len(part)),
            "churners": churn_count,
            "non_churners": int(
                (part[TARGET_COL] == 0).sum()
            ),
            "churn_rate": round(
                float(part[TARGET_COL].mean()),
                6,
            ),
        }

    return summary


# Main execution
def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading modelling dataset...")
    df = load_model_dataset()

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    numeric_cols, categorical_cols = (
        get_feature_columns(df)
    )

    print("\nPredictor configuration")
    print(f"Numeric predictors: {len(numeric_cols)}")
    print(f"Categorical predictors: {len(categorical_cols)}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    assignments = create_splits(df)

    summary = build_summary(
        df,
        assignments,
    )

    assignments.to_parquet(
        SPLIT_PATH,
        index=False,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nSplit validation")
    for split_name, values in summary["splits"].items():
        print(
            f"{split_name}: "
            f"{values['customers']:,} customers | "
            f"{values['churners']:,} churners | "
            f"churn rate: {values['churn_rate']:.2%}"
        )

    print("\nSaved split assignments:")
    print(SPLIT_PATH)

    print("\nSaved preprocessing summary:")
    print(SUMMARY_PATH)

    print("\nPreprocessing setup completed.")


if __name__ == "__main__":
    main()