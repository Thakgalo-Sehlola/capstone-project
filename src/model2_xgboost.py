
from pathlib import Path
import json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing import (
    DATA_PATH,
    SPLIT_PATH,
    ID_COL,
    TARGET_COL,
    CATEGORICAL_COLS,
    load_model_dataset,
    get_feature_columns,
    prepare_predictors,
    build_preprocessor,
)


# Paths and configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

MODEL_PATH = MODEL_DIR / "xgboost_pipeline.joblib"
METRICS_PATH = RESULTS_DIR / "model2_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "model2_test_predictions.csv"
IMPORTANCE_PATH = RESULTS_DIR / "model2_feature_importance.csv"
CONFIG_PATH = RESULTS_DIR / "model2_configuration.json"

RANDOM_STATE = 42

# Initial, deliberately small hyperparameter search.
PARAMETER_GRID = [
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.10},
]

N_ESTIMATORS = 400
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8


# Load saved partitions
def load_partitions():

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Saved split assignments not found: {SPLIT_PATH}"
        )

    df = load_model_dataset()
    assignments = pd.read_parquet(SPLIT_PATH)

    required_assignment_cols = {ID_COL, "split"}

    if not required_assignment_cols.issubset(assignments.columns):
        raise ValueError(
            "Split file must contain msno and split columns."
        )

    if assignments[ID_COL].isna().any():
        raise ValueError("Split assignments contain null IDs.")

    if assignments[ID_COL].duplicated().any():
        raise ValueError(
            "A customer appears more than once in split assignments."
        )

    if set(assignments[ID_COL]) != set(df[ID_COL]):
        raise ValueError(
            "Saved split assignments do not match the modelling dataset."
        )

    valid_splits = {"train", "validation", "test"}

    if not set(assignments["split"].unique()).issubset(valid_splits):
        raise ValueError("Unexpected split labels.")

    df = df.merge(
        assignments,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        raise ValueError("Some customers have no split assignment.")

    train_df = df[df["split"] == "train"].copy()
    validation_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    for name, part in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        if part.empty:
            raise ValueError(f"{name} partition is empty.")

    print(f"Training customers: {len(train_df):,}")
    print(f"Validation customers: {len(validation_df):,}")
    print(f"Test customers: {len(test_df):,}")

    return train_df, validation_df, test_df


# Metrics and threshold selection
def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc_average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def select_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    # The final precision/recall point has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    max_f1 = np.max(f1_values)

    # F1 tie-break: select the tied threshold with highest precision.
    tied_indices = np.flatnonzero(
        np.isclose(f1_values, max_f1)
    )

    best_index = tied_indices[
        np.argmax(precision[tied_indices])
    ]

    return float(thresholds[best_index])


# Main modelling workflow
def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved data partitions...")

    train_df, validation_df, test_df = load_partitions()

    numeric_cols, categorical_cols = get_feature_columns(
        train_df
    )

    feature_cols = numeric_cols + categorical_cols

    X_train = prepare_predictors(
        train_df[feature_cols]
    )
    X_validation = prepare_predictors(
        validation_df[feature_cols]
    )
    X_test = prepare_predictors(
        test_df[feature_cols]
    )

    y_train = train_df[TARGET_COL].astype(int)
    y_validation = validation_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    # Calculate class imbalance from training data only
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = negative_count / positive_count

    print(f"\nTraining negative cases: {negative_count:,}")
    print(f"Training positive cases: {positive_count:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")

    # Fit imputation and category encoding on training data only
    print("\nFitting shared XGBoost preprocessing...")

    preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
        scale_numeric=False,
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    print(
        "Transformed training matrix:",
        X_train_transformed.shape,
    )

    print(
        "Transformed validation matrix:",
        X_validation_transformed.shape,
    )


    # Hyperparameter selection
    print("\nSelecting XGBoost configuration...")
    print("Selection metric: validation average precision")

    candidate_results = []
    candidate_models = []

    for candidate_number, parameters in enumerate(
        PARAMETER_GRID,
        start=1,
    ):

        print(
            f"\nTraining candidate {candidate_number}: "
            f"{parameters}"
        )

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=N_ESTIMATORS,
            max_depth=parameters["max_depth"],
            learning_rate=parameters["learning_rate"],
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=1,
        )

        model.fit(
            X_train_transformed,
            y_train,
            verbose=False,
        )

        validation_probabilities = model.predict_proba(
            X_validation_transformed
        )[:, 1]

        validation_ap = average_precision_score(
            y_validation,
            validation_probabilities,
        )

        validation_roc_auc = roc_auc_score(
            y_validation,
            validation_probabilities,
        )

        print(
            f"Validation average precision: {validation_ap:.6f}"
        )
        print(
            f"Validation ROC-AUC: {validation_roc_auc:.6f}"
        )

        candidate_results.append({
            "candidate": candidate_number,
            "parameters": parameters,
            "validation_average_precision": float(validation_ap),
            "validation_roc_auc": float(validation_roc_auc),
        })

        candidate_models.append(model)

    # Select the candidate with the highest validation AP
    best_index = int(
        np.argmax([
            result["validation_average_precision"]
            for result in candidate_results
        ])
    )

    best_model = candidate_models[best_index]
    best_parameters = PARAMETER_GRID[best_index]

    print("\nSelected XGBoost configuration:")
    print(best_parameters)

    print(
        "Validation average precision:",
        f"{candidate_results[best_index]['validation_average_precision']:.6f}",
    )

    # Select threshold on validation only
    validation_probabilities = best_model.predict_proba(
        X_validation_transformed
    )[:, 1]

    threshold = select_threshold(
        y_validation,
        validation_probabilities,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold,
    )

    print("\nValidation metrics at selected threshold")
    print(json.dumps(validation_metrics, indent=4))

    # Final test evaluation
    print("\nTransforming test data...")

    X_test_transformed = preprocessor.transform(X_test)

    print("Evaluating selected model on test data...")

    test_probabilities = best_model.predict_proba(
        X_test_transformed
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    print("\nTest metrics")
    print(json.dumps(test_metrics, indent=4))


    # Save model and outputs

    # Save the fitted preprocessing transformer together
    # with the classifier so inference uses identical encoding.
    import joblib

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": best_model,
            "feature_columns": feature_cols,
            "threshold": threshold,
        },
        MODEL_PATH,
    )

    test_predictions = pd.DataFrame({
        ID_COL: test_df[ID_COL].values,
        "actual_is_churn": y_test.values,
        "predicted_probability": test_probabilities,
        "predicted_is_churn": (
            test_probabilities >= threshold
        ).astype(int),
    })

    test_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    feature_names = preprocessor.get_feature_names_out()

    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": best_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    configuration = {
        "model": "XGBoost binary classifier",
        "selection_metric": "validation average precision",
        "selected_parameters": best_parameters,
        "n_estimators": N_ESTIMATORS,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "threshold_selection": (
            "Maximum validation F1; precision used as tie-breaker"
        ),
        "selected_threshold": threshold,
        "candidate_results": candidate_results,
        "training_customers": len(train_df),
        "validation_customers": len(validation_df),
        "test_customers": len(test_df),
    }

    metrics_output = {
        "model": "XGBoost",
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
        )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_output,
            file,
            indent=4,
        )

    print("\nSaved outputs:")
    print(MODEL_PATH)
    print(METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(IMPORTANCE_PATH)
    print(CONFIG_PATH)

    print("\nModel 2 completed.")


if __name__ == "__main__":
    main()