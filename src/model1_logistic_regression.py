from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline

from preprocessing import (
    DATA_PATH,
    SPLIT_PATH,
    OUTPUT_DIR,
    ID_COL,
    TARGET_COL,
    load_model_dataset,
    get_feature_columns,
    prepare_predictors,
    build_preprocessor,
)


# Configuration

RANDOM_STATE = 42

# Candidate inverse regularisation strengths
# Larger C means weaker regularisation
C_VALUES = [0.1, 1.0, 10.0]

MAX_ITER = 2000
TOL = 1e-3

RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "results"
)

MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
)

METRICS_PATH = RESULTS_DIR / "model1_metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "model1_test_predictions.csv"
COEFFICIENTS_PATH = RESULTS_DIR / "model1_coefficients.csv"
CONFIG_PATH = RESULTS_DIR / "model1_configuration.json"


# Load the shared partitions
def load_split_data():
    df = load_model_dataset()

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Shared split assignments not found. "
            "Run src/preprocessing.py first."
        )

    assignments = pd.read_parquet(SPLIT_PATH)

    if assignments[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs in split assignments."
        )

    if set(assignments[ID_COL]) != set(df[ID_COL]):
        raise ValueError(
            "Split assignments do not match the modelling cohort."
        )

    df = df.merge(
        assignments,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        raise ValueError(
            "Some customers have no partition assignment."
        )

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One or more partitions are empty.")

    return train, validation, test


# Evaluation utilities
def evaluate_probabilities(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

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


def select_f1_threshold(y_true, probabilities):
    """
    Select the threshold with maximum F1 on validation data.

    If multiple thresholds have the same F1, choose the
    one with the highest precision among those candidates.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to calculate validation thresholds."
        )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    maximum_f1 = np.max(f1_values)

    candidates = np.flatnonzero(
        np.isclose(f1_values, maximum_f1)
    )

    best_index = candidates[
        np.argmax(precision[candidates])
    ]

    return float(thresholds[best_index])


# Main model workflow
def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading shared data partitions...")
    train, validation, test = load_split_data()

    print(
        f"Training customers: {len(train):,}"
    )
    print(
        f"Validation customers: {len(validation):,}"
    )
    print(
        f"Test customers: {len(test):,}"
    )

    numeric_cols, categorical_cols = (
        get_feature_columns(train)
    )

    feature_cols = (
        numeric_cols + categorical_cols
    )

    # Separate predictors and targets.
    X_train = prepare_predictors(
        train[feature_cols]
    )
    X_validation = prepare_predictors(
        validation[feature_cols]
    )
    X_test = prepare_predictors(
        test[feature_cols]
    )

    y_train = train[TARGET_COL].to_numpy()
    y_validation = validation[TARGET_COL].to_numpy()
    y_test = test[TARGET_COL].to_numpy()

    # Hyperparameter selection

    print("\nSelecting regularisation strength...")
    print("Selection metric: validation average precision")

    candidate_results = []

    best_model = None
    best_C: float | None = None
    best_validation_ap = -np.inf

    for C in C_VALUES:
        print(f"\nTraining Logistic Regression: C={C}")

        preprocessor = build_preprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            scale_numeric=True,
        )

        classifier = LogisticRegression(
            C=C,
            penalty="l2",
            solver="saga",
            class_weight="balanced",
            max_iter=MAX_ITER,
            tol=TOL,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            pipeline.fit(
                X_train,
                y_train,
            )

        convergence_warnings = [
            str(w.message)
            for w in caught
            if issubclass(
                w.category,
                ConvergenceWarning,
            )
        ]

        if convergence_warnings:
            print("CONVERGENCE WARNINGS:")
            for warning_message in convergence_warnings:
                print(warning_message)

        validation_probabilities = (
            pipeline.predict_proba(
                X_validation
            )[:, 1]
        )

        validation_ap = (
            average_precision_score(
                y_validation,
                validation_probabilities,
            )
        )

        validation_roc_auc = (
            roc_auc_score(
                y_validation,
                validation_probabilities,
            )
        )

        result = {
            "C": float(C),
            "validation_average_precision": float(
                validation_ap
            ),
            "validation_roc_auc": float(
                validation_roc_auc
            ),
            "convergence_warnings": (
                convergence_warnings
            ),
        }

        candidate_results.append(result)

        print(
            f"Validation average precision: "
            f"{validation_ap:.6f}"
        )
        print(
            f"Validation ROC-AUC: "
            f"{validation_roc_auc:.6f}"
        )

        if validation_ap > best_validation_ap:
            best_validation_ap = validation_ap
            best_C = C
            best_model = pipeline

    if best_model is None:
        raise RuntimeError(
            "No Logistic Regression model was selected."
        )

    if best_C is None:
        raise RuntimeError(
            "No regularisation strength was selected."
        )

    print("\nSelected regularisation strength:")
    print(f"C = {best_C}")
    print(
        f"Validation average precision = "
        f"{best_validation_ap:.6f}"
    )

    # Select classification threshold on validation
    validation_probabilities = (
        best_model.predict_proba(
            X_validation
        )[:, 1]
    )

    selected_threshold = select_f1_threshold(
        y_validation,
        validation_probabilities,
    )

    validation_metrics = evaluate_probabilities(
        y_validation,
        validation_probabilities,
        selected_threshold,
    )

    print("\nValidation metrics at selected threshold")
    print(
        json.dumps(
            validation_metrics,
            indent=4,
        )
    )

    # Final evaluation on untouched test partition
    print("\nEvaluating selected model on test data...")

    test_probabilities = (
        best_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = evaluate_probabilities(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    print("\nTest metrics")
    print(
        json.dumps(
            test_metrics,
            indent=4,
        )
    )

    # Save test predictions
    test_predictions = pd.DataFrame({
        ID_COL: test[ID_COL].to_numpy(),
        "actual_is_churn": y_test,
        "predicted_probability": test_probabilities,
        "predicted_is_churn": (
            test_probabilities >= selected_threshold
        ).astype(int),
    })

    test_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Save model coefficients
    fitted_preprocessor = (
        best_model.named_steps["preprocessor"]
    )

    fitted_classifier = (
        best_model.named_steps["classifier"]
    )

    feature_names = (
        fitted_preprocessor.get_feature_names_out()
    )

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": fitted_classifier.coef_[0],
    })

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    coefficients.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    # Save metrics and configuration
    metrics = {
        "model": "Logistic Regression",
        "target": TARGET_COL,
        "selected_C": float(best_C),
        "selected_threshold": float(
            selected_threshold
        ),
        "selection_metric": (
            "Validation average precision"
        ),
        "threshold_selection_metric": (
            "Validation F1"
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "hyperparameter_results": candidate_results,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    configuration = {
        "model": "Logistic Regression",
        "solver": "saga",
        "penalty": "l2",
        "class_weight": "balanced",
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": RANDOM_STATE,
        "candidate_C_values": C_VALUES,
        "selected_C": float(best_C),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "threshold": float(selected_threshold),
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

    # Save the fitted pipeline for reproducibility.
    import joblib

    model_path = MODEL_DIR / "logistic_regression.joblib"

    joblib.dump(
        best_model,
        model_path,
    )

    print("\nSaved outputs:")
    print(METRICS_PATH)
    print(PREDICTIONS_PATH)
    print(COEFFICIENTS_PATH)
    print(CONFIG_PATH)
    print(model_path)

    print("\nModel 1 completed.")


if __name__ == "__main__":
    main()