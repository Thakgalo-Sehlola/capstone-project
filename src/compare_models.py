
"""
Part C: Paired comparison of Logistic Regression and XGBoost.

Run from the repository root:
    python src/compare_models.py

If your prediction CSV headers differ from the default column names:
    python src/compare_models.py --help

The script:
1. Loads the two saved test prediction files.
2. Checks that both contain the same test customers and actual labels.
3. Calculates test performance metrics using the saved predictions.
4. Calculates XGBoost-minus-Logistic-Regression metric differences.
5. Uses a paired, stratified bootstrap to estimate 95% confidence
   intervals for differences in AP, ROC-AUC and F1.
6. Saves comparison results under experiments/results/.

It does not retrain either model or select a new threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

LR_PATH = ROOT / "experiments/results/model1_test_predictions.csv"
XGB_PATH = ROOT / "experiments/results/model2_test_predictions.csv"

OUTPUT_JSON = ROOT / "experiments/results/model_comparison.json"
OUTPUT_CSV = ROOT / "experiments/results/model_comparison_metrics.csv"
OUTPUT_BOOTSTRAP = (
    ROOT / "experiments/results/model_comparison_bootstrap.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare saved test predictions from Model 1 and Model 2."
    )

    parser.add_argument(
        "--lr-file",
        type=Path,
        default=LR_PATH,
        help="Path to Model 1 test prediction CSV.",
    )
    parser.add_argument(
        "--xgb-file",
        type=Path,
        default=XGB_PATH,
        help="Path to Model 2 test prediction CSV.",
    )

    parser.add_argument(
        "--id-col",
        default="msno",
        help="Customer identifier column. Use 'none' if unavailable.",
    )
    parser.add_argument(
        "--actual-col",
        default="actual",
        help="Actual target column in both CSV files.",
    )
    parser.add_argument(
        "--prob-col",
        default="predicted_probability",
        help="Predicted churn probability column in both CSV files.",
    )
    parser.add_argument(
        "--pred-col",
        default="predicted_class",
        help="Predicted binary class column in both CSV files.",
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Number of paired bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )

    return parser.parse_args()


def load_predictions(
    path,
    id_col,
    actual_col,
    prob_col,
    pred_col,
    model_name,
):
    if not path.exists():
        raise FileNotFoundError(f"{model_name} prediction file not found: {path}")

    df = pd.read_csv(path)

    required = [actual_col, prob_col, pred_col]
    if id_col.lower() != "none":
        required.append(id_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name}: missing columns {missing} in {path.name}.\n"
            f"Available columns: {list(df.columns)}\n"
            "Rerun with the appropriate --id-col, --actual-col, "
            "--prob-col, and --pred-col arguments."
        )

    keep = [actual_col, prob_col, pred_col]
    rename = {
        actual_col: "actual",
        prob_col: "probability",
        pred_col: "prediction",
    }

    if id_col.lower() != "none":
        keep.append(id_col)
        rename[id_col] = "customer_id"

    df = df[keep].rename(columns=rename)

    if df.isnull().any().any():
        raise ValueError(f"{model_name}: missing values detected in predictions.")

    df["actual"] = pd.to_numeric(df["actual"], errors="raise").astype(int)
    df["probability"] = pd.to_numeric(
        df["probability"], errors="raise"
    ).astype(float)
    df["prediction"] = pd.to_numeric(
        df["prediction"], errors="raise"
    ).astype(int)

    if not set(df["actual"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: actual labels must be 0 or 1.")

    if not set(df["prediction"].unique()).issubset({0, 1}):
        raise ValueError(f"{model_name}: predicted classes must be 0 or 1.")

    if not df["probability"].between(0, 1).all():
        raise ValueError(
            f"{model_name}: predicted probabilities must be between 0 and 1."
        )

    if df.empty:
        raise ValueError(f"{model_name}: prediction file is empty.")

    return df


def align_predictions(lr, xgb, id_col):
    if len(lr) != len(xgb):
        raise ValueError(
            f"Different test-set sizes: LR={len(lr)}, XGBoost={len(xgb)}."
        )

    if id_col.lower() != "none":
        if lr["customer_id"].duplicated().any():
            raise ValueError("Duplicate customer IDs in Model 1 predictions.")
        if xgb["customer_id"].duplicated().any():
            raise ValueError("Duplicate customer IDs in Model 2 predictions.")

        merged = lr.merge(
            xgb,
            on="customer_id",
            how="outer",
            suffixes=("_lr", "_xgb"),
            indicator=True,
            validate="one_to_one",
        )

        if not (merged["_merge"] == "both").all():
            raise ValueError(
                "The two prediction files do not contain identical test customers."
            )

        if not (
            merged["actual_lr"].to_numpy()
            == merged["actual_xgb"].to_numpy()
        ).all():
            raise ValueError(
                "Actual labels differ between models for at least one customer."
            )

        return (
            merged["actual_lr"].to_numpy(dtype=int),
            merged["probability_lr"].to_numpy(dtype=float),
            merged["prediction_lr"].to_numpy(dtype=int),
            merged["probability_xgb"].to_numpy(dtype=float),
            merged["prediction_xgb"].to_numpy(dtype=int),
        )

    # Without customer IDs, row order must be identical.
    if not np.array_equal(lr["actual"].to_numpy(), xgb["actual"].to_numpy()):
        raise ValueError(
            "Actual labels are not identical row-by-row. "
            "Provide a customer ID column to align predictions safely."
        )

    return (
        lr["actual"].to_numpy(dtype=int),
        lr["probability"].to_numpy(dtype=float),
        lr["prediction"].to_numpy(dtype=int),
        xgb["probability"].to_numpy(dtype=float),
        xgb["prediction"].to_numpy(dtype=int),
    )


def calculate_metrics(y, probability, prediction):
    tn, fp, fn, tp = confusion_matrix(
        y, prediction, labels=[0, 1]
    ).ravel()

    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "log_loss": float(
            log_loss(y, probability, labels=[0, 1])
        ),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
    }


def bootstrap_metric_differences(
    y,
    lr_prob,
    lr_pred,
    xgb_prob,
    xgb_pred,
    replicates,
    seed,
):
    """
    Paired stratified bootstrap.

    Each replicate samples churn and non-churn observations separately
    with replacement, preserving the observed class counts.

    The same sampled indices are used for both models.
    """
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")

    rng = np.random.default_rng(seed)

    negative_idx = np.flatnonzero(y == 0)
    positive_idx = np.flatnonzero(y == 1)

    if len(negative_idx) == 0 or len(positive_idx) == 0:
        raise ValueError("Both target classes are required for bootstrap.")

    metric_functions = {
        "average_precision": lambda yy, pp, cc: average_precision_score(
            yy, pp
        ),
        "roc_auc": lambda yy, pp, cc: roc_auc_score(yy, pp),
        "f1": lambda yy, pp, cc: f1_score(
            yy, cc, zero_division=0
        ),
    }

    differences = {name: [] for name in metric_functions}

    for _ in range(replicates):
        sampled_neg = rng.choice(
            negative_idx, size=len(negative_idx), replace=True
        )
        sampled_pos = rng.choice(
            positive_idx, size=len(positive_idx), replace=True
        )

        idx = np.concatenate([sampled_neg, sampled_pos])
        rng.shuffle(idx)

        yy = y[idx]

        for name, metric_fn in metric_functions.items():
            lr_value = metric_fn(yy, lr_prob[idx], lr_pred[idx])
            xgb_value = metric_fn(yy, xgb_prob[idx], xgb_pred[idx])

            differences[name].append(xgb_value - lr_value)

    results = {}

    for name, values in differences.items():
        values = np.asarray(values)

        results[name] = {
            "difference_definition": "XGBoost minus Logistic Regression",
            "observed_difference": None,
            "bootstrap_replicates": int(replicates),
            "confidence_level": 0.95,
            "confidence_interval_method": "Paired stratified percentile bootstrap",
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
            "bootstrap_mean_difference": float(np.mean(values)),
        }

    return results


def main():
    args = parse_args()

    lr = load_predictions(
        args.lr_file,
        args.id_col,
        args.actual_col,
        args.prob_col,
        args.pred_col,
        "Logistic Regression",
    )

    xgb = load_predictions(
        args.xgb_file,
        args.id_col,
        args.actual_col,
        args.prob_col,
        args.pred_col,
        "XGBoost",
    )

    y, lr_prob, lr_pred, xgb_prob, xgb_pred = align_predictions(
        lr, xgb, args.id_col
    )

    lr_metrics = calculate_metrics(y, lr_prob, lr_pred)
    xgb_metrics = calculate_metrics(y, xgb_prob, xgb_pred)

    metric_names = [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "log_loss",
    ]

    metric_differences = {
        name: float(xgb_metrics[name] - lr_metrics[name])
        for name in metric_names
    }

    bootstrap_results = bootstrap_metric_differences(
        y,
        lr_prob,
        lr_pred,
        xgb_prob,
        xgb_pred,
        args.bootstrap_replicates,
        args.seed,
    )

    for name in bootstrap_results:
        bootstrap_results[name]["observed_difference"] = (
            metric_differences[name]
        )

    comparison = {
        "comparison": "XGBoost minus Logistic Regression",
        "test_observations": int(len(y)),
        "test_churn_cases": int(np.sum(y == 1)),
        "test_non_churn_cases": int(np.sum(y == 0)),
        "test_churn_prevalence": float(np.mean(y)),
        "models": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics,
        },
        "metric_differences": metric_differences,
        "paired_bootstrap": bootstrap_results,
        "bootstrap_settings": {
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "sampling": "Stratified by actual target; paired indices across models",
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    rows = []

    for metric in metric_names:
        row = {
            "metric": metric,
            "logistic_regression": lr_metrics[metric],
            "xgboost": xgb_metrics[metric],
            "difference_xgboost_minus_logistic": metric_differences[metric],
        }

        if metric in bootstrap_results:
            row["bootstrap_ci_lower"] = bootstrap_results[metric]["ci_lower"]
            row["bootstrap_ci_upper"] = bootstrap_results[metric]["ci_upper"]

        rows.append(row)

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    bootstrap_rows = []
    for metric, result in bootstrap_results.items():
        bootstrap_rows.append({"metric": metric, **result})

    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT_BOOTSTRAP, index=False
    )

    print("\nTEST SET COMPARISON")
    print(f"Observations: {len(y):,}")
    print(f"Churn cases: {np.sum(y == 1):,}")
    print(f"Churn prevalence: {np.mean(y):.4%}")

    print("\nMETRICS")
    print(
        f"{'Metric':<22}"
        f"{'Logistic Regression':>22}"
        f"{'XGBoost':>14}"
        f"{'XGB - LR':>14}"
    )

    for metric in metric_names:
        print(
            f"{metric:<22}"
            f"{lr_metrics[metric]:>22.6f}"
            f"{xgb_metrics[metric]:>14.6f}"
            f"{metric_differences[metric]:>14.6f}"
        )

    print("\nPAIRED BOOTSTRAP 95% CONFIDENCE INTERVALS")
    for metric, result in bootstrap_results.items():
        print(
            f"{metric}: difference={result['observed_difference']:.6f}, "
            f"95% CI=({result['ci_lower']:.6f}, "
            f"{result['ci_upper']:.6f})"
        )

    print("\nSaved:")
    print(OUTPUT_JSON.relative_to(ROOT))
    print(OUTPUT_CSV.relative_to(ROOT))
    print(OUTPUT_BOOTSTRAP.relative_to(ROOT))


if __name__ == "__main__":
    main()