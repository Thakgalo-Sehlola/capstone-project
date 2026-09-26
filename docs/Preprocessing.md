
# Data Preprocessing

## 1. Objective

This stage prepares the KKBox customer churn dataset for two supervised binary classification models:

1. Logistic Regression
2. XGBoost Classifier

The preprocessing workflow creates a reproducible, stratified training, validation and test split shared by both models. It also defines model-specific transformations to ensure that preprocessing parameters are learned from training data only.

## 2. Input dataset

Input file: `data/kkbox/processed/kkbox_model_features.parquet`

The dataset contains 992,931 customer records and 32 columns:

- Customer identifier: `msno`
- Target: `is_churn`
- Predictors: 30 engineered behavioural, transaction and member features

The target is binary:

| Value | Meaning |
|---|---|
| 0 | Non-churn |
| 1 | Churn |

The target distribution is 929,460 non-churn customers (93.61%) and 63,471 churn customers (6.39%).

The customer identifier is retained for partition assignment and prediction traceability but excluded from model predictors.

## 3. Data partitioning

The dataset is divided using stratified random sampling:

| Partition | Proportion | Customers |
|---|---:|---:|
| Training | 70% | 695,051 |
| Validation | 15% | 148,940 |
| Test | 15% | 148,940 |

Random seed: 42.

Stratification uses `is_churn` to preserve the target-class proportions across the partitions.

The customer-to-partition mapping is saved to:

`data/kkbox/processed/split_assignments.parquet`

Both models must use this same mapping to ensure a consistent experimental comparison.

The split summary is saved to:

`data/kkbox/processed/preprocessing_summary.json`

## 4. Predictor classification

The 30 predictors are divided into 26 numerical and four categorical variables.

### Numerical predictors

Behavioural features:
- Monthly active days, total listening seconds, unique-song counts, song-event counts and average daily listening seconds for M1, M2 and M3.

Transaction features:
- Transaction count, total and average amount paid, total plan list price, average payment-plan days, auto-renewal rate, cancellation rate, transaction recency and valid expiry count.

Member features:
- Age and registration year.

### Categorical predictors

| Variable | Treatment |
|---|---|
| `city` | Categorical code |
| `gender` | Categorical value |
| `registered_via` | Categorical code |
| `registration_month` | Categorical month |

Category codes are not treated as continuous numerical measurements. Their meanings are not inferred beyond the values supplied in the dataset.

## 5. Missing-value handling and transformations

### Numerical features

Missing and infinite numerical values are treated as missing.

A median imputer is fitted on the training partition and subsequently applied to validation and test data.

For Logistic Regression, numerical features are also standardised using `StandardScaler(with_mean=False)`. Disabling centring preserves sparse-compatible output.

For XGBoost, numerical features are not scaled.

### Categorical features

Categorical values are converted to strings, with missing values represented by `__MISSING__`.

A constant-value imputer and one-hot encoder are applied. The encoder uses `handle_unknown="ignore"` so categories not observed during training can be processed without failure.

The encoder is fitted on training data only.

## 6. Leakage prevention

The following controls are applied:

1. The customer ID is excluded from the predictors.
2. The target is used only as the classification label and for stratification.
3. Imputation statistics, scaling parameters and category encodings are fitted on training data only.
4. Validation data is used for model selection and threshold selection.
5. Test data is reserved for final model evaluation.

The feature-engineering window is December 2016 to February 2017. The exact temporal relationship between this window and the competition's churn-label construction has not been independently established. Therefore, the workflow does not claim that temporal leakage has been conclusively ruled out.

## 7. Implementation

Implementation: `src/preprocessing.py`

Run from the repository root:

```powershell
python src/preprocessing.py
```

Generated local outputs:

- `data/kkbox/processed/split_assignments.parquet`
- `data/kkbox/processed/preprocessing_summary.json`

The generated data files are excluded from version control. The preprocessing script, documentation and requirements are maintained in the repository.

## 8. Validation results

The preprocessing script completed successfully.

| Partition | Customers | Churners | Churn rate |
|---|---:|---:|---:|
| Training | 695,051 | 44,430 | 6.39% |
| Validation | 148,940 | 9,521 | 6.39% |
| Test | 148,940 | 9,520 | 6.39% |
| Total | 992,931 | 63,471 | 6.39% |

All customers were assigned to one of the three partitions. The partition sizes sum to the full modelling cohort, and the churn proportions are consistent across partitions.