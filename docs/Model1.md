# Model 1: Logistic Regression

## 1. Model objective

The objective is to develop a supervised binary classification model that estimates the probability of customer churn using engineered KKBox customer features.

Logistic Regression serves as the baseline model against which Model 2, XGBoost Classifier, will be compared.

The target is `is_churn`:

- 0: Non-churn
- 1: Churn

The model produces a churn probability and a threshold-based classification.

## 2. Algorithm

Logistic Regression estimates the probability of churn using the logistic function:

$$P(Y=1\mid X)=\frac{1}{1+e^{-(\beta_0+\sum_{j=1}^{p}\beta_jX_j)}}$$

where $Y$ is the binary churn target, $X_j$ is an engineered predictor, and $\beta_j$ is its estimated coefficient.

L2 regularisation is used to penalise large coefficients. The inverse regularisation strength, $C$, controls the penalty: smaller values apply stronger regularisation.

The implementation uses scikit-learn's `LogisticRegression` with the `saga` solver.

## 3. Input features

The model uses the same engineered predictors and customer cohort as Model 2.

The features cover:

1. Listening behaviour for December 2016, January 2017 and February 2017.
2. Transaction activity and payment characteristics.
3. Customer demographic and registration characteristics.

The customer identifier `msno` is excluded from the predictors. The target `is_churn` is used as the classification label.

The feature-engineering window is December 2016 to February 2017. The exact temporal relationship between this period and the competition's churn-label construction has not been independently established. Consequently, the model is not claimed to be conclusively free of temporal leakage.

## 4. Preprocessing

The model reuses the shared preprocessing implementation in `src/preprocessing.py`.

Numerical predictors:
- Missing values are imputed using training-set medians.
- Numerical variables are standardised using `StandardScaler(with_mean=False)`.

Categorical predictors:
- Missing values are represented by `__MISSING__`.
- One-hot encoding is applied, with unknown categories ignored.

All learned transformations are fitted on the training partition only.

Class imbalance is addressed using `class_weight="balanced"`, which calculates class weights from the training labels.

The complete preprocessing and partitioning methodology is documented in [Preprocessing.MD](Preprocessing.MD).

## 5. Experimental setup

### 5.1 Data partitions

The modelling cohort contains 992,931 customers, of whom 63,471 (6.39%) are churners.

| Partition | Customers | Churners | Churn rate |
|---|---:|---:|---:|
| Training | 695,051 | 44,430 | 6.39% |
| Validation | 148,940 | 9,521 | 6.39% |
| Test | 148,940 | 9,520 | 6.39% |

The partitions were created using stratified random sampling with random seed 42.

Both models use the same saved customer-to-partition assignments.

### 5.2 Hyperparameter selection

The inverse regularisation strength $C$ was evaluated using the following predefined candidates:

| Hyperparameter | Value or candidates |
|---|---|
| `C` | 0.1, 1.0, 10.0 |
| `penalty` | L2 |
| `solver` | saga |
| `class_weight` | balanced |
| `max_iter` | 2,000 |
| `tol` | 0.001 |
| `random_state` | 42 |

The candidate with the highest validation average precision was selected.

Average precision summarises the precision-recall curve and is used as the primary selection metric because churn is the minority class.

### 5.3 Classification threshold

The classification threshold was selected by maximising F1 on the validation partition.

Where multiple thresholds achieved the same F1, the threshold with the highest precision was selected.

The resulting threshold was applied unchanged to the test partition. Test labels were not used for hyperparameter or threshold selection.

## 6. Evaluation metrics

The model was evaluated using:

| Metric | Purpose |
|---|---|
| Precision | Proportion of predicted churners who actually churned |
| Recall | Proportion of actual churners detected |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Discrimination across classification thresholds |
| Average precision | Summary of precision-recall performance |
| Balanced accuracy | Average recall across both classes |
| Log loss | Quality of predicted probabilities |
| Confusion matrix | Counts of true negatives, false positives, false negatives and true positives |

Accuracy is not used as the primary selection metric because 93.61% of the cohort belongs to the non-churn class.

## 7. Results

### 7.1 Hyperparameter selection

The following validation results were obtained:

| C | Validation average precision | Validation ROC-AUC |
|---:|---:|---:|
| 0.1 | 0.711381 | 0.967802 |
| 1.0 | 0.713401 | 0.968036 |
| 10.0 | 0.713384 | 0.967724 |

The selected regularisation strength was $C=1.0$, which achieved the highest validation average precision of 0.713401.

### 7.2 Selected threshold

The validation-selected classification threshold was:

**0.892256**

This threshold maximised validation F1 under the implemented selection procedure, with precision used as the tie-breaker.

### 7.3 Validation performance

The selected candidate achieved the following validation metrics:

| Metric | Validation result |
|---|---:|
| Precision | 0.6582 |
| Recall | 0.7281 |
| F1 | 0.6914 |
| Balanced accuracy | 0.8511 |
| ROC-AUC | 0.9680 |
| Average precision | 0.7134 |
| Log loss | 0.2244 |

Validation confusion matrix:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 135,819 | 3,600 |
| Actual churn | 2,589 | 6,932 |

### 7.4 Test performance

The selected candidate and validation-derived threshold were applied to the untouched test partition.

| Metric | Test result |
|---|---:|
| Precision | 0.6455 |
| Recall | 0.7183 |
| F1 | 0.6800 |
| Balanced accuracy | 0.8457 |
| ROC-AUC | 0.9670 |
| Average precision | 0.6993 |
| Log loss | 0.2281 |

Test confusion matrix:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 135,665 | 3,755 |
| Actual churn | 2,682 | 6,838 |

The model correctly identified 6,838 of the 9,520 churners in the test partition, corresponding to a recall of approximately 71.83%.

It also generated 3,755 false-positive churn predictions.

### 7.5 Interpretation

The test ROC-AUC of 0.9670 indicates strong ranking discrimination between churn and non-churn customers on the test partition. The test average precision of 0.6993 provides the corresponding precision-recall summary under the imbalanced target distribution.

At the selected threshold, the model detected approximately 71.83% of actual churners, with precision of approximately 64.55%.

The difference between validation and test average precision is approximately 0.0141. This indicates some reduction in performance on the held-out test partition, although the test results remain close to the validation results.

These findings describe predictive performance on this dataset. They do not establish that acting on the predictions will reduce churn or that the identified relationships are causal.

## 8. Convergence and model validation

Using the `saga` solver with `max_iter=2000` and `tol=0.001` resolved previous convergence warnings, ensuring a stable optimisation solution across all candidate regularisation strengths.

## 9. Implementation and execution

Implementation: `src/model1_logistic_regression.py`

Run from the repository root:

```powershell
python src/model1_logistic_regression.py
```

The script loads the combined feature dataset and shared split assignments, trains candidate pipelines, selects the regularisation strength on validation average precision, selects the classification threshold on validation F1, and evaluates the selected candidate on the test partition.

## 10. Saved outputs

| File | Description |
|---|---|
| `models/logistic_regression.joblib` | Fitted preprocessing and Logistic Regression pipeline |
| `experiments/results/model1_metrics.json` | Hyperparameter results and validation/test metrics |
| `experiments/results/model1_test_predictions.csv` | Test customer IDs, actual labels and predicted probabilities |
| `experiments/results/model1_coefficients.csv` | Fitted coefficients and absolute coefficient magnitudes |
| `experiments/results/model1_configuration.json` | Model configuration and selected parameters |

The coefficients describe the fitted model on the transformed feature scale. Their magnitudes should not automatically be interpreted as causal effects.

## 11. Limitations

1. Logistic Regression models a linear relationship between transformed predictors and the log-odds of churn.
2. One-hot encoding does not automatically capture complex interactions between customer features.
3. Class weighting changes the training objective and can affect probability calibration.
4. The competition's label-generation timing has not been independently verified against the feature cutoff.
5. The convergence warnings require resolution before the model can be treated as fully validated.
6. The practical value of the model depends on the costs of false positives and false negatives, which have not been specified.

The same customer partitions, predictor set and evaluation metrics will be used for XGBoost to support a consistent comparison.