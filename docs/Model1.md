
# Model 1: Logistic Regression

## 1. Model objective

The objective is to develop a supervised binary classification model that estimates the probability of customer churn using engineered KKBox customer features.

Logistic Regression serves as the baseline model against which Model 2, the XGBoost Classifier, is compared.

The target variable is `is_churn`:

- 0: Non-churn
- 1: Churn

The model produces a churn probability and a threshold-based binary classification.

## 2. Algorithm

Logistic Regression estimates the probability of churn using the logistic function:

\[
$$
P(Y=1\mid X)=\frac{1}{1+e^{-(\beta_0+\sum_{j=1}^{p}\beta_jX_j)}}
$$
\]

where $Y$ is the binary churn target, $X_j$ is an engineered predictor, and $beta_j$ is its estimated coefficient.

L2 regularisation penalises large coefficients. The inverse regularisation strength, $C$, controls the penalty: smaller values apply stronger regularisation.

The implementation uses scikit-learn's `LogisticRegression` with the `saga` solver.

## 3. Input features

The model uses the same engineered predictors and customer cohort as Model 2.

The features cover:

1. Listening behaviour for December 2016, January 2017 and February 2017.
2. Transaction activity and payment characteristics.
3. Customer demographic and registration characteristics.

The customer identifier `msno` is excluded from the predictors. The target `is_churn` is used as the classification label.

The feature-engineering window is December 2016 to February 2017. The exact temporal relationship between this period and the competition's churn-label construction has not been independently established. Consequently, the model is not claimed to be conclusively free of temporal leakage.

The feature definitions and engineering procedures are documented in the project's data preparation documentation and source code.

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

The inverse regularisation strength \(C\) was evaluated using the following candidates:

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

Average precision summarises precision across recall increments and is used as the primary selection metric because churn is the minority class.

### 5.3 Classification threshold

The classification threshold was selected by maximising F1 on the validation partition.

Where multiple thresholds achieved the same F1, the threshold with the highest precision was selected.

The resulting threshold was applied unchanged to the test partition. Test labels were not used for hyperparameter or threshold selection in the final evaluation procedure.

## 6. Evaluation metrics

The model was evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Precision | Proportion of predicted churners who actually churned |
| Recall | Proportion of actual churners detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ranking discrimination across classification thresholds |
| Average precision | Summary of precision-recall performance |
| Balanced accuracy | Average of sensitivity and specificity |
| Log loss | Quality of predicted probabilities |
| Confusion matrix | Counts of true negatives, false positives, false negatives and true positives |

Accuracy is not used as the primary selection metric because 93.61% of the cohort belongs to the non-churn class.

The definitions and final performance results are presented in [Model1Performance.MD](Model1Performance.MD).

## 7. Model selection results

The final converged run produced the following validation ranking results:

| C | Validation average precision | Validation ROC-AUC |
|---:|---:|---:|
| 0.1 | 0.722036 | 0.967736 |
| 1.0 | 0.723098 | 0.967762 |
| 10.0 | 0.722838 | 0.967750 |

The selected regularisation strength was \(C=1.0\), which achieved the highest validation average precision of 0.723098.

The selected classification threshold was 0.981076, determined by the validation F1 procedure described in Section 5.3.

## 8. Final model performance

The final converged model achieved the following results:

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.671042 | 0.659078 |
| Recall | 0.719462 | 0.711555 |
| F1-score | 0.694409 | 0.684312 |
| Balanced accuracy | 0.847688 | 0.843211 |
| ROC-AUC | 0.967762 | 0.966862 |
| Average precision | 0.723098 | 0.710419 |
| Log loss | 0.323401 | 0.329232 |

The test confusion matrix was:

| | Predicted non-churn | Predicted churn |
|---|---:|---:|
| Actual non-churn | 135,916 | 3,504 |
| Actual churn | 2,746 | 6,774 |

The final test results are documented and interpreted in [Model1Performance.MD](Model1Performance.MD).

## 9. Convergence and model validation

The final implementation uses the `saga` solver with `max_iter=2000` and `tol=0.001`.

The final run completed without convergence warnings across the evaluated regularisation candidates.

This addresses the convergence warnings encountered in earlier development runs. The final reported results correspond to the converged run rather than the superseded development results.

## 10. Implementation and execution

Implementation: `src/model1_logistic_regression.py`

Run from the repository root:

```powershell
python src/model1_logistic_regression.py
```

The script loads the prepared feature dataset and shared split assignments, trains candidate pipelines, selects the regularisation strength using validation average precision, selects the classification threshold using validation F1, and evaluates the selected candidate on the test partition.

The script uses the shared preprocessing functions in `src/preprocessing.py`.

## 11. Saved outputs

| File | Description |
|---|---|
| `models/logistic_regression.joblib` | Fitted preprocessing and Logistic Regression pipeline |
| `experiments/results/model1_metrics.json` | Hyperparameter results and validation/test metrics |
| `experiments/results/model1_test_predictions.csv` | Test customer IDs, actual labels and predicted probabilities/classifications |
| `experiments/results/model1_coefficients.csv` | Fitted coefficients and absolute coefficient magnitudes |
| `experiments/results/model1_configuration.json` | Model configuration and selected parameters |

The coefficients describe the fitted model on the transformed feature scale. Their magnitudes should not automatically be interpreted as causal effects.

## 12. Limitations

1. Logistic Regression models a linear relationship between transformed predictors and the log-odds of churn.
2. One-hot encoding does not automatically capture complex interactions between customer features.
3. Class weighting changes the training objective and can affect probability calibration.
4. The competition's label-generation timing has not been independently verified against the feature cutoff.
5. The practical value of the model depends on the costs of false positives and false negatives, which have not been specified.

The same customer partitions, predictor set and evaluation metrics are used for XGBoost to support a consistent comparison.